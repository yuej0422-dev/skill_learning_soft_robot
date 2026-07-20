from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from soft_vla.schemas import GRIPPER_ACTION_INDEX


@dataclass(frozen=True)
class GripperSamplingReport:
    raw_transition_frame_ratio: float
    weighted_transition_mass_ratio: float
    open_frame_ratio: float
    closed_frame_ratio: float
    transition_indices: list[int]
    episode_transition_counts: dict[int, int]


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def apply_hybrid_action_stats(stats: dict[str, Any], *, gripper_index: int = GRIPPER_ACTION_INDEX) -> dict[str, Any]:
    """Return stats where action gripper normalization is identity under MEAN_STD processors.

    LeRobot 0.4.4 normalizes a whole action vector using one mode. Setting the
    selected action dimension to mean=0/std=1 gives identity behavior for that
    dimension while preserving mean/std for TCP dimensions.
    """
    patched = copy.deepcopy(stats)
    action_stats = patched.get("action")
    if not isinstance(action_stats, dict):
        raise KeyError("stats must contain action statistics")
    for key, value in {"mean": 0.0, "std": 1.0}.items():
        arr = list(action_stats[key])
        if gripper_index >= len(arr):
            raise IndexError(f"gripper_index {gripper_index} out of range for action.{key}")
        arr[gripper_index] = value
        action_stats[key] = arr
    return patched


def apply_identity_stats_for_indices(stats: dict[str, Any], identity_indices: dict[str, list[int]]) -> dict[str, Any]:
    """Patch mean/std stats so selected vector dimensions pass through unchanged.

    LeRobot normalizers apply one normalization mode to a whole feature vector.
    For mixed continuous/binary vectors, setting mean=0 and std=1 for selected
    dimensions keeps those dimensions in identity form while preserving mean/std
    normalization for the remaining dimensions.
    """
    patched = copy.deepcopy(stats)
    for feature_key, indices in identity_indices.items():
        feature_stats = patched.get(feature_key)
        if not isinstance(feature_stats, dict):
            raise KeyError(f"stats must contain {feature_key} statistics")
        for stat_key, value in {"mean": 0.0, "std": 1.0}.items():
            arr = list(feature_stats[stat_key])
            for idx in indices:
                if idx >= len(arr):
                    raise IndexError(f"{feature_key}.{stat_key} index {idx} out of range for length {len(arr)}")
                arr[idx] = value
            feature_stats[stat_key] = arr
    return patched


def apply_sigmoid_to_gripper_action(actions: torch.Tensor, *, gripper_index: int = GRIPPER_ACTION_INDEX) -> torch.Tensor:
    """Treat the selected action dimension as a logit and bound it to 0..1."""
    bounded = actions.clone()
    bounded[..., gripper_index] = torch.sigmoid(bounded[..., gripper_index])
    return bounded


def extract_dataset_arrays(dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None:
        actions = np.stack([np.asarray(x, dtype=np.float32) for x in hf_dataset["action"]])
        episodes = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
        frames = np.asarray(hf_dataset["frame_index"], dtype=np.int64)
        indices = np.asarray(hf_dataset["index"], dtype=np.int64) if "index" in hf_dataset.column_names else np.arange(len(actions))
        return actions, episodes, frames, indices
    actions: list[np.ndarray] = []
    episodes: list[int] = []
    frames: list[int] = []
    indices: list[int] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        actions.append(_as_numpy(sample["action"]).astype(np.float32))
        episodes.append(int(_as_numpy(sample.get("episode_index", -1)).reshape(-1)[0]))
        frames.append(int(_as_numpy(sample.get("frame_index", i)).reshape(-1)[0]))
        indices.append(i)
    return np.stack(actions), np.asarray(episodes), np.asarray(frames), np.asarray(indices)


def find_transition_indices(actions: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    gripper = actions[:, GRIPPER_ACTION_INDEX].astype(np.float32)
    transitions: list[int] = []
    for i in range(1, len(gripper)):
        if episodes[i] != episodes[i - 1]:
            continue
        if gripper[i] != gripper[i - 1]:
            transitions.append(i)
    return np.asarray(transitions, dtype=np.int64)


def transition_window_mask(
    actions: np.ndarray,
    episodes: np.ndarray,
    *,
    before_steps: int = 5,
    after_steps: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    transition_indices = find_transition_indices(actions, episodes)
    mask = np.zeros(len(actions), dtype=bool)
    for idx in transition_indices:
        ep = episodes[idx]
        lo = max(0, idx - before_steps)
        hi = min(len(actions), idx + after_steps + 1)
        for j in range(lo, hi):
            if episodes[j] == ep:
                mask[j] = True
    return mask, transition_indices


def build_transition_weights(
    dataset,
    *,
    enabled: bool = True,
    before_steps: int = 5,
    after_steps: int = 5,
    transition_weight: float = 4.0,
    normal_weight: float = 1.0,
) -> tuple[torch.DoubleTensor, GripperSamplingReport]:
    actions, episodes, _frames, _indices = extract_dataset_arrays(dataset)
    transition_mask, transition_indices = transition_window_mask(
        actions, episodes, before_steps=before_steps, after_steps=after_steps
    )
    weights = np.full(len(actions), float(normal_weight), dtype=np.float64)
    if enabled:
        weights[transition_mask] = float(transition_weight)
    gripper = actions[:, GRIPPER_ACTION_INDEX]
    episode_transition_counts = {
        int(ep): int(np.sum(episodes[transition_indices] == ep)) for ep in sorted(set(episodes.tolist()))
    }
    report = GripperSamplingReport(
        raw_transition_frame_ratio=float(np.mean(transition_mask)) if len(transition_mask) else 0.0,
        weighted_transition_mass_ratio=float(weights[transition_mask].sum() / weights.sum()) if weights.sum() else 0.0,
        open_frame_ratio=float(np.mean(gripper == 0.0)) if len(gripper) else 0.0,
        closed_frame_ratio=float(np.mean(gripper == 1.0)) if len(gripper) else 0.0,
        transition_indices=transition_indices.astype(int).tolist(),
        episode_transition_counts=episode_transition_counts,
    )
    return torch.as_tensor(weights, dtype=torch.double), report


def smolvla_weighted_action_loss(
    policy,
    batch: dict[str, torch.Tensor],
    *,
    tcp_weight: float = 1.0,
    gripper_weight: float = 3.0,
    noise=None,
    time=None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute SmolVLA flow loss with per-action-dimension weights.

    This mirrors the public SmolVLAPolicy.forward path up to the unreduced
    `[B, T, D]` loss tensor, then applies a 7D action weight vector.
    """
    from lerobot.policies.smolvla.modeling_smolvla import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    if policy.config.adapt_to_pi_aloha:
        batch[OBS_STATE] = policy._pi_aloha_decode_state(batch[OBS_STATE])
        batch[ACTION] = policy._pi_aloha_encode_actions_inv(batch[ACTION])

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
    lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
    actions = policy.prepare_action(batch)
    losses = policy.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
    losses = losses[:, :, : policy.config.max_action_dim]

    pad = batch.get("action_is_pad")
    if pad is None:
        pad = batch.get("actions_id_pad")
    valid = None
    if pad is not None:
        valid = (~pad).to(losses.device).bool()
        losses = losses * valid.unsqueeze(-1)

    weights = torch.ones(policy.config.max_action_dim, device=losses.device, dtype=losses.dtype) * float(tcp_weight)
    weights[GRIPPER_ACTION_INDEX] = float(gripper_weight)
    weighted = losses * weights.view(1, 1, -1)

    if valid is not None:
        denom_time = valid.sum().clamp_min(1).to(losses.dtype)
        tcp_loss = losses[:, :, :GRIPPER_ACTION_INDEX].sum() / (denom_time * GRIPPER_ACTION_INDEX)
        gripper_loss = losses[:, :, GRIPPER_ACTION_INDEX].sum() / denom_time
        total_loss = weighted.sum() / (denom_time * policy.config.max_action_dim)
    else:
        tcp_loss = losses[:, :, :GRIPPER_ACTION_INDEX].mean()
        gripper_loss = losses[:, :, GRIPPER_ACTION_INDEX].mean()
        total_loss = weighted.mean()

    return total_loss, {
        "loss": float(total_loss.detach().cpu()),
        "tcp_loss": float(tcp_loss.detach().cpu()),
        "gripper_loss": float(gripper_loss.detach().cpu()),
        "weighted_gripper_loss": float((gripper_loss * float(gripper_weight)).detach().cpu()),
        "tcp_weight": float(tcp_weight),
        "gripper_weight": float(gripper_weight),
    }


def smolvla_sigmoid_gripper_loss(
    policy,
    batch: dict[str, torch.Tensor],
    *,
    gripper_index: int = GRIPPER_ACTION_INDEX,
    noise=None,
    time=None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """SmolVLA flow loss with only gripper final action passed through sigmoid.

    The non-gripper dimensions keep the standard flow-matching MSE. For the
    gripper dimension, reconstruct the final action prediction from the current
    flow step, treat it as a logit, apply sigmoid, and compare directly against
    the 0/1 target with weight 1 like every other action dimension.
    """
    import torch.nn.functional as F
    from lerobot.policies.smolvla.modeling_smolvla import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
        make_att_2d_masks,
    )

    if policy.config.adapt_to_pi_aloha:
        batch[OBS_STATE] = policy._pi_aloha_decode_state(batch[OBS_STATE])
        batch[ACTION] = policy._pi_aloha_encode_actions_inv(batch[ACTION])

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
    lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
    actions = policy.prepare_action(batch)

    if noise is None:
        noise = policy.model.sample_noise(actions.shape, actions.device)
    if time is None:
        time = policy.model.sample_time(actions.shape[0], actions.device)

    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    suffix_embs, suffix_pad_masks, suffix_att_masks = policy.model.embed_suffix(x_t, time)

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    (_, suffix_out), _ = policy.model.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        fill_kv_cache=False,
    )
    suffix_out = suffix_out[:, -policy.model.config.chunk_size :]
    suffix_out = suffix_out.to(dtype=torch.float32)
    v_t = policy.model.action_out_proj(suffix_out)

    losses = F.mse_loss(u_t, v_t, reduction="none")
    losses = losses[:, :, : policy.config.max_action_dim]
    final_action_logits = x_t - time_expanded * v_t
    gripper_pred = torch.sigmoid(final_action_logits[:, :, gripper_index])
    gripper_target = actions[:, :, gripper_index]
    gripper_losses = F.mse_loss(gripper_pred, gripper_target, reduction="none")
    losses[:, :, gripper_index] = gripper_losses

    pad = batch.get("action_is_pad")
    if pad is None:
        pad = batch.get("actions_id_pad")
    if pad is not None:
        valid = (~pad).to(losses.device).bool()
        losses = losses * valid.unsqueeze(-1)

    loss = losses.mean()
    return loss, {
        "losses_after_forward": float(losses.detach().mean().cpu()),
        "losses_after_sigmoid_gripper": float(loss.detach().cpu()),
        "loss": float(loss.detach().cpu()),
        "gripper_loss": float(gripper_losses.detach().mean().cpu()),
        "gripper_pred_min": float(gripper_pred.detach().min().cpu()),
        "gripper_pred_max": float(gripper_pred.detach().max().cpu()),
        "gripper_index": int(gripper_index),
        "gripper_weight": 1.0,
    }


def compute_gripper_metrics(pred_raw: np.ndarray, gt: np.ndarray, *, threshold: float = 0.5) -> dict[str, Any]:
    pred_raw = np.asarray(pred_raw, dtype=np.float32).reshape(-1)
    gt = np.asarray(gt, dtype=np.float32).reshape(-1)
    pred = pred_raw >= threshold
    true = gt >= threshold
    tp = int(np.sum(pred & true))
    tn = int(np.sum(~pred & ~true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    total = max(1, len(true))
    acc = (tp + tn) / total
    recall_pos = tp / max(1, tp + fn)
    recall_neg = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    recall = recall_pos
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float((recall_pos + recall_neg) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "pred_open_ratio": float(np.mean(~pred)) if len(pred) else 0.0,
        "pred_closed_ratio": float(np.mean(pred)) if len(pred) else 0.0,
        "gt_open_ratio": float(np.mean(~true)) if len(true) else 0.0,
        "gt_closed_ratio": float(np.mean(true)) if len(true) else 0.0,
        "mae": float(np.mean(np.abs(pred_raw - gt))) if len(gt) else 0.0,
        "rmse": float(np.sqrt(np.mean((pred_raw - gt) ** 2))) if len(gt) else 0.0,
    }
