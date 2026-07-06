from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()

from soft_vla.config import load_yaml
from soft_vla.schemas import GRIPPER_ACTION_INDEX, GRIPPER_STATE_INDEX
from soft_vla.training.gripper import extract_dataset_arrays, find_transition_indices, transition_window_mask


IMAGE_KEYS = ["observation.images.main", "observation.images.wrist_left", "observation.images.wrist_right"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.real_records.yaml")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=8)
    return parser.parse_args()


def to_numpy(value) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def image_from_sample(sample: dict, key: str) -> Image.Image:
    arr = to_numpy(sample[key])
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def save_transition_sample(dataset, indices: list[int], out_path: Path) -> None:
    thumbs = []
    labels = []
    for idx in indices:
        if idx < 0 or idx >= len(dataset):
            continue
        sample = dataset[idx]
        row_imgs = [image_from_sample(sample, key).resize((128, 128)) for key in IMAGE_KEYS]
        state_g = float(to_numpy(sample["observation.state"]).reshape(-1)[GRIPPER_STATE_INDEX])
        action_g = float(to_numpy(sample["action"]).reshape(-1)[GRIPPER_ACTION_INDEX])
        strip = Image.new("RGB", (128 * len(row_imgs), 150), (255, 255, 255))
        for i, img in enumerate(row_imgs):
            strip.paste(img, (128 * i, 0))
        draw = ImageDraw.Draw(strip)
        draw.text((4, 132), f"idx={idx} state={state_g:.0f} action={action_g:.0f}", fill=(0, 0, 0))
        thumbs.append(strip)
        labels.append(str(idx))
    if not thumbs:
        return
    mosaic = Image.new("RGB", (thumbs[0].width, thumbs[0].height * len(thumbs)), (255, 255, 255))
    for i, thumb in enumerate(thumbs):
        mosaic.paste(thumb, (0, i * thumb.height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(out_path)


def run(config_path: str, chunk_size: int, window: int, max_samples: int) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cfg = load_yaml(PROJECT_ROOT / config_path)["dataset"]
    root = Path(cfg["root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    dataset = LeRobotDataset(repo_id=cfg["repo_id"], root=root)
    actions, episodes, frames, _indices = extract_dataset_arrays(dataset)
    states = []
    for i in range(len(dataset)):
        states.append(to_numpy(dataset[i]["observation.state"]).astype(np.float32))
    states_np = np.stack(states)
    gripper = actions[:, GRIPPER_ACTION_INDEX]
    transitions = find_transition_indices(actions, episodes)
    transition_rows = []
    for idx in transitions:
        direction = "open_to_closed" if gripper[idx - 1] == 0 and gripper[idx] == 1 else "closed_to_open"
        lo = max(0, idx - window)
        hi = min(len(actions), idx + window + 1)
        transition_rows.append(
            {
                "dataset_index": int(idx),
                "episode_index": int(episodes[idx]),
                "frame_index": int(frames[idx]),
                "direction": direction,
                "before_action": float(gripper[idx - 1]),
                "after_action": float(gripper[idx]),
                "tcp_delta_mean_abs_window": np.mean(np.abs(actions[lo:hi, :6]), axis=0).tolist(),
                "image_frame_indices": list(range(int(frames[lo]), int(frames[hi - 1]) + 1)),
            }
        )

    segments = []
    if len(gripper):
        start = 0
        for i in range(1, len(gripper) + 1):
            if i == len(gripper) or episodes[i] != episodes[start] or gripper[i] != gripper[start]:
                segments.append(
                    {
                        "episode_index": int(episodes[start]),
                        "value": int(gripper[start]),
                        "start_dataset_index": int(start),
                        "end_dataset_index": int(i - 1),
                        "length": int(i - start),
                    }
                )
                start = i

    chunk_has_transition = []
    for ep in sorted(set(episodes.tolist())):
        ep_idx = np.where(episodes == ep)[0]
        if len(ep_idx) < chunk_size:
            continue
        trans_set = set(transitions.tolist())
        for start in ep_idx[: len(ep_idx) - chunk_size + 1]:
            chunk_has_transition.append(any(i in trans_set for i in range(start + 1, start + chunk_size)))

    per_episode = {}
    for ep in sorted(set(episodes.tolist())):
        m = episodes == ep
        vals = gripper[m]
        ep_trans = [r for r in transition_rows if r["episode_index"] == int(ep)]
        per_episode[str(int(ep))] = {
            "frames": int(np.sum(m)),
            "open": int(np.sum(vals == 0)),
            "closed": int(np.sum(vals == 1)),
            "open_ratio": float(np.mean(vals == 0)),
            "closed_ratio": float(np.mean(vals == 1)),
            "transition_count": len(ep_trans),
            "transitions": ep_trans,
            "constant_gripper": len(np.unique(vals)) == 1,
        }

    report = {
        "dataset_root": str(root),
        "frames": int(len(actions)),
        "open_count": int(np.sum(gripper == 0)),
        "closed_count": int(np.sum(gripper == 1)),
        "open_ratio": float(np.mean(gripper == 0)),
        "closed_ratio": float(np.mean(gripper == 1)),
        "open_to_closed_count": int(sum(r["direction"] == "open_to_closed" for r in transition_rows)),
        "closed_to_open_count": int(sum(r["direction"] == "closed_to_open" for r in transition_rows)),
        "transition_count": int(len(transition_rows)),
        "per_episode": per_episode,
        "open_segment_lengths": [s["length"] for s in segments if s["value"] == 0],
        "closed_segment_lengths": [s["length"] for s in segments if s["value"] == 1],
        "action_chunk_transition_ratio": float(np.mean(chunk_has_transition)) if chunk_has_transition else 0.0,
        "state_gripper_matches_action_same_frame_ratio": float(np.mean(states_np[:, GRIPPER_STATE_INDEX] == gripper)),
        "transitions": transition_rows,
    }

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "gripper_action_statistics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (reports / "gripper_transition_frames.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset_index", "episode_index", "frame_index", "direction", "before_action", "after_action"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(transition_rows)
    sample_dir = reports / "gripper_transition_samples"
    if sample_dir.exists():
        import shutil

        shutil.rmtree(sample_dir)
    for k, row in enumerate(transition_rows[:max_samples]):
        idx = row["dataset_index"]
        save_transition_sample(dataset, list(range(idx - window, idx + window + 1)), sample_dir / f"transition_{k:03d}_{row['direction']}.png")
    lines = [
        "# Gripper Action Statistics",
        "",
        f"- Frames: `{report['frames']}`",
        f"- Open count: `{report['open_count']}`",
        f"- Closed count: `{report['closed_count']}`",
        f"- Open ratio: `{report['open_ratio']}`",
        f"- Closed ratio: `{report['closed_ratio']}`",
        f"- Open to closed transitions: `{report['open_to_closed_count']}`",
        f"- Closed to open transitions: `{report['closed_to_open_count']}`",
        f"- Action chunk transition ratio: `{report['action_chunk_transition_ratio']}`",
        f"- Same-frame state/action gripper match ratio: `{report['state_gripper_matches_action_same_frame_ratio']}`",
        "",
        "## Per Episode",
        "",
    ]
    for ep, item in per_episode.items():
        lines.append(
            f"- Episode {ep}: frames=`{item['frames']}`, open_ratio=`{item['open_ratio']}`, "
            f"closed_ratio=`{item['closed_ratio']}`, transitions=`{item['transition_count']}`, "
            f"constant_gripper=`{item['constant_gripper']}`"
        )
    lines.extend(["", "## Transition Samples", "", f"- Directory: `{sample_dir}`", ""])
    (reports / "gripper_action_statistics.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    report = run(args.config, args.chunk_size, args.window, args.max_samples)
    print(json.dumps({k: report[k] for k in ["frames", "open_count", "closed_count", "transition_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
