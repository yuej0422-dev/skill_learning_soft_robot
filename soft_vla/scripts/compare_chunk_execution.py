from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()

from soft_vla.config import load_yaml
from soft_vla.data.replay_source import LeRobotReplaySource
from soft_vla.inference.chunk_execution import RTCUnavailableError, make_chunk_executor, probe_official_rtc
from soft_vla.inference.chunk_execution.metrics import chunk_action_metrics
from soft_vla.schemas import validate_action


DEFAULT_MODES = ["single_step", "chunk", "receding_horizon", "temporal_ensemble", "rtc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference_receding_horizon.yaml")
    parser.add_argument("--modes", nargs="*", default=DEFAULT_MODES)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def mode_config(mode: str, base: dict) -> dict:
    cfg = dict(base)
    cfg["mode"] = mode
    if mode == "single_step":
        return cfg
    if mode == "chunk":
        cfg["chunk_size"] = int(cfg.get("chunk_size", 50))
        cfg["execution_horizon"] = 10
    elif mode == "receding_horizon":
        cfg["chunk_size"] = int(cfg.get("chunk_size", 50))
        cfg["execution_horizon"] = 5
        cfg["replan_interval"] = 5
    elif mode == "temporal_ensemble":
        cfg["replan_interval"] = 1
        cfg["max_history_chunks"] = int(cfg.get("max_history_chunks", 10))
        cfg["weight_type"] = str(cfg.get("weight_type", "exponential"))
        cfg["decay"] = float(cfg.get("decay", 0.25))
        cfg["prefer_newer_predictions"] = bool(cfg.get("prefer_newer_predictions", True))
    return cfg


def load_policy(cfg: dict):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint = Path(cfg["policy"].get("checkpoint") or cfg["inference"].get("checkpoint"))
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    device = cfg["policy"].get("device", "cuda")
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
    policy.config.device = device
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor, checkpoint


def predict_chunk(policy, preprocessor, postprocessor, sample: dict) -> tuple[np.ndarray, float]:
    obs = {k: v for k, v in sample.items() if k not in {"action", "action_is_pad"}}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    batch = preprocessor(obs)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=str(policy.config.device).startswith("cuda")):
        chunk = policy.predict_action_chunk(batch)
    raw_chunk = postprocessor.process_action(chunk).detach().cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return raw_chunk, latency_ms


def run_mode(mode: str, cfg: dict, samples: list[dict], policy, preprocessor, postprocessor, out_root: Path) -> dict:
    out_dir = out_root / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "rtc":
        compat = probe_official_rtc()
        summary = {"mode": mode, "status": "NOT_RUN", "official_rtc_api": compat}
        try:
            make_chunk_executor(mode_config(mode, cfg.get("chunk_execution", {})))
        except RTCUnavailableError as exc:
            summary["rtc_error"] = str(exc)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out_dir / "timing.json").write_text(json.dumps({"rtc": summary}, indent=2), encoding="utf-8")
        return summary

    executor = make_chunk_executor(mode_config(mode, cfg.get("chunk_execution", {})))
    control_hz = float(cfg.get("control", {}).get("control_frequency_hz", 10))
    period_s = 1.0 / control_hz
    maximum_steps = int(cfg.get("control", {}).get("maximum_steps", 40))
    if cfg.get("_max_steps_override") is not None:
        maximum_steps = int(cfg["_max_steps_override"])
    maximum_steps = min(maximum_steps, len(samples))

    actions = []
    gt_actions = []
    records = []
    chunks = []
    latencies = []
    replan_steps = []
    queue_lengths = []
    for step in range(maximum_steps):
        control_ts = step * period_s
        if executor.needs_replan(step, control_ts):
            inf_start = time.perf_counter()
            chunk, latency_ms = predict_chunk(policy, preprocessor, postprocessor, samples[step])
            inf_end = inf_start + latency_ms / 1000.0
            executor.submit_chunk(chunk, observation_timestamp=float(step), inference_start_timestamp=inf_start, inference_end_timestamp=inf_end)
            chunks.append(chunk[0])
            latencies.append(latency_ms)
            replan_steps.append(step)
        rec = executor.get_action(step, control_ts)
        action = validate_action(rec.action, require_binary_gripper=False)
        gt = validate_action(np.asarray(samples[step]["action"], dtype=np.float32))
        actions.append(action)
        gt_actions.append(gt)
        device_gripper_command = int(action[6] >= 0.5)
        state = executor.get_debug_state()
        queue_lengths.append(int(state.get("queue_len", 0)))
        records.append(
            {
                "control_step": step,
                "control_timestamp": control_ts,
                "source": rec.source,
                "chunk_id": rec.chunk_id,
                "chunk_step": rec.chunk_step,
                "absolute_step": rec.absolute_step,
                "action_age_steps": rec.action_age_steps,
                "raw_gripper_output": float(action[6]),
                "postprocessed_gripper_output": float(action[6]),
                "device_gripper_command": device_gripper_command,
                "gt_gripper": float(gt[6]),
                **{f"action_{i}": float(action[i]) for i in range(7)},
                **{f"gt_{i}": float(gt[i]) for i in range(7)},
                "debug": json.dumps(rec.debug),
            }
        )

    actions_np = np.stack(actions) if actions else np.zeros((0, 7), dtype=np.float32)
    gt_np = np.stack(gt_actions) if gt_actions else np.zeros((0, 7), dtype=np.float32)
    chunks_np = np.stack(chunks) if chunks else np.zeros((0, 50, 7), dtype=np.float32)
    summary = chunk_action_metrics(actions_np, gt_np, latencies)
    summary.update(
        {
            "mode": mode,
            "status": "PASS",
            "replan_count": len(replan_steps),
            "replan_steps": replan_steps,
            "queue_min_length": int(min(queue_lengths)) if queue_lengths else 0,
            "queue_underrun_count": int(executor.get_debug_state().get("underruns", 0)),
            "expired_chunk_count": 0,
            "mean_action_age": float(np.mean([r["action_age_steps"] for r in records])) if records else 0.0,
            "max_action_age": int(max([r["action_age_steps"] for r in records])) if records else 0,
            "inferences_per_second": float(len(replan_steps) / max(1e-9, maximum_steps * period_s)),
            "hardware_enabled": False,
            "dry_run": True,
            "gripper_inference_postprocessing": "NONE; device command uses fixed threshold 0.5 only",
        }
    )
    with (out_dir / "actions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["control_step"])
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(out_dir / "chunks.npz", chunks=chunks_np, actions=actions_np, gt_actions=gt_np)
    (out_dir / "timing.json").write_text(
        json.dumps({"latencies_ms": latencies, "replan_steps": replan_steps, "control_period_s": period_s}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    cfg = load_yaml(PROJECT_ROOT / args.config)
    if args.max_steps is not None:
        cfg["_max_steps_override"] = args.max_steps
    ds_cfg = load_yaml(PROJECT_ROOT / cfg["dataset"]["config"])["dataset"]
    root = Path(cfg["dataset"].get("root") or ds_cfg["root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    source = LeRobotReplaySource(root, ds_cfg.get("repo_id"), cfg["dataset"].get("episode_index"))
    samples = list(source)
    policy, preprocessor, postprocessor, _checkpoint = load_policy(cfg)
    out_root = PROJECT_ROOT / "outputs" / "chunk_execution_comparison"
    summaries = {}
    for mode in args.modes:
        summaries[mode] = run_mode(mode, cfg, samples, policy, preprocessor, postprocessor, out_root)
    (out_root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    lines = ["# Chunk Execution Comparison", ""]
    for mode, summary in summaries.items():
        lines.append(f"## {mode}")
        lines.append("")
        for key in ["status", "frames", "overall_mae", "tcp_overall_mae", "replan_count", "queue_underrun_count"]:
            if key in summary:
                lines.append(f"- {key}: `{summary[key]}`")
        if "gripper" in summary:
            lines.append(f"- gripper metrics: `{summary['gripper']}`")
        if mode == "rtc":
            lines.append(f"- official RTC API: `{summary.get('official_rtc_api')}`")
            lines.append(f"- RTC error: `{summary.get('rtc_error')}`")
        lines.append("")
    (reports / "chunk_execution_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summaries, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
