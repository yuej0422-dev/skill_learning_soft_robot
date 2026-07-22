from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path

_COMPONENTS_DIR = _Path(__file__).resolve().parents[1] / "components"
if str(_COMPONENTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_COMPONENTS_DIR))

from bootstrap import add_src_to_path

add_src_to_path()

from soft_vla.motion_control.controller_runtime import MotionControlRuntime
from soft_vla.motion_control.feedforward_adapters import (
    AwacFeedforwardAdapter,
    AwacFeedforwardConfig,
    FeedforwardPressureConfig,
    FeedforwardPressureMLPAdapter,
    ZeroFeedforwardPolicy,
)
from soft_vla.motion_control.feedback_controllers import (
    IntegralFeedbackConfig,
    IntegralFeedbackController,
    load_fixed_gain,
    make_integral_lqr_q_weights,
    solve_integral_lqr,
)
from soft_vla.motion_control.koopman_adapter import KoopmanAdapter, KoopmanAdapterConfig
from soft_vla.motion_control.reference_generator import ReferenceGenerator, ReferenceGeneratorConfig
from soft_vla.real_robot.safety_manager import SafetyLimits, SafetyManager
from soft_vla.runtime.shared_state import UpperAction


def load_episode(root: Path, episode_index: int) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required. Use the soft_vla_cuda conda env.") from exc
    rows: list[dict] = []
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"])
        data = table.to_pydict()
        for i, ep in enumerate(data["episode_index"]):
            if int(ep) != episode_index:
                continue
            rows.append(
                {
                    "frame_index": int(data["frame_index"][i]),
                    "timestamp": float(data["timestamp"][i]),
                    "state": np.asarray(data["observation.state"][i], dtype=np.float32),
                    "action": np.asarray(data["action"][i], dtype=np.float32),
                }
            )
    rows.sort(key=lambda item: item["frame_index"])
    return rows


def build_feedback(
    kind: str,
    fixed_k_path: Path | None,
    *,
    koopman: KoopmanAdapter | None,
    feedback_gain_scale: float,
    max_integral_error: float,
    q_tcp6_weight: float,
    q_state_tail_weight: float,
    q_latent_weight: float,
    q_integral_weight: float,
    r_weight: float,
):
    if kind == "none":
        return None
    if koopman is None:
        raise ValueError("koopman adapter is required for feedback dry-run")
    C = koopman.output_matrix(6)
    if kind == "fixed_k_integral" and fixed_k_path is not None:
        K = load_fixed_gain(fixed_k_path)
    elif kind == "fixed_k_integral":
        raise ValueError("--fixed-k-path is required for fixed_k_integral")
    else:
        q_weights = make_integral_lqr_q_weights(
            n_koopman=koopman.n_koopman,
            ny=6,
            tcp6_weight=q_tcp6_weight,
            state_tail_weight=q_state_tail_weight,
            latent_weight=q_latent_weight,
            integral_weight=q_integral_weight,
        )
        K, _, _ = solve_integral_lqr(koopman.A_lift, koopman.B, C, q_weights=q_weights, r_weight=r_weight)
    return IntegralFeedbackController(
        K=K,
        C=C,
        config=IntegralFeedbackConfig(
            ny=6,
            max_integral_error=max_integral_error,
            feedback_gain_scale=feedback_gain_scale,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run episode replay through motion-control plumbing without hardware.")
    parser.add_argument("--dataset-root", type=Path, default=Path("lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp"))
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--feedforward", choices=["zero", "pressure_model", "awac"], default="zero")
    parser.add_argument("--pressure-checkpoint", type=Path, default=Path("motion_control_training/feedforward_pressure/runs/optimized_state12_raw_pressure/best.pt"))
    parser.add_argument("--awac-checkpoint", type=Path, default=Path("motion_control_training/KORL/runs/feedforward/awac_quadq_2k_eval_2x256/best.pt"))
    parser.add_argument("--feedback", choices=["none", "integral_lqr", "fixed_k_integral"], default="none")
    parser.add_argument(
        "--koopman-checkpoint",
        type=Path,
        default=Path("motion_control_training/koopman/runs/robot_records_7_03_1_delta_tcp_10hz_to_50hz_k50_epoch1500_wandb_online_20260706_2159/best.pt"),
    )
    parser.add_argument("--fixed-k-path", type=Path, default=None)
    parser.add_argument("--delta-tcp-scale", type=float, default=0.2)
    parser.add_argument("--pressure-scale", type=float, default=0.5)
    parser.add_argument("--feedback-gain-scale", type=float, default=0.2)
    parser.add_argument("--max-integral-error", type=float, default=0.5)
    parser.add_argument("--q-tcp6-weight", type=float, default=1.0)
    parser.add_argument("--q-state-tail-weight", type=float, default=0.1)
    parser.add_argument("--q-latent-weight", type=float, default=0.1)
    parser.add_argument("--q-integral-weight", type=float, default=0.5)
    parser.add_argument("--r-weight", type=float, default=10.0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = load_episode(args.dataset_root, args.episode_index)
    if not rows:
        raise SystemExit(f"episode {args.episode_index} not found")
    rows = rows[: args.max_frames]
    reference_generator = ReferenceGenerator(ReferenceGeneratorConfig(delta_tcp_scale=args.delta_tcp_scale))
    if args.feedforward == "pressure_model":
        feedforward = FeedforwardPressureMLPAdapter(
            FeedforwardPressureConfig(checkpoint=args.pressure_checkpoint, device=args.device, input_mode="target_state")
        )
    elif args.feedforward == "awac":
        feedforward = AwacFeedforwardAdapter(AwacFeedforwardConfig(checkpoint=args.awac_checkpoint, device=args.device))
    else:
        feedforward = ZeroFeedforwardPolicy()
    koopman = None
    if args.feedback != "none":
        koopman = KoopmanAdapter(KoopmanAdapterConfig(checkpoint=args.koopman_checkpoint, device=args.device))
    feedback = build_feedback(
        args.feedback,
        args.fixed_k_path,
        koopman=koopman,
        feedback_gain_scale=args.feedback_gain_scale,
        max_integral_error=args.max_integral_error,
        q_tcp6_weight=args.q_tcp6_weight,
        q_state_tail_weight=args.q_state_tail_weight,
        q_latent_weight=args.q_latent_weight,
        q_integral_weight=args.q_integral_weight,
        r_weight=args.r_weight,
    )
    runtime = MotionControlRuntime(
        feedforward=feedforward,
        feedback=feedback,
        safety=SafetyManager(SafetyLimits(slew_rate_physical_per_s=None)),
    )

    records = []
    t0 = time.perf_counter()
    for upper_step, row in enumerate(rows):
        state12 = row["state"][:12]
        action = UpperAction(
            delta_tcp6=row["action"][:6],
            gripper_open=float(row["action"][6]),
            upper_step=upper_step,
            timestamp=row["timestamp"],
            frame_index=row["frame_index"],
            episode_index=args.episode_index,
            source="episode_replay",
        )
        segment = reference_generator.build(current_state12=state12, action=action)
        for substep in range(segment.reference_states12.shape[0]):
            ref = segment.reference_for_substep(substep)
            lifted_error = None if feedback is None else koopman.tracking_error(state12, ref)
            cmd = runtime.compute(
                current_state12=state12,
                reference_state12=ref,
                delta_tcp6=action.delta_tcp6,
                gripper_open=action.gripper_open,
                lifted_error=lifted_error,
                pressure_scale=args.pressure_scale,
            )
            records.append(
                {
                    "upper_step": upper_step,
                    "substep": substep,
                    "frame_index": row["frame_index"],
                    "timestamp": row["timestamp"],
                    "delta_tcp": action.delta_tcp6.tolist(),
                    "reference_state": ref.tolist(),
                    "pressure_norm": cmd.motion_norm12.tolist(),
                    "pressure_physical": cmd.final_physical.tolist(),
                    "integral_state": [] if feedback is None else feedback.q.tolist(),
                    "safety_flags": list(cmd.safety_flags),
                }
            )
    summary = {
        "episode_index": args.episode_index,
        "upper_frames": len(rows),
        "control_steps": len(records),
        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        "max_pressure_physical": float(np.max([max(r["pressure_physical"]) for r in records])) if records else 0.0,
        "feedforward": args.feedforward,
        "feedback": args.feedback,
        "q_tcp6_weight": args.q_tcp6_weight,
        "q_state_tail_weight": args.q_state_tail_weight,
        "q_latent_weight": args.q_latent_weight,
        "q_integral_weight": args.q_integral_weight,
        "r_weight": args.r_weight,
        "safety_flags": sorted({flag for r in records for flag in r["safety_flags"]}),
    }
    result = {"summary": summary, "records": records}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["empty"])
            writer.writeheader()
            for record in records:
                writer.writerow(record)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
