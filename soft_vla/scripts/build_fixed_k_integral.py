from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from soft_vla.motion_control.feedback_controllers import make_integral_lqr_q_weights, save_fixed_gain, solve_integral_lqr
from soft_vla.motion_control.fulla_history_adapters import FullAHistoryKoopmanAdapter, FullAHistoryKoopmanConfig
from soft_vla.motion_control.koopman_adapter import KoopmanAdapter, KoopmanAdapterConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed integral-feedback K from a Koopman checkpoint.")
    parser.add_argument(
        "--koopman-checkpoint",
        type=Path,
        default=Path("motion_control_training/koopman/runs/robot_records_7_03_1_delta_tcp_10hz_to_50hz_k50_epoch1500_wandb_online_20260706_2159/best.pt"),
    )
    parser.add_argument(
        "--koopman-architecture",
        choices=["legacy", "fullA_history_v2"],
        default="legacy",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--q-tcp6-weight", type=float, default=1.0)
    parser.add_argument("--q-state-tail-weight", type=float, default=0.1)
    parser.add_argument("--q-latent-weight", type=float, default=0.1)
    parser.add_argument("--q-integral-weight", type=float, default=0.5)
    parser.add_argument("--r-weight", type=float, default=10.0)
    args = parser.parse_args()

    if args.koopman_architecture == "fullA_history_v2":
        koopman = FullAHistoryKoopmanAdapter(
            FullAHistoryKoopmanConfig(checkpoint=args.koopman_checkpoint, device=args.device)
        )
    else:
        koopman = KoopmanAdapter(KoopmanAdapterConfig(checkpoint=args.koopman_checkpoint, device=args.device))
    C = koopman.output_matrix(args.ny)
    q_weights = make_integral_lqr_q_weights(
        n_koopman=koopman.n_koopman,
        ny=args.ny,
        tcp6_weight=args.q_tcp6_weight,
        state_tail_weight=args.q_state_tail_weight,
        latent_weight=args.q_latent_weight,
        integral_weight=args.q_integral_weight,
    )
    K, At, Bt = solve_integral_lqr(
        koopman.A_lift,
        koopman.B,
        C,
        dt=args.dt,
        q_weights=q_weights,
        r_weight=args.r_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_fixed_gain(args.output, K, At=At, Bt=Bt)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "K_shape": list(K.shape),
                "At_shape": list(At.shape),
                "Bt_shape": list(Bt.shape),
                "koopman_checkpoint": str(args.koopman_checkpoint),
                "koopman_architecture": args.koopman_architecture,
                "ny": args.ny,
                "q_tcp6_weight": args.q_tcp6_weight,
                "q_state_tail_weight": args.q_state_tail_weight,
                "q_latent_weight": args.q_latent_weight,
                "q_integral_weight": args.q_integral_weight,
                "r_weight": args.r_weight,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
