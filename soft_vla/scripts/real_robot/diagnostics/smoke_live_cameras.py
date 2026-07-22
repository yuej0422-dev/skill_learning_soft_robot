from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path

_COMPONENTS_DIR = _Path(__file__).resolve().parents[1] / "components"
if str(_COMPONENTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_COMPONENTS_DIR))

from bootstrap import add_src_to_path

add_src_to_path()

from soft_vla.real_robot.live_cameras import LiveCameraConfig, LiveThreeCameraSource


def main() -> None:
    parser = argparse.ArgumentParser(description="Read the live 3-camera stack and reject black/pure-color frames.")
    parser.add_argument("--zed-index", type=int, default=None)
    parser.add_argument("--zed-eye", choices=["left", "right"], default="left")
    parser.add_argument("--zed-width", type=int, default=2560)
    parser.add_argument("--zed-height", type=int, default=720)
    parser.add_argument("--zed-fps", type=int, default=30)
    parser.add_argument("--realsense-serial-cam2", default="401522072797")
    parser.add_argument("--realsense-serial-cam3", default="408322072769")
    parser.add_argument("--output-dir", type=Path, default=Path("soft_vla/artifacts/real_robot/camera_smoke"))
    parser.add_argument("--min-gray-std", type=float, default=2.0)
    parser.add_argument("--min-nonblack-fraction", type=float, default=0.05)
    parser.add_argument("--zed-warmup-usable-frames", type=int, default=10)
    parser.add_argument("--realsense-warmup-usable-frames", type=int, default=10)
    parser.add_argument("--min-realsense-mean", type=float, default=40.0)
    args = parser.parse_args()

    source = LiveThreeCameraSource(
        LiveCameraConfig(
            zed_index=args.zed_index,
            zed_eye=args.zed_eye,
            zed_width=args.zed_width,
            zed_height=args.zed_height,
            zed_fps=args.zed_fps,
            realsense_serial_cam2=args.realsense_serial_cam2,
            realsense_serial_cam3=args.realsense_serial_cam3,
            min_gray_std=args.min_gray_std,
            min_nonblack_fraction=args.min_nonblack_fraction,
            zed_warmup_usable_frames=args.zed_warmup_usable_frames,
            realsense_warmup_usable_frames=args.realsense_warmup_usable_frames,
            min_realsense_mean=args.min_realsense_mean,
        )
    )
    try:
        source.open()
        report = source.smoke_report(args.output_dir)
    finally:
        source.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
