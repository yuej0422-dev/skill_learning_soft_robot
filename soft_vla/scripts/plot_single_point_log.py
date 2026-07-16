from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from soft_vla.real_robot.single_point_plot import save_single_point_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot single-point real/mock debug log.")
    parser.add_argument("--log-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frequency", type=float, default=50.0)
    parser.add_argument("--run", choices=["latest", "all"], default="latest")
    args = parser.parse_args()
    output = save_single_point_plot(args.log_jsonl, args.output, frequency=args.frequency, run=args.run)
    print(output)


if __name__ == "__main__":
    main()
