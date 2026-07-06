from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()

from soft_vla.data.dataset_inspector import inspect_dataset, write_reports
from soft_vla.data.synthetic import SyntheticConfig, generate_dataset, save_sample_mosaics
from soft_vla.schemas import lerobot_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/synthetic_soft_robot_vla")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--frames-per-episode", type=int, default=40)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    repo_id = args.repo_id or f"local/{output_dir.name}"
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Re-run with --overwrite to replace it.")
        shutil.rmtree(output_dir)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        raise RuntimeError(
            "LeRobot is required to create an official LeRobotDataset. "
            "Use the existing /home/yuej/miniconda3/envs/lerobot_v3_convert/bin/python "
            "or create environment.cuda.yml."
        ) from exc

    config = SyntheticConfig(
        episodes=args.episodes,
        frames_per_episode=args.frames_per_episode,
        fps=args.fps,
        image_height=args.image_height,
        image_width=args.image_width,
        seed=args.seed,
    )
    episodes = generate_dataset(config)
    features = lerobot_features(args.image_height, args.image_width, use_videos=args.use_videos)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=args.fps,
        features=features,
        root=output_dir,
        robot_type="soft_robot_synthetic",
        use_videos=args.use_videos,
        image_writer_processes=0,
        image_writer_threads=0,
        batch_encoding_size=1,
    )

    for episode in episodes:
        for frame in episode:
            payload = {
                **frame.images,
                "observation.state": frame.state,
                "action": frame.action,
                "task": frame.task,
            }
            dataset.add_frame(payload)
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()

    reports_dir = PROJECT_ROOT / "reports"
    save_sample_mosaics(episodes, reports_dir / "synthetic_dataset_samples")
    result = inspect_dataset(output_dir, repo_id=repo_id, expected_episodes=args.episodes)
    write_reports(result, reports_dir)

    resolved_config = {
        "dataset": {
            "source": "local",
            "root": str(output_dir.relative_to(PROJECT_ROOT) if output_dir.is_relative_to(PROJECT_ROOT) else output_dir),
            "repo_id": repo_id,
            "fps": args.fps,
            "episodes": args.episodes,
            "frames_per_episode": args.frames_per_episode,
            "image_keys": [
                "observation.images.main",
                "observation.images.wrist_left",
                "observation.images.wrist_right",
            ],
            "state_key": "observation.state",
            "action_key": "action",
            "task_key": "task",
        }
    }
    (PROJECT_ROOT / "configs" / "dataset.synthetic.resolved.json").write_text(
        json.dumps(resolved_config, indent=2), encoding="utf-8"
    )
    print(f"created: {output_dir}")
    print(f"inspection_ok: {result.ok}")
    if result.errors:
        print("errors:")
        for error in result.errors:
            print(f"- {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
