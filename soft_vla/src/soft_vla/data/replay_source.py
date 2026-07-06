from __future__ import annotations

from pathlib import Path
from typing import Iterator


class LeRobotReplaySource:
    def __init__(
        self,
        root: str | Path,
        repo_id: str | None = None,
        episode_index: int | None = None,
        video_backend: str | None = None,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        kwargs = {"repo_id": repo_id or "local/synthetic_soft_robot_vla", "root": Path(root)}
        if video_backend is not None:
            kwargs["video_backend"] = video_backend
        self.dataset = LeRobotDataset(**kwargs)
        self.episode_index = episode_index

    def __iter__(self) -> Iterator[dict]:
        for sample in self.dataset:
            if self.episode_index is not None and int(sample.get("episode_index", -1)) != self.episode_index:
                continue
            yield sample
