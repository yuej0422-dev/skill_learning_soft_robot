from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureMapping:
    image_keys: tuple[str, ...]
    state_key: str
    action_key: str
    task_key: str = "task"


def infer_feature_mapping(features: dict, preferred_image_keys: list[str] | None = None) -> FeatureMapping:
    camera_keys = [k for k, v in features.items() if v.get("dtype") in {"image", "video"}]
    if preferred_image_keys:
        image_keys = [k for k in preferred_image_keys if k in camera_keys]
        image_keys.extend(k for k in camera_keys if k not in image_keys)
    else:
        image_keys = camera_keys

    state_key = "observation.state" if "observation.state" in features else ""
    if not state_key:
        state_candidates = [k for k in features if k.startswith("observation") and features[k].get("dtype") == "float32"]
        state_key = state_candidates[0] if state_candidates else ""

    action_key = "action" if "action" in features else ""
    if not action_key:
        action_candidates = [k for k in features if k.startswith("action")]
        action_key = action_candidates[0] if action_candidates else ""

    return FeatureMapping(tuple(image_keys), state_key, action_key)

