from __future__ import annotations

import importlib.util


def require_smolvla_dependencies() -> None:
    missing = [name for name in ["torch", "lerobot", "transformers"] if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "SmolVLA dependencies are missing: "
            + ", ".join(missing)
            + ". Create the environment from environment.cuda.yml or install lerobot[smolvla,peft]."
        )


def probe_smolvla_api() -> dict:
    require_smolvla_dependencies()
    import inspect
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    return {
        "SmolVLAConfig": str(inspect.signature(SmolVLAConfig)),
        "SmolVLAPolicy": str(inspect.signature(SmolVLAPolicy)),
    }

