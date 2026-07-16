from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _select_run(rows: list[dict], run: str) -> list[dict]:
    if run == "all":
        return rows
    if run != "latest":
        raise ValueError(f"unsupported run selector: {run}")
    start = 0
    prev_step = rows[0].get("step", 0)
    for idx, row in enumerate(rows[1:], start=1):
        step = row.get("step", idx)
        if step <= prev_step:
            start = idx
        prev_step = step
    return rows[start:]


def load_single_point_log(path: str | Path, *, run: str = "latest") -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty single-point log: {path}")
    rows = _select_run(rows, run)
    state_key = "state" if "state" in rows[0] else "measured_state"
    reference_key = "reference" if "reference" in rows[0] else "reference_state"
    state = np.asarray([row[state_key][:12] for row in rows], dtype=np.float64)
    reference = np.asarray([row[reference_key] for row in rows], dtype=np.float64)
    if state.shape != reference.shape or state.shape[1] != 12:
        raise ValueError(f"expected state/reference shape [T,12], got {state.shape} and {reference.shape}")
    if "motion_norm12" in rows[0]:
        action12 = np.asarray([row["motion_norm12"] for row in rows], dtype=np.float64)
    else:
        action12 = np.asarray([row["pressure"][:12] for row in rows], dtype=np.float64) / 3.0
    if action12.shape != state.shape:
        raise ValueError(f"expected action shape [T,12], got {action12.shape}")
    steps = np.asarray([row.get("step", i) for i, row in enumerate(rows)], dtype=np.float64)
    return {"steps": steps, "state": state, "reference": reference, "action12": action12}


def save_single_point_plot(
    log_path: str | Path,
    output_path: str | Path,
    *,
    frequency: float = 50.0,
    run: str = "latest",
) -> Path:
    data = load_single_point_log(log_path, run=run)
    steps = data["steps"]
    time_s = steps / float(frequency)
    error = data["state"] - data["reference"]
    action12 = np.clip(data["action12"], 0.0, 1.0)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    ax_xyz, ax_rot, ax_vel, ax_action = axes.ravel()

    labels_xyz = ["x", "y", "z"]
    for idx, label in enumerate(labels_xyz):
        ax_xyz.plot(time_s, error[:, idx], label=label)
    ax_xyz.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax_xyz.set_title("XYZ Tracking Error")
    ax_xyz.set_xlabel("time (s)")
    ax_xyz.set_ylabel("error (m)")
    ax_xyz.grid(True, alpha=0.3)
    ax_xyz.legend(ncol=3, fontsize=8)

    labels_rot = ["rx", "ry", "rz"]
    for offset, label in enumerate(labels_rot, start=3):
        ax_rot.plot(time_s, error[:, offset], label=label)
    ax_rot.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax_rot.set_title("Rotation XYZ Tracking Error")
    ax_rot.set_xlabel("time (s)")
    ax_rot.set_ylabel("error (rad)")
    ax_rot.grid(True, alpha=0.3)
    ax_rot.legend(ncol=3, fontsize=8)

    linear_vel_error = np.linalg.norm(error[:, 6:9], axis=1)
    angular_vel_error = np.linalg.norm(error[:, 9:12], axis=1)
    ax_vel.plot(time_s, linear_vel_error, label="linear norm")
    ax_vel.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax_vel.set_title("Velocity Tracking Error")
    ax_vel.set_xlabel("time (s)")
    ax_vel.set_ylabel("linear error (m/s)")
    ax_vel.grid(True, alpha=0.3)
    ax_vel_ang = ax_vel.twinx()
    ax_vel_ang.plot(time_s, angular_vel_error, linestyle="--", color="tab:orange", label="angular norm")
    ax_vel_ang.set_ylabel("angular error (rad/s)")
    lines, labels = ax_vel.get_legend_handles_labels()
    lines_ang, labels_ang = ax_vel_ang.get_legend_handles_labels()
    ax_vel.legend(lines + lines_ang, labels + labels_ang, ncol=3, fontsize=8)

    for idx in range(12):
        ax_action.plot(time_s, action12[:, idx], label=f"u{idx + 1}")
    ax_action.set_title("Action First 12 Channels")
    ax_action.set_xlabel("time (s)")
    ax_action.set_ylabel("normalized pressure")
    ax_action.set_ylim(-0.02, 1.02)
    ax_action.grid(True, alpha=0.3)
    ax_action.legend(ncol=4, fontsize=7)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
