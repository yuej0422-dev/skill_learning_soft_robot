from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import sys as _sys
from pathlib import Path as _Path

_DATA_COLLECTION_DIR = _Path(__file__).resolve().parent
_REPO_ROOT = _DATA_COLLECTION_DIR.parent
_SOFT_VLA_SRC = _REPO_ROOT / "soft_vla" / "src"
if str(_SOFT_VLA_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SOFT_VLA_SRC))

from soft_vla.motion_control.reference_generator import gripper_open_to_pressure4
from soft_vla.real_robot.pressure_driver import (
    SerialPressureDriver,
    SerialPressureDriverConfig,
    resolve_default_serial_port,
)
from soft_vla.real_robot.robot_io import LuMoStateSource, LuMoStateSourceConfig
from soft_vla.runtime.timing import PeriodicTimer


DEFAULT_AMPLITUDES = 0.45 * np.asarray(
    [0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.75, 1.9, 2.0] * 2,
    dtype=np.float64,
)
MOTION_CHANNELS = 12
PACKET_CHANNELS = 16
GRIPPER_CLOSED = 0.0
PHYSICAL_PRESSURE_MAX = 3.0


@dataclass(frozen=True)
class ExcitationParameters:
    frequency: np.ndarray
    ratio: np.ndarray
    phase: np.ndarray


def parse_index_spec(spec: str, *, upper_bound: int) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            fields = part.split(":")
            if len(fields) not in (2, 3):
                raise ValueError(f"invalid index range: {part}")
            start = int(fields[0]) if fields[0] else 0
            stop = int(fields[1]) if fields[1] else upper_bound
            step = int(fields[2]) if len(fields) == 3 and fields[2] else 1
            values.extend(range(start, stop, step))
        else:
            values.append(int(part))
    if not values:
        raise ValueError("index selection is empty")
    if len(set(values)) != len(values):
        raise ValueError(f"index selection contains duplicates: {spec}")
    if min(values) < 0 or max(values) >= upper_bound:
        raise ValueError(f"indices must be in [0, {upper_bound}), got {values}")
    return values


def sample_excitation_parameters(amplitude: float, rng: np.random.Generator) -> ExcitationParameters:
    frequency = np.clip(rng.random(6), 0.1, 0.4)
    ratio = np.clip(float(amplitude) * rng.random(4), 0.5 * float(amplitude), 1.0)
    phase_pair = 6.0 * rng.random(4) - 3.0
    opposite_phase = 2.0 * rng.random(2) - 1.0
    segment_phase = 3.0 * rng.random(8) - 1.5

    base_phase = np.asarray(
        [
            phase_pair[0],
            phase_pair[1],
            phase_pair[0] + opposite_phase[0] + np.pi,
            phase_pair[1] + opposite_phase[1] + np.pi,
        ],
        dtype=np.float64,
    )
    phase = np.concatenate(
        [
            base_phase,
            base_phase + segment_phase[:4],
            base_phase + segment_phase[4:],
        ]
    )
    return ExcitationParameters(frequency=frequency, ratio=ratio, phase=phase)


def excitation_motion_norm12(
    term: int,
    step: int,
    sample_period_s: float,
    parameters: ExcitationParameters,
    *,
    noise_lambda: float,
) -> np.ndarray:
    """Reproduce the eight excitation cases in Seg3_Collect_Orig_Data.py."""
    if term < 0 or term >= 8:
        raise ValueError(f"term must be in [0, 8), got {term}")

    wave_case = term % 4
    low_variation = term >= 4
    values = np.empty(MOTION_CHANNELS, dtype=np.float64)
    for channel in range(MOTION_CHANNELS):
        local = channel % 4
        segment = channel // 4
        frequency = parameters.frequency[2 * segment + (local % 2)]
        phase = parameters.phase[channel]
        theta = frequency * np.pi * float(step) * float(sample_period_s) + phase
        use_cos = wave_case == 3 or (wave_case == 1 and local % 2 == 0) or (wave_case == 2 and local % 2 == 1)
        wave = np.cos if use_cos else np.sin
        ratio = parameters.ratio[local]
        if low_variation:
            values[channel] = ratio * (float(noise_lambda) * wave(theta) + wave(phase) + 0.98)
        else:
            values[channel] = ratio * (wave(theta) + 0.98)
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def build_pressure_packet(motion_norm12: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    motion = np.asarray(motion_norm12, dtype=np.float32).reshape(-1)
    if motion.shape != (MOTION_CHANNELS,):
        raise ValueError(f"motion pressure must have shape (12,), got {motion.shape}")
    if not np.all(np.isfinite(motion)):
        raise ValueError("motion pressure contains NaN or Inf")
    motion = np.clip(motion, 0.0, 1.0)
    gripper_physical4 = gripper_open_to_pressure4(GRIPPER_CLOSED)
    physical16 = np.concatenate([motion * PHYSICAL_PRESSURE_MAX, gripper_physical4]).astype(np.float32)
    norm16 = physical16 / PHYSICAL_PRESSURE_MAX
    return norm16.astype(np.float32), physical16


def collect_episode(
    *,
    state_source: LuMoStateSource,
    pressure_driver: SerialPressureDriver,
    repeat_index: int,
    term: int,
    amplitude: float,
    steps: int,
    frequency_hz: float,
    noise_lambda: float,
    parameters: ExcitationParameters,
) -> dict[str, np.ndarray]:
    state12 = np.empty((steps, 12), dtype=np.float32)
    pressure_norm16 = np.empty((steps, PACKET_CHANNELS), dtype=np.float32)
    pressure_physical16 = np.empty((steps, PACKET_CHANNELS), dtype=np.float32)
    command_time_ns = np.empty(steps, dtype=np.int64)
    state_time_ns = np.empty(steps, dtype=np.int64)
    overrun = np.zeros(steps, dtype=np.uint8)

    timer = PeriodicTimer(frequency_hz)
    sample_period_s = 1.0 / float(frequency_hz)
    for step in range(steps):
        # Record x_t immediately before applying u_t so each MAT row is
        # correctly aligned for the Koopman transition x_{t+1}=f(x_t, u_t).
        measured = state_source.read_state(blocking=True)
        motion = excitation_motion_norm12(
            term,
            step,
            sample_period_s,
            parameters,
            noise_lambda=noise_lambda,
        )
        norm16, physical16 = build_pressure_packet(motion)
        command_time_ns[step] = time.monotonic_ns()
        pressure_driver.send_physical(physical16)
        overrun[step] = np.uint8(not timer.wait_next())

        state12[step] = measured.state12
        state_time_ns[step] = measured.monotonic_ns
        pressure_norm16[step] = norm16
        pressure_physical16[step] = physical16

    # X/U can be consumed directly by the reference Koopman collect_data().
    # U remains 12-D because the final four fixed gripper channels contain no
    # system-identification excitation. The full transmitted packet is retained.
    return {
        "X": state12,
        "U": pressure_norm16[:, :MOTION_CHANNELS],
        "U16": pressure_norm16,
        "state12": state12,
        "position": state12[:, 0:3],
        "angle": state12[:, 3:6],
        "pos_vel": state12[:, 6:9],
        "angle_vel": state12[:, 9:12],
        "action": pressure_norm16[:, :MOTION_CHANNELS],
        "action16": pressure_norm16,
        "pressure_norm16": pressure_norm16,
        "pressure_physical16": pressure_physical16,
        "gripper_pressure4": pressure_physical16[:, 12:16],
        "command_monotonic_ns": command_time_ns,
        "state_monotonic_ns": state_time_ns,
        "sample_time_s": (state_time_ns - state_time_ns[0]).astype(np.float64) / 1e9,
        "timing_overrun": overrun,
        "repeat_index": np.asarray([[repeat_index]], dtype=np.int32),
        "term": np.asarray([[term]], dtype=np.int32),
        "amplitude": np.asarray([[amplitude]], dtype=np.float64),
        "frequency_hz": np.asarray([[frequency_hz]], dtype=np.float64),
        "noise_lambda": np.asarray([[noise_lambda]], dtype=np.float64),
        "excitation_frequency": parameters.frequency.reshape(1, -1),
        "excitation_ratio": parameters.ratio.reshape(1, -1),
        "excitation_phase": parameters.phase.reshape(1, -1),
    }


def save_mat(path: Path, data: dict[str, np.ndarray]) -> None:
    scipy_io = require_scipy_io()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    scipy_io.savemat(temporary_path, data, do_compression=True, appendmat=False)
    temporary_path.replace(path)


def require_scipy_io():
    try:
        import scipy.io
    except ImportError as exc:  # pragma: no cover - deployment environment dependency
        raise RuntimeError(
            "scipy is required to save Koopman .mat files. Install it before enabling hardware."
        ) from exc
    return scipy.io


def existing_episode_count(output_dir: Path) -> int:
    pattern = re.compile(r"soft_robot_data_(\d{4})\.mat$")
    indices = sorted(
        int(match.group(1))
        for path in output_dir.glob("soft_robot_data_*.mat")
        if (match := pattern.fullmatch(path.name)) is not None
    )
    if indices != list(range(len(indices))):
        raise ValueError(
            "existing episode files must be contiguous from soft_robot_data_0000.mat; "
            f"found indices={indices}"
        )
    return len(indices)


def resume_config_errors(existing: dict, requested: dict) -> list[str]:
    keys = (
        "format_version",
        "state_layout",
        "pressure_layout",
        "koopman_layout",
        "row_alignment",
        "terms",
        "amplitude_indices",
        "amplitudes",
        "steps_per_episode",
        "frequency_hz",
        "noise_lambda",
        "seed",
        "lumo",
    )
    errors = []
    for key in keys:
        if existing.get(key) != requested.get(key):
            errors.append(f"{key}: existing={existing.get(key)!r}, requested={requested.get(key)!r}")
    existing_repeats = int(existing.get("repeats", 1))
    requested_repeats = int(requested.get("repeats", 1))
    if requested_repeats < existing_repeats:
        errors.append(
            f"repeats cannot decrease when resuming: existing={existing_repeats}, requested={requested_repeats}"
        )
    existing_serial = existing.get("serial", {})
    requested_serial = requested.get("serial", {})
    for key in ("baudrate", "packet_channels"):
        if existing_serial.get(key) != requested_serial.get(key):
            errors.append(
                f"serial.{key}: existing={existing_serial.get(key)!r}, requested={requested_serial.get(key)!r}"
            )
    return errors


def build_manifest(args: argparse.Namespace, terms: Sequence[int], amplitude_indices: Sequence[int]) -> dict:
    return {
        "format_version": 1,
        "source_reference": str(Path(__file__).resolve().parents[2] / "Seg3_Collect_Orig_Data.py"),
        "state_layout": ["x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz", "wx", "wy", "wz"],
        "pressure_layout": {
            "motion_channels": "0:12",
            "gripper_channels": "12:16",
            "gripper_closed_physical": [0.0, 3.0, 0.0, 0.0],
            "physical_pressure_max": PHYSICAL_PRESSURE_MAX,
        },
        "koopman_layout": {"X": "state12", "U": "pressure_norm16[:, 0:12]", "full_packet": "U16"},
        "row_alignment": "state X[t] is measured immediately before pressure U[t] is sent",
        "terms": list(terms),
        "amplitude_indices": list(amplitude_indices),
        "amplitudes": DEFAULT_AMPLITUDES[list(amplitude_indices)].tolist(),
        "repeats": int(args.repeats),
        "steps_per_episode": int(args.steps),
        "frequency_hz": float(args.frequency),
        "noise_lambda": float(args.noise_lambda),
        "seed": int(args.seed),
        "lumo": {"ip": args.ip, "rigid_body_id": int(args.rigid_body_id)},
        "serial": {"port": args.port, "baudrate": int(args.baudrate), "packet_channels": PACKET_CHANNELS},
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect 50 Hz LuMo state and 16-channel pressure packets for Koopman training."
    )
    parser.add_argument("--hardware-enabled", action="store_true", help="Required before opening LuMo and pressure serial.")
    parser.add_argument("--preview-only", action="store_true", help="Validate selections and print the manifest without hardware.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a matching interrupted run, skipping contiguous existing episode files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DATA_COLLECTION_DIR / "Collected_Data" / "koopman_pressure16",
    )
    parser.add_argument("--ip", default="192.168.140.1")
    parser.add_argument("--rigid-body-id", type=int, default=1)
    parser.add_argument("--receive-timeout-ms", type=int, default=1000)
    parser.add_argument("--port", default=resolve_default_serial_port())
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--frequency", type=float, default=50.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat the complete term/amplitude grid; default selections produce 160 episodes per repeat.",
    )
    parser.add_argument("--terms", default="0:8", help="Excitation cases, e.g. 0:8 or 0,4.")
    parser.add_argument("--amplitude-indices", default="0:20", help="Indices into the reference 20-value amplitude list.")
    parser.add_argument("--noise-lambda", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.frequency <= 0:
        parser.error("--frequency must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    terms = parse_index_spec(args.terms, upper_bound=8)
    amplitude_indices = parse_index_spec(args.amplitude_indices, upper_bound=len(DEFAULT_AMPLITUDES))
    manifest = build_manifest(args, terms, amplitude_indices)

    if args.preview_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if not args.hardware_enabled:
        raise SystemExit("Refusing to actuate the robot without --hardware-enabled. Use --preview-only to inspect the run.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    existing_count = existing_episode_count(output_dir)
    existing_manifest = None
    if args.resume:
        if manifest_path.exists():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            errors = resume_config_errors(existing_manifest, manifest)
            if errors:
                raise SystemExit("Cannot resume because collection settings changed:\n- " + "\n- ".join(errors))
        elif existing_count:
            raise SystemExit("Cannot resume existing MAT files without manifest.json.")
    elif manifest_path.exists() or existing_count:
        raise SystemExit(
            f"Output directory already contains a collection: {output_dir}. "
            "Use --resume with identical settings, or choose a new --output-dir."
        )

    total = args.repeats * len(terms) * len(amplitude_indices)
    if existing_count > total:
        raise SystemExit(f"Found {existing_count} episodes but the requested plan contains only {total}.")
    if existing_count == total:
        print(f"collection already complete: episodes={total} output_dir={output_dir}")
        return

    # Fail before connecting either hardware device if MAT output is unavailable.
    require_scipy_io()
    extending_repeats = (
        existing_manifest is not None and int(existing_manifest.get("repeats", 1)) < args.repeats
    )
    if not manifest_path.exists() or extending_repeats:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    state_source = LuMoStateSource(
        LuMoStateSourceConfig(
            ip=args.ip,
            rigid_body_id=args.rigid_body_id,
            receive_timeout_ms=args.receive_timeout_ms,
        )
    )
    pressure_driver = SerialPressureDriver(
        SerialPressureDriverConfig(
            port=args.port,
            baudrate=args.baudrate,
            packet_channels=PACKET_CHANNELS,
        )
    )
    rng = np.random.default_rng(args.seed)
    episode_index = 0
    if existing_count:
        print(f"resuming: completed={existing_count} remaining={total - existing_count}", flush=True)
    state_source.open()
    try:
        pressure_driver.open()
        try:
            for repeat_index in range(args.repeats):
                for term in terms:
                    for amplitude_index in amplitude_indices:
                        amplitude = float(DEFAULT_AMPLITUDES[amplitude_index])
                        # Always advance the seeded generator, including skipped
                        # episodes, so resumed runs preserve the original sequence.
                        parameters = sample_excitation_parameters(amplitude, rng)
                        if episode_index < existing_count:
                            episode_index += 1
                            continue
                        print(
                            f"[{episode_index + 1}/{total}] repeat={repeat_index + 1}/{args.repeats} "
                            f"term={term} amplitude_index={amplitude_index} amplitude={amplitude:.3f}",
                            flush=True,
                        )
                        data = collect_episode(
                            state_source=state_source,
                            pressure_driver=pressure_driver,
                            repeat_index=repeat_index,
                            term=term,
                            amplitude=amplitude,
                            steps=args.steps,
                            frequency_hz=args.frequency,
                            noise_lambda=args.noise_lambda,
                            parameters=parameters,
                        )
                        path = output_dir / f"soft_robot_data_{episode_index:04d}.mat"
                        save_mat(path, data)
                        print(
                            f"saved={path} samples={args.steps} overruns={int(data['timing_overrun'].sum())}",
                            flush=True,
                        )
                        episode_index += 1
        finally:
            # SerialPressureDriver.close() sends a full 16-channel zero packet.
            pressure_driver.close()
    finally:
        state_source.close()

    print(f"collection complete: episodes={episode_index} output_dir={output_dir}")


if __name__ == "__main__":
    main()
