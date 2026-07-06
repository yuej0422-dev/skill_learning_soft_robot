# Feedforward Pressure Policy

Train MLP feedforward policies from low-dimensional robot state to the first
12 raw pressure channels. The pressure targets are used directly in their
native control range; only the state input is normalized with LeRobot
`observation.state` mean/std from `meta/stats.json`.

## Final Results

| Policy | Input | Training input mode | Best val loss | Checkpoint |
|---|---|---|---:|---|
| TCP 6D target | target TCP pose `[x,y,z,rx,ry,rz]` | `observation.state[:6] + action[:6]` | `0.01460110` | `runs/tcp6_target_raw_pressure/best.pt` |
| State 12D | pose + velocity `observation.state[:12]` | `observation.state[:12]` | `0.01218724` | `runs/optimized_state12_raw_pressure/best.pt` |

The 6D policy is the quasi-static feedforward controller: pass the desired TCP
target state directly at inference time, and it outputs 12 pressure values.
The 12D policy uses velocity as extra context and has lower validation error on
this dataset.

## Train

Use the environment with `torch`, `numpy`, and `pyarrow`:

```bash
conda run -n soft_vla_cuda python motion_control_training/feedforward_pressure/train_feedforward_pressure.py \
  --state-indices 0:6 \
  --input-mode target_state \
  --run-name tcp6_target_raw_pressure
```

```bash
conda run -n soft_vla_cuda python motion_control_training/feedforward_pressure/train_feedforward_pressure.py \
  --state-indices 0:12 \
  --input-mode observation_state \
  --run-name optimized_state12_raw_pressure
```

Outputs are written to `motion_control_training/feedforward_pressure/runs/<run-name>/`.
Each run contains `best.pt`, `last.pt`, `metrics.csv`, and `config.json`.

## Infer

6D target TCP state:

```bash
conda run -n soft_vla_cuda python motion_control_training/feedforward_pressure/infer_pressure.py \
  --checkpoint motion_control_training/feedforward_pressure/runs/tcp6_target_raw_pressure/best.pt \
  --state "0.06,0.65,0.07,0.05,0.02,0.01" \
  --clip-min 0 --clip-max 1
```

12D pose + velocity state:

```bash
conda run -n soft_vla_cuda python motion_control_training/feedforward_pressure/infer_pressure.py \
  --checkpoint motion_control_training/feedforward_pressure/runs/optimized_state12_raw_pressure/best.pt \
  --state "0.06,0.65,0.07,0.05,0.02,0.01,0,0,0,0,0,0" \
  --clip-min 0 --clip-max 1
```

Both policies also accept a full LeRobot state with extra trailing dimensions;
the selected state indices are sliced automatically before normalization.
