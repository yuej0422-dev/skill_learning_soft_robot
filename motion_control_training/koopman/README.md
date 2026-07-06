# Koopman Dynamics Training

Train a Deep Koopman dynamics model using LeRobot `observation.state[:12]`
and the first 12 raw pressure channels. State inputs are normalized with
LeRobot `meta/stats.json`; pressure inputs are kept in their native 0-1 range.
The raw LeRobot trajectories are 10 Hz; the default data pipeline upsamples
each episode to 50 Hz before constructing Koopman windows.

This implementation follows
`motion_control_training/reference/Learning_Koopman_with_Reg_HPN.py` for the
Koopman network, AB construction, and loss terms. The data loader is changed
from `.mat` files to LeRobot episodes.

## Data Flow

Source dataset:

```text
lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp
```

Inputs used for Koopman training:

```text
state   = observation.state[0:12]
control = raw_pressure[0:12]
```

The training buffer layout matches the HPN reference script:

```text
[sample, Ksteps + 1, raw_pressure_12 + normalized_state_12]
```

Before windowing, each episode is converted from 10 Hz to 50 Hz with:

```text
state:    linear interpolation between adjacent 10 Hz samples
pressure: zero-order hold; u_t is repeated for all 5 substeps until t+1
```

For original samples `x_t, x_{t+1}`, `upsample_factor=5` creates states at
fractions `0/5, 1/5, 2/5, 3/5, 4/5`, then appends the final original state.
Pressure rows use `u_t` for those 5 substeps. Unlike the original `.mat`
loader, windows are constructed inside each episode only, so no training
sequence crosses an episode boundary.

## Default Parameters

Default training parameters follow the reference script for model/loss shape,
with two engineering choices enabled by default:

```text
Ksteps          = 50
source_hz       = 10
target_hz       = 50
upsample_factor = 5
u_dim           = 12
state_dim       = 12
encode_dim      = 12
encoder layers  = [12, 64, 128, 64, 12]
Nkoopman        = 24
batch_size      = 4096
epochs          = 700
learning_rate   = 3e-4
gamma           = 0.99
grad_clip       = 1.0
train drop_last = False
```

Use `--drop-last --grad-clip 0` if you need the training loop to match the
reference script more literally.

Loss weights:

```text
koopman_lam = 10.0
a_eig_lam   = 0.003
svd_lam     = 0.003
augment_lam = 1.0
pred_lam    = 1.0
```

The model constructs trainable Koopman matrices as:

```text
z = [x, encoder(x)]
z_next = z @ A + u @ B
```

`A` is parameterized from real/complex-conjugate eigenvalue blocks just like the
reference. `B` is a trainable `[u_dim, Nkoopman]` matrix. The loss keeps the
reference terms: lifted linear consistency, state prediction, augmentation
consistency, eigenvalue regularization, and controllability SVD regularization.

At `Ksteps=50`, the current LeRobot dataset would build 26,819 train windows
without upsampling. With default 10 Hz to 50 Hz upsampling, this becomes
150,167 train windows, and `Ksteps=50` is about a 1-second horizon. With batch
size 4096 and default `drop_last=False`, one epoch has
`ceil(150167 / 4096) = 37` optimizer steps, plus validation every epoch.

## Train

```bash
conda run -n soft_vla_cuda python motion_control_training/koopman/train_koopman_lerobot.py \
  --run-name robot_records_7_03_1_state12_pressure12_k50
```

Train with WandB logging:

```bash
conda run -n soft_vla_cuda python motion_control_training/koopman/train_koopman_lerobot.py \
  --run-name robot_records_7_03_1_state12_pressure12_k50_wandb \
  --epochs 2000 \
  --patience 0 \
  --wandb \
  --wandb-project soft-robot-koopman
```

Offline WandB logging, useful on AutoDL or unstable network:

```bash
conda run -n soft_vla_cuda python motion_control_training/koopman/train_koopman_lerobot.py \
  --run-name robot_records_7_03_1_state12_pressure12_k50_wandb_offline \
  --epochs 2000 \
  --patience 0 \
  --wandb \
  --wandb-mode offline
```

Short 50-epoch check with full `Ksteps=50`:

```bash
conda run -n soft_vla_cuda python motion_control_training/koopman/train_koopman_lerobot.py \
  --run-name robot_records_7_03_1_state12_pressure12_k50_epoch50 \
  --epochs 50 \
  --log-every 5 \
  --patience 0
```

Useful quick smoke test:

```bash
conda run -n soft_vla_cuda python motion_control_training/koopman/train_koopman_lerobot.py \
  --run-name upsample50hz_smoke \
  --epochs 1 \
  --ksteps 50 \
  --max-train-windows 512 \
  --max-val-windows 256
```

Outputs are written to `motion_control_training/koopman/runs/<run-name>/`:
`best.pt`, `last.pt`, `metrics.csv`, and `config.json`.

`metrics.csv` and WandB record total loss, each component for
train/validation, learning rate, and epoch wall time:

```text
loss, linear_loss, a_eig_loss, svd_loss, augment_loss, pred_loss, lr, epoch_seconds
```

WandB logs are enabled only when `--wandb` is passed.

## Validate / Infer

Run one-step and rollout validation from a trained checkpoint:

```bash
conda run -n soft_vla_cuda python motion_control_training/koopman/validate_koopman_rollout.py \
  --checkpoint motion_control_training/koopman/runs/robot_records_7_03_1_state12_pressure12_k50/best.pt \
  --split val \
  --max-episodes 8 \
  --rollout-steps 50
```

This writes `validation_rollout.json` next to the checkpoint with normalized
and raw-state RMSE for teacher-forced one-step prediction and multi-step
rollout. New checkpoints store `upsample_factor=5`, so validation uses the same
50 Hz episode trajectories by default. Add `--save-npz` to save predicted
normalized trajectories for plotting or inspection.

Validation modes:

```text
one-step: z_t is encoded from real x_t, then predicts x_{t+1}
rollout: starts from x_0, repeatedly applies real pressure controls for N steps
```

## Timing Estimate

Benchmark command with default training loop, full 50 Hz `Ksteps=50`, full
windows, and WandB code path disabled from upload:

```bash
/usr/bin/time -p conda run -n soft_vla_cuda python motion_control_training/koopman/train_koopman_lerobot.py \
  --run-name timing_50hz_k50_1epoch_wandb_disabled \
  --epochs 1 \
  --log-every 1 \
  --patience 0 \
  --wandb \
  --wandb-mode disabled
```

Observed on RTX 4060 Laptop GPU:

```text
train_windows=150167
val_windows=39566
optimizer_steps_per_epoch=37
val_batches_per_epoch=10
epoch_seconds=7.00
process_real_seconds_for_1_epoch=12.75
```

Projected 2000-epoch training time:

```text
epoch-only estimate: 7.00 * 2000 = 14000s = 3.89h
practical estimate with startup/WandB/checkpoint overhead: about 4.0-4.5h
```

Projected 3000-epoch training time:

```text
epoch-only estimate: 7.00 * 3000 = 21000s = 5.83h
practical estimate: about 6.0-6.8h
```

## Legacy 10 Hz Strict Reference Check Run

This check was run before enabling default 10 Hz to 50 Hz upsampling.

```text
run: motion_control_training/koopman/runs/robot_records_7_03_1_state12_pressure12_k50_epoch50_refstrict
Ksteps: 50
epochs: 50
train_windows: 26819
val_windows: 7090
optimizer_steps_per_epoch: 6
elapsed_real_seconds: 62.32
```

Final training CSV row:

```text
epoch=50
train_loss=4.60342081
train_pred_loss=0.76134206
val_loss=4.71002728
val_pred_loss=0.77924297
```

Validation on 8 validation episodes with 50-step rollout:

```text
episodes=[0, 2, 4, 18, 21, 24, 25, 27]
one_step_raw_rmse_mean=0.07506914
one_step_normalized_rmse_mean=0.82711595
rollout_raw_rmse_mean=0.09088114
rollout_normalized_rmse_mean=0.98536968
```

Artifacts:

```text
best.pt
last.pt
metrics.csv
config.json
validation_rollout.json
validation_rollout.npz
```
