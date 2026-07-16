# Full-A History Koopman v2

This experiment trains a history-context Deep Koopman model with a full
trainable `A` matrix. The legacy rollout loss is preserved: `linear_loss` is
computed on the full lifted state, and `pred_loss` is computed on the physical
state slice.

The original Koopman files are not modified:

```text
motion_control_training/koopman/model.py
motion_control_training/koopman/train_koopman_lerobot.py
```

## Current Data

Default dataset:

```text
/home/yuej/skill_learning_soft_robot/data_collection/koopman_pressure16
```

The trainer supports `--dataset-format auto|mat|lerobot`. For this MAT dataset:

```text
state_key = X
action_key = U
n_state = 12
u_dim = 12
frequency_hz = 50
upsample_factor = 1
```

State is normalized with mean/std computed from the selected MAT state columns.
Action is not normalized; `U[:, 0:12]` is already in `[0, 1]`.

The MAT reader does not require scipy. It reads numeric MATLAB v5 variables
directly, including compressed variables written by `scipy.io.savemat`.

## Data Check

Checked on the current dataset:

```text
mat_files = 480
episode_len = 400 for every episode
selected state shape = [400, 12]
selected action shape = [400, 12]
action range = [0, 1]
NaN/Inf = none
timing_overrun_sum = 0
zero_velocity_rows_lt_1e-8 = 0
identical_consecutive_state_rows_lt_1e-9 = 0
sample_dt_p50 = 0.0200029 s
sample_dt_p95 = 0.0202436 s
```

Selected state global ranges:

```text
min = [-0.1626, 0.5789, -0.0573, -0.5071, -0.2532, -0.7324,
       -0.4248, -0.1217, -0.2896, -7.1639, -3.6887, -4.5415]
max = [ 0.2067, 0.8325,  0.2169,  0.8025,  0.2238,  1.0698,
        0.3871, 0.2336,  0.3976,  9.0839,  3.4357,  4.9037]
```

## Timing

No interpolation or action hold expansion is used:

```text
source_hz = 50
target_hz = 50
upsample_factor = 1
upsample_method = none
```

For prediction from `t` to `t+1`:

```text
state history  = x[t-history_steps+1], ..., x[t]
action history = u[t-history_steps], ..., u[t-1]
current action = u[t]
```

Current context:

```python
context_t = concat(
    flatten(x[t-history_steps+1:t+1]),
    flatten(u[t-history_steps:t]),
)
```

So `u[t]` is not in `context_t`, but enters the transition and the next
context. No `x[t+1]` or `u[t+1]` leaks into `context_t`.

## Default Hyperparameters

Current defaults are tuned for the 50 Hz MAT dataset:

```text
history_steps = 30
ksteps = 50
encode_dim = 36
hidden_sizes = 512,512,256,128
batch_size = 4096
eval_batch_size = 4096
lr = 3e-4
gamma = 0.99
grad_clip = 1.0
buffer_mode = lazy_window_dataset
precompute_contexts = true
```

Dimensions with defaults:

```text
context_dim = 30 * 12 + 30 * 12 = 720
z_dim = n_state + encode_dim = 48
A = [48, 48]
B = [12, 48]
```

## Buffer Memory

Training uses a lazy window dataset by default. It stores normalized episode
arrays, window indices, and one context cache per episode instead of
materializing every `[K+1, context_dim]` rollout window.

For the full current MAT dataset:

```text
train_windows = 122880
val_windows = 30720
train lazy storage = 404 MB
val lazy storage = 101 MB
```

This replaces the older full materialized buffer, which could peak at tens of
GB on CPU because every rollout window was expanded before training.

Use this only if the machine is extremely memory constrained:

```bash
--no-precompute-contexts
```

That lowers CPU memory further, but batch loading is slower. Keep
`--num-workers 0` unless you intentionally want worker processes to hold their
own dataset copies.

Loss weights:

```text
koopman_lam = 10.0
pred_lam = 1.0
stability_lam = 0.01
std_lam = 0.1
identity_lam = 0.0001
svd_lam = 0.0
augment_lam = 0.0
```

## Loss

The rollout is fully open-loop. Only the initial lifted state is built from
real data:

```python
z_current = net.encode(current_state_sequence[:, 0], context_sequence[:, 0])
for i in range(ksteps):
    z_current = net.forward(z_current, control_sequence[:, i])
```

Full lifted-state linear loss:

```python
true_next_phi = net.encode_only(context_sequence[:, i + 1])
z_target = torch.cat([true_next_state, true_next_phi], dim=-1)
linear_loss += beta * mse(z_current, z_target)
```

Physical-state prediction loss:

```python
pred_loss += beta * mse(z_current[:, :n_state], true_next_state)
```

Both losses are gamma-weighted and divided by `beta_sum`.

Total loss:

```text
loss = 10.0 * linear_loss
     + 1.0 * pred_loss
     + 0.01 * stability_loss
     + 0.1 * std_loss
     + 0.0001 * identity_loss
```

## Train

Short smoke test:

```bash
conda run -n soft_vla_cuda python motion_control_training/koopman/experiments/fullA_history_v2/train_fullA_history.py \
  --dataset-root /home/yuej/skill_learning_soft_robot/data_collection/koopman_pressure16 \
  --dataset-format mat \
  --source-hz 50 \
  --target-hz 50 \
  --upsample-factor 1 \
  --history-steps 30 \
  --ksteps 50 \
  --encode-dim 36 \
  --hidden-sizes 512,512,256,128 \
  --batch-size 4096 \
  --eval-batch-size 4096 \
  --epochs 2 \
  --max-train-windows 4096 \
  --max-val-windows 2048 \
  --log-every 1 \
  --patience 0 \
  --device auto \
  --run-name smoke_mat50_hist30_edim36_batch4096_lazy
```

Smoke result:

```text
device = cuda
train_windows = 4096
val_windows = 2048
epoch 1 val_loss = 3.76243329
epoch 2 val_loss = 3.35702872
best_epoch = 2
```

Output:

```text
motion_control_training/koopman/experiments/fullA_history_v2/runs/smoke_mat50_hist30_edim36_batch4096_lazy
```

Full-window one-epoch check:

```bash
conda run -n soft_vla_cuda python motion_control_training/koopman/experiments/fullA_history_v2/train_fullA_history.py \
  --dataset-root /home/yuej/skill_learning_soft_robot/data_collection/koopman_pressure16 \
  --dataset-format mat \
  --source-hz 50 \
  --target-hz 50 \
  --upsample-factor 1 \
  --history-steps 30 \
  --ksteps 50 \
  --encode-dim 36 \
  --hidden-sizes 512,512,256,128 \
  --batch-size 4096 \
  --eval-batch-size 4096 \
  --epochs 1 \
  --log-every 1 \
  --patience 0 \
  --device auto \
  --run-name smoke_mat50_hist30_edim36_batch4096_full_lazy_1ep
```

Result on the current 4090 machine:

```text
device = cuda
train_windows = 122880
val_windows = 30720
epoch 1 val_loss = 1.47831705
epoch_seconds = 23.07
```

## Tests

Run:

```bash
conda run -n soft_vla_cuda python motion_control_training/koopman/experiments/fullA_history_v2/test_fullA_history.py
```

Result:

```text
All Full-A history v2 tests passed.
```

Covered:

```text
history timing and no cross-episode windows
Full-A model shapes and gradients
linear_loss zero/positive behavior
target-std loss
checkpoint save/load
real MAT loader check when the dataset is present
```
