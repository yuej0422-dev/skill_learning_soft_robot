# Full-A History Koopman v2

This experiment is a minimal, isolated extension of the legacy Koopman trainer.
It keeps the original open-loop rollout loss structure while changing only the
requested parts: 10 Hz data, history-context encoder, full `A`, stability loss,
target-std anti-collapse loss, disabled augment loss, and disabled SVD loss by
default.

The legacy files are not modified:

```text
motion_control_training/koopman/model.py
motion_control_training/koopman/train_koopman_lerobot.py
```

## Audit Of Previous Attempt

The previous `fullA_history` attempt was removed because it deviated from the
new spec:

```text
linear_loss was split into latent-only loss instead of full lifted-state loss
koopman_lam default was changed to 1.0 instead of preserving 10.0
the 50 epoch real-data run used ksteps=10 instead of required ksteps=50
metrics used state_pred/latent_linear names instead of legacy linear/pred loss
```

This v2 implementation restores the legacy `linear_loss + pred_loss` design.

## Files

```text
model_fullA_history.py   Full-A history-context Koopman model and legacy-style loss
train_fullA_history.py   10 Hz data loader, window builder, trainer, checkpoints
test_fullA_history.py    Timing, no interpolation, loss, gradient, checkpoint tests
README.md               This experiment note
```

## Data And Timing

No interpolation or action hold expansion is used:

```text
source_hz = 10
target_hz = 10
upsample_factor = 1
upsample_method = none
```

The trainer rejects `--upsample-factor` values other than `1`.

For prediction from `t` to `t+1`:

```text
state history  = x[t-9], ..., x[t]
action history = u[t-10], ..., u[t-1]
current action = u[t]
```

Current context:

```python
context_t = concat(
    flatten(x[t-9:t+1]),
    flatten(u[t-10:t]),
)
```

Next context:

```python
context_next = concat(
    flatten(x[t-8:t+2]),
    flatten(u[t-9:t+1]),
)
```

So `u[t]` is not in `context_t`, but enters `context_next`. No `x[t+1]` or
`u[t+1]` leaks into `context_t`.

Default dimensions:

```text
n_state = 12
u_dim = 12
history_steps = 10
context_dim = 240
encode_dim = 12
z_dim = 24
ksteps = 50
```

`ksteps=50` at 10 Hz is a 5 second open-loop rollout.

## Model

The lifted state is:

```text
z_t = [x_t, encoder(context_t)]
```

Full-A row-vector transition:

```text
z_next = z @ A + u @ B + bias
A: [24, 24], initialized as I + 0.001 noise
B: [12, 24], initialized as 0.01 noise
bias: [24]
```

For row-vector convention:

```python
A_latent_to_state = A[n_state:, :n_state]
A_state_to_latent = A[:n_state, n_state:]
```

## Loss

The rollout is fully open-loop. Only the initial lifted state is built from
real data:

```python
z_current = net.encode(current_state_sequence[:, 0], context_sequence[:, 0])
for i in range(ksteps):
    z_current = net.forward(z_current, control_sequence[:, i])
```

No true state is re-injected into the prediction path.

Legacy-style full lifted-state linear loss:

```python
true_next_phi = net.encode_only(context_sequence[:, i + 1])
z_target = torch.cat([true_next_state, true_next_phi], dim=-1)
linear_loss += beta * mse(z_current, z_target)
```

Physical-state prediction loss is retained separately:

```python
pred_loss += beta * mse(z_current[:, :n_state], true_next_state)
```

Both use gamma weighting and are divided by `beta_sum`.

Total loss:

```text
loss = 10.0 * linear_loss
     + 1.0 * pred_loss
     + 0.01 * stability_loss
     + 0.1 * std_loss
     + 0.0001 * identity_loss
```

Defaults:

```text
koopman_lam = 10.0
pred_lam = 1.0
stability_lam = 0.01
std_lam = 0.1
identity_lam = 0.0001
svd_lam = 0.0
augment_lam = 0.0
```

`augment_loss` is recorded as zero and not used.

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
manual timing with x[t]=t and u[t]=100+t
no interpolation: processed_frames == original_frames
no cross-episode windows
shape checks: context=240, phi=12, z=24, A=[24,24], B=[12,24]
linear_loss zero/positive behavior
Full-A gradients, including A[n_state:, :n_state]
target-std loss
checkpoint save/load for model, optimizer, metadata
```

## 50 Epoch Run

Command:

```bash
conda run -n soft_vla_cuda python motion_control_training/koopman/experiments/fullA_history_v2/train_fullA_history.py \
  --dataset-root /home/yuej/skill_learning_soft_robot/lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp \
  --epochs 50 \
  --history-steps 10 \
  --ksteps 50 \
  --encode-dim 12 \
  --hidden-sizes 128,128,64 \
  --batch-size 1024 \
  --eval-batch-size 1024 \
  --lr 3e-4 \
  --gamma 0.99 \
  --target-std 1.0 \
  --koopman-lam 10.0 \
  --pred-lam 1.0 \
  --stability-lam 0.01 \
  --std-lam 0.1 \
  --identity-lam 0.0001 \
  --svd-lam 0.0 \
  --augment-lam 0.0 \
  --spectral-radius-limit 1.0 \
  --upsample-factor 1 \
  --device auto \
  --run-name smoke_fullA_hist10_k50_50ep
```

Output:

```text
motion_control_training/koopman/experiments/fullA_history_v2/runs/smoke_fullA_hist10_k50_50ep
```

Checkpoints:

```text
best.pt
last.pt
```

Best epoch:

```text
49
```

Data:

```text
train_windows = 25999
val_windows = 6880
processed_frames == original_frames
source_hz = target_hz = 10
upsample_factor = 1
```

Metrics:

| Metric | Epoch 1 | Best Epoch 49 | Epoch 50 |
|---|---:|---:|---:|
| train_loss | 3.91341518 | 1.46089225 | 1.45766503 |
| val_loss | 3.23798229 | 1.62105027 | 1.62710107 |
| train_linear_loss | 0.32087185 | 0.12202936 | 0.12176307 |
| val_linear_loss | 0.26379427 | 0.13567135 | 0.13616821 |
| train_pred_loss | 0.61267101 | 0.23600129 | 0.23551499 |
| val_pred_loss | 0.51360438 | 0.26025789 | 0.26129077 |
| train_std_loss | 0.92025702 | 0.04597261 | 0.04519229 |
| val_std_loss | 0.86435221 | 0.04078714 | 0.04128138 |
| train_latent_std_min | 0.02073872 | 0.59095611 | 0.59546315 |
| train_latent_std_mean | 0.04097532 | 0.80429349 | 0.80556983 |
| train_latent_std_max | 0.06483263 | 0.89485729 | 0.89646169 |
| val_latent_std_min | 0.02070734 | 0.60564383 | 0.59954988 |
| val_latent_std_mean | 0.07077865 | 0.83370523 | 0.83370211 |
| val_latent_std_max | 0.12964320 | 0.97408966 | 0.97681986 |
| spectral_radius | 1.00657868 | 0.99913329 | 0.99824780 |
| A_latent_to_state_norm | 0.06298931 | 0.60481268 | 0.60834795 |

Checks:

```text
NaN/Inf: none
loss decreased: yes
encoder collapse: no, latent_std_mean rose from near 0 to about 0.83
latent participates in physical-state prediction: yes, A_latent_to_state_norm and gradient are nonzero
spectral radius controlled: yes, final spectral_radius < 1.0
overfitting: no severe split; val tracks train with a moderate gap
```
