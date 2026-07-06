# KORL LeRobot Training

This directory adapts the reference KORL code in `motion_control_training/reference/KORL`
to the local LeRobot dataset and raw-pressure sidecars.

## Data Layout

Both training scripts build offline RL transitions from
`lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp` by default.

Each observation is:

```text
[normalized_current_state_12, normalized_future_target_state_12]
```

The default future target uses `--target-offset 5`. The action is the 12-dim raw
pressure vector from `meta/extra/raw_pressure_metadata.json`.

## Feedforward AWAC

```bash
/home/yuej/miniconda3/envs/soft_vla_cuda/bin/python \
  motion_control_training/KORL/train_awac_feedforward.py \
  --run-name awac_1k_eval_2x256 \
  --max-timesteps 1000 \
  --eval-freq 250 \
  --batch-size 1024 \
  --hidden-sizes 256,256 \
  --device auto
```

Outputs are written under `motion_control_training/KORL/runs/feedforward/<run-name>/`.
The main validation metric is deterministic action RMSE against held-out episodes.

## Feedback KORL

Run feedback after feedforward and pass the feedforward checkpoint:

```bash
/home/yuej/miniconda3/envs/soft_vla_cuda/bin/python \
  motion_control_training/KORL/train_korl_feedback.py \
  --run-name feedback_1k_eval_2x256_bc \
  --feedforward-checkpoint motion_control_training/KORL/runs/feedforward/awac_1k_eval_2x256/best.pt \
  --max-timesteps 1000 \
  --eval-freq 250 \
  --batch-size 512 \
  --cql-n-actions 6 \
  --bc-steps 1000 \
  --device auto
```

Outputs are written under `motion_control_training/KORL/runs/feedback/<run-name>/`.
The script reports:

- `val_ff_action_rmse_mean`: actor/feedforward action RMSE.
- `val_feedback_total_action_rmse_mean`: RMSE after adding the learned feedback residual.
- `val_koopman_linear_mse`: one-step lifted Koopman linearity MSE.

## Notes From Initial Runs

The 1k-step feedforward run `awac_1k_eval_2x256` reached validation action RMSE
`0.128476`.

The 1k-step feedback run `feedback_1k_eval_2x256_bc` improved the feedforward
actor RMSE to `0.115857` and Koopman linear MSE from `0.500441` to `0.426542`.
However, the learned feedback residual worsened action-imitation RMSE in this
short run (`0.192519` total RMSE at step 1000). Treat the feedback checkpoint as
only a runnable baseline until tuned with a control-oriented validation loop.
