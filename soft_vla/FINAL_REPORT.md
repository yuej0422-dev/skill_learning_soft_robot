# Current SmolVLA Report

Date: 2026-07-03

## Status

The current active pipeline is based on:

```text
/home/yuej/skill_learning_soft_robot/lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp
```

This dataset is usable for local SmolVLA fine-tuning with `video_backend: pyav`.

## Dataset

- Episodes: `103`
- Frames: `39059`
- FPS: `10`
- State: `13D`
- Action: `7D`
- Images: three RGB cameras
- Action type: delta TCP plus binary gripper
- Gripper: `u_paw2`, converted to `0 = closed`, `1 = open`

## Training Smoke

Command:

```bash
cd /home/yuej/skill_learning_soft_robot/soft_vla
/home/yuej/miniconda3/envs/soft_vla_cuda/bin/python scripts/train.py \
  --config configs/smolvla_real_7_03_1_full_finetune_smoke.yaml \
  --overwrite
```

Result:

- Mode: full-parameter fine-tuning
- Steps: `2`
- Trainable parameters: `402737376`
- Total parameters: `450046176`
- Trainable ratio: `0.8948801200346161`
- Peak training GPU memory: `4.99719762802124 GiB`
- Loss step 1: `0.8205887675285339`
- Loss step 2: `0.7135990858078003`
- Weight update: passed
- Updated parameter: `model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight`
- Max absolute parameter difference: `0.00018310546875`

Checkpoint:

```text
outputs/smolvla_real_7_03_1_full_finetune_smoke/checkpoints/last/pretrained_model
```

## Inference Smoke

Command:

```bash
cd /home/yuej/skill_learning_soft_robot/soft_vla
/home/yuej/miniconda3/envs/soft_vla_cuda/bin/python scripts/offline_inference.py \
  --config configs/inference_real_7_03_1_smoke.yaml
```

Result:

- Frames: `3`
- Action chunk shape: `[1, 50, 7]`
- Mean latency: `227.62550354003906 ms`
- P95 latency: `334.092333984375 ms`
- Peak inference GPU memory: `0.9292097091674805 GiB`
- Dry run: `true`

This is a software smoke test, not evidence of task success or convergence.

## Important Implementation Notes

- The default `torchcodec` video backend failed in this environment.
- The active training and inference configs use `video_backend: pyav`.
- Gripper normalization is identity for action dimension `6`; TCP action dimensions still use mean/std normalization.
- Depth sidecars are ignored by the current SmolVLA path.

## Archive

Historical reports and outputs were moved to:

```text
archive/legacy_before_7_03_smolvla_smoke
```

Current work should reference this report and the active configs above.
