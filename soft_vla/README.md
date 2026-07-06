# soft_vla Current Entry

This project now treats the 2026-07-03 delta-TCP LeRobot dataset as the active training source.

## Current Dataset

- Root: `/home/yuej/skill_learning_soft_robot/lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp`
- Format: LeRobotDataset v3.0
- Episodes: `103`
- Frames: `39059`
- FPS: `10`
- Images: `observation.images.cam_1`, `observation.images.cam_2`, `observation.images.cam_3`
- State: `observation.state`, shape `[13]`
- Action: `action`, shape `[7]`
- Action semantics: `action[:6]` is delta TCP, `action[6]` is gripper target
- Gripper semantics: `0 = closed`, `1 = open`
- Gripper source: `u_paw2`

Depth files may exist as sidecars, but the active SmolVLA path does not use depth.

## Latest Local Result

Full-parameter SmolVLA smoke fine-tuning succeeded locally.

```bash
cd /home/yuej/skill_learning_soft_robot/soft_vla
/home/yuej/miniconda3/envs/soft_vla_cuda/bin/python scripts/train.py \
  --config configs/smolvla_real_7_03_1_full_finetune_smoke.yaml \
  --overwrite
```

Latest checkpoint:

```text
outputs/smolvla_real_7_03_1_full_finetune_smoke/checkpoints/last/pretrained_model
```

Offline inference smoke also succeeded:

```bash
cd /home/yuej/skill_learning_soft_robot/soft_vla
/home/yuej/miniconda3/envs/soft_vla_cuda/bin/python scripts/offline_inference.py \
  --config configs/inference_real_7_03_1_smoke.yaml
```

## Active Files

- Training config: `configs/smolvla_real_7_03_1_full_finetune_smoke.yaml`
- Dataset config: `configs/dataset.real_records_7_03_1_delta_tcp.yaml`
- Inference config: `configs/inference_real_7_03_1_smoke.yaml`
- Latest report: `FINAL_REPORT.md`
- Latest output: `outputs/smolvla_real_7_03_1_full_finetune_smoke`
- Latest inference output: `outputs/offline_inference_real_7_03_1_smoke`

## Archive

Older synthetic, 5-episode, gripper-comparison, chunk-execution, and deployment-bundle artifacts were moved to:

```text
archive/legacy_before_7_03_smolvla_smoke
```

Use the archive only for historical reference. For current work, use the files listed above.
