# Real 7/03/1 SmolVLA Smoke

Dataset:

```text
/home/yuej/skill_learning_soft_robot/lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp
```

Training command:

```bash
cd /home/yuej/skill_learning_soft_robot/soft_vla
/home/yuej/miniconda3/envs/soft_vla_cuda/bin/python scripts/train.py \
  --config configs/smolvla_real_7_03_1_full_finetune_smoke.yaml \
  --overwrite
```

Training result:

- Full-parameter fine-tuning: passed
- Steps: `2`
- Peak GPU memory: `4.99719762802124 GiB`
- Checkpoint: `outputs/smolvla_real_7_03_1_full_finetune_smoke/checkpoints/last/pretrained_model`
- Weight update: passed

Offline inference command:

```bash
cd /home/yuej/skill_learning_soft_robot/soft_vla
/home/yuej/miniconda3/envs/soft_vla_cuda/bin/python scripts/offline_inference.py \
  --config configs/inference_real_7_03_1_smoke.yaml
```

Inference result:

- Frames: `3`
- Action chunk shape: `[1, 50, 7]`
- Dry run: `true`
