# Current Roadmap

## Immediate

1. Re-run the same smoke command on the AutoDL server after copying the dataset and project.
2. Increase `steps` from `2` to a small real run, for example `100` or `300`.
3. Keep `video_backend: pyav` unless the remote environment has a verified working `torchcodec`.
4. Use full-parameter training first if GPU memory allows; fall back to expert/action-head training only if memory or speed is unacceptable.

## Recommended Remote Training Defaults

- Dataset: `/home/yuej/skill_learning_soft_robot/lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp`
- Config base: `configs/smolvla_real_7_03_1_full_finetune_smoke.yaml`
- Batch size: `1`
- AMP: `true`
- Gripper action normalization: identity
- TCP action normalization: mean/std
- Save frequency: match final step for short runs

## After Remote Smoke

1. Save the remote `train_summary.json`.
2. Run offline inference with `configs/inference_real_7_03_1_smoke.yaml`.
3. Compare training loss, GPU memory, checkpoint size, and first action chunk shape against the local smoke result.
