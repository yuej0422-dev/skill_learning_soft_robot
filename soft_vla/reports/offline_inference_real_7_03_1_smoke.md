# Offline Inference SmolVLA

- Checkpoint: `/home/yuej/skill_learning_soft_robot/soft_vla/outputs/smolvla_real_7_03_1_full_finetune_smoke/checkpoints/last/pretrained_model`
- Policy type: `smolvla`
- Episode index: `0`
- Frames: `3`
- Action chunk shape: `[1, 50, 7]`
- Mean latency ms: `227.62550354003906`
- P95 latency ms: `334.092333984375`
- Peak GPU memory GiB: `0.9292097091674805`
- Overall MAE: `0.14648021757602692`
- Overall RMSE: `0.3780074119567871`
- Per-dimension MAE: `[0.005717812571674585, 0.0003388906770851463, 0.0014845541445538402, 0.00261492352001369, 0.003659074893221259, 0.011546239256858826, 1.0]`
- Gripper prediction values after safety filter: `[0.0]`

This is offline action fitting error on synthetic data, not real task success rate.
