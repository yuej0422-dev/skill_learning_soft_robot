# Offline Inference SmolVLA

- Checkpoint: `/home/yuej/skill_learning_soft_robot/soft_vla/outputs/smolvla_real_7_03_1_pressure_state_crop_cam1_sigmoid_gripper_smoke/checkpoints/last/pretrained_model`
- Policy type: `smolvla`
- Episode index: `0`
- Frames: `3`
- Action chunk shape: `[1, 50, 19]`
- Mean latency ms: `235.2166493733724`
- P95 latency ms: `353.2808837890625`
- Peak GPU memory GiB: `0.9265170097351074`
- Overall MAE: `0.02758067660033703`
- Overall RMSE: `0.11174867302179337`
- Per-dimension MAE: `[0.0008467739098705351, 0.0009129443787969649, 0.0018487594788894057, 0.0028549947310239077, 0.001196356606669724, 0.009934104047715664, 0.4867610037326813, 0.005383269861340523, 0.0011397396447136998, 0.0022905918303877115, 0.0019925932865589857, 0.0012066890485584736, 0.0011845496483147144, 0.0019893913995474577, 0.0013285070890560746, 0.0006583582144230604, 0.0008786955731920898, 0.0008261001785285771, 0.0007994251209311187]`
- Gripper prediction values after safety filter: `[0.500116765499115, 0.5102718472480774, 0.5293283462524414]`
- Gripper postprocess: `sigmoid(action[6])`

This is offline action fitting error on replay data, not real task success rate.
