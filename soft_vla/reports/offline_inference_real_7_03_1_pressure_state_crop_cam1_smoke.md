# Offline Inference SmolVLA

- Checkpoint: `/home/yuej/skill_learning_soft_robot/soft_vla/outputs/smolvla_real_7_03_1_pressure_state_crop_cam1_smoke/checkpoints/last/pretrained_model`
- Policy type: `smolvla`
- Episode index: `0`
- Frames: `3`
- Action chunk shape: `[1, 50, 19]`
- Mean latency ms: `227.26605733235678`
- P95 latency ms: `341.2377990722656`
- Peak GPU memory GiB: `0.9265170097351074`
- Overall MAE: `0.05378807708621025`
- Overall RMSE: `0.22942298650741577`
- Per-dimension MAE: `[0.0011389004066586494, 0.0010572168976068497, 0.0005371840670704842, 0.0014113993383944035, 0.003568575019016862, 0.004808599594980478, 1.0, 0.002263806527480483, 0.0026280090678483248, 0.0010152369504794478, 0.0010257870890200138, 0.00041252459050156176, 0.0006833905936218798, 0.00018257969350088388, 0.0003760112449526787, 0.0001775904092937708, 0.0001637734821997583, 7.36614820198156e-05, 0.0004492221924010664]`
- Gripper prediction values after safety filter: `[0.0]`

This is offline action fitting error on replay data, not real task success rate.
