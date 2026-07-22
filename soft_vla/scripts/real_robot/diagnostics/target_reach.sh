/home/cao/miniconda3/envs/soft_vla_cuda/bin/python \
  soft_vla/scripts/real_robot/diagnostics/debug_single_point_target_real.py \
  --hardware-enabled \
  --ip 192.168.140.1 \
  --rigid-body-id 1 \
  --port /dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0 \
  --packet-channels 16 \
  --target-delta 0.001,0,0,0,0,0 \
  --duration-s 5 \
  --feedforward pressure_model \
  --feedback none \
  --pressure-scale 1 \
  --delta-tcp-scale 1.0 \
  --log-jsonl /home/cao/skill_learning_soft_robot/soft_vla/artifacts/real_robot/soft_vla_single_point.jsonl \
  --plot-path /home/cao/skill_learning_soft_robot/soft_vla/artifacts/real_robot/soft_vla_single_point.png