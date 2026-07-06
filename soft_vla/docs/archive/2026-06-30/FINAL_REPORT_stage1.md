# FINAL_REPORT

## Summary

Created a new project folder at `soft_vla/` for the soft-robot VLA pipeline.

The current interface is fixed as:

- `observation.state`: 13D.
- `state[12]`: current gripper open/close state, binary `0/1`.
- `action`: 7D.
- `action[0:6]`: TCP delta translation/rotation.
- `action[6]`: target gripper position, binary `0/1`.

## Completed

- Built project docs: `README.md`, `PLAN.md`, `ARCHITECTURE.md`.
- Added CUDA-capable environment spec: `environment.cuda.yml`.
- Added configs for synthetic data, public smoke metadata, SmolVLA 8GB/16GB, offline inference, and hardware template.
- Implemented schema validation in `src/soft_vla/schemas.py`.
- Implemented deterministic synthetic visual servoing data generation with three synchronized RGB views.
- Created official LeRobotDataset v3.0 synthetic datasets through installed LeRobot 0.4.4:
  - CI dataset: `data/synthetic_soft_robot_vla_ci`, 2 episodes x 12 frames.
  - Default smoke dataset: `data/synthetic_soft_robot_vla`, 12 episodes x 40 frames.
- Generated sample mosaics in `reports/synthetic_dataset_samples/`.
- Implemented dataset inspection and wrote:
  - `reports/synthetic_dataset_report.md`
  - `reports/synthetic_dataset_report.json`
- Implemented offline dry-run inference using replay, `SafetyFilter`, and `NullRobotController`.
- Ran offline inference on one 40-frame episode:
  - frames: 40
  - mean latency: about 0.029 ms with the oracle fallback
  - MAE/RMSE: 0.0 against the oracle action
- Added unit tests and ran them successfully with `unittest`.

## Environment Findings

The workspace root is not a git repository.

The existing environment `/home/yuej/miniconda3/envs/lerobot_v3_convert` has:

- Python 3.10.20
- LeRobot 0.4.4
- Torch 2.6.0+cpu
- NumPy/Pillow/YAML available
- `transformers` missing
- `peft` missing
- CUDA unavailable in that Torch build

The separate environment `/home/yuej/miniconda3/envs/soft_robot_state` has CUDA Torch 2.6.0+cu124, but does not have LeRobot/Transformers/PEFT installed.

A new environment was created:

- `/home/yuej/miniconda3/envs/soft_vla_cuda`
- Python 3.11.15
- Torch 2.6.0+cu124
- CUDA 12.4 available
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- visible VRAM: about 7.747 GiB
- bf16 supported: true
- LeRobot 0.4.4 installed
- Transformers 4.57.6 installed
- PEFT 0.19.1 installed

The first unconstrained pip install attempted to upgrade Torch to 2.10.0 and
was cancelled while downloading a 915MB wheel. The successful install pinned
Torch/Torchvision:

```bash
/home/yuej/miniconda3/envs/soft_vla_cuda/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0+cu124" \
  "torchvision==0.21.0+cu124" \
  "lerobot[smolvla,peft]" \
  "transformers<5" \
  peft
```

## SmolVLA Status

SmolVLA training was not faked.

In `soft_vla_cuda`, the official `SmolVLAConfig` and `SmolVLAPolicy` imports
succeeded. The probed API is recorded in `reports/smolvla_smoke_status.md`.

Full single-batch SmolVLA forward/backward was not executed yet because that
requires wiring this project to the official LeRobot trainer/model input
processor and may require downloading `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`.
The environment is now ready for that next step.

## Public Dataset Smoke

The public metadata check for `lerobot/pusht` was attempted and recorded in
`reports/public_smoke_dataset_metadata.json`. It failed with an SSL EOF while
contacting Hugging Face Hub. This does not affect the target synthetic soft-robot
pipeline.

## Verification Commands Run

```bash
/home/yuej/miniconda3/envs/lerobot_v3_convert/bin/python scripts/create_synthetic_soft_robot_dataset.py --output-dir data/synthetic_soft_robot_vla_ci --episodes 2 --frames-per-episode 12 --image-height 64 --image-width 64 --fps 10 --seed 42
/home/yuej/miniconda3/envs/lerobot_v3_convert/bin/python scripts/create_synthetic_soft_robot_dataset.py --output-dir data/synthetic_soft_robot_vla --episodes 12 --frames-per-episode 40 --image-height 128 --image-width 128 --fps 10 --seed 42
/home/yuej/miniconda3/envs/lerobot_v3_convert/bin/python scripts/inspect_dataset.py --config configs/dataset.synthetic.yaml
/home/yuej/miniconda3/envs/lerobot_v3_convert/bin/python scripts/offline_inference.py --config configs/inference_offline.yaml
PYTHONPATH=src /home/yuej/miniconda3/envs/lerobot_v3_convert/bin/python -m unittest discover -s tests -v
/home/yuej/miniconda3/envs/soft_vla_cuda/bin/python scripts/verify_installation.py
/home/yuej/miniconda3/envs/soft_vla_cuda/bin/python scripts/smoke_train.py --config configs/smolvla_smoke_8gb.yaml --single-batch
```

## Required Limitations

- Synthetic data is only for engineering pipeline verification.
- Loss decrease on synthetic data would not prove real robot performance.
- Synthetic action error does not imply real task success.
- Real soft-robot demonstration data must still be collected.
- Real data must keep the same three camera keys.
- Real data must keep the 13D state order.
- Real data must keep the 7D action semantics.
- Real camera synchronization needs separate validation.
- TCP pose and delta rotation coordinate frames must be confirmed before hardware use.
- `gripper_action` is currently a binary absolute target position; confirm again before implementing the real controller.
- Policy frequency and low-level control frequency should be decoupled.
- Real execution requires recalibrated action safety bounds.
