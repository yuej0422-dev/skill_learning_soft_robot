# Command-line scripts

The reusable implementation lives under `src/soft_vla/`. This directory only
contains command-line entry points and task orchestration.

## Stable entry points

- `train.py`: SmolVLA training.
- `offline_inference.py`: offline inference and evaluation.
- `real_robot/`: real-robot deployment, replay, components, and diagnostics.

## Supporting tools

- `data/`: dataset conversion, generation, preparation, and inspection.
- `evaluation/`: checkpoint, gripper, inference, and execution-mode analysis.
- `diagnostics/`: installation, processor, preprocessing, and training smoke checks.

Run commands from the `soft_vla` project directory. For example:

```bash
python scripts/data/inspect_dataset.py --config configs/dataset.real_records.yaml
python scripts/evaluation/audit_checkpoint.py --checkpoint /path/to/checkpoint
python scripts/diagnostics/verify_installation.py
```
