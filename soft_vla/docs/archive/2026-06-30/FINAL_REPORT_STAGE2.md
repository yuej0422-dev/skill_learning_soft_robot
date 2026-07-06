# FINAL_REPORT_STAGE2

## Status

REAL SMOLVLA TRAINING: PASS
BACKWARD: PASS
OPTIMIZER STEP: PASS
WEIGHTS UPDATED: PASS
CHECKPOINT SAVED: PASS
PROCESSORS SAVED: PASS
CHECKPOINT RELOAD IN NEW PROCESS: PASS
TRAIN/INFERENCE PREPROCESSING PARITY: PASS
REAL SMOLVLA INFERENCE: PASS
ORACLE USED IN MODEL REPORT: NO
DRY RUN: TRUE

## Code Audit

See `reports/training_pipeline_audit.md`. Previous code only probed SmolVLA imports and used an oracle fallback for offline inference. Formal inference now defaults to `smolvla`; oracle is only available by explicit `--policy-type oracle_baseline`.

## Environment

- Python: `/home/yuej/miniconda3/envs/soft_vla_cuda/bin/python`
- LeRobot: `0.4.4`
- Torch: `2.6.0+cu124`
- CUDA available: `true`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- GPU total memory bytes: `8318484480`
- Transformers: `4.57.6`
- PEFT: `0.19.1`

## Training Mode

- Mode used: `expert-only fine-tuning`
- Pretrained model: `lerobot/smolvla_base`
- Steps: `20`
- Batch size: `1`
- AMP: `true`
- Vision encoder frozen: `true`
- Train expert only: `true`
- Train state projection: `true`
- LoRA config generated: `configs/smolvla_lora_8gb.yaml`; not used for this successful run.
- Full-parameter 16GB config generated: `configs/smolvla_full_finetune_16gb.yaml`; not launched on 8GB.

Actual command:

```bash
/home/yuej/miniconda3/envs/soft_vla_cuda/bin/python scripts/train.py --config configs/smolvla_smoke_8gb.yaml --overwrite
```

## Loss And Memory

- Loss sequence: `[0.177691, 0.396508, 0.175502, 0.186138, 0.381635, 0.172782, 0.316795, 0.292776, 0.133117, 0.124158, 0.187231, 0.096888, 0.064294, 0.077139, 0.721347, 0.369445, 0.047783, 0.140491, 0.081548, 0.065105]`
- Peak GPU memory during training GiB: `2.0368971824645996`
- Trainable parameters: `99880992`
- Total parameters: `450046176`
- Trainable ratio: `0.2219349865112508`

## Weight Update

- Parameter checked: `model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight`
- Checksum before: `-21.1252498626709`
- Checksum after: `-21.318771362304688`
- Norm before: `16.79098129272461`
- Norm after: `16.794078826904297`
- Max absolute difference: `0.001220703125`

## Checkpoint

- Checkpoint path: `outputs/smolvla_expert_smoke/checkpoints/last/pretrained_model`
- Contains state normalization stats: `True`
- Contains action normalization stats: `True`
- Contains action unnormalization: `True`
- Contains camera key/order: `True`
- Contains image resize/padding: `True`
- Contains custom crop: `False`
- Depends on dataset stats directory: `False`

Files:

- `config.json`
- `model.safetensors`
- `policy_postprocessor.json`
- `policy_postprocessor_step_0_unnormalizer_processor.safetensors`
- `policy_preprocessor.json`
- `policy_preprocessor_step_5_normalizer_processor.safetensors`
- `train_config.json`
- `training_state/optimizer_param_groups.json`
- `training_state/optimizer_state.safetensors`
- `training_state/rng_state.safetensors`
- `training_state/scheduler_state.json`
- `training_state/training_step.json`

## Normalization

- Normalization mapping: `{'VISUAL': 'IDENTITY', 'STATE': 'MEAN_STD', 'ACTION': 'MEAN_STD'}`
- State mean: `[-0.01120109  0.01457049  0.21231704 -0.00377844 -0.08432929 -0.00173457
 -0.01448736  0.0119699  -0.01323384 -0.01692368 -0.01125596 -0.01532461
  0.52708333]`
- State std: `[0.08394588 0.10465676 0.07108154 0.13814745 0.14106746 0.1408053
 0.04998578 0.05857641 0.05105406 0.11239236 0.08986587 0.10472582
 0.49926595]`
- Action mean: `[-0.00145628  0.00119715 -0.00132228 -0.00169034 -0.00112101 -0.0015268
  0.55208334]`
- Action std: `[0.00499704 0.00585789 0.00510671 0.01124028 0.00898885 0.01047588
 0.49727993]`
- State normalization max error: `1.9069688539374852e-07`
- Action normalization max error: `5.417192248557967e-06`
- Gripper state std: `0.4992659514757442`
- Gripper action std: `0.4972799331032125`

## Image Pipeline

- Camera keys/order: `['observation.images.main', 'observation.images.wrist_left', 'observation.images.wrist_right']`
- Custom crop: `false`
- Raw color space: `RGB`
- Raw image shape: `[1, 1, 3, 128, 128]`
- Raw image range: `[0.10980392247438431, 1.0]`
- SmolVLA resize/padding: `[512, 512]`
- Model image normalization: `[0,1] -> [-1,1]` inside `prepare_images`.

## Processor Parity

- State max abs error: `0.0`
- Action max abs error: `0.0`
- Image max abs error: `0.0`

## Real SmolVLA Offline Inference

- Checkpoint: `/home/yuej/skill_learning_soft_robot/soft_vla/outputs/smolvla_expert_smoke/checkpoints/last/pretrained_model`
- Frames: `40`
- Action chunk shape: `[1, 50, 7]`
- Mean latency ms: `157.7246166229248`
- Median latency ms: `152.46644592285156`
- P90 latency ms: `156.4658645629883`
- P95 latency ms: `157.98596267700194`
- Peak GPU memory GiB: `0.9117417335510254`
- Overall MAE: `0.061950378119945526`
- Overall RMSE: `0.23915070295333862`
- Per-dimension MAE: `[0.003254918847233057, 0.004212043713778257, 0.0061873323284089565, 0.008445775136351585, 0.0051802461966872215, 0.00637233629822731, 0.4000000059604645]`
- Gripper prediction values after safety filter: `[1.0]`

This is offline action fitting error on synthetic data, not real robot task success rate.

## Deployment Bundle

- Bundle: `outputs/deployment_bundle_smolvla`
- LoRA merged: `false` because this run was expert-only, not LoRA.
- Bundle offline verification: `PASS`
- Action chunk shape: `[1, 50, 7]`
- Null controller actions: `1`
- No GT action input: `True`

## Remaining Issues

- This is synthetic data only; it does not prove real soft-robot success.
- LoRA training config exists, but LoRA training was not executed because expert-only succeeded and satisfied the real training loop requirement.
- Full-parameter fine-tuning was not launched on the 8GB GPU; use the generated 16GB config on the RTX 4090 workstation.
- Real hardware execution remains disabled; all inference uses `NullRobotController` dry run.
