# Current Architecture

## Data Flow

```text
robot_records_7_03_1_delta_tcp LeRobotDataset
  -> LeRobotDataset / make_dataset with video_backend=pyav
  -> SmolVLA preprocessor
  -> SmolVLA full-parameter fine-tuning
  -> checkpoint
  -> offline inference dry run
```

## Active Dataset Contract

- `observation.images.cam_1`: RGB video
- `observation.images.cam_2`: RGB video
- `observation.images.cam_3`: RGB video
- `observation.state`: 13D TCP pose, velocity, gripper-open state
- `action`: 7D delta TCP plus gripper target

Action convention:

```text
action[0:6] = delta TCP
action[6] = gripper target, 0 closed, 1 open
```

## Current Training Path

Entry point:

```text
scripts/train.py
```

Active config:

```text
configs/smolvla_real_7_03_1_full_finetune_smoke.yaml
```

The training helper passes `video_backend` into `DatasetConfig`, which is required locally because `torchcodec` is not working in the current environment.

## Current Inference Path

Entry point:

```text
scripts/offline_inference.py
```

Active config:

```text
configs/inference_real_7_03_1_smoke.yaml
```

The replay source also accepts `video_backend`, so checkpoint-loaded inference uses the same image decoding backend as training.

## Historical Material

Older synthetic and 5-episode validation artifacts are archived under:

```text
archive/legacy_before_7_03_smolvla_smoke
```
