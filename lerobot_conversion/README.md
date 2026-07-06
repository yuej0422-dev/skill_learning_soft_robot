# LeRobot 转换项目

当前目录维护软体机器人三相机数据到 LeRobotDataset v3 的转换流程。主数据集已经转换完成，位于：

```text
/home/yuej/skill_learning_soft_robot/lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp
```

## 当前数据集

原始数据：

```text
/home/yuej/skill_learning_soft_robot/data_collection/robot_records_7_03_1
```

正式配置：

```text
lerobot_conversion/conversion_config_7_03_1_delta_tcp.yaml
```

转换结果：

- 103 episodes
- 39059 frames
- 10 FPS
- task: `pick up the apple and place it on the electronic scale`
- 三路 RGB video: `observation.images.cam_1/cam_2/cam_3`
- `observation.state`: 13 维，12D TCP state + `gripper_open`
- `action`: 7 维，6D delta TCP pose + `gripper_open`
- depth 不保存 raw `.npy`，只保留两路 preview video sidecar
- raw pressure 额外 sidecar：16 维 `u_p1..u_p12 + u_paw1..u_paw4`

夹爪映射：

```text
u_paw2 == 0 -> gripper_open = 1
u_paw2 == 3 -> gripper_open = 0
```

最后一帧没有下一帧时，delta TCP action 置 0。

## 输出结构

```text
outputs/robot_records_7_03_1_delta_tcp/
  data/                         # LeRobot parquet
  videos/                       # LeRobot RGB videos
  depth_videos/                 # depth preview sidecar, 2 cameras x 103 episodes
  raw_pressure/                 # raw pressure sidecar, 103 npy files
  meta/
    stats.json
    extra/
      depth_index.parquet
      depth_metadata.json
      depth_stats.json
      raw_pressure_index.parquet
      raw_pressure_metadata.json
      raw_pressure_stats.json
  source_to_lerobot_mapping.json
  conversion_report.json
  validation_report.json
  action_outlier_fix_report.json
```

`raw_pressure/episode_*.npy` 每帧 shape 为 `(16,)`，列顺序记录在：

```text
meta/extra/raw_pressure_metadata.json
```

通过 `episode_index + frame_index` 与主数据和 depth sidecar 对齐。

## 环境

```bash
cd /home/yuej/skill_learning_soft_robot
conda activate lerobot_v3_convert
```

如果环境不存在，按项目依赖安装 LeRobot、PyAV、PyArrow、NumPy、Pillow、PyYAML、tqdm 等包。

## 全量转换

正式转换命令：

```bash
python lerobot_conversion/convert_zips_to_lerobot_v3.py \
  --config lerobot_conversion/conversion_config_7_03_1_delta_tcp.yaml \
  --repair-zero-velocity-dropouts \
  --save-raw-pressure-sidecar \
  --overwrite
```

该命令会一次性重新生成：

- 主 LeRobot parquet 和 RGB videos
- depth video sidecar，不保存 depth raw `.npy`
- velocity 全 0 掉点段的 state 插值补全，并重算 delta TCP action
- 16 维 raw pressure sidecar

修补报告写入：

```text
action_state_dropout_repair_report.json
```

raw pressure 生成内容：

```text
raw_pressure/episode_000000.npy ...
meta/extra/raw_pressure_index.parquet
meta/extra/raw_pressure_metadata.json
meta/extra/raw_pressure_stats.json
```

配置文件里默认关闭这两个增强项，正式跑时推荐用上面的 CLI 参数显式开启：

```yaml
postprocess.zero_velocity_dropout_repair.enabled: false
sidecars.raw_pressure.enabled: false
```

给已有数据补 raw pressure sidecar 时，可单独使用：

```bash
python lerobot_conversion/add_raw_pressure_sidecar.py \
  --root lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp \
  --source-root data_collection/robot_records_7_03_1 \
  --overwrite
```

## 子集测试

正式全量转换前可以先跑 1/4 数据：

```bash
python lerobot_conversion/convert_zips_to_lerobot_v3.py \
  --config lerobot_conversion/conversion_config_7_03_1_delta_tcp.yaml \
  --output-root lerobot_conversion/outputs/test_integrated_quarter_delta_tcp \
  --repo-id local/test_integrated_quarter_delta_tcp \
  --max-episodes 26 \
  --repair-zero-velocity-dropouts \
  --save-raw-pressure-sidecar \
  --overwrite
```

## 验证

```bash
python lerobot_conversion/validate_lerobot_v3.py \
  --root lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp \
  --repo-id local/soft_robot_7_03_1_delta_tcp \
  --check-depth-alignment \
  --check-video-decode \
  --check-quantiles
```

验证内容：

- LeRobotDataset 可以加载
- `data` 行数与 episode metadata 一致
- 三路 RGB video 可以解码
- depth video sidecar 索引存在且对齐
- `observation.state` shape 为 `[13]`
- `action` shape 为 `[7]`
- `meta/stats.json` 中 q01/q10/q50/q90/q99 正常

raw pressure 对齐可用脚本抽查：

```bash
python - <<'PY'
import csv, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

root = Path("lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp")
cols = json.loads((root / "meta/extra/raw_pressure_metadata.json").read_text())["columns"]
index = pq.read_table(root / "meta/extra/raw_pressure_index.parquet").to_pylist()
for pos in [0, len(index) // 2, len(index) - 1]:
    row = index[pos]
    vec = np.load(root / row["raw_pressure_path"])[int(row["raw_array_index"])]
    csv_path = Path("data_collection/robot_records_7_03_1") / row["source_csv"]
    source_rows = list(csv.DictReader(csv_path.open()))
    expected = np.array([float(source_rows[int(row["frame_index"])][c]) for c in cols], dtype=np.float32)
    print(pos, np.array_equal(vec, expected))
PY
```

## State/Action 掉点修补

`--repair-zero-velocity-dropouts` 会在转换时处理状态 velocity 全 0 的掉点段：

- 检测 `observation.state[6:12]` 全 0 且长度大于等于配置阈值的连续区间
- 对 state 前 12 维使用邻近有效帧线性插值
- 根据修补后的 state 重新计算 6D delta TCP action
- 夹爪 state/action 仍按 `u_paw2` 映射为 0/1

```text
outputs/robot_records_7_03_1_delta_tcp/action_state_dropout_repair_report.json
```

历史上对已生成数据做过一次单独的 action 离群修补；新流程推荐在转换时直接使用上面的集成参数。

## 脚本说明

- `convert_zips_to_lerobot_v3.py`: 主转换脚本，支持 zip 和普通 episode 目录。
- `lerobot_conversion_common.py`: CSV、图像、depth、统计和通用工具。
- `validate_lerobot_v3.py`: LeRobot 数据与 sidecar 验证。
- `add_raw_pressure_sidecar.py`: 给已有数据追加 16 维 raw pressure sidecar。
- `inspect_source_schema.py`: 旧数据扫描辅助脚本。

## Legacy 配置

以下配置保留用于旧 ZIP 数据或历史批次，不属于当前 `robot_records_7_03_1_delta_tcp` 正式流程：

```text
conversion_config.yaml
conversion_config_cam2_file002_24m18s.yaml
```
