# Data collection

这里集中保存实物数据采集入口和本地采集结果，与 `soft_vla` 的训练、部署脚本分开。

## Koopman 气压数据采集

从仓库根目录运行：

```bash
bash data_collection/collect_koopman_data.sh
```

- `collect_koopman_data.sh`：硬件参数和采集流程入口。
- `collect_koopman_pressure_data.py`：16 路气压与 LuMo 状态采集实现。
- `Collected_Data/koopman_pressure16/`：默认输出目录，属于本地数据，不提交 Git。

可以通过环境变量覆盖串口、输出目录和采样参数，例如：

```bash
PORT=/dev/ttyUSB0 OUTPUT_DIR=/path/to/output bash data_collection/collect_koopman_data.sh
```
