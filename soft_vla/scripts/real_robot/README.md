# 实物运行脚本

这里集中保存所有直接服务于软体机器人实物运行的命令。除非脚本另有说明，均从仓库根目录
`skill_learning_soft_robot` 执行。

## 目录划分

- `deploy/`：稳定的 SmolVLA 实物部署入口，日常部署只需要关注这里。
- `replay/`：episode 回放入口与回放执行器。
- `components/`：部署/回放依赖的 Python 入口，以及 K 构建、打包和校验工具。
- `diagnostics/`：仅保留 LuMo、气压、相机、手柄、安全空载和人工介入 dry-run 等关键检查。

Koopman 气压数据采集已集中到仓库根目录的 `data_collection/`，与本地采集数据放在一起。

## 正式部署入口

```bash
bash soft_vla/scripts/real_robot/deploy/smolvla_deploy.sh
bash soft_vla/scripts/real_robot/deploy/smolvla_deploy_human_intervention.sh
bash soft_vla/scripts/real_robot/deploy/smolvla_pressure_state_deploy.sh
bash soft_vla/scripts/real_robot/deploy/smolvla_pressure_state_human_intervention.sh
```

## Replay 入口

```bash
bash soft_vla/scripts/real_robot/replay/epis_replay.sh
bash soft_vla/scripts/real_robot/replay/epis_replay_fullA_history_v2.sh
```

`diagnostics/` 下的命令仅用于硬件接入、诊断和阶段性验证，不是正式部署入口。可复用的运行逻辑
继续放在 `src/soft_vla/`，这里仅保留命令行编排代码。
