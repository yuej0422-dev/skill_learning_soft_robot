# SmolVLA 异步推理执行模式报告

本文档说明当前 SmolVLA 异步部署框架中四种 action chunk 执行器的运行逻辑：

- `single_step.py`
- `fixed_chunk.py`
- `receding_horizon.py`
- `temporal_ensemble.py`

相关代码目录：

```text
soft_vla/src/soft_vla/inference/chunk_execution/
```

四进程部署入口：

```text
soft_vla/src/soft_vla/runtime/smolvla_async_runtime.py
```

## 统一 Tick 语义

上层 action dispatch 固定为 10Hz：

```text
1 tick = 100 ms
```

SmolVLA 每次输出一个 `[chunk_size, 7]` action chunk。当前默认：

```text
chunk_size = 50
execution_horizon = 10
replan_interval = 5
expected_stale_steps = 2
trigger_margin = 1
worst_stale_steps = 5
```

统一时间对齐字段：

```text
request_tick        提交推理请求时的 10Hz tick
result_tick         推理结果返回时对应的 10Hz tick
next_dispatch_tick  上层下一次尚未执行的 tick
effective_tick      新 chunk 最早允许生效的 tick
stale_steps         推理延迟导致 chunk 前端过期的 action 数
```

当前 CHUNK / Receding Horizon / Single Step 使用：

```python
effective_tick = max(result_tick, next_dispatch_tick)
stale_steps = max(0, effective_tick - request_tick)
valid_actions = chunk[stale_steps:]
valid_start_tick = effective_tick
```

这保证不会从 `chunk[0]` 执行已经过期的动作。

## 运行时异步结构

四进程部署框架中，10Hz 上层 dispatch loop 不直接调用 SmolVLA 推理。

当前关键队列：

```text
inference_request_queue  upper 进程向 inference 进程提交异步推理请求
chunk_queue              inference 进程返回 action chunk
reference_queue          upper 进程向 50Hz 控制进程发送 reference segment
upper_state_queue        50Hz 控制进程向 upper 进程发布最新 state
inference_state_queue    50Hz 控制进程向 inference 进程发布最新 state
```

启动逻辑：

```text
1. inference 进程 bootstrap 生成第一段 chunk
2. first_action_chunk_ready 后 upper/control 才进入正式 runtime loop
3. 之后 upper 根据 executor.needs_replan() 发 request
4. inference 收到 request 后异步生成新 chunk
5. upper 每个 10Hz tick 只做非阻塞队列读取和 action dispatch
```

因此 SmolVLA 155-176ms 的推理延迟不会阻塞 10Hz dispatch loop。

## single_step.py

### 目标

`single_step` 每次只消费一个 action。它适合作为 debug 模式，不适合作为当前真实 SmolVLA 的 10Hz 实时部署主模式。

原因：

```text
SmolVLA p50/p95 推理延迟约 155-176ms > 100ms
```

如果要求每个 tick 都有新推理结果，模型无法稳定满足 10Hz。

### submit_chunk 逻辑

输入一个 chunk 后，执行器计算：

```python
effective_tick = max(result_tick, next_dispatch_tick)
stale_steps = effective_tick - request_tick
local_idx = min(stale_steps, len(chunk) - 1)
```

也就是说，如果 request 在 tick 10，结果到 tick 12，则不会执行 `chunk[0]`，而是执行 `chunk[2]`。

### get_action 逻辑

`get_action()` 只允许调用一次。

```text
第一次调用：返回 chunk[local_idx]
第二次调用：抛 RuntimeError，表示需要 fresh chunk
```

runtime 捕获异常后会 fallback，不会阻塞等模型。

### replan 逻辑

```python
needs_replan = self.chunk is None or self.used
```

因此 single_step 基本每个 tick 都要求新 chunk。

### 当前状态

```text
可用于 debug
不推荐真实 10Hz 实物部署
已支持 stale action 处理
```

## fixed_chunk.py

### 目标

`fixed_chunk` 每次收到一个 chunk 后：

```text
1. 丢弃 stale actions
2. 取 valid_actions 前 execution_horizon 步
3. 按 10Hz tick 顺序 dispatch
4. 队列快耗尽时提前触发下一次推理
```

### submit_chunk 逻辑

核心代码语义：

```python
request_tick = ...
result_tick = ...
next_dispatch_tick = ...

effective_tick = max(result_tick, next_dispatch_tick)
stale_steps = max(0, effective_tick - request_tick)
valid = chunk[stale_steps:]

queue.clear()
for j, action in enumerate(valid[:execution_horizon]):
    local_idx = stale_steps + j
    absolute_tick = effective_tick + j
    queue.append((action, chunk_id, local_idx, absolute_tick))
```

示例：

```text
request_tick = 10
result_tick = 12
next_dispatch_tick = 12
effective_tick = 12
stale_steps = 2
```

第一个执行动作是：

```text
chunk[2] -> tick 12
```

不是：

```text
chunk[0]
```

### get_action 逻辑

队列非空时：

```text
pop 左侧第一个 action
返回 source="chunk"
记录 chunk_step/local_idx/absolute_step
```

队列为空时：

```text
返回 queue_underrun_fallback
action 前 6 维为 0
gripper 保持 last_gripper
```

不会阻塞。

### replan 逻辑

当前触发条件：

```python
threshold = expected_stale_steps + trigger_margin
needs_replan = not queue or len(queue) <= threshold
```

默认：

```text
expected_stale_steps = 2
trigger_margin = 1
threshold = 3
```

如果 `execution_horizon=10`，大约执行到队列剩 3 步时触发下一次推理，即约第 7 步，符合当前延迟估计。

### 特点

```text
优点：简单、稳定、易 debug
缺点：新 chunk 到达时会替换当前未执行队列
适用：初期实物部署、验证 action 和底层 controller
```

## receding_horizon.py

### 目标

`receding_horizon` 是当前推荐默认模式。它继承 `FixedChunkExecutor` 的 stale action 处理和队列执行逻辑，但 replan 触发方式不同。

### submit_chunk 逻辑

`receding_horizon` 调用父类 `FixedChunkExecutor.submit_chunk()`，因此同样执行：

```python
effective_tick = max(result_tick, next_dispatch_tick)
stale_steps = effective_tick - request_tick
valid_actions = chunk[stale_steps:]
```

随后记录 boundary 信息：

```text
old_unexecuted
old_last_action
new_first_action
request_tick
effective_tick
stale_steps
```

这些字段用于检查是否在规划边界发生动作突变。

### get_action 逻辑

与 `fixed_chunk` 相同：

```text
队列非空：顺序 pop
队列为空：queue_underrun_fallback
```

### replan 逻辑

当前触发条件：

```python
needs_replan = not queue or (control_step > 0 and control_step % replan_interval == 0)
```

默认：

```text
replan_interval = 5
```

即每 5 个 10Hz tick 触发一次异步推理请求，大约 2Hz 重规划。

### 与 fixed_chunk 的区别

`fixed_chunk` 主要根据队列剩余长度提前触发。

`receding_horizon` 主要根据固定 tick 间隔触发。

两者都不会阻塞 10Hz dispatch loop。

### 当前语义

新 chunk 返回后：

```text
1. 根据 request_tick/result_tick/next_dispatch_tick 丢弃 stale actions
2. 从 effective_tick 开始建立新的未来动作队列
3. 不会回写已经执行过的 tick
```

注意：当前实现是“最新 chunk 替换未执行队列”，不是复杂的多 chunk future map 合并。这是一个保守的最小实现，便于实物部署调试。

## temporal_ensemble.py

### 目标

`temporal_ensemble` 保留多个历史 chunk，在每个 target tick 选择所有能覆盖该 tick 的候选动作，并做加权融合。

它适合降低 action 抖动，但对时间对齐更敏感。

### submit_chunk 逻辑

TE 不丢弃 chunk 前端，也不直接建立 FIFO 队列。

它保存：

```python
HistoricalChunk(
    chunk_id=...,
    start_step=request_tick,
    actions=chunk.copy(),
)
```

其中：

```text
start_step = request_tick
chunk[i] 对应真实 tick = request_tick + i
```

history 长度受 `max_history_chunks` 限制。

### get_action 逻辑

给定当前 `control_step / target_tick`：

```python
candidates = []

for record in history:
    local_idx = target_tick - record.start_step
    if 0 <= local_idx < len(record.actions):
        candidates.append(record.actions[local_idx])
```

示例：

```text
chunk_0.start_tick = 0
chunk_1.start_tick = 5
target_tick = 7
```

参与融合的是：

```text
chunk_0[7]
chunk_1[2]
```

不是：

```text
chunk_0[7]
chunk_1[7]
```

### 权重逻辑

当前默认：

```text
weight_type = exponential
decay = 0.25
prefer_newer_predictions = True
```

实现里使用 chunk 新旧程度作为 age：

```python
age = latest_chunk_id - candidate_chunk_id
weight = exp(-decay * age)
```

因此更新的 chunk 权重更高。

### 夹爪处理

TCP 前 6 维是连续值，按权重平均。

第 7 维 gripper 是离散 open/close，不直接保留连续平均值，而是：

```python
weighted_gripper >= 0.5 -> 1.0
else -> 0.0
```

### fallback

如果没有任何历史 chunk 覆盖当前 tick：

```text
返回 te_underrun_fallback
前 6 维为 0
gripper 保持 last_gripper
```

不会阻塞等待模型。

### 当前状态

```text
已按绝对 tick 对齐
已限制 history 长度
已记录 candidate chunk ids / local indices / weights
适合后续平滑策略对比
真实实物初期建议先用 receding_horizon 验证
```

## 10Hz 到 50Hz Reference Handoff

上层 executor 输出的是 10Hz delta TCP action。

底层 50Hz 控制不直接重复完整 delta 5 次，而是由 `ReferenceGenerator` 展开为 5 个 reference state：

```python
fractions = [1/5, 2/5, 3/5, 4/5, 5/5]
refs[:, :6] = current[:6] + fractions * scaled_delta
```

因此当前是“目标插值”逻辑，不是错误的完整 delta 重复累加。

## 推荐部署选择

当前根据实测 SmolVLA 延迟：

```text
p50/p95: 约 2 ticks
max: 约 5 ticks
```

推荐：

```text
默认：receding_horizon
调试：chunk
平滑对比：temporal_ensemble
仅 debug：single_step
```

`soft_vla/scripts/real_robot/deploy/smolvla_deploy.sh` 当前默认：

```bash
MODE=${MODE:-receding_horizon}
CHUNK_EXPECTED_STALE_STEPS=${CHUNK_EXPECTED_STALE_STEPS:-2}
CHUNK_WORST_STALE_STEPS=${CHUNK_WORST_STALE_STEPS:-5}
CHUNK_TRIGGER_MARGIN=${CHUNK_TRIGGER_MARGIN:-1}
```

## 已有测试覆盖

测试文件：

```text
soft_vla/validation/automated/test_async_chunk_time_alignment.py
```

覆盖内容：

```text
CHUNK stale=2 时从 chunk[2] 开始
CHUNK stale=5 时从 chunk[5] 开始
Receding Horizon 只从 future tick 更新
Temporal Ensemble 按绝对 tick 对齐
160/180/405ms 延迟映射为 2/2/5 ticks
queue underflow 返回 fallback，不阻塞
```

最近一次全测试结果：

```text
41 passed
```

## 人工检查建议

实物运行时建议重点看 jsonl 中以下字段：

```text
process=upper_10hz
period_ms
request_tick
result_tick
effective_tick
stale_steps
chunk_id
chunk_local_idx
queue_underflow
fallback_used
replan_triggered
inference_running
te_candidate_chunk_ids
te_candidate_local_indices
te_weights
```

判断逻辑：

```text
period_ms 应接近 100ms
stale_steps 应与推理延迟 tick 数一致
chunk_local_idx 不应长期从 0 开始，除非 bootstrap 或极低延迟
queue_underflow 不应频繁出现
replan_triggered 应按模式规律出现
TE candidate local indices 应能对应同一个真实 target tick
```
