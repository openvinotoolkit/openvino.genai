# Session 快照保存/恢复（Session KV+LA Cache Snapshot Save/Restore）用户手册

> 面向使用者的功能说明与操作手册。设计/实现细节见提交记录与
> [`session_snapshot.hpp`](../src/cpp/src/continuous_batching/session_snapshot.hpp)。
> 本文适合直接在 VS Code 中使用 **Markdown: Open Preview** 查看。

## 1. 这个功能解决什么问题

场景：用户在和模型进行一个很长的对话（或处理一段很长的上下文），这时进程需要退出（关机、重启、
服务滚动升级等）。下次再打开时，如果重新把整段历史对话喂给模型，就要重新跑一遍完整的 prefill ——
历史越长，这笔开销越大。

`save_state()` / `restore_state()` 让你在进程退出前，把某个请求当前的 **全部 KV cache（以及混合
模型的 linear-attention 状态）+ token 历史** 存成一个文件；下次（哪怕是全新的进程）用
`restore_state()` 直接把这些状态注入到新 pipeline 里，跳过 prefill，直接从"已经处理完这段历史"的
状态继续生成。

这是对已有的"机会性/LRU 磁盘 offload cache"（`use_cache_offload`）的补充，两者解决的问题不同：

| | 机会性 offload（`use_cache_offload`） | 本功能（session snapshot） |
|---|---|---|
| 触发方式 | 自动、按 LRU 淘汰 | 显式调用 `save_state()` |
| 命中条件 | 依赖下次请求 token 前缀完全一致 | 恢复时直接拿到 token 历史，无需重新发送完整 prompt |
| 生命周期 | 跟随进程/磁盘文件（进程内，文件名带 PID） | 显式文件路径，可以跨进程/跨重启持久化 |
| 适用场景 | 短期内同一批请求反复命中相同前缀 | 显式"保存会话 - 下次整体恢复"场景 |

两者可以同时启用，互不冲突。

## 2. 效果有多大（已验证数据）

在 Qwen3-8B、约 4000 token 的历史对话上实测，"直接完整 prefill" vs "从 session snapshot 恢复"：

| 设备 | 测试方式 | 不走 restore (TTFT) | 走 restore (TTFT) | 加速比 |
|---|---|---|---|---|
| CPU | C++ 测试 (`tests/cpp/session_snapshot_e2e.cpp`) | ~71542 ms | ~5960 ms | ~12.0x |
| CPU | Python 脚本 (`tools/llm_bench/session_snapshot_bench.py`) | ~67242 ms | ~6181 ms | ~10.9x |
| CPU | `benchmark.py` CLI | ~68547 ms | ~4514 ms | ~15.2x |
| GPU | C++ 测试 | ~4261 ms | ~588 ms | ~7.25x |
| GPU | `benchmark.py` CLI | ~3829 ms | ~611 ms | ~6.27x |
| CPU | Qwen3.5-9B（混合 KV+LA）C++ 测试 | ~135602 ms | ~5788 ms | **~23.4x** |

Qwen3.5 的加速比更高，因为模型更大、prefill 成本更高，而 restore 的开销基本是常数级。

历史越长，理论上加速比越大，因为 restore 的开销基本是常数级别（只需要把最后一个 token 重新走一次
解码），而完整 prefill 的开销随历史长度线性增长。

## 3. 使用限制（V1 范围）

当前实现有以下限制，调用时若不满足会直接抛出 `ov::Exception`，不会静默产生错误结果：

1. **只支持 `ContinuousBatchingPipeline`**，不支持 `LLMPipeline`/`VLMPipeline` 的高层封装。
2. **只支持单序列（非 beam search、非 parallel sampling）请求**：`num_beams=1` 且没有被 fork 过。
3. **不支持 `use_cache_eviction=true`** 的调度配置（被淘汰的历史块内容已经不完整，无法可靠恢复）。
4. **支持走 input-embeddings 输入的模型（例如 Qwen3.5），但仅限纯文本对话**：某些原生多模态
   模型即使只喂纯文本 prompt，底层也会自动走 `InputsEmbedder`（而不是普通 token id）。这类模型
   已经支持，但对话历史里不能包含图片/视频，也不能用到 `token_type_ids`/`deepstack_visual_embeds`/
   `per_layer_inputs` 这类多模态额外状态——一旦检测到这些，`save_state()` 会明确拒绝，而不是悄悄
   保存一份内容错乱的快照。
5. **`save_state()` 必须在请求仍处于 `RUNNING` 状态时调用**，不能等它自然生成完/达到
   `max_new_tokens` 后再保存——请求一旦结束，它的缓存块会在同一个 `step()` 内立刻被释放，之后就
   没有机会再保存了。实际使用时这通常不是问题：真实场景本来就是"进程被打断时，请求还在生成中"。
6. **推测解码（`draft_model` 配置了草稿模型）现已支持**：内部会分别对主模型和草稿模型
   各自调用一次单模型的 save_state/restore_state（存成两个独立文件，文件名分别加 `.main`/`.draft`
   后缀）。注意：DFlash 策略特有的中间状态（`pending_hidden_deltas`）恢复后会重置为空，还没有用真
   实草稿模型做过端到端验证。

## 4. C++ API 用法

```cpp
#include "openvino/genai/continuous_batching_pipeline.hpp"

using namespace ov::genai;

// ===== 保存（进程退出前） =====
ContinuousBatchingPipeline pipeline(model_path, scheduler_config, device);
auto handle = pipeline.add_request(request_id, prompt, sampling_params);

// 正常驱动生成若干步……
while (pipeline.has_non_finished_requests()) {
    pipeline.step();
    // 这里可以用 handle->can_read() + handle->read() 读取已经生成的 token，
    // 但注意 handle->read_all() 会阻塞到请求结束，保存前不要调用它。
}

// 在请求仍处于 RUNNING 状态时保存（不要等它自然结束）：
pipeline.save_state(request_id, "/path/to/session.bin");

// ===== 恢复（新进程 / 新 pipeline 实例） =====
ContinuousBatchingPipeline pipeline(model_path, scheduler_config, device);
auto handle = pipeline.restore_state(new_request_id, "/path/to/session.bin", sampling_params);
while (pipeline.has_non_finished_requests()) {
    pipeline.step();
}
auto outputs = handle->read_all();  // 正常读取后续生成的 token
```

`restore_state()` 的 `sampling_params` 只影响 **恢复之后新生成的部分**（`max_new_tokens` 等），
不包含已经保存的历史 token 数。

## 5. Python API 用法

```python
import openvino_genai as ov_genai

# ===== 保存 =====
pipeline = ov_genai.ContinuousBatchingPipeline(model_path, scheduler_config, device)
handle = pipeline.add_request(request_id, prompt, generation_config)
for _ in range(num_steps):
    if handle.get_status() != ov_genai.GenerationStatus.RUNNING:
        break
    pipeline.step()
pipeline.save_state(request_id, "session.bin")

# ===== 恢复（新进程） =====
pipeline = ov_genai.ContinuousBatchingPipeline(model_path, scheduler_config, device)
handle = pipeline.restore_state(new_request_id, "session.bin", generation_config)
while pipeline.has_non_finished_requests():
    pipeline.step()
outputs = handle.read_all()
```

可直接运行的完整示例脚本：[`tools/llm_bench/session_snapshot_bench.py`](../tools/llm_bench/session_snapshot_bench.py)
（对比有/无 restore 的 TTFT）。

## 6. 用 `benchmark.py` 测试

`tools/llm_bench/benchmark.py` 新增了三个参数，可以直接用它跑保存/恢复两次独立进程来测量真实收益：

```powershell
# 第一次运行：正常生成若干步后，保存快照（不是完整生成，--infer_count 在此模式下不生效）
python benchmark.py -m <model_dir> -d CPU -pf prompt.jsonl -n 0 -ic 1 `
    --cb_config cb_config.json `
    --session_snapshot_save session.bin `
    --session_snapshot_save_steps 3

# 第二次运行（模拟进程重启）：从快照恢复并继续生成，走标准 benchmark 指标输出
python benchmark.py -m <model_dir> -d CPU -p "ignored" -n 0 -ic 1 `
    --cb_config cb_config.json `
    --session_snapshot_restore session.bin
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--session_snapshot_save PATH` | 保存模式：跑 `--session_snapshot_save_steps` 步后调用 `save_state()`，而不是跑满 `--infer_count` 的完整生成。 |
| `--session_snapshot_save_steps N` | 保存前要跑的 continuous-batching 步数（默认 5），需要覆盖完整 prefill + 至少 1-2 个 decode 步。 |
| `--session_snapshot_restore PATH` | 恢复模式：从快照恢复并继续生成 `--infer_count` 个新 token，走标准指标（TTFT/MD5 等）输出。此时 `-p`/`-pf` 的内容会被忽略（只是为了满足 CLI 必须有个 prompt 来源的要求）。 |

两个参数互斥，且都要求 `--batch_size 1 --num_beams 1`（这是本功能本身的限制，见第 3 节）。两者任一
被设置时会自动强制走 `ContinuousBatchingPipeline`（和 `cache_eviction`/`sparse_attention`/
`use_cache_offload` 一样），不需要额外在 `--cb_config` 里开启其他开关。

## 7. 异常/报错速查

| 报错信息（节选） | 原因 | 处理方式 |
|---|---|---|
| `No active request with id ... to save a session snapshot for` | 请求已经结束（或 id 不存在）才调用 `save_state()` | 必须在请求仍是 `RUNNING` 时保存，见第 3 节第 5 条 |
| `Session snapshot is not supported together with use_cache_eviction` | 调度配置开启了 `use_cache_eviction` | 关闭 cache eviction，或不对这类请求使用本功能 |
| `Session snapshot only supports models that take plain token ids as input` | 模型走的是 input-embeddings 路径（例如原生多模态模型） | 暂不支持；这是 V1 的已知限制，非 bug |
| `Session snapshot only supports a single (non-beam-search, non-forked) sequence` | 请求是 beam search 或已经被 fork | 使用 `num_beams=1` 的普通贪心/采样请求 |
| `Session snapshot file not found` | `restore_state()` 的路径不存在 | 检查保存时用的路径是否一致 |
| `... failed its integrity check (file may be corrupted)` | 快照文件内容被破坏，或和当前文件不是同一次保存 | 重新保存一份；不要手动编辑快照文件 |
| `Session snapshot was captured with an incompatible KV cache layout` / `on device '...' but the current pipeline uses '...'` | 恢复时用的模型/设备和保存时不一致 | 确保恢复用的是同一个模型、同一个推理设备 |

## 8. 已知待办

- **混合（linear-attention）模型现已支持并用 Qwen3.5 真实验证通过**（2026-10-13）：保存/恢复后续写与
  未打断的基线完全一致。仅限纯文本对话（见第 3 节第 4 条），带图片/视频的历史还不支持。
- **推测解码现已支持**（2026-10-13），但只用到了已验证的单模型 save/restore 机制，没有用真实草稿
  模型做过端到端验证（环境里没有现成的草稿模型可用），DFlash 策略特有的 `pending_hidden_deltas`
  中间状态恢复后会重置为空。
