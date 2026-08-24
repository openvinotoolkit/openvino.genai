# OpenVINO GenAI Paged KV Cache 磁盘 Offload 迁移规格

本文是给“另一个 AI / 实现者”的工程规格，目标是把 vLLM V1 的 paged KV cache 磁盘 offload 思路迁移到 OpenVINO GenAI 的 Continuous Batching pipeline。

本文只描述设计和代码落点，不直接修改实现。所有结论基于当前仓库源码；实现时应以源码为准，并先补测试再改生产代码。

## 1. 结论先行

OpenVINO 不应照搬 vLLM 的 Python `SimpleCPUOffloadScheduler` / `DiskBackend` 类名，而应复用现有的 C++ 分层：

```text
Scheduler
  -> CacheOrchestrator
       -> BlockManager / BlockAllocator
       -> KVCacheManager
  -> ModelRunner
  -> ContinuousBatchingImpl::step()
```

推荐的最小实现边界是：

```text
Scheduler / BlockManager:
    决定哪些“已不在当前 active block table 中、但可被 prefix hash 找到”的物理 block 要保存
    决定 prefix hit 对应的 disk slot 是否需要恢复

KVCacheManager:
    提供按 physical block、decoder layer、key/value tensor 读写一个 block 的接口
    管理异步传输完成后的状态，不直接决定 eviction policy

CacheOrchestrator:
    协调 KV block manager 和 KV physical manager 的 store/load 状态
    确保 block table 在 load 完成前不能暴露给 ModelRunner

ContinuousBatchingImpl::step:
    在 infer 完成后提交 GPU -> disk store
    在下一次 forward 前确保 disk -> GPU load 已完成
```

不要把“逻辑 block 被 eviction”直接当成“物理 block 可以立即复用”。OpenVINO 的 `free_blocks_from_sequence()` 会从 block table 删除逻辑项，并通过 allocator 把物理 block 放入 free pool 或 prefix-cache overwrite store；offload 需要在物理 block 复用前建立自己的 pinned/in-flight 保护。

## 1.1 勘误（以本节为准）

本文后续章节写于逐行核对代码之前，有两处结论已被证伪。实现时以本节和
[paged_kv_cache_offload_to_disk_implementation_plan_zh.md](paged_kv_cache_offload_to_disk_implementation_plan_zh.md) 为准。

### 勘误 1：`BlocksPerLayer` 不是“每个 decoder layer 一个 block”

它的长度是 **block-table 层数**，由 `CacheOrchestrator::register_kv_cache()` 决定：

```cpp
const bool per_layer_control = config.use_cache_eviction;
const size_t num_block_table_layers = per_layer_control ? kv_manager->get_num_layers() : 1;
```

- `use_cache_eviction == false`（默认）：`BlocksPerLayer.size() == 1`。单个 physical block ID 同时索引所有 decoder layer 的 `key_cache.L` / `value_cache.L`。
- `use_cache_eviction == true`：`BlocksPerLayer.size() == decoder 层数`，每层持有不同的 physical block ID。

因此第 2.2 节“同一个 prefix block 在所有 layer 上必须同步释放”只在 eviction 模式下才有实际含义；第 14 节“错误 6”的描述也应按此理解。磁盘 slot 的划分必须区分这两种布局。

### 勘误 2：不应以 eviction 作为第一版 store 来源

第 7.1 节和第 12 节建议的 eager eviction-driven store 不可行：

1. `_maybe_evict_cache_blocks()` 只在 `use_cache_eviction == true` 时运行，而该模式会把 block table 切为 per-decoder-layer。
2. 该模式下每层 evict 的 logical block 各自独立，同一次释放的跨层 block-set 通常 hash 不一致。`BlockAllocator::free()` 只在跨层 hash 全部相同时才写入 `OverwritableBlocksHashStore`，否则直接进 free pool。因此这些块本来就不可被 prefix 命中，写盘也只会产生死数据。

正确的第一版 store 触发点是内存 prefix cache 的生命周期：block 进入 `OverwritableBlocksHashStore` 时成为候选，在 `get_lru_block_to_overwrite()` 把它交出去覆写之前必须完成落盘。

### 补充：restore 的线程上下文

`restore_cached_blocks()` 在 `ContinuousBatchingImpl::add_request()` 中调用，运行在调用方线程，与 `step()` 并发。任何阻塞式 disk load 都不能放在这个位置。

## 2. OpenVINO 当前 KV cache 对象模型

### 2.1 物理 tensor

`KVCacheManager` 为每个 decoder layer 持有两组 tensor：

```text
m_key_cache[layer]
m_value_cache[layer]
```

每个 tensor 的第 0 维是 physical block ID，形状来自模型的 `key_cache.N` / `value_cache.N` 输入。一个 block 的总大小是所有 layer 的 key/value block bytes 之和：

```text
block_bytes = sum(layer_key_block_bytes + layer_value_block_bytes)
```

当前 `get_block_size_in_bytes()` 返回这个“跨所有 decoder layer 的一个 block”大小。磁盘实现应该以此作为单个 slot 的逻辑大小；底层文件布局可以按 layer/key/value 分段，但必须保持可逆。

OpenVINO 当前设备分支：

- CPU：普通 `ov::Tensor`，数据可通过 `data()` / `copy_to()` 访问。
- GPU：通过 `RemoteContext::create_tensor()` 创建 `ov::RemoteTensor`，由 GPU plugin 持有实际存储。

`KVCacheManager::allocate_cache_if_needed()` 扩展 cache 时会创建更大的 tensor，然后把旧 tensor 内容复制到新 tensor。这个过程是 cache capacity growth，不应被误认为 offload；offload 的目标是保持 physical GPU cache 容量不变，把完整 block 的内容放到 host/disk。

### 2.2 BlockAllocator 与 BlockManager

`BlockAllocator` 每层有一个 free list，但同一个 logical block 的所有 decoder layer 通过 `BlocksPerLayer` 成组操作。关键不变量：

```text
同一个 prefix block 在所有 layer 上必须同步释放、同步复用、同步恢复
```

`CacheBlock` 当前保存：

- `m_index`：physical block ID。
- `m_hash`：prefix hash。
- `m_ref_count`：sequence / shared fork / temporary ownership 的引用计数。
- `m_timestamp`：overwrite store 的 LRU 时间。

当 prefix caching 打开时：

- `m_prefix_hash_to_cached_blocks`：active/cache map，hash -> 每层 physical block。
- `OverwritableBlocksHashStore`：已无 sequence owner、可以被 prefix hit 恢复或被 LRU 覆写的 block 组。

`BlockManager::free_cached_blocks()` 最终会调用 allocator 的 `free(BlocksPerLayer, cached_blocks)`。如果所有 layer 的 block 都 free 且 hash 一致，它们会进入 `OverwritableBlocksHashStore`，并可能继续由 `get_cached_block()` 恢复。

这已经很接近 vLLM 的 CPU block pool，但 OpenVINO 的 prefix cache 是“同一组物理 block 跨层管理”，而 vLLM 的实现把 scheduler-side offload block 作为独立 pool 管理。迁移时应保留 OpenVINO 的跨层原子性。

### 2.3 Logical block table 与 ModelRunner

每个 sequence 有 per-layer block table：

```text
logical block index -> CacheBlock::get_index() -> physical block ID
```

`Scheduler::Output::KVPagedAttentionData::block_tables` 保存这些表。`ModelRunner::_fill_indices_from_block_tables()` 将 physical IDs 填入：

```text
block_indices
或
block_indices.0, block_indices.1, ...
```

Paged Attention 只认识 physical block indices 和 `key_cache.N` / `value_cache.N`。因此 disk slot 不能直接写入 `block_indices`：在 forward 前，所有要参与 attention 的 disk block 必须先恢复到合法的 GPU physical block。

## 3. 当前 OpenVINO 完整运行时顺序

### 3.1 初始化

```text
ContinuousBatchingImpl 构造
  -> prepare_model_for_paged_attention()
  -> compile_model()
  -> create_infer_request()
  -> CacheOrchestrator::create()
       -> detect_cache_managers()
       -> KVCacheManager
       -> BlockManager
  -> Scheduler
  -> ModelRunner
```

`CacheOrchestrator::create()` 根据模型 cache inputs 和 available memory 归一化 `num_kv_blocks`，随后注册 KV cache manager 和 block manager。

### 3.2 每个 step

```text
step()
  1. _pull_awaiting_requests()
  2. Scheduler::schedule()
       - clean_empty_blocks()
       - allocate_tokens()/append_slots()
       - allocate_cache_if_needed()
       - copy_blocks()  // fork / copy-on-write
       - 生成 Scheduler::Output
  3. ModelRunner::forward()
       - 填 inputs、past_lens、block_indices
       - m_request.infer()
  4. _maybe_evict_cache_blocks()
       - 读取上一轮 attention scores
       - 计算逻辑 block eviction
       - Scheduler::free_blocks_from_sequence()
  5. sampler
  6. fork_sequence()/free_sequence()
  7. 下一次 step
```

这个顺序决定了 store/load 插入点：

- store 的源 block 在 `infer()` 过程中可能刚被写入，所以至少要在该次 `infer()` 返回后提交。
- eviction 发生在 `infer()` 后；被 eviction 的物理 block 可以被选为 store 候选，但在 store 完成前不能被 allocator 复用。
- load 必须在下一次 `ModelRunner::forward()` 使用 block table 前完成。

## 4. vLLM 设计中真正值得迁移的部分

参考 vLLM：

- [vllm/v1/simple_kv_offload/manager.py](../../vllm/vllm/v1/simple_kv_offload/manager.py)：scheduler 侧 store/load 状态机。
- [vllm/v1/simple_kv_offload/worker.py](../../vllm/vllm/v1/simple_kv_offload/worker.py)：异步 transfer、event high-water mark、preemption flush。
- [vllm/v1/simple_kv_offload/disk_backend.py](../../vllm/vllm/v1/simple_kv_offload/disk_backend.py)：独立 load/store 队列、pinned staging buffer、固定 slot 文件、流水线 I/O。
- [vllm/v1/simple_kv_offload/cuda_mem_ops.py](../../vllm/vllm/v1/simple_kv_offload/cuda_mem_ops.py)：按 block bytes 进行批量内存复制的思路。
- [vllm/v1/simple_kv_offload/metadata.py](../../vllm/vllm/v1/simple_kv_offload/metadata.py)：用 event ID 将 scheduler 决策与 worker 完成通知解耦。

需要迁移的是这些原则，而不是 Python 具体实现：

1. scheduler 不执行 I/O，只生成 transfer metadata。
2. store 和 load 分离成两个队列/线程，避免后台 store 阻塞 latency-sensitive load。
3. 一个 transfer event 对应一批 block pairs；完成用单调 event ID 追踪。
4. store 必须等待 compute 完成；load 必须在模型 forward 前完成。
5. block 在 transfer in-flight 时增加引用或进入 `IN_FLIGHT` 状态，不能被 allocator 重用。
6. store 完成后才把 hash 注册成可命中的 offload cache。
7. reset/preemption 时不能直接清空 metadata；必须等待旧 transfer 完成后释放引用。

## 5. 推荐 OpenVINO 架构

### 5.1 新增 `KVCacheOffloadManager`

建议新增独立的 C++ 类，例如：

```cpp
class KVCacheOffloadManager {
public:
    void init(const KVCacheManager& cache_manager,
              const KVCacheOffloadConfig& config);

    void enqueue_store(const std::vector<BlocksPerLayer>& blocks,
                       uint64_t store_event);

    void enqueue_load(const std::vector<BlocksPerLayer>& source_slots,
                      const std::vector<BlocksPerLayer>& destination_blocks,
                      uint64_t load_event);

    void poll_completed(std::vector<uint64_t>& completed_stores,
                        std::vector<uint64_t>& completed_loads);

    void flush();
    void reset();
};
```

实际命名可按仓库风格调整。这个类负责：

- disk file 生命周期。
- per-rank 文件路径。
- slot 分配。
- store/load worker queue。
- staging buffers。
- block copy / read / write。
- in-flight event 和异常传播。

不要让它直接修改 `BlockManager` 的 hash map；完成回收应由 `CacheOrchestrator` 或专门的 offload-aware block manager 执行。

### 5.2 配置

建议在 `SchedulerConfig` 增加或扩展配置，而不是从环境变量读取所有选项：

```cpp
struct KVCacheOffloadConfig {
    bool enabled = false;
    std::string backend = "disk"; // future: cpu, disk
    std::string path;
    size_t capacity_bytes = 0;
    size_t buffer_slots = 2;
    bool use_page_cache = false;
    bool lazy_offload = false;
};
```

初版建议只支持 CPU host tensor 或 CPU device，以验证状态机；GPU RemoteTensor 需要额外确认 OpenVINO GPU plugin 的 host/device copy 和同步语义，再开启 disk backend。

## 6. Block 状态和元数据设计

### 6.1 不要复制 vLLM 的 request metadata

OpenVINO 的基本单位是跨 layer 的 `BlocksPerLayer`，所以建议使用 block-set 级别 metadata：

```cpp
enum class OffloadLocation {
    DEVICE,
    DISK,
    STORE_IN_FLIGHT,
    LOAD_IN_FLIGHT,
};

struct OffloadBlockSet {
    size_t hash = 0;
    BlocksPerLayer device_blocks; // 每层一个 device physical block
    size_t disk_slot = 0;
    size_t ref_count = 0;
    OffloadLocation location = OffloadLocation::DEVICE;
    uint64_t store_event = 0;
    uint64_t load_event = 0;
};
```

每个 layer 的 block ID 目前通常相同，但不要依赖这个偶然事实；以 `BlocksPerLayer[layer]` 的 index 作为真实访问方式。

### 6.2 必须维护的索引

至少需要：

```text
hash -> OffloadBlockSet / disk slot
physical block ID -> in-flight protection
store_event -> OffloadBlockSet
load_event -> sequence/request IDs
```

已有 `m_prefix_hash_to_cached_blocks` 和 `OverwritableBlocksHashStore` 可以继续承担 active/overwritable prefix cache 的职责，但 disk cache 必须有独立索引，因为 disk 数据没有常驻 device block。

建议新增：

```cpp
std::map<size_t, DiskCacheEntry> m_disk_hash_to_entry;
std::map<uint64_t, TransferRecord> m_store_events;
std::map<uint64_t, TransferRecord> m_load_events;
std::set<size_t> m_in_flight_physical_block_ids;
```

同一个 hash 如果同时存在 device active entry、overwritable host entry 和 disk entry，需要明确优先级：

```text
active device block > host/overwritable block > disk block
```

并保证一个 hash 只产生一次有效 restore 来源，避免重复占用物理 block。

### 6.3 Disk slot 文件布局

推荐固定长度 slot：

```text
slot_bytes = sum(all layers: key_block_bytes + value_block_bytes)
file_offset = slot_id * slot_bytes
file_size = num_slots * slot_bytes
```

单个 slot 内推荐固定布局：

```text
[layer 0 key][layer 0 value][layer 1 key][layer 1 value]...
```

也可以按所有 key 再所有 value，但必须由同一个 layout helper 生成读写 offset，不能在 store/load 两侧各自手算。

初版建议使用：

- 临时文件或 run-specific 文件名。
- `0600` 权限。
- `O_EXCL` 或等价的安全创建方式。
- pipeline shutdown 时 unlink。
- 不把该文件当作跨进程、跨模型、跨版本持久化 cache。

如果支持 `O_DIRECT`，必须保证 file offset、I/O size、buffer address 满足平台对齐要求。否则第一版使用 page cache 更容易先验证正确性。

## 7. Store 链路设计

### 7.1 候选选择

> 已修正，见第 1.1 节勘误 2。下面的 eviction-driven 流程仅在后续支持 `use_cache_eviction == true` 时才适用；第一版应改为以 `OverwritableBlocksHashStore` 的进入和 LRU 覆写作为 store 触发点。

后续 eviction 模式下的 store 流程：

```text
infer() 返回
  -> _maybe_evict_cache_blocks() 计算 logical blocks
  -> 通过 BlockManager 找到被移除的 BlocksPerLayer
  -> 对每个 block-set 创建 store record
  -> touch / pin 物理 block
  -> enqueue_store()
  -> store 完成
  -> 注册 disk hash entry
  -> 释放 device block 的 transfer 引用
```

不要第一版就实现 vLLM 的 lazy free-queue cursor。OpenVINO 当前 eviction 已经给出明确的 `logical_block_indices_to_free`，先围绕这个路径建立 correctness 更可控。

关键修改点：`BlockManager::free_blocks_from_sequence()` 当前会先 `free_cached_blocks()`，然后重建 block table。若 store 需要在 block 失去 owner 后仍访问其内容，必须在释放前捕获 block-set 和增加 offload pin，或者改变 `free_cached_blocks()` 的返回值：

```cpp
std::vector<BlocksPerLayer> free_blocks_from_sequence(...);
```

建议不要让 `KVCacheOffloadManager` 事后从已经修改的 block table 猜测被释放的物理块。

### 7.2 Store 时序

必须满足：

```text
model infer/write 完成
  -> capture blocks
  -> enqueue GPU/host -> staging -> disk
  -> disk write 完成
  -> publish disk hash index
  -> release device/offload refs
```

对于 CPU `ov::Tensor`，`infer()` 返回后通常可以读 host memory，但仍需保持 pipeline 线程上的明确 happens-before。对于 GPU `ov::RemoteTensor`，不能假设 `infer()` 返回即代表所有 plugin queue 的 host-visible copy 已完成；需要使用 OpenVINO 支持的 `copy_to` / `copy_from` 或 `RemoteTensor` 机制，并做专门同步测试。

### 7.3 被 store 的 block 不能立即复用

现有 allocator 只知道 `m_ref_count`。推荐把 store in-flight pin 映射为额外引用：

```text
store begin: block.increment()
store complete: block.release()
```

但要确认 `free_cached_blocks()` 的语义不会把“只有 offload pin、没有 sequence owner”的 block 提前放入 overwrite store。更稳妥的办法是给 `CacheBlock` 增加显式状态：

```cpp
bool is_transfer_pinned() const;
```

并让 allocator 在 `transfer_pinned` 时不把 block 放入可复用 free/overwrite pool。

## 8. Load 链路设计

### 8.1 发现 disk prefix hit

当前 `BlockManager::get_prefix_restore_plan()` / `get_cached_block()` 只查：

1. `OverwritableBlocksHashStore`。
2. `m_prefix_hash_to_cached_blocks`。

需要把 disk index 作为第三个来源：

```text
get_cached_block(hash)
  -> active device cache
  -> overwritable host cache
  -> disk cache
```

但 disk hit 不能直接返回普通 `CacheBlock` 并加入 block table，因为它还没有 device 数据。建议返回带来源的 restore plan：

```cpp
struct PrefixRestoreSource {
    enum class Kind { DEVICE, HOST, DISK };
    Kind kind;
    size_t hash;
    size_t disk_slot;
    BlocksPerLayer source_blocks;
};
```

对于 disk source，流程应是：

```text
发现连续 disk hash hit
  -> 先分配 destination device BlocksPerLayer
  -> 把 block table 暂时标记为 LOAD_IN_FLIGHT 或不发布给 scheduler output
  -> enqueue_load(disk_slot -> destination physical IDs)
  -> load complete
  -> publish block table / update processed token count
```

### 8.2 与现有 `restore_cached_blocks()` 的关系

当前 `CacheOrchestrator::restore_cached_blocks()` 在 `add_request()` 阶段调用，且会直接让 block manager 把命中的 block 放入 sequence block table，并更新 processed tokens。

对 disk offload，需要拆成两阶段：

```text
plan_prefix_restore(): 只发现 hit，不修改可执行 block table
commit_prefix_restore(): load 完成后把 blocks 放入 block table并更新 tokens
```

或者保留现有 API，增加异步返回：

```cpp
RestoreResult restore_cached_blocks(...)
// RestoreResult::READY / LOAD_PENDING / MISS
```

推荐第一种，因为它更清楚地保证 ModelRunner 不会看到未恢复的 physical block。

### 8.3 Load 必须在 forward 前完成

OpenVINO 的 `step()` 当前直接：

```text
scheduler_output = m_scheduler->schedule(...)
logits = m_model_runner->forward(...)
```

因此必须在这两个调用之间增加：

```text
m_cache_orchestrator->start_pending_loads()
m_cache_orchestrator->wait_for_required_loads(scheduler_output)
```

或者把 load 纳入 `Scheduler::schedule()` 的输出，并由 `ContinuousBatchingImpl` 在 `forward()` 前等待。

不能把 load 放到 `infer()` 返回之后再做；那样本 step 的 paged attention 已经读取了旧数据。

### 8.4 Load 完成后的引用释放

load 完成后：

1. destination physical block 的 load pin 保留到 block table 正式可执行。
2. disk entry 的引用保持到所有需要该 block 的 sequence 完成或 block 被重新缓存。
3. load event 完成后才释放 source/destination 的 transfer pin。
4. request 被取消或 preempt 时，若 load in-flight，必须延迟释放。

## 9. Copy-on-write、fork 和 eviction 的特殊处理

### 9.1 Copy-on-write

`BlockManager::append_slots()` 在 fork 后发现 `last_blocks[0]->copy_on_write()` 时，会分配新 block，并返回：

```text
source physical block -> destination physical blocks
```

随后 `CacheOrchestrator::copy_blocks()` 调用 `KVCacheManager::copy_blocks()`。

如果 source 在 disk：

```text
disk source -> temporary/device source block -> destination block
```

不能让 `KVCacheManager::copy_blocks()` 直接从不存在于 device 的 physical block 读取。可行做法：

1. 在 copy 前检测 source location。
2. 先同步恢复 source 到 device temporary block。
3. 执行现有 device copy。
4. 释放 temporary block。

初版也可以规定：有 disk source 的 fork 必须先完成 prefix restore，再允许 fork，作为功能限制。

### 9.2 Partial eviction

`_maybe_evict_cache_blocks()` 使用 attention score 选择每层 logical block index，并调用：

```cpp
m_scheduler->free_blocks_from_sequence(
    seq_id,
    logical_blocks_to_evict,
    CacheType::KV_CACHE);
```

当前实现要求每层释放数量相同，并从 block table 移除这些 logical entries。Offload record 必须在这个调用内部捕获每层对应的 physical block，不能只保存 logical index，因为 eviction 后 logical index 可能被压缩重排。

同时要保留 OpenVINO 的：

- `m_block_table_logical_start`。
- `SequenceGroup::register_token_eviction()`。
- sparse attention skip set。
- SnapKV / Adaptive RKV 的逻辑 block 选择。

Offload 只改变 block 内容的位置，不应改变 eviction algorithm 的 score、logical index 或 token accounting。

### 9.3 完整 request free

`BlockManager::free_sequence()` 会释放整个 sequence 的 blocks。对于 prefix caching：

- 如果希望后续 request 命中内存 prefix，保留现有 overwrite store。
- 如果要把冷 prefix 进一步放到 disk，需要在进入 overwrite store 后由独立策略选择 disk store。
- request free 本身不等于必须写盘；否则会把所有短生命周期 request 都转化成磁盘 I/O。

## 10. 推荐的最小状态机

### 10.1 Block-set 状态

```text
DEVICE_ACTIVE
  -> STORE_PINNED
  -> STORE_IN_FLIGHT
  -> DISK_CACHED
  -> LOAD_ALLOCATED
  -> LOAD_IN_FLIGHT
  -> DEVICE_ACTIVE
```

失败路径：

```text
STORE_IN_FLIGHT -> DEVICE_FREE / DROP
LOAD_IN_FLIGHT  -> release destination + request retry / MISS
```

### 10.2 Event 状态

```text
CREATED
  -> QUEUED
  -> IO_RUNNING
  -> CUDA_OR_HOST_COPY_DONE
  -> PUBLISHED
  -> CLEANED
```

每个 event 需要：

```text
monotonic event_id
store/load kind
block pairs
affected hashes
affected request IDs
exception state
```

### 10.3 不允许的状态

```text
DISK_CACHED + 没有 disk hash index
LOAD_IN_FLIGHT + block table 已交给 ModelRunner
STORE_IN_FLIGHT + physical block 已进入 allocator 可复用 free pool
同一 hash 同时指向两个可写 destination
```

## 11. OpenVINO Runtime 层的同步风险

### 11.1 CPU

CPU 版本最适合作为第一阶段：

- `ov::Tensor::data()` 可用于连续 block copy。
- 使用固定 staging buffer + `pread/pwrite` 或 C++ 文件流即可。
- 先实现同步 store/load，验证状态机后再加入后台线程。

### 11.2 GPU RemoteTensor

GPU 是迁移的主要风险：

- `m_context.create_tensor()` 返回 RemoteTensor，不能把 `data()` 当作普通 host pointer。
- `KVCacheManager::copy_blocks()` 当前对 remote tensor 遇到 `is_remote` 时直接 return，说明 GPU copy 的实际工作由模型/plugin 或其他机制承担，不能从这个函数推断 host/device copy 已完成。
- `ov::RemoteTensor::copy_to()` / `copy_from()` 的同步与设备队列语义必须通过 OpenVINO Runtime 文档和 GPU plugin 测试确认。
- `InferRequest::infer()`、RemoteTensor copy、host readback 之间要有明确的完成点。

建议分阶段：

```text
Phase 1: CPU device + synchronous disk backend
Phase 2: CPU device + background I/O + event state machine
Phase 3: GPU device + explicit RemoteTensor -> host staging copy
Phase 4: GPU copy overlap / pinned host / direct I/O optimization
```

不要在 Phase 1 直接引入 `O_DIRECT`、多线程和 GPU 异步 copy；否则很难区分状态机 bug、alignment bug 和 plugin synchronization bug。

## 12. 推荐实现步骤

### Phase 0：只读验证和测试夹具

1. 给 `KVCacheManager` 增加可测试的 `read_block()` / `write_block()` 或 layer/key/value ROI helper。
2. 给 `BlockManager` 增加能返回“即将释放的 BlocksPerLayer”的内部接口。
3. 使用 CPU tensor 填充可识别 pattern：

```text
byte = layer_id + key_or_value_id + physical_block_id + offset
```

4. 验证一个 block 写盘后读回到另一个 physical block，逐字节比较。

### Phase 1：同步 CPU disk store/load

只支持：

- CPU。
- prefix caching。
- full prefix restore。
- prefix-cache 生命周期驱动的 store（见第 1.1 节勘误 2）。
- 同步 load before forward。
- 暂不支持 `use_cache_eviction == true`、GPU、partial async、COW from disk、O_DIRECT。

验收：eager store/load roundtrip、重复 hash 去重、disk slot 复用、load 后 logits 与 baseline 相同。

### Phase 2：独立 I/O 线程和 event

加入：

- store/load 两个队列。
- 固定 staging buffer。
- monotonic event ID。
- `poll_completed()`。
- preemption/reset flush。
- in-flight block pin。

验收：store 不阻塞 load、取消 request 不崩溃、异常能回传主线程。

### Phase 3：GPU RemoteTensor

加入明确的：

```text
RemoteTensor block -> host staging
host staging -> disk
 disk -> host staging -> RemoteTensor block
```

先使用同步 copy，确认 plugin 正确性后再优化异步 overlap。

### Phase 4：优化

最后才考虑：

- `O_DIRECT` 和 4096 对齐。
- 多 buffer slot 流水线。
- lazy free queue scanning。
- 批量 pwritev/preadv。
- 更复杂的 host + disk 多级 cache。

## 13. 必须覆盖的测试

### Block / storage 单元测试

- 每层 key/value block 的 file offset 正确。
- 不同 physical block 不互相覆盖。
- 同一 hash 的所有 layer 一致。
- short read/write 被检测。
- disk slot 分配、释放和复用。
- 文件权限和 shutdown unlink。

### Scheduler / state machine 测试

- eviction 后 store record 捕获正确 physical block，而不是错误的 logical index。
- store 未完成时 block 不能再次分配。
- store 完成前 hash 不可命中。
- load hit 只在 load 完成后更新 processed token 和 block table。
- 两个 request 同时命中同一 disk hash 时不会重复破坏 source。
- request 在 load in-flight 时取消，引用延迟释放。
- preemption/reset 后旧 transfer 完成不会重新注册 stale cache。
- COW / fork 在 device source 与 disk source 两种情况下行为正确。

### Model correctness 测试

- 无 offload baseline 与 offload 输出 token 完全一致。
- prompt prefix 命中后输出一致。
- partial eviction + disk restore 输出一致。
- 多 decoder layer 一致。
- key/value precision 包括 f16、bf16、u4/i4 需要分别验证。
- CPU device 先验证；GPU device 单独验证 RemoteTensor 同步。
- hybrid model 的 linear attention cache 不应被 KV disk backend 错误处理。

### 性能测试

记录：

```text
TTFT
TPOT
throughput
store bytes / load bytes
store queue wait
load queue wait
blocking wait before forward
cache hit rate
```

功能正确后再比较：

```text
baseline
synchronous disk
background disk
background disk + staging pipeline
```

## 14. 常见错误方案

### 错误 1：只在 `model_runner.hpp` 中看到 block index 就做 load

`ModelRunner` 只有逻辑到 physical block index 的输入填充职责，不适合拥有 disk cache 索引和请求生命周期。load 需要在 scheduler 已经决定 restore、且 forward 使用 block table 之前完成。

### 错误 2：在 `free_blocks_from_sequence()` 之后再查找被释放块

该函数会修改 block table。应在释放前捕获 `BlocksPerLayer`，并增加 transfer pin。

### 错误 3：把 disk slot ID 直接当成 physical block ID

二者属于不同地址空间：

```text
physical block ID -> key_cache/value_cache tensor
 disk slot ID     -> offload file
```

load 必须显式指定 source disk slot 和 destination physical block。

### 错误 4：store 完成前把 hash 加入 prefix cache

磁盘 I/O 或 GPU/host copy 仍未完成时命中，会读到半写数据。只有 event 完成后才能 publish hash index。

### 错误 5：用 request ID 作为唯一 cache key

prefix cache 共享的是 block hash，不是 request。request 只用于 load 完成通知和生命周期清理。

### 错误 6：忽略 per-layer block set

OpenVINO 的 KV cache 是每层一对 key/value tensor，`BlockManager` 还要求各层同步。任何只处理 layer 0 的实现都可能生成 silent correctness bug。

### 错误 7：第一版直接照搬 GPU + O_DIRECT + async

这会同时引入 RemoteTensor 同步、pinned memory、文件对齐、线程异常和 block 生命周期问题。应先用 CPU + synchronous disk 验证语义。

## 15. 源码附录：OpenVINO GenAI

### 核心调用链

- [docs/paged_kv_cache_call_chain.md](paged_kv_cache_call_chain.md)：已有 paged KV cache 初始化和 step 调用链。
- [src/cpp/src/continuous_batching/pipeline_impl.cpp](../src/cpp/src/continuous_batching/pipeline_impl.cpp)：模型编译、cache orchestrator 创建、`step()`、infer 后 eviction。
- [src/cpp/src/continuous_batching/pipeline_impl.hpp](../src/cpp/src/continuous_batching/pipeline_impl.hpp)：`_maybe_evict_cache_blocks()` 声明和 pipeline 成员。
- [src/cpp/src/continuous_batching/scheduler.hpp](../src/cpp/src/continuous_batching/scheduler.hpp)：schedule、Scheduler::Output、block table 输出、preemption/free API。
- [src/cpp/src/continuous_batching/model_runner.hpp](../src/cpp/src/continuous_batching/model_runner.hpp)：forward、infer 输入准备、block_indices 填充。

### Cache abstraction

- [src/cpp/src/continuous_batching/cache/cache_orchestrator.hpp](../src/cpp/src/continuous_batching/cache/cache_orchestrator.hpp)：cache type 注册、物理 cache allocation、restore、free、copy 路由。
- [src/cpp/src/continuous_batching/cache/block_manager.hpp](../src/cpp/src/continuous_batching/cache/block_manager.hpp)：`CacheBlock`、`BlockAllocator`、`OverwritableBlocksHashStore`、logical/physical table、prefix restore、free/append/COW。
- [src/cpp/src/continuous_batching/cache/kv_cache_manager.hpp](../src/cpp/src/continuous_batching/cache/kv_cache_manager.hpp)：per-layer key/value tensor、block size、物理 cache allocation、copy blocks。
- [src/cpp/src/continuous_batching/cache/i_cache_manager.hpp](../src/cpp/src/continuous_batching/cache/i_cache_manager.hpp)：物理 cache manager 抽象接口。
- [src/cpp/src/continuous_batching/cache/cache_type.hpp](../src/cpp/src/continuous_batching/cache/cache_type.hpp)：KV_CACHE 与其他 cache type。

### Eviction

- [src/cpp/src/continuous_batching/cache/cache_eviction.hpp](../src/cpp/src/continuous_batching/cache/cache_eviction.hpp)：eviction algorithm 接口和 score tracking。
- [src/cpp/src/continuous_batching/cache/cache_eviction.cpp](../src/cpp/src/continuous_batching/cache/cache_eviction.cpp)：score 聚合、logical block eviction。
- [src/cpp/src/continuous_batching/cache/kvcrush.hpp](../src/cpp/src/continuous_batching/cache/kvcrush.hpp)：KVCRUSH 相关策略。
- [src/cpp/src/continuous_batching/sparse_attention.cpp](../src/cpp/src/continuous_batching/sparse_attention.cpp)：sparse attention 与跳过 logical block 的相关逻辑。

### Tests

- [tests/cpp/cache_manager.cpp](../tests/cpp/cache_manager.cpp)：KVCacheManager 物理 cache allocation 测试。
- [tests/cpp/cache_orchestrator_hybrid.cpp](../tests/cpp/cache_orchestrator_hybrid.cpp)：cache orchestrator 与 hybrid cache 测试。
- [tests/cpp/block_manager.cpp](../tests/cpp/block_manager.cpp)：block table、prefix cache、物理 block 生命周期测试。
- [tests/cpp/scheduler.cpp](../tests/cpp/scheduler.cpp)：scheduler、block allocation、preemption、eviction 调用测试。
- [tests/cpp/cache_eviction.cpp](../tests/cpp/cache_eviction.cpp)：eviction algorithm 和 logical block 选择测试。
- [tests/python_tests/test_kv_cache_eviction/test_kv_cache_eviction_1.py](../tests/python_tests/test_kv_cache_eviction/test_kv_cache_eviction_1.py)：端到端 KV eviction 配置与行为测试。

## 16. 与 vLLM 导读的关系

已有 vLLM 磁盘 offload 说明：

- [vllm/docs/paged_kv_cache_offload_to_disk_zh.md](../../vllm/docs/paged_kv_cache_offload_to_disk_zh.md)：vLLM 的完整 store/load、event、disk backend 说明。

阅读顺序建议：

```text
1. 本文第 2、3 节：理解 OpenVINO 当前真实对象和 step 顺序
2. 本文第 5、6、7、8、9 节：实现迁移规格
3. vllm/docs/paged_kv_cache_offload_to_disk_zh.md：对照 vLLM 的状态机细节
4. OpenVINO 的 block_manager.hpp / kv_cache_manager.hpp：落到 C++ 生命周期
5. 先实现 Phase 1 测试，再逐步增加 async/GPU 优化
```
