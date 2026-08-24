# OpenVINO GenAI Paged KV Cache `offload_to_disk` 施工方案

## 1. 目标与范围

本文是基于当前 `openvino.genai` 源码和 `paged_kv_cache_offload_migration_spec_zh.md` 制定的实施方案，目标是在 Continuous Batching + Paged Attention pipeline 中增加 KV cache 的磁盘 offload 能力。

第一版只追求正确性和可验证性：

- 设备侧继续使用现有 `KVCacheManager` 和 paged-attention block table。
- 即将失去内存副本的完整 KV block 写入临时磁盘文件。
- 后续 prefix 命中时，从磁盘恢复到新的设备 physical block，再交给 `ModelRunner`。
- scheduler 只做 block/transfer 决策，不执行文件 I/O。
- 首个可交付版本只支持 CPU device、同步 store/load、page cache I/O。

第一版的强制前提（不满足时直接拒绝启用 offload）：

```text
enable_prefix_caching == true    // 磁盘条目只能靠 block hash 重新发现
use_cache_eviction   == false    // 见 3.1，eviction 会改变 block table 的层语义
device               == CPU
```

第一版明确不做：`use_cache_eviction = true`、GPU `RemoteTensor` 异步拷贝、`O_DIRECT`、跨进程持久化 cache、跨模型复用、linear-attention cache offload 和基于复杂优先级的后台预取。

## 2. 当前代码控制点

| 位置 | 当前职责 | 施工改造 |
| --- | --- | --- |
| `src/cpp/include/openvino/genai/scheduler_config.hpp` | 暴露 `SchedulerConfig` | 增加可序列化的 `KVCacheOffloadConfig`，先支持 `enabled`、`path`、`capacity_bytes` |
| `src/cpp/src/continuous_batching/cache/kv_cache_manager.hpp` | 持有每层 key/value tensor；计算 block bytes；扩容和 device block copy | 增加按 layer/key/value/block 的读写 helper；由 offload manager 调用，不在这里做 eviction policy |
| `src/cpp/src/continuous_batching/cache/block_manager.hpp` | 维护 logical table、physical block、prefix cache、free/overwrite store | 增加 block-set 捕获、transfer pin 和 disk restore plan；保证 in-flight block 不进入可复用池 |
| `src/cpp/src/continuous_batching/cache/cache_orchestrator.hpp` | 注册 cache manager/block manager，协调 restore/free/copy | 持有 `KVCacheOffloadManager`；协调 store/load 生命周期和 block-table commit |
| `src/cpp/src/continuous_batching/pipeline_impl.cpp` | 创建 orchestrator/scheduler/model runner，并执行每个 step | 在 eviction 后提交 store；在 forward 前等待本 step 所需 load |
| `src/cpp/src/continuous_batching/model_runner.hpp` | 将 scheduler 输出的 block table 填入模型输入 | 保持只消费已完成 load 的 device physical block，不直接感知 disk slot |
| `src/python/py_continuous_batching_pipeline.cpp` | Python `SchedulerConfig` 绑定 | 暴露 offload 配置字段并做参数校验 |
| `src/js` 相关 helper/binding | JavaScript `SchedulerConfig` 转换 | 与 Python/C++ 保持字段一致；可在第二阶段接入 |

实际新增文件建议放在：

```text
src/cpp/src/continuous_batching/cache/kv_cache_disk_layout.hpp
src/cpp/src/continuous_batching/cache/kv_cache_offload_manager.hpp
src/cpp/src/continuous_batching/cache/kv_cache_offload_manager.cpp
src/cpp/src/continuous_batching/cache/kv_cache_offload_types.hpp
```

## 3. 设计约束

### 3.1 `BlocksPerLayer` 是 block-table 层，不是 decoder 层

这是最容易写错的一点。`BlocksPerLayer` 的长度由 `CacheOrchestrator::register_kv_cache()` 决定：

```cpp
const bool per_layer_control = config.use_cache_eviction;
const size_t num_block_table_layers = per_layer_control ? kv_manager->get_num_layers() : 1;
```

因此存在两种完全不同的物理布局：

| 模式 | `BlocksPerLayer.size()` | 一个逻辑 block 的数据位置 |
| --- | --- | --- |
| `use_cache_eviction == false`（默认） | 1 | 单个 physical block ID `p`，所有 decoder layer 都在 `key_cache.L[p]` / `value_cache.L[p]` |
| `use_cache_eviction == true` | decoder 层数 | 每个 decoder layer 有各自的 physical block ID `p_L`，位于 `key_cache.L[p_L]` |

`ModelRunner::_fill_indices_from_block_tables()` 中的注释也确认了这一点：非 eviction 模式下所有层共用一张 block table。

直接后果：

- 非 eviction 模式下，「一个 physical block ID 跨所有 decoder layer 的 key/value」是一个自洽的 offload 单位，可以整体读写成一个磁盘 slot。Phase 0 的 `KVCacheManager::read_block()` / `write_block()` 正是按这个模型实现的。
- eviction 模式下该模型不成立：一个逻辑 block 的内容分散在 N 个互不相同的 physical index 上，必须按 `(decoder_layer, physical_block_id)` 粒度读写。Phase 0 的 API 在该模式下会搬错数据。

所以第一版限定 `use_cache_eviction == false`。要支持 eviction 模式，必须先给 `KVCacheManager` 增加按 decoder layer 单独读写 block 的重载，并让 `KVCacheDiskLayout` 支持每层独立 slot。

### 3.2 slot 大小必须来自 layout

不要用 `KVCacheManager::get_block_size_in_bytes()` 当作 slot 大小。该值用 `ov::element::Type::size()` 累加，而 `size()` 返回 `(bitwidth + 7) / 8`，因此 `u4` / `i4` KV cache 会被按每元素 1 字节高估。它可以继续用于内存预算，但 slot 大小、offset 和 I/O 长度必须统一来自 `KVCacheDiskLayout`。

推荐 slot 布局（此处 layer 指 decoder layer）：

```text
[layer 0 key][layer 0 value][layer 1 key][layer 1 value]...
```

由一个 `KVCacheDiskLayout` 同时生成 store/load offset，避免两侧手算产生不可逆布局。文件采用固定 slot 大小，容量为 `capacity_bytes / slot_size` 向下取整。子字节精度必须保证单个 block 的比特数按字节对齐，否则拒绝 offload。

### 3.3 disk slot 不是 block table

Paged Attention 只接受设备 physical block ID。disk hit 必须经过以下两阶段：

```text
disk hash hit
  -> 分配 destination BlocksPerLayer
  -> load slot 到 destination
  -> load 成功后 commit block table
  -> ModelRunner::forward()
```

在 `LOAD_IN_FLIGHT` 期间不得把 destination block table 放入 `Scheduler::Output`。

### 3.4 正确的 store 触发点是内存 prefix cache 的生命周期

`OverwritableBlocksHashStore` 就是现成的「已无 sequence 持有、但内容仍然有效且可被 hash 命中」的内存池，它天然是磁盘的上一级。相关事实：

- `BlockAllocator::free()` 只有在跨层 hash 全部一致时才把 block 放入 `m_overwriteable_blocks`；否则直接进 free pool，内容不再可被 prefix 命中。
- `OverwritableBlocksHashStore::get_lru_block_to_overwrite()` 是内容被真正破坏的唯一时刻，它把 block 交出去用于覆写。

因此：

```text
store 候选：block 进入 OverwritableBlocksHashStore 时
store 必须完成：该 block 被 get_lru_block_to_overwrite() 取走覆写之前
```

不要把 `_maybe_evict_cache_blocks()` 当作第一版的 store 来源，理由见 Phase 2。

### 3.5 store 不能与 allocator 复用竞争

被选中的 block 可能已从 sequence table 移除，但其数据仍是 store 的源。必须先捕获 physical block-set，再增加 transfer pin；磁盘写入成功或失败后才能释放该 pin。建议在 `CacheBlock` 上增加独立的 `transfer_pin_count`，不要把 transfer pin 隐式混入 prefix-cache 引用计数。

`BlockManager::free_blocks_from_sequence()` 返回的 block-set 是在本次调用释放任何块之前捕获的，但这些块并没有被 pin。它们的内容只在下一次 allocator 分配之前有效，因此调用方必须在释放路径内部就完成 pin，而不是拿到返回值之后再补。

### 3.6 active、memory prefix、disk 的优先级

同一 prefix hash 可能同时存在于多个位置，查找顺序固定为：

```text
active device block > overwritable memory block > disk entry
```

disk entry 只有在 store 完成后才能发布；load 失败时必须删除或标记无效，不能留下可重复命中的半成品。

## 4. 建议的数据结构与接口

```cpp
enum class OffloadLocation {
    DEVICE,
    STORE_IN_FLIGHT,
    DISK,
    LOAD_IN_FLIGHT,
};

struct KVCacheOffloadConfig {
    bool enabled = false;
    std::string path;
    size_t capacity_bytes = 0;
    size_t buffer_slots = 2;
    bool use_page_cache = true;
};

struct DiskCacheEntry {
    size_t prefix_hash = 0;
    size_t slot_id = 0;
    uint64_t generation = 0;
};

struct TransferRecord {
    uint64_t event_id = 0;
    std::vector<BlocksPerLayer> source_blocks;
    std::vector<BlocksPerLayer> destination_blocks;
    std::vector<size_t> slot_ids;
};
```

第一版接口可以保持同步，但保留 event 返回值，便于后续替换为后台 I/O：

```cpp
uint64_t store(const BlocksPerLayer& blocks, size_t prefix_hash);
uint64_t load(size_t slot_id, const BlocksPerLayer& destination);
void wait(uint64_t event_id);
void flush();
void reset();
```

`KVCacheOffloadManager` 只负责文件、slot、layout、读写和 transfer 记录，不修改 `BlockManager` 的 hash map。index 的发布和 block pin 的回收由 `CacheOrchestrator` 统一完成。

## 5. 分阶段施工步骤

### Phase 0：夹具与边界确认（已完成）

1. 在 `KVCacheManager` 增加只读 accessor：layer 数、key/value tensor、block byte stride、remote/device 类型。
2. 增加 `KVCacheDiskLayout` 与 `read_block()` / `write_block()`，把一个 physical block 的 key/value 内容在 CPU tensor 和字节缓冲之间双向搬运。
3. `BlockManager::free_blocks_from_sequence()` 返回释放前捕获的 `BlocksPerLayer`。
4. 用 CPU tensor 写入可识别 pattern，验证 block layout、sub-byte precision 和多 layer 对称性。

验收：不启用 offload 时现有测试行为不变；layout slot size 与实际分配 tensor 的 per-block 字节数一致；sub-byte 未对齐时给出明确错误。

已知限制：`read_block()` / `write_block()` 把一个 physical block ID 当作跨所有 decoder layer 的整体，仅在 `use_cache_eviction == false` 时成立（见 3.1）。

### Phase 1：同步 CPU disk backend

1. 新增 `KVCacheOffloadManager`，复用 Phase 0 的 `KVCacheDiskLayout`。
2. 创建 run-specific 临时文件：父目录存在、文件使用独占创建、权限限制为当前用户可读写，pipeline 销毁时清理。
3. 以 `pread/pwrite` 或等价的二进制随机 I/O 实现固定 slot store/load；检查完整读写长度和整数溢出。
4. 在 `SchedulerConfig` 增加配置，并在 C++/Python 绑定中暴露。
5. 只允许 CPU device 启用该功能；GPU/NPU 配置直接报清晰错误。

验收：单 block、多 layer、多个 slot、重启前后临时文件清理、磁盘空间不足和短读写均有单元测试。

### Phase 2：接入 prefix-cache 生命周期的 store

不要以 `_maybe_evict_cache_blocks()` 作为第一版的 store 来源。原因有三个，都是代码层面的硬约束：

1. `_maybe_evict_cache_blocks()` 仅在 `use_cache_eviction == true` 时执行，而该模式会把 block table 切换成 per-decoder-layer，与 3.1 的 slot 模型和 Phase 0 的 API 直接冲突。
2. 该模式下每层 evict 的是各自独立的 logical block，所以同一次 `free_blocks_from_sequence()` 释放的跨层 block-set 通常 hash 不一致。`BlockAllocator::free()` 对这种情况直接走 free pool 分支，不进 `OverwritableBlocksHashStore`，也就永远不会被 prefix 命中。把它们写盘只会产生永远命中不了的死数据。
3. eviction 的语义是「这些 token 已被放弃」，与 offload 的「内容换个地方保存」并不同，两者耐久性模型不一致。

第一版改为挂在内存 prefix cache 的生命周期上：

```text
free_sequence() / free_sequence_partially() / append_slots() 的 COW 释放
  -> BlockAllocator::free()
  -> 跨层 hash 一致，block 进入 OverwritableBlocksHashStore
  -> 标记为 store 候选（可以立即写，也可以延迟写）
  -> 写盘完成后发布 hash -> slot

OverwritableBlocksHashStore::get_lru_block_to_overwrite()
  -> 该 block 内容即将被覆写
  -> 如果尚未落盘，必须先完成 store，或放弃该条目
```

这条路径同时满足：block 已无 sequence 持有（内容稳定）、跨层 hash 一致（可被 prefix 命中）、且存在明确的「内容即将丢失」时刻（store 的真正截止点）。

实现上建议给 `BlockManager` 增加 offload 回调或观察接口，而不是在 `BlockAllocator` 内部直接做 I/O；`BlockAllocator` 应继续只管块的所有权。

### Phase 3：接入 disk prefix restore

当前 `CacheOrchestrator::restore_cached_blocks()` 会直接调用 block manager 恢复并更新 processed tokens。需要拆出或扩展为：

```text
plan_prefix_restore()   // 只查 active/memory/disk，不发布 disk block
allocate_restore_blocks()
load_disk_blocks()
commit_prefix_restore() // load 成功后更新 block table 和 processed tokens
```

必须注意调用位置和线程：`restore_cached_blocks()` 是在 `ContinuousBatchingImpl::add_request()` 里调用的，运行在**调用方线程**，与 `step()` 并发，并且发生在请求进入 `m_awaiting_requests` 之前。因此：

- 不能在这里同步做大块磁盘 I/O，否则会直接阻塞用户的 `add_request()` 调用。
- 在这里分配 destination block 会与 `step()` 中的分配竞争 allocator，必须走 `BlockManager` 已有的 `m_cached_blocks_map_mutex` 保护路径。
- 推荐做法：`add_request()` 只生成 restore plan 并记录到 `SequenceGroup`，真正的 allocate + load + commit 放到 `step()` 中 `schedule()` 之后、`ModelRunner::forward()` 之前执行。

失败处理为释放 destination、使 disk entry 失效，并将该 prefix 当作 miss 重新调度，不能用未初始化 block 继续 forward。

### Phase 4：fork、copy-on-write、取消和清理

1. `copy_blocks()` 遇到 disk source 时，第一版先同步 restore 到临时 device block，再走现有 device copy；也可以在 scheduler 层暂时禁止 disk source fork。
2. request 完成、取消或 preemption 时，若关联 load/store 仍在进行，延迟释放 block 和 transfer record。
3. pipeline 析构和异常路径统一调用 `flush/reset`，保证没有后台任务访问已销毁的 tensor 或文件。注意 `BlockManager::clear()` 开头就断言 `m_enable_prefix_caching == false`，而 offload 要求 prefix caching 必须打开，所以 offload 的清理不能挂在 `clear()` 上，需要独立的释放入口。
4. 同一 hash 的新 generation 发布时，旧 slot 延迟回收，避免旧 load 完成后覆盖新 index。

### Phase 5：后台 I/O 与 GPU 支持

在 CPU 同步链路稳定后再实现：

- 独立 store/load 队列，load 优先级高于 store。
- 单调 event ID 和异常传播。
- staging buffer pool 和有限 buffer slots。
- GPU `RemoteTensor` 的显式 host staging copy 及完成同步。
- 性能优化：批量 I/O、预取、pinned host memory、可选 direct I/O。

GPU 版本必须先用实际 plugin 验证 `InferRequest::infer()`、`RemoteTensor::copy_to/copy_from()` 和 host readback 的 happens-before，不能由普通 CPU `ov::Tensor` 的语义推断。

## 6. 测试方案

### 6.1 单元测试

新增 cache 目录下的 C++ 测试，覆盖：

- `KVCacheDiskLayout` offset、slot size、不同 layer shape/precision。
- layout slot size 与实际分配 tensor 的 per-block 字节数一致。
- block store/load byte-for-byte round-trip。
- slot 分配、容量耗尽、slot 重用和 generation 校验。
- store/load 失败时 index 不发布、pin 必须回收。
- `LOAD_IN_FLIGHT` 不可见于 scheduler output。
- transfer-pinned block 不可被 allocator 分配。
- active/memory/disk 三种 prefix source 的优先级。

### 6.2 Continuous Batching 集成测试

在 `tests/python_tests/test_continuous_batching.py` 增加 CPU-only 测试夹具：

1. 关闭 offload 与开启 offload 使用相同模型、prompt、greedy generation，比较 token/text 结果。
2. 让 cache 容量小于多个请求的共享 prefix 工作集，强迫内存 prefix cache 发生 LRU 覆写，验证后续命中 disk prefix 仍得到相同结果。
3. 多 request 交错执行，覆盖一个 request store、另一个 request load 的情况。
4. 覆盖 request cancel、preemption、pipeline clear 和临时目录清理。
5. 用 debug trace 或测试 accessor 验证 forward 前所有 block table 都是 device physical ID。
6. 验证 `use_cache_eviction = true` 与 offload 同时开启时报出明确错误而不是静默跑错。

GPU 测试在 Phase 5 单独加入，不能把 CPU 测试的通过当作 GPU 同步正确性的证明。

### 6.3 性能与容量测试

记录以下指标：

- store/load 次数、bytes、平均/尾延迟。
- disk hit/miss、memory hit/miss。
- forward 等待 load 的时间。
- device cache 使用量与磁盘占用量。
- 同一 workload 下 offload 开关前后的生成吞吐和首 token 延迟。

## 7. 配置与兼容性

建议配置语义：

```text
offload.enabled = false       // 默认关闭，保持完全兼容
offload.path                    // 空路径由实现创建临时目录；显式路径必须可写
offload.capacity_bytes          // 0 表示按 device cache block 数推导或直接拒绝，需固定语义
offload.buffer_slots = 2
offload.use_page_cache = true
```

配置必须参与 `SchedulerConfig::operator==`、`to_string()`、Python binding 和配置校验。默认关闭时不得创建文件、线程或改变已有 prefix-cache 行为。

`validate()` 必须在 `enabled == true` 时强制检查：

```text
enable_prefix_caching == true   // 否则 disk 条目永远无法被命中
use_cache_eviction   == false   // 否则 block table 层语义与 slot 模型不一致
```

linear attention cache、speculative decoding 的 draft pipeline 和 visual-language pipeline 在第一版应显式标记为 unsupported，直到它们的 cache 生命周期被单独验证。

## 8. 主要风险与处理原则

| 风险 | 处理原则 |
| --- | --- |
| 把 `BlocksPerLayer` 当成 decoder 层 | 它是 block-table 层；非 eviction 模式下 size 为 1，一个 physical ID 覆盖所有 decoder layer |
| 在 eviction 模式下复用单 slot 模型 | 第一版直接拒绝 `use_cache_eviction == true` |
| store 了永远命中不了的 block | 只 store 跨层 hash 一致、已进入 `OverwritableBlocksHashStore` 的 block |
| 内存 block 被 LRU 覆写前未落盘 | 以 `get_lru_block_to_overwrite()` 为 store 的硬截止点 |
| physical block 已被复用 | 捕获 block-set 后立即 pin，在释放路径内部完成 |
| slot 大小取错导致数据截断 | slot 大小和 offset 统一来自 `KVCacheDiskLayout`，不使用 `get_block_size_in_bytes()` |
| disk load 后 table 提前发布 | 采用 plan/load/commit 三阶段，forward 前强制 wait |
| 在 `add_request()` 里做阻塞 I/O | 该函数跑在调用方线程；只生成 plan，load 放到 `step()` |
| GPU RemoteTensor 不可直接读 host | CPU first；GPU 使用显式 staging 和 plugin 同步测试 |
| hash/index 与 I/O 不一致 | store 完成后才 publish；generation 防止旧 event 覆盖新 entry |
| 文件被误当持久化 cache | run-specific 文件，版本/模型校验不作为第一版跨运行协议 |
| I/O 阻塞生成 | 第一版允许同步以验证正确性；性能阶段再拆 store/load worker |

## 9. 交付顺序与完成标准

建议按以下变更拆分 review：

1. Phase 0：layout、ROI 和 block capture 测试。
2. Phase 1：独立 disk backend 与配置/绑定。
3. Phase 2：prefix-cache 生命周期驱动的 store。
4. Phase 3：disk prefix restore。
5. Phase 4：异常、fork、取消和清理。
6. Phase 5：异步 I/O/GPU，单独评审性能和设备同步。
7. 后续单独立项：`use_cache_eviction == true` 的 per-decoder-layer slot 支持。

Phase 3 完成才算 `offload_to_disk` 功能闭环。最终验收必须同时满足：

- 默认配置下已有 Continuous Batching 测试无回归。
- CPU 上 disk store/load round-trip 通过多 layer/precision 测试。
- disk hit 产生的 block 在 forward 前已恢复到设备 tensor。
- store/load in-flight block 永不被 allocator 复用。
- I/O、容量、取消和异常路径不会泄漏文件、引用或线程。
- 开关关闭时不产生额外 I/O 和可观察的调度语义变化。