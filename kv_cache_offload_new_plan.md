<!-- Copyright (C) 2026 Intel Corporation -->

# upstream openvino.genai 多级 KV Cache Offloading 架构设计与实施计划（含可插拔 SSD Plugin 扩展）

> **文档定位**：本文档为将单进程 KV Cache Offloading 方案从 `pipeline.mx` 移植并升级至 upstream `openvino.genai`（`genai.mx`）与 `openvino` runtime（`openvino.mx`）的权威架构设计规范与落地工程计划。
> **核心目标**：在 upstream `openvino.genai` 的 Continuous Batching 框架中构建原生的三级缓存体系（L0 Device → L1 Host DRAM → L2 Local Storage/SSD），实现跨 Session 极速前缀复用、断电/重启持久化缓存、强租户隔离，并提供 **Vendor SSD Plugin 扩展架构**，支持第三方 SSD 硬件厂商（如 Solidigm、Samsung、Kioxia、Intel 等）接入定制化驱动（SPDK、io_uring、ZNS/FDP、GPUDirect/USM Direct I/O）以最大化硬件吞吐。
> **基线代码**：`thirdparty/openvino.genai`（ov_master）与 `thirdparty/openvino`（ov_master）。

---

## 1. 架构目标与设计原则

### 1.1 核心诉求与业务场景
1. **单 Session 内与跨 Session Prefix Caching**：长系统提示词（System Prompt）、多轮对话（Multi-turn Chat）、Few-shot 模板等前缀 KV Block 零重复计算。
2. **跨进程 / 实例重启持久化**：服务重启或模型重新加载后，历史高频 Prefix KV Block 从本地 SSD 毫秒级恢复，消除冷启动延迟。
3. **超长上下文（Long Context）显存卸载**：当 GPU/NPU 显存吃紧时，自动将非活跃、远端或即将被淘汰的 KV Block 下沉至 Host 内存或 SSD，避免 OOM 并大幅提升并发 Capacity。
4. **强租户与会话安全隔离**：原生内置 `tenant_id` / `cache_salt` 链式哈希污染机制，防止跨租户越权命中与侧信道信息泄漏。
5. **硬件厂商生态开放（SSD Plugin）**：提供轻量、高性能、解耦的存储抽象接口与 C/C++ ABI 动态载入机制，允许 SSD 厂商提供专属优化的硬件插件，替代默认 POSIX 文件后端。

### 1.2 架构设计原则（Karpathy Guidelines & OpenVINO Core）
- **In-Tree 原生契合**：深度复用 `openvino.genai` 已有的 `Sequence::_make_hash`、`OverwritableBlocksHashStore`、`CacheOrchestrator` 等基础设施，不破坏现有连续批处理调度模型。
- **正交分层与极简契约**：L0/L1/L2 各层只对抽象 Block 及其 Hash 负责，存储后端与设备计算流水线解耦。
- **Surgical Changes**：对核心调度主循环改动保持最小化，仅在 Block 分配（Allocate）、淘汰（Evict）、恢复（Restore）三个生命周期节点插入分层感知逻辑。
- **渐进增强（Graceful Degradation）**：若未配置 SSD 路径或插件加载失败，系统平滑退化为 L0 单层或 L0+L1 双层模式，保证零崩溃。

---

## 2. 总体分层架构图

```
+----------------------------------------------------------------------------------------------------+
|                                    User Application / Facade                                       |
|  GenerationConfig(tenant_id, cache_salt)  |  SchedulerConfig(enable_kv_cache_offloading, ssd_...)    |
+----------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+----------------------------------------------------------------------------------------------------+
|                                  openvino.genai Continuous Batching                                 |
|                                                                                                    |
|  +-----------------------------------------------------------------------------------------------+  |
|  |                                      Scheduler & Pipeline                                     |  |
|  |   - SequenceGroup / Sequence (tenant_id 混入虚拟 Block-0 链式哈希)                                |  |
|  |   - PrefixRestorePlan (生成包含 L0 / L1 / L2 来源的多级恢复计划)                                 |  |
|  +-----------------------------------------------------------------------------------------------+  |
|                                                |                                                   |
|                                                v                                                   |
|  +-----------------------------------------------------------------------------------------------+  |
|  |                                       CacheOrchestrator                                       |  |
|  |  +----------------------------+  +----------------------------------------------------------+ |  |
|  |  |   BlockManager (L0 元数据)   |  |            TieredCacheManager (分级缓存协调器)            | |  |
|  |  | - m_prefix_hash_to_blocks  |  | - L0/L1/L2 块迁移状态追踪                                  | |  |
|  |  | - OverwritableBlocksStore  |  | - 异步 Swap-Out / Swap-In 队列与流控                       | |  |
|  |  | - LRU / Eviction Policy    |  | - 多级淘汰与命中判定策略 (Hit/Miss Decision Engine)         | |  |
|  |  +----------------------------+  +----------------------------------------------------------+ |  |
|  +-----------------------------------------------------------------------------------------------+  |
+----------------------------------------------------------------------------------------------------+
          |                                      |                                    |
          v                                      v                                    v
+-----------------------+              +-------------------+               +-------------------------+
|   L0: Device Cache    |  D2H Copy    |  L1: Host Cache   |  Storage IO   |    L2: Storage Cache    |
| (GPU VRAM / NPU USM)  | <=========>  | (Pinned Host RAM) | <===========> |    (Local NVMe / SSD)   |
|                       |  H2D Copy    |                   |               |                         |
| KVCacheManager        |              | HostBlockPool     |               | IKVCacheStorageBackend  |
| (OpenVINO Tensor/USM) |              | (Aligned Memory)  |               | (Plugin Architecture)   |
+-----------------------+              +-------------------+               +-------------------------+
                                                                                        |
                                                          +-----------------------------+-----------------------------+
                                                          |                                                           |
                                                          v                                                           v
                                            +---------------------------+                               +---------------------------+
                                            |   DefaultFileStorage      |                               |   Vendor SSD Plugin       |
                                            | (In-Tree Default Backend) |                               | (Solidigm/Samsung/Custom) |
                                            |                           |                               |                           |
                                            | - POSIX pread/pwrite/mmap |                               | - SPDK Polling Driver     |
                                            | - Container Block File    |                               | - Linux io_uring / Direct |
                                            | - JSON/Binary Manifest    |                               | - ZNS / FDP Flash Stream  |
                                            | - Zero-dependency         |                               | - USM/GPUDirect P2P DMA   |
                                            +---------------------------+                               +---------------------------+
```

---

## 3. 核心子系统详细设计

### 3.1 安全隔离与链式哈希扩展（Tenant Isolation & Hash Scheme）

#### 3.1.1 漏洞根因与防线突破
标准 vLLM 的 `cache_salt` 仅混入第 0 个物理块的哈希计算中；如果匹配请求从第 1 个块之后切入，或者攻击者构造了相同后缀，存在越权利用公共前缀的风险。

#### 3.1.2 链式传播哈希设计
在 `openvino.genai` 中，`Sequence::_make_hash`（`src/cpp/src/sequence_group.cpp`）基于前一个块的哈希值与当前块 Token 序列迭代计算。
我们将 `tenant_id` 与可选 `cache_salt` 封装为 **Sequence 根种子（Virtual Parent Hash）**：

```cpp
// Sequence Group / Sequence 内部哈希计算更新
size_t Sequence::_make_hash(size_t content_length, size_t block_size) {
    auto sequence_group = get_sequence_group_ptr();
    size_t block_start_idx = content_length - (content_length % block_size);
    if (block_start_idx == content_length) {
        block_start_idx -= block_size;
    }

    std::vector<int64_t> content;
    size_t filled_blocks_count = block_start_idx / block_size;

    if (filled_blocks_count > 0) {
        // 继承前驱块哈希
        content.emplace_back(m_prefix_hashes[filled_blocks_count - 1]);
    } else {
        // Block-0: 注入租户与 Salt 虚拟根种子
        const std::string& salt = sequence_group->get_cache_salt();
        const std::string& tenant_id = sequence_group->get_tenant_id();
        if (!salt.empty() || !tenant_id.empty()) {
            std::string root_seed = tenant_id + ":" + salt;
            size_t seed_hash = std::hash<std::string_view>{}(root_seed);
            content.emplace_back(static_cast<int64_t>(seed_hash));
        }
    }

    // 压入当前块的 Tokens / Embeddings ...
    // [原有 Token/Embedding 填充逻辑保持不变]

    const char* data = reinterpret_cast<const char*>(content.data());
    std::size_t size = content.size() * sizeof(content[0]);
    return std::hash<std::string_view>{}(std::string_view(data, size));
}
```

- **数学性质保证**：由于哈希链结构，只要 Block-0 的种子不同，后续所有衍生 Block 的哈希值均严格隔离，数学上彻底根除跨租户碰撞。
- **安全默认行为**：若未显式指定 `tenant_id`，系统默认生成单次会话级 Ephemeral 随机 Salt，**仅允许当前 Session 内复用，禁止跨 Session/磁盘复用**。

---

### 3.2 L1 Host Cache 与 L0↔L1 搬运机制

#### 3.2.1 HostBlockPool（主机内存池）
- 预分配锁页内存（Pinned Host Memory），避免 OS 换页引入抖动。
- 维护固定大小对齐内存块（按 `block_size_in_bytes` 划分），由 `HostBlockAllocator` 统一管理。
- 维护 `m_l1_hash_to_block` 哈希索引与 LRU 淘汰链表。

#### 3.2.2 跨设备搬运（D2H / H2D）
针对不同推理硬件，在 `KVCacheManager` 中实现两阶段搬运能力：
1. **CPU 推理后端**：
   - GPU/NPU 显存与 Host 内存为同一物理介质。
   - `swap_out` / `swap_in` 退化为 `std::memcpy` 或指针所有权置换，吞吐可达 40~60 GB/s。
2. **GPU 推理后端（阶段 1 - 标准 OpenVINO USM / RemoteTensor）**：
   - 利用 `ov::RemoteContext` 与 `ov::RemoteTensor` 的 ROI 切片拷贝：
   - Host 侧使用 USM Host 内存，通过 Level Zero / OpenCL 异步 DMA 队列进行传输。
3. **GPU 推理后端（阶段 2 - 专用 PagedAttention 批量 Swap Kernel）**：
   - 在 `openvino` GPU plugin 中扩展算子/执行器：支持传入 `std::vector<size_t> src_block_ids, dst_block_ids`，一个 Kernel Launch 完成数十个分散 Block 的并发搬移。

---

### 3.3 可插拔 SSD 插件架构（Extensible SSD Plugin Architecture）

为支持 Solidigm、Samsung、Intel 及其他存储硬件厂商提供极致优化的 SSD 驱动实现，采用 **C++ 抽象接口 + 动态 C ABI 插件载入机制**。

```
                    +------------------------------------------+
                    |          IKVCacheStorageBackend          |
                    |           (C++ Core Interface)           |
                    +------------------------------------------+
                                         ^
                                         | C++ Adapter Wrapper
                    +------------------------------------------+
                    |          ov_genai_ssd_plugin_t           |
                    |          (C ABI Plugin Interface)        |
                    +------------------------------------------+
                                         ^
                                         | dlopen / LoadLibrary
                    +--------------------+---------------------+
                    |                                          |
        +-----------------------+                  +-----------------------+
        | libov_genai_ssd_samsung.so |             | libov_genai_ssd_solidigm.so |
        | (e.g. io_uring / FDP) |                  | (e.g. SPDK / CSAL / ZNS)    |
        +-----------------------+                  +-----------------------+
```

#### 3.3.1 C++ 抽象契约（`IKVCacheStorageBackend`）

```cpp
namespace ov::genai {

struct StorageModelMetadata {
    std::string model_name;
    std::string model_fingerprint; // 结构/权重 Hash
    std::string tokenizer_hash;
    ov::element::Type kv_precision;
    size_t kv_block_size_tokens;
    size_t num_layers;
    size_t block_bytes_per_layer;
};

struct BlockIORequest {
    uint64_t block_hash;
    void* host_buffer;         // 对齐的 Host 内存指针（支持 USM/Pinned）
    size_t buffer_size_bytes;
    uint32_t layer_idx;
};

using AsyncIOCompletionCallback = std::function<void(int status_code, size_t completed_blocks)>;

class IKVCacheStorageBackend {
public:
    virtual ~IKVCacheStorageBackend() = default;

    /// 初始化插件环境、挂载存储路径、校验/重建元数据
    virtual void initialize(const std::string& storage_path, 
                            const StorageModelMetadata& metadata,
                            const ov::AnyMap& custom_properties) = 0;

    /// 检查指定 Hash 的 Block 在存储介质中是否存在
    virtual bool has_block(uint64_t block_hash) const = 0;

    /// 批量查询哪些 Hash 存在于存储中
    virtual std::vector<bool> has_blocks(const std::vector<uint64_t>& block_hashes) const = 0;

    /// 同步写入 Block 数据（包含所有 Layer）
    virtual bool write_blocks(const std::vector<BlockIORequest>& requests) = 0;

    /// 同步读取 Block 数据（包含所有 Layer）
    virtual bool read_blocks(const std::vector<BlockIORequest>& requests) = 0;

    /// 异步批量读取（供 Prefetch 流水线使用）
    virtual void async_read_blocks(const std::vector<BlockIORequest>& requests,
                                   AsyncIOCompletionCallback callback) {
        // 默认实现退化为同步调用
        bool ok = read_blocks(requests);
        callback(ok ? 0 : -1, ok ? requests.size() : 0);
    }

    /// 淘汰指定 Block 或根据存储预算执行清理
    virtual void evict_blocks(const std::vector<uint64_t>& block_hashes) = 0;

    /// 存储元数据落盘与同步
    virtual void flush_manifest() = 0;

    /// 释放存储资源
    virtual void shutdown() = 0;
};

} // namespace ov::genai
```

#### 3.3.2 C ABI 导出规范（供第三方 vendor 编写动态库）

第三方厂商只需编译独立的 `.so` / `.dll`，无需依赖完整的 OpenVINO 构建树：

```c
// include/openvino/genai/c/ssd_plugin_interface.h
#ifndef OPENVINO_GENAI_SSD_PLUGIN_INTERFACE_H
#define OPENVINO_GENAI_SSD_PLUGIN_INTERFACE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* model_name;
    const char* model_fingerprint;
    const char* tokenizer_hash;
    uint32_t element_type;
    size_t kv_block_size;
    size_t num_layers;
    size_t block_bytes_per_layer;
} ov_genai_storage_metadata_t;

typedef struct {
    uint64_t block_hash;
    void* buffer;
    size_t buffer_bytes;
    uint32_t layer_idx;
} ov_genai_block_io_desc_t;

typedef struct ov_genai_ssd_plugin {
    uint32_t api_version; // 校验 API 兼容版本，例如 0x00010000
    const char* vendor_name; // 如 "Solidigm-SPDK", "Samsung-FDP"
    
    void* (*create_instance)(void);
    void (*destroy_instance)(void* handle);
    
    int (*init)(void* handle, const char* path, const ov_genai_storage_metadata_t* meta, const char* json_props);
    int (*has_block)(void* handle, uint64_t hash);
    int (*read_blocks)(void* handle, const ov_genai_block_io_desc_t* descs, size_t count);
    int (*write_blocks)(void* handle, const ov_genai_block_io_desc_t* descs, size_t count);
    int (*evict_blocks)(void* handle, const uint64_t* hashes, size_t count);
    int (*flush)(void* handle);
} ov_genai_ssd_plugin_t;

// 动态库必须导出的入口符号
ov_genai_ssd_plugin_t* ov_genai_get_ssd_plugin(void);

#ifdef __cplusplus
}
#endif

#endif // OPENVINO_GENAI_SSD_PLUGIN_INTERFACE_H
```

#### 3.3.3 DefaultFileStorageBackend（内置默认实现）
- **存储文件结构**：采用 `kv_cache_data.bin` 紧凑连续大文件（按 4KB 对齐） + `manifest.json` 元数据索引。
- **I/O 引擎**：Linux 环境优先使用 `pread`/`pwrite` 或 `io_uring`，Windows 环境使用 `ReadFile`/`WriteFile` 重叠 I/O。
- **元数据安全性**：每次持久化写入追加 CRC32 校验码，启动时校验模型签名，若不一致则自动清空重建，杜绝脏数据注入。

#### 3.3.4 厂商特定扩展点与高级加速特性
1. **Solidigm / Intel CSAL & SPDK 插件**：利用用户态 NVMe 轮询驱动绕过 Linux Kernel VFS 与中断，4K/16K Block 读延迟从 15μs 降至 3~5μs。
2. **Samsung FDP (Flexible Data Placement) / ZNS 插件**：将不同 Tenant / 模型的 KV Block 分配到不同 Flash Stream / Physical Zone，彻底消除 SSD 内部垃圾回收（GC）引起的尾延迟（Tail Latency）。
3. **GPUDirect Storage (GDS) / USM Direct DMA**：厂商插件可直接通过 PCIe Peer-to-Peer DMA 从 NVMe 读写 GPU USM / Pinned 内存，完全绕过 CPU Host 中转。

---

## 4. 多级生命周期调度与决策时序

### 4.1 Block 淘汰时序（Eviction / Swap-Out）

```
[Scheduler Step Finished]
         |
         v
[BlockAllocator::allocate_block] --> L0 物理块已满?
         |
        YES
         v
[OverwritableBlocksHashStore::get_lru_block_to_overwrite]
         |
         v
[TieredCacheManager::notify_l0_eviction(hash, blocks)]
         |
         +---> 1. 触发 D2H 拷贝将 Block 搬运至 L1 HostBlockPool
         |
         +---> 2. 若 L1 Host 内存亦满，挑选 L1 LRU Block 触发 Storage 写入：
                   IKVCacheStorageBackend::write_blocks(...)
         |
         v
[L0 物理槽位被安全覆写给新 Sequence]
```

### 4.2 Block 命中与多级恢复时序（Restore / Swap-In）

```
[New Request Arrived: Prompt Tokens]
         |
         v
[SequenceGroup::compute_prefix_hashes(tenant_id, salt)]
         |
         v
[BlockManager::get_prefix_restore_plan]
         |
         +--- Tier 0 (Device) 命中? ----> [YES] --> 标记 L0 复用 (Plan.l0_blocks)
         |
         +--- Tier 1 (Host) 命中? ------> [YES] --> 标记 L1 恢复 (Plan.l1_blocks)，分配 L0 块并在 prefill 前下发 H2D Swap-In
         |
         +--- Tier 2 (SSD Plugin) 命中? -> [YES] --> 标记 L2 恢复 (Plan.l2_blocks)，分配 L0/L1 块并在 prefill 前下发 SSD->Host->Device Swap-In
         |
         v
[执行增量 Prefill (仅对未命中的剩余 Tokens 计算)]
```

---

## 5. 公共 API 与配置定义

### 5.1 `SchedulerConfig` 扩展（C++ & Python）

```cpp
struct SchedulerConfig {
    // ... [原有参数保持] ...

    // 是否开启多级 KV Cache Offloading
    bool enable_kv_cache_offloading = false;

    // L1 Host RAM 缓存大小（单位：GB，0 表示不使用 L1 Host 缓存）
    size_t host_cache_size = 0;

    // L2 存储路径（若为空则不开启 L2 存储缓存）
    std::string storage_cache_dir = "";

    // L2 存储大小上限（单位：GB）
    size_t storage_cache_size = 0;

    // 存储后端类型："default" 或 "plugin"
    std::string storage_backend_type = "default";

    // 动态载入的第三方 SSD 插件路径（例如 "libov_genai_ssd_solidigm.so"）
    std::string storage_plugin_path = "";

    // 传递给 SSD 插件的厂商专有参数
    ov::AnyMap storage_plugin_properties = {};
};
```

### 5.2 `GenerationConfig` 扩展

```cpp
struct GenerationConfig {
    // ... [原有采样参数保持] ...

    // 租户隔离标识符（强制作用于 Block 链式哈希根种子）
    std::string tenant_id = "";

    // 用户自定义 Cache Salt 种子
    std::string cache_salt = "";
};
```

---

## 6. 实施计划与里程碑（Implementation Roadmap）

| 阶段 | 交付模块 | 涉及源码路径 | 验收标准 / 测试用例 | 依赖项 |
|---|---|---|---|---|
| **Phase 1** | **安全隔离与链式根哈希改造** | `src/cpp/src/sequence_group.cpp`<br>`src/cpp/include/openvino/genai/generation_config.hpp` | 编写单元测试：验证不同 `tenant_id`/`salt` 下相同 Prompt 生成完全不同的 Prefix Hash 链，相同 `tenant_id` 生成一致 Hash。 | 无 |
| **Phase 2** | **L1 Host Memory Pool 与 L0↔L1 搬运** | `src/cpp/src/continuous_batching/cache/host_block_pool.hpp`<br>`src/cpp/src/continuous_batching/cache/kv_cache_manager.hpp`<br>`src/cpp/src/continuous_batching/cache/block_manager.hpp` | 单元测试：CPU/GPU 模拟高负载显存挤压，验证被淘汰 Block 成功迁入 L1 Host，二次请求命中后从 L1 恢复且生成 Token 完全一致。 | Phase 1 |
| **Phase 3** | **L2 存储抽象接口与 DefaultFileStorage** | `src/cpp/include/openvino/genai/cache/i_storage_backend.hpp`<br>`src/cpp/src/continuous_batching/cache/default_file_storage.cpp` | 单测与集成测试：进程退出后重启 Pipeline，验证从 `manifest.json` 与 `.bin` 成功加载历史前缀，TTFT 降低 80% 以上。 | Phase 2 |
| **Phase 4** | **SSD Plugin 动态加载架构与 C ABI 封装** | `src/cpp/include/openvino/genai/c/ssd_plugin_interface.h`<br>`src/cpp/src/continuous_batching/cache/ssd_plugin_loader.cpp` | 编写 Mock Vendor Plugin（`mock_ssd_plugin.so`），测试动态发现、载入、读写委托与异常故障隔离测试。 | Phase 3 |
| **Phase 5** | **GPU 运行时加速与端到端评测** | `thirdparty/openvino` GPU plugin PagedAttention swap kernel 适配与 Benchmark | 在 Qwen/Llama 32k 长文本场景下，对比开启 Offloading 前后的吞吐、并发容量与 TTFT 加速比。 | Phase 4 |

---

## 7. 风险评估与应对策略

1. **Host ↔ SSD I/O 阻塞主推理调度线程**：
   - *应对*：`TieredCacheManager` 内部维护独立的后台 I/O 线程池与环形队列，Swap-Out 过程完全异步化；Swap-In 采用预取管线（Prefetch Pipeline），在分词与调度阶段提前发出磁盘读请求。
2. **第三方 SSD 插件崩溃或内存泄漏**：
   - *应对*：在 `SSDPluginLoader` 中封装强防御边界，校验 C ABI 版本号；对插件返回的状态码进行严格检查，一旦插件 I/O 报错自动切断并回退至内存或默认文件模式，打印告警日志。
3. **模型换版 / Tokenizer 变更产生脏 Cache**：
   - *应对*：L2 存储 Manifest 强制绑定 Model Topology Fingerprint 与 Tokenizer Vocab Hash，指纹不一致时拒绝加载并原子清空历史存储。
