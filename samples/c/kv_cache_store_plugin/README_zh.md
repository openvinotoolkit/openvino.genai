# KV Cache Store 插件示例（`sample_kv_cache_store_plugin`）

[English](README.md)

这是 KV cache store 插件 ABI 的一个最小、独立实现，对应头文件
[`openvino/genai/c/kv_cache_store_provider_abi.h`](../../../src/cpp/include/openvino/genai/c/kv_cache_store_provider_abi.h)。
可以直接拿它当作真实厂商存储插件（例如 SSD 存储后端）的起点，也可以用它来验证你自己的 host 侧
集成代码能否正确加载并驱动一个插件。

## 这是什么

OpenVINO GenAI 的 KV cache 磁盘 offload（`SchedulerConfig::use_cache_offload`）默认会把被淘汰的
block 写到一个普通文件里（`FileKVCacheStore`）。这个 ABI 让第三方可以用自己的存储后端替换掉这个
默认实现——不需要链接 `openvino::genai`，也不需要写任何 C++ 代码——只需要提供一个导出单个 C 入口
符号的动态库。

这份示例用一个加了互斥锁的、进程内存里的链表实现了**完整**的 ABI 接口。它**不是生产可用的代码**
（没有持久化、没有真正的 I/O、O(n) 的 key 查找、内存会无限增长）——它唯一的作用是展示一个厂商插件
到底需要实现哪些东西，并证明这套机制能和真实模型跑通整个流程。

这份示例展示的几个关键点：

- **完全不依赖 genai。** `.c` 文件只 include 了这一个 ABI 头文件；对应的 CMake target 也没有链接
  `openvino::genai`。
- **单一入口点。** host 只通过一个导出的 C 符号 `ov_pipeline_query_kvcache_store_providers` 来发现
  插件（见 [`sample_kv_cache_store_plugin.c`](sample_kv_cache_store_plugin.c) 文件末尾）。
- **插件命中只是"发现"，不是"真相"。** host 在信任这个插件返回的任何字节之前，总是会重新计算一遍
  内容哈希，并校验插件返回的 `witness`。

## 编译

跟着 GenAI 常规 samples 一起编译，不需要额外开关（见
[`samples/CMakeLists.txt`](../../CMakeLists.txt)）：

```sh
cmake -S . -B build -DENABLE_SAMPLES=ON ...
cmake --build build --target sample_kv_cache_store_plugin
```

编译产物是一个动态库，和其它 sample 的二进制放在一起（Windows 上是
`sample_kv_cache_store_plugin.dll`，Linux/macOS 上是
`libsample_kv_cache_store_plugin.so`/`.dylib`）。

## 怎么接到 host pipeline 里

把 `CacheOffloadConfig` 指向编译好的动态库即可。`allow_untrusted_store_plugin` 必须显式设置——
它默认是 `false`，这样任何第三方代码都不会在没有明确开启的情况下被加载：

```python
import openvino_genai as ov_genai

scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.use_cache_offload = True
scheduler_config.enable_prefix_caching = True  # 被 offload 的 block 要靠前缀哈希重新发现
scheduler_config.cache_offload_config = ov_genai.CacheOffloadConfig(
    capacity_bytes=1 * 1024 * 1024 * 1024,
    buffer_slots=128,
    store_library_path="/path/to/sample_kv_cache_store_plugin.dll",
    store_provider_name="sample-in-memory",  # 这份示例唯一的 provider；如果一个模块只导出一个 provider，这个字段可以不填
    allow_untrusted_store_plugin=True,
)
```

或者通过 `tools/llm_bench` 的 `--cb_config` JSON 文件：

```json
{
  "use_cache_offload": true,
  "enable_prefix_caching": true,
  "cache_offload_config": {
    "capacity_bytes": 1073741824,
    "buffer_slots": 128,
    "store_library_path": "/path/to/sample_kv_cache_store_plugin.dll",
    "store_provider_name": "sample-in-memory",
    "allow_untrusted_store_plugin": true
  }
}
```

如果动态库加载失败、ABI 握手失败、或者指定的 provider 名字找不到，host 只会打一条警告日志，然后
回退到默认的 `FileKVCacheStore`——offload 永远是尽力而为（best-effort），不会因为插件不可用而影响
正常推理。

## 怎么写自己的插件

1. 以 [`sample_kv_cache_store_plugin.c`](sample_kv_cache_store_plugin.c) 为起点复制一份。
2. 把 `store_read_blocks` / `store_write_blocks` / `store_drop_blocks` 里的内存链表换成真正对接你
   存储设备的 I/O，并把这里简单的链表查找换成你自己的 key -> 位置 映射。host 只会给你一个不透明的
   `OvKVCacheBlockKey{group, hash}` 加上原始字节；怎么把它映射到你自己的文件/对象/偏移量，完全由
   你自己决定。
3. 保持 ABI 头文件里规定的绑定契约：纯 C、不能有异常/STL/`ov::*` 类型跨越这个边界、不能回调 host、
   每个 vtable 的第一个字段必须是 `struct_bytes`（用于向前兼容的边界检查）、`witness` 必须是一个
   强的二级摘要——绝不能是原始 token id。
4. 只能按照头文件 `VERSIONING` 这一节描述的方式去演进 `OV_KVCACHE_STORE_ABI_MAJOR`/`MINOR` 和能力
   位（capability bits）；绝不能复用或者重新编号已有的状态码、能力位、或 vtable 条目。

完整的绑定契约（线程/生命周期规则、寻址模型、这个 ABI 里明确不包含哪些东西）请看头文件开头的大段
注释。
