---
sidebar_position: 4
---

# Prefix Caching

## Overview
**Prefix Caching** reuses the KV cache of a prompt prefix that was already processed by an earlier request, instead of computing it again. When a new prompt starts with the same tokens as a previous one, for example the same system prompt, the same document in a RAG pipeline, or the earlier turns of a chat, only the new part of the prompt has to go through prefill. This mainly reduces time to first token (TTFT) for long, repeated prefixes.

Prefix caching is built on the PagedAttention backend used by [Continuous Batching](./continuous-batching.md), so it is available wherever that backend is used.

## How It Works
With PagedAttention, the KV cache is stored in fixed-size blocks. When prefix caching is enabled:

1. Each full block gets a hash computed from its own content and the hash of the block before it. Two blocks have the same hash only if the whole prefix up to and including that block is the same.
2. When a request finishes, its blocks are not released. They stay in the cache, indexed by their hashes.
3. When a new request arrives, the pipeline looks up the hashes of its prompt blocks and restores the longest matching prefix from the cache. Prefill then starts after the restored part.
4. When the cache is full and a new block is needed, the least recently used cached block is overwritten.

Because a block hash depends on everything before it, a cache hit requires an identical prefix starting from the first token. A prompt that shares text with an earlier prompt only in the middle or at the end does not benefit.

## When It Is Enabled
| Pipeline | Default |
|---|---|
| `LLMPipeline` / `VLMPipeline` on the PagedAttention backend | Enabled |
| `LLMPipeline` / `VLMPipeline` with a user-provided `SchedulerConfig` | Value of `enable_prefix_caching` in that config (`False` by default) |
| `ContinuousBatchingPipeline` | Value of `enable_prefix_caching` in the provided `SchedulerConfig` (`False` by default) |

When no `SchedulerConfig` is passed, `LLMPipeline` and `VLMPipeline` use a latency-oriented scheduler configuration with prefix caching enabled. Passing your own `SchedulerConfig` replaces that configuration entirely, so set `enable_prefix_caching` explicitly if you still want prefix caching. See [Supported Pipelines and Configuration](./continuous-batching.md#supported-pipelines-and-configuration) for how the attention backend is selected.

## Configuration
```python
import openvino_genai as ov_genai

scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.enable_prefix_caching = True
scheduler_config.cache_size = 2  # KV cache budget, in GB

pipe = ov_genai.LLMPipeline(models_path, "CPU", scheduler_config=scheduler_config)

system_prompt = "You are a helpful assistant. Answer briefly.\n"
for question in ["What is OpenVINO?", "What is a KV cache?"]:
    # The second call reuses the KV cache of the shared system prompt
    result = pipe.generate([system_prompt + question], max_new_tokens=50)
    print(result.texts[0])
    print(f"TTFT: {result.perf_metrics.get_ttft().mean:.2f} ms")
```

```cpp
#include "openvino/genai/llm_pipeline.hpp"

ov::genai::SchedulerConfig scheduler_config;
scheduler_config.enable_prefix_caching = true;
scheduler_config.cache_size = 2;  // KV cache budget, in GB

ov::genai::LLMPipeline pipe(models_path, "CPU", ov::genai::scheduler_config(scheduler_config));

std::string system_prompt = "You are a helpful assistant. Answer briefly.\n";
for (const std::string question : {"What is OpenVINO?", "What is a KV cache?"}) {
    // The second call reuses the KV cache of the shared system prompt
    auto result = pipe.generate(std::vector<std::string>{system_prompt + question}, ov::genai::max_new_tokens(50));
    std::cout << result.texts[0] << std::endl;
    std::cout << "TTFT: " << result.perf_metrics.get_ttft().mean << " ms" << std::endl;
}
```

The same `scheduler_config` argument is accepted by `VLMPipeline` and `ContinuousBatchingPipeline`.

## Visual Language Models
For `VLMPipeline`, block hashes are computed from the input embeddings rather than only from token IDs, so images and videos are part of the prefix. A cached prefix is reused only if both the text and the visual inputs in that prefix are identical. Changing the image invalidates the cache from the position where the image appears onwards.

`VLMPipeline` on NPU does not use the PagedAttention backend and does not accept `scheduler_config`.

## Memory Usage
With prefix caching enabled, KV cache blocks of finished requests are kept for reuse, so cache memory usage grows up to the configured limit instead of shrinking when requests finish. The limit is set by `cache_size` (in GB) or `num_kv_blocks`. When both are `0`, the cache is allocated dynamically. When the limit is reached, the least recently used cached blocks are overwritten, so a prefix that was not used for a long time may need to be computed again.

For models that combine attention layers with linear-attention layers, prefix caching also changes how the linear-attention state is stored. See [Hybrid Attention Model Cache Management](./hybrid-attention-models-cache.md).

## Related Pages
* [Continuous Batching](./continuous-batching.md): the PagedAttention backend and `SchedulerConfig`.
* [Hybrid Attention Model Cache Management](./hybrid-attention-models-cache.md): prefix caching for hybrid models.
* [KVCache Token Eviction Algorithm](./kvcache-eviction-algorithm.md): limiting KV cache size per sequence.
* [src/cpp/include/openvino/genai/scheduler_config.hpp](../../../../src/cpp/include/openvino/genai/scheduler_config.hpp): all `SchedulerConfig` parameters and their default values.
