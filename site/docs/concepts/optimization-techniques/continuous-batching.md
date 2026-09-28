---
sidebar_position: 3
---

# Continuous Batching

## Overview
**Continuous Batching** is an optimization strategy for LLM inference that improves throughput by dynamically composing a batch of independent requests at each model inference step. Instead of handling each request separately from start to finish, the scheduler continuously combines active requests into a single batch. The algorithm evicts a request from the batch when it has produced its final token, allowing other requests to continue or new ones to join. Continuous Batching in OpenVINO GenAI is built on top of the **PagedAttention (PA)** layer. Read more in [src/cpp/include/openvino/genai/continuous_batching_pipeline.hpp](../../../../src/cpp/include/openvino/genai/continuous_batching_pipeline.hpp).

## PagedAttention Backend
**SDPA (Scaled Dot-Product Attention)** pipelines execute one request (or a fixed static batch) at a time with a contiguous KV cache `[batch, num_heads, seq_len, head_dim]` tensor, which grows as the sequence length increases. **PA** stores K/V in a global pool of fixed-size blocks. Each sequence owns a block table and the attention kernel gathers K/V through that table. New blocks are allocated to a sequence only when the previous one fills up, and blocks are returned to the pool as soon as the sequence finishes. The same pool is shared between all in-flight requests. This makes possible such optimization techniques as multi-request batching, prefix caching, chunked prefill, and cache eviction.

PA approach can provide higher throughput and better memory utilization.


## Supported Pipelines and Configuration
On supported x86_64/ARM64 platforms, PagedAttention is the default attention backend on CPU and GPU for:
* `ov::genai::LLMPipeline` / `openvino_genai.LLMPipeline`
* `ov::genai::VLMPipeline` / `openvino_genai.VLMPipeline`
* `ov::genai::ContinuousBatchingPipeline` / `openvino_genai.ContinuousBatchingPipeline`

If **PagedAttention backend initialization fails**, `LLMPipeline` and `VLMPipeline` **automatically fall back to the SDPA backend.**

To avoid fallback, set the backend explicitly by passing `ATTENTION_BACKEND="PA"` or `SchedulerConfig` to `LLMPipeline` and `VLMPipeline` constructors. In this case, if the PA backend can't be enabled, pipeline creation fails instead of falling back to SDPA.

Set `ATTENTION_BACKEND="SDPA"` to force SDPA instead of PA.

Scheduling and paged KV cache behavior are configured with `ov::genai::SchedulerConfig` (`openvino_genai.SchedulerConfig`). Please read more about supported parameters and default values in [src/cpp/include/openvino/genai/scheduler_config.hpp](../../../../src/cpp/include/openvino/genai/scheduler_config.hpp).

## Sample Usage

### Client scenario

```python
import openvino_genai as ov_genai

scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.cache_size = 2                 # KV cache budget, in GB
scheduler_config.max_num_seqs = 64              # max concurrent decode sequences
scheduler_config.dynamic_split_fuse = True      # chunked prefill
scheduler_config.enable_prefix_caching = True   # reuse shared prompt prefixes

pipe = ov_genai.LLMPipeline(models_path, "CPU", scheduler_config=scheduler_config)
print(pipe.generate("The Sun is yellow because", max_new_tokens=100))
```

```cpp
#include "openvino/genai/llm_pipeline.hpp"

ov::genai::SchedulerConfig scheduler_config;
scheduler_config.cache_size = 2;                 // KV cache budget, in GB
scheduler_config.max_num_seqs = 64;
scheduler_config.dynamic_split_fuse = true;
scheduler_config.enable_prefix_caching = true;

ov::genai::LLMPipeline pipe(models_path, "CPU", ov::genai::scheduler_config(scheduler_config));
std::cout << pipe.generate("The Sun is yellow because", ov::genai::max_new_tokens(100));
```

### Server scenario
For serving workloads, `ContinuousBatchingPipeline` exposes the scheduler directly. Requests are pushed with `add_request()` and the pipeline is driven one iteration at a time via `step()`, so many clients can be served concurrently.

```python
import openvino_genai as ov_genai

scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.max_num_batched_tokens = 2048
scheduler_config.max_num_seqs = 128
scheduler_config.cache_size = 8                 # in GB
scheduler_config.dynamic_split_fuse = True
scheduler_config.enable_prefix_caching = True

pipe = ov_genai.ContinuousBatchingPipeline(models_path, scheduler_config, "CPU")

generation_config = ov_genai.GenerationConfig()
generation_config.max_new_tokens = 128

# High-level batched API
results = pipe.generate(
    ["Hello, ", "The Sun is yellow because"],
    [generation_config, generation_config],
)
for r in results:
    print(r.m_generation_ids[0])

# Low-level streaming API
handle = pipe.add_request(request_id=0, prompt="Explain paged attention", generation_config=generation_config)
while pipe.has_non_finished_requests():
    pipe.step()
print(handle.read_all())
```

```cpp
#include "openvino/genai/continuous_batching_pipeline.hpp"

ov::genai::SchedulerConfig scheduler_config;
scheduler_config.max_num_batched_tokens = 2048;
scheduler_config.max_num_seqs           = 128;
scheduler_config.cache_size             = 8;    // in GB
scheduler_config.dynamic_split_fuse     = true;
scheduler_config.enable_prefix_caching  = true;

ov::genai::ContinuousBatchingPipeline pipe(models_path, scheduler_config, "CPU");

ov::genai::GenerationConfig generation_config;
generation_config.max_new_tokens = 128;

auto results = pipe.generate(
    {"Hello, ", "The Sun is yellow because"},
    {generation_config, generation_config});
```

## Tools and samples:
* [tools/continuous_batching/accuracy/continuous_batching_accuracy.cpp](../../../../tools/continuous_batching/accuracy/continuous_batching_accuracy.cpp) — batched generation with mixed sampling configs.
* [tools/continuous_batching/benchmark/continuous_batching_benchmark.cpp](../../../../tools/continuous_batching/benchmark/continuous_batching_benchmark.cpp) — throughput benchmark for server-like workloads.
* [samples/python/text_generation/benchmark_genai.py](../../../../samples/python/text_generation/benchmark_genai.py) — `LLMPipeline` driven by a custom `SchedulerConfig`.
