---
sidebar_position: 5
---

# Hybrid Attention Model Cache Management

## Overview

Some models combine more than one cache type during continuous batching.
The most common hybrid case is a model that uses both:

- regular KV-cache inputs for attention layers
- linear-attention state tables for layers such as CausalConv1D or GatedDeltaNet

These models do not have a single cache pool.
They have at least two different cache pools with different growth rules:

- KV cache is token-driven
- linear-attention cache is either sequence-driven or interval-driven, depending on whether prefix caching is enabled

This distinction matters when configuring `SchedulerConfig`, because the same setting can affect the two cache types differently.

## Cache Types in Hybrid Models

### KV cache

KV cache capacity is measured in blocks, and each block stores a fixed number of tokens determined by the target device.
Increasing KV capacity increases the number of tokens that can remain resident across active sequences.

Relevant settings:

- `num_kv_blocks`
- `cache_size`
- `use_cache_eviction`

### Linear-attention cache without prefix caching

When `enable_prefix_caching=false`, linear-attention cache behaves like a fixed-size state per live sequence.
Each sequence needs a fixed number of linear-attention blocks, regardless of prompt length.

When `num_linear_attention_blocks=0`, the runtime derives this capacity differently for two common cases:

- if `max_num_batched_tokens == std::numeric_limits<size_t>::max()`, it treats the configuration as a client-style latency scenario and starts with `1` linear-attention block
- otherwise it treats the configuration as bounded batching and derives linear-attention capacity from `max_num_seqs`

Relevant settings:

- `num_linear_attention_blocks`
- `max_num_batched_tokens`
- `max_num_seqs`

### Linear-attention cache with prefix caching

When `enable_prefix_caching=true`, linear-attention cache switches to paged checkpointing mode.
Instead of allocating one fixed state block per sequence, the runtime stores checkpoints every `cache_interval` tokens.

Relevant settings:

- `num_linear_attention_blocks`
- `cache_interval`
- `num_kv_blocks`
- `cache_size`

Smaller `cache_interval` values create more checkpoints and consume more linear-attention memory.
Larger values reduce memory usage but make checkpointing coarser.

## How SchedulerConfig Is Interpreted

For hybrid-attention models, the following rules are the most useful mental model.

### Constructor defaults matter

The starting `SchedulerConfig` depends on which pipeline constructor reaches Continuous Batching.

`LLMPipeline` constructors that create the Continuous Batching adapter use a latency-oriented default scheduler configuration when the user does not pass an explicit scheduler config in properties.
That default changes two important fields:

- `max_num_batched_tokens = std::numeric_limits<size_t>::max()`
- `enable_prefix_caching = true`

This matters for hybrid-attention models because unlimited `max_num_batched_tokens` is treated as the client-style signal in non-prefix mode.

`ContinuousBatchingPipeline` constructors do not infer that latency-oriented profile on their own.
They use the `SchedulerConfig` object that the caller provides.
If the caller default-constructs `SchedulerConfig`, the relevant defaults remain:

- `max_num_batched_tokens = 256`
- `max_num_seqs = 256`
- `enable_prefix_caching = false`

As a result, the same hybrid-attention model can start with different automatic linear-attention sizing depending on the constructor path unless these fields are set explicitly.

### Explicit KV capacity

If `num_kv_blocks > 0`, it is treated as the explicit KV target.
If `num_linear_attention_blocks` is left at `0`, the runtime derives linear-attention capacity automatically:

- with `enable_prefix_caching=false` and unlimited `max_num_batched_tokens`, it starts with `1` linear-attention block
- with `enable_prefix_caching=false` and bounded `max_num_batched_tokens`, it derives linear-attention blocks from `max_num_seqs`
- with `enable_prefix_caching=true`, it derives linear-attention blocks from the KV token target and `cache_interval`

This is the best option when deterministic capacity matters more than fitting into a precise byte budget.

### Shared memory budget

If `cache_size > 0` and `num_kv_blocks == 0`, the runtime treats `cache_size` as a shared cache budget for all registered cache types in the model.

The runtime then derives:

- KV blocks
- linear-attention blocks

under one combined memory limit.

For non-prefix mode, the same fallback split still applies before KV blocks are computed from the remaining budget:

- unlimited `max_num_batched_tokens` reserves one fixed linear-attention block
- bounded `max_num_batched_tokens` reserves linear-attention capacity based on `max_num_seqs`

For hybrid models this is usually the simplest way to tell the runtime, "fit all cache types into this memory budget".

### Fully dynamic mode

If both `num_kv_blocks == 0` and `cache_size == 0`, the cache starts with no preallocated capacity and grows on demand.
This is the most flexible mode, but it is less deterministic than explicit sizing.

### Expert override for linear-attention blocks

If `num_linear_attention_blocks > 0`, it overrides the automatic linear-attention sizing logic.
This is useful only when the required capacity is already known.

If the model does not expose linear-attention cache inputs, `num_linear_attention_blocks` should not be set.

## Which Configuration Fits Which Scenario

### Scenario 1: Stable production throughput on a known workload

Use this when the number of active requests and expected context lengths are already understood.

Recommended settings:

- set `num_kv_blocks` explicitly
- keep `cache_size=0`
- leave `num_linear_attention_blocks=0` unless manual control is needed
- set `max_num_seqs` to the intended concurrency if `enable_prefix_caching=false`
- set `cache_interval` explicitly if `enable_prefix_caching=true`

Pros:

- deterministic cache capacity
- easy to reason about admission limits
- easier benchmarking and capacity planning

Cons:

- requires up-front sizing work
- can over-allocate memory for bursty or variable workloads

### Scenario 2: Constrained device memory with mixed cache types

Use this when the main requirement is "do not exceed this cache memory budget".

Recommended settings:

- set `cache_size`
- keep `num_kv_blocks=0`
- keep `num_linear_attention_blocks=0` unless manual override is required
- choose `cache_interval` according to the desired prefix-checkpoint granularity if `enable_prefix_caching=true`

Pros:

- one budget applies to both KV and linear-attention caches
- avoids KV-only sizing on hybrid models
- good default for memory-limited deployments

Cons:

- less direct than explicit block counts
- derived capacities depend on model cache layout and `cache_interval`

### Scenario 3: High concurrency, no prefix reuse

Use this when requests are short-lived or reuse across requests is not expected, and sequence concurrency matters more than prefix reuse.

Recommended settings:

- set `enable_prefix_caching=false`
- set `num_kv_blocks` explicitly or use `cache_size`
- set `max_num_seqs` to the intended concurrent sequence count
- keep `num_linear_attention_blocks=0` unless a manual override is required

Why this works:

With prefix caching disabled, linear-attention cache is sequence-driven rather than token-driven.
If the runtime sizes it automatically, bounded batching uses `max_num_seqs` as the target for linear-attention capacity.

Pros:

- simple mental model
- predictable linear-attention capacity for concurrent requests
- avoids unnecessary checkpoint storage

Cons:

- no prefix reuse across compatible requests
- less effective for chat-style repeated-prefix workloads

### Scenario 4: Prefix-heavy chat or repeated-prompt workloads

Use this when prefix reuse matters and the model has linear-attention cache inputs.

Recommended settings:

- set `enable_prefix_caching=true`
- either set `num_kv_blocks` explicitly or provide `cache_size`
- keep `num_linear_attention_blocks=0` unless manual tuning is necessary
- choose `cache_interval` carefully

Guidance for `cache_interval`:

- smaller interval: more checkpoints, more linear-attention memory, finer-grained reuse
- larger interval: fewer checkpoints, lower linear-attention memory, coarser reuse

Pros:

- aligns linear-attention capacity with token capacity
- supports prefix checkpoint reuse in hybrid models
- good fit for chat and recurring-prefix scenarios

Cons:

- `cache_interval` becomes part of memory planning
- too small an interval can consume linear-attention memory aggressively

### Scenario 5: Single-stream or interactive client inference without prefix reuse

Use this when the pipeline is effectively latency-oriented and you do not want fixed linear-attention capacity to scale with server-style concurrency limits.

Recommended settings:

- set `enable_prefix_caching=false`
- keep `num_linear_attention_blocks=0`
- leave `max_num_batched_tokens` unlimited
- set `num_kv_blocks` explicitly or use `cache_size`, depending on whether you want deterministic capacity or a shared budget

Why this works:

With prefix caching disabled and unlimited `max_num_batched_tokens`, the runtime treats the configuration as a client-style scenario and starts with one fixed linear-attention block instead of reserving `max_num_seqs` blocks.

Pros:

- avoids over-reserving fixed linear-attention state for single-stream inference
- keeps the initial memory footprint lower on large hybrid models
- still allows KV capacity to be controlled independently by `num_kv_blocks` or `cache_size`

Cons:

- not appropriate if multiple concurrent sequences are expected immediately
- fixed linear-attention capacity may need to grow later if concurrency increases

### Scenario 6: Exploratory tuning or highly variable traffic

Use this when workload shape is not stable enough to pre-size confidently.

Recommended settings:

- set `num_kv_blocks=0`
- set `cache_size=0`
- keep `num_linear_attention_blocks=0`
- enable prefix caching only if the workload benefits from it

Pros:

- no up-front capacity planning required
- adapts to traffic that is hard to predict

Cons:

- less deterministic memory growth
- harder to compare across benchmark runs
- not the best fit when hard admission or latency targets must be guaranteed

## Recommended Starting Points

If no manual tuning has been done yet, these are reasonable defaults to start with.

### General deployment default

- use `cache_size`
- keep `num_kv_blocks=0`
- keep `num_linear_attention_blocks=0`

This is the safest default for hybrid models when the main goal is to respect a memory budget.

### Throughput-tuned deployment default

- use explicit `num_kv_blocks`
- keep `num_linear_attention_blocks=0`
- set bounded `max_num_batched_tokens` and `max_num_seqs` for non-prefix mode
- set `cache_interval` intentionally for prefix mode

This is the better default when concurrency and token capacity have already been characterized.

### Interactive client default

- set `enable_prefix_caching=false`
- keep `num_linear_attention_blocks=0`
- leave `max_num_batched_tokens` unlimited

This is the better default when non-prefix hybrid inference is effectively single-stream and initial linear-attention state should stay minimal.

### Prefix-heavy deployment default

- set `enable_prefix_caching=true`
- keep `num_linear_attention_blocks=0`
- start with the default `cache_interval`
- tune `cache_interval` only if memory pressure or reuse granularity requires it

## When to Set num_linear_attention_blocks Manually

Manual `num_linear_attention_blocks` is useful only when one of the following is true:

- the deployment has a known fixed sequence budget and the value must be pinned exactly
- benchmark repeatability requires removing one more derived quantity
- a custom allocation split between KV and linear-attention cache is intentionally required

In most cases, leaving `num_linear_attention_blocks=0` is preferred because it lets the runtime derive a value consistent with the selected mode.

## Related Topics

- [Continuous Batching](./continuous-batching.md)
- [Prefix Caching](./prefix-caching.md)

## Summary

For hybrid-attention models, the best configuration depends on whether the deployment is optimized for:

- deterministic token capacity
- a shared memory budget
- fixed concurrent sequence count
- prefix reuse efficiency
- dynamic flexibility

The simplest rule is:

- use `num_kv_blocks` when explicit capacity matters most
- use `cache_size` when a shared memory budget matters most
- keep `num_linear_attention_blocks=0` unless there is a strong reason to override the derived value
- treat `cache_interval` as a memory-versus-checkpoint-granularity knob when prefix caching is enabled