---
sidebar_position: 2
---

# KVCache Token Eviction Algorithm


## Overview
The cache eviction algorithm is designed to manage KV (Key-Value) cache memory for large language models (LLMs) during text generation. It determines which blocks of tokens should be evicted from the KV cache based on importance scores calculated from attention scores across different attention layers.

## Conceptual Model
The KV cache for each sequence is divided into three logical areas:

![KV cache layout with cache eviction](./../../../static/img/kv-cache-areas-diagram.svg)

* Start Area: Initial tokens that are never evicted
* Evictable Area: Tokens that can be evicted based on importance scores
* Recent Area: Most recent tokens that are preserved (not evicted while in this area, but naturally migrating toward the evictable area as the text generation goes on)

The sizes of all three areas can be configured by modifying corresponding fields in a `CacheEvictionConfig` struct, which itself is a part of the pipeline-wide `SchedulerConfig`.
As the generation starts, the blocks in respective logical areas are filled token-by-token, and once at least one block past the "recent" area is filled, eviction may take place. 
The tokens are evicted based on accumulated importance scores following the [H2O](https://arxiv.org/abs/2306.14048) approach.
The scores are accumulated throughout the entire generation process and their weighting may be changed by adjusting the `CacheEvictionConfig.aggregation_mode` parameter.
Eviction occurs with a block-wise granularity, and only the completely filled blocks from the "evictable" area are evicted.
By default the start area is 32 tokens, evictable area is 512 tokens and recent area is 128 tokens, which amounts to a total maximum cache usage by sequence during the generation phase of 672 tokens.

This approach allows LLMs to handle long sequences efficiently by keeping the most contextually important tokens in the cache while evicting those of lesser importance.
The downside of the eviction procedure is potential loss of generation accuracy, since the cache no longer contains the entire context for the generation, but only the most "important" token blocks.
The user can adjust the individual sizes of the eviction sub-areas to hit the optimal point of accuracy/memory usage tradeoff in their particular case.

Note that currently the eviction only starts after the full prompt has been processed, i.e. no eviction takes place during the prefill phase.
This means that for longer prompt sizes the maximum cache usage may exceed the limit defined by the `CacheEvictionConfig` parameters. 

After the prefill phase, however, the maximum cache occupancy for each sequence currently being processed is strictly limited by the combined sizes of the 3 areas described above. 
`CacheEvictionConfig.get_max_cache_size_after_eviction()` can be queried to get this cache size limit in tokens.


## Sample - impact of cache eviction on possible generation length and prompt throughput
[limit_checker.py](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/text_generation/limit_checker.py) can be used to visualize the impact of the cache eviction algorithm on the end performance of the generation pipeline.
The script is paramaterized to allow specifying own model (by its `huggingface_hub` ID) and the base cache size.

With `--mode gen_length`, the script will run the generation pipeline with increasing requested length of generation until it either hits 100% maximum cache usage or times out. 
With cache eviction disabled, the pipeline will eventually exhaust the cache size, and the generation length will be capped at the output token count determined by the base cache size. 
With eviction enabled, however, the pipeline is able to generate sequences of arbitrary length (as long as the cache size is at least `max(prompt_size, max_cache_size_after_eviction)`, and the script will instead finish with a timeout.

With `--mode gen_throughput`, the script will run a binary search to determine the minimum number of concurrently processed sequences to hit the 100% cache utilization.


## (Optional) Cache Rotation
By default, no additional cache modification is performed during eviction. 
Most LLMs employ some kind of positional embedding at some point in the inferencing, which effectively becomes associated with each per-token KV cache vector as well. 
The popular RoPE positional embedding is more or less continuous in the linear space of the token positions, but when token eviction takes place, the continuity of the remaining blocks is disrupted.
This may impact the ability of the model to correctly recognize the relative positions of the remaining blocks and degrade the generation accuracy.

Cache rotation seeks to alleviate this by "re-rotating" corresponding blocks so that the blocks that remain after each eviction are once again "continuous" in terms of the effective RoPE embedding. 
It can be enabled by setting the `CacheEvictionConfig.apply_rotation` field to `true` (default is `false`).

## Current limitations

* Cache rotation is only targeted for the regular, linear LLaMa-like RoPE application and may degrade accuracy on models that use other RoPE schemes.

* Cache rotation is currently only supported for the models with uniform V embedding sizes across the layers.
