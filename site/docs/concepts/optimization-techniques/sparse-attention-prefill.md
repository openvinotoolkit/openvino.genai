---
sidebar_position: 2
---

# Sparse Attention prefill algorithms

## Overview
The sparse attention prefill algorithms enable speedups during the prompt processing (prefill) stage of the generation process by reducing the amount of computation taken up by the attention operation.
During the prefill stage, the attention operation could only be applied to the subset of blocks determined to be most "important" for the current generation context, with importance estimation method determined by the algorithm.

The KV-cache blocks that are deemed "unimportant" at a certain stage of prefill are not discarded entirely, but are preserved as-is so that they may still be considered for usage in the attention computation in latter stages of prompt processing.
Moreover, the sparse prefill algorithms do not apply during the generation stage. 
The sparse prefill algorithms therefore do not lead to decreased maximum and average memory consumption throughout the generation process, but they do lead to the decreased total generation and first token latency times due to enabling faster prefills.

To achieve overall memory savings and generation stage memory and compute optimizations, the [cache eviction algorithm](./kvcache-eviction-algorithm.md) can be used along with sparse prefill algorithms, or separately.


## Usage with openvino.genai
The sparse attention prefill can be enabled by setting `.use_sparse_attention` field to `True` in the `openvino_genai.SchedulerConfig` structure that serves as input to most of the `openvino_genai` model pipeline objects.
Further configuration of the sparse prefill algorithm is done using the `.sparse_attention_config` field of `openvino_genai.SchedulerConfig`, which accepts objects of `openvino_genai.SparseAttentionConfig`. See the [in-code documentation](../../../../src/cpp/include/openvino/genai/sparse_attention.hpp) for the description of individual fields; in particular, the `.mode` field of `SparseAttentionConfig` selects the type of the sparse prefill algorithm to be applied.

Currently two sparse prefill algorithms are supported - the tri-shape algorithm (https://arxiv.org/pdf/2412.10319) and the XAttention algorithm (https://arxiv.org/pdf/2503.16428).

### Refresher on the openvino.genai paged attention implementation
In a vLLM-like approach, which underlies `openvino.genai` inference, the KV cache of a sequence is divided into blocks of fixed size.
The size of the blocks is hardware-defined (i.e. 32 tokens for CPU, 16 for GPU, etc.) and, depending on the algorithm, imposes limitations on the minimum granularity of sparsity that can be introduced into the attention operation.

The `openvino.genai` API allows for generation of multiple sequences at once, which are divided into chunks of arbitrary size by the internal scheduler and submitted to the paged attention kernels in one go to achieve better throughput. In practice this means that the full prefill stage for each sequence may take several time steps, between which the adjustments of KV cache blocks assigned to each sequence are possible. This impacts the operation of the sparse prefill algorithms as implemented in `openvino`/`openvino.genai` as described below.

### Tri-shape
For the tri-shape algorithm, the majority of the prefill occurs with as little as 2-3 KV cache blocks (depending on the configuration) being utilized as previous KV cache data to process each new prompt chunk.
These retained blocks are the blocks in the chronological beginning of the prompt, the last-processed full blocks of the same prompt and the last KV cache block not currently completely filled (if it exists).
The sizes of the retained areas can be adjusted using the `SparseAttentionConfig.num_retained_start_tokens_in_cache` and `SparseAttentionConfig.num_retained_recent_tokens_in_cache`.


![Tri-shape sparse prefill illustrated](./../../../static/img/trishape.svg)

The picture above illustrates the tri-shape algorithm in more detail. For simplicity, it is presumed that the prompt takes up 8 full KV cache blocks and is filled within 5 chunks. The `.num_retained_start_tokens_in_cache` and `.num_retained_recent_tokens_in_cache` are both set to 1 HW-dependent block size in tokens.


The prompt processing occurs as usual until at least two KV cache blocks have been completely filled (`t = 0, 1`).
After that, for the next prompt chunks only the first and the last/second-last blocks processed will be visible as KV cache contents, effectively introducing sparsity in the attention computation for the rest of the KV cache "body" (`t = 2-4`).

Upon reaching the tail of the prompt the KV cache for the entire prompt is used in attention again, effectively switching back from the sparse attention mode to "dense" attention (`t = 5`).
Apart from improving the generation accuracy, this also makes it possible to effectively combine the tri-shape sparse prefill algorithm with the cache eviction algorithm, which relies on the model having "seen" the entire prompt KV cache when processing the last tokens of the prompt. The "dense attention" portion of the prompt can be configured using the `SparseAttentionConfig.num_retained_recent_tokens_in_cache` field. 


### XAttention
TBA



