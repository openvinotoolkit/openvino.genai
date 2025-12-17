---
sidebar_position: 5
---

# Visual Token Pruning (CDPruner)

## Overview
Visual Token Pruning is a context compression technique for Multimodal / Visual Language Models (VLMs) that aims to enhance inference efficiency without significant performance degradation by identifying and removing redundant or less informative tokens. A representative approach is CDPruner, introduced in the paper [Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs](https://arxiv.org/pdf/2506.10967). Its main goal is to lower inference latency and memory footprint while retaining the visual information most relevant to the user's query.

Unlike traditional attention-based or similarity-based pruning techniques, which can either retain redundant tokens or neglect instruction relevance, CDPruner focuses on maximizing the conditional diversity of the retained visual tokens. Pruned tokens are removed from further attention computations, shrinking KV cache footprint, reducing Time To First Token (TTFT) and improving throughput. A relevance weighting factor controls the influence of instruction relevance during pruning, helping balance token reduction against the preservation of important visual details.

## Conceptual Model
CDPruner operates on the sequence of visual token embeddings produced by the vision encoder before they are passed to the language model. Instead of forwarding all tokens, it selects a subset based on conditional diversity, combining token similarity and instruction relevance.

### Token Partitioning

The visual tokens are conceptually divided into:
* Retained Tokens: A selected subset that provides diverse and instruction-relevant visual information.
* Pruned Tokens: Tokens excluded from further processing because they contribute redundant or low-relevance information.

High-level flow:
1. Encode image producing N visual tokens (embeddings).
2. Compute pairwise token similarity and per-token relevance scores.
3. Relevance and similarity are combined into a conditional kernel. A greedy DPP-based MAP algorithm identifies the least important tokens to discard according to `pruning_ratio`, adjusting scores using `relevance_weight` to control the trade-off between diversity and relevance.
4. Build reduced token set; subsequent generation attends only to retained tokens.

Improvement beyond the paper's approach:
In step 3, when applying the DPP-based token selection algorithm, this implementation provides a splitting strategy option in addition to the original CDPruner approach. While the original approach processes the entire kernel matrix at once, the splitting strategy divides the kernel matrix into two separate blocks for parallel processing when the visual token count exceeds a threshold (default : 1, can be set via environment variable `CDPRUNER_SPLIT_THRESHOLD`). 
*Note:* The split variant is not semantically equivalent to running DPP on the full kernel. In the split approach, an equal number of tokens are selected from each half and then merged, whereas a single full-kernel DPP call may select all the top-K tokens from one half if those tokens are most diverse/relevant. This constraint in the split variant can change the token selection set and may affect accuracy differently depending on the model and input. In practice, this splitting strategy has shown: (a) improved accuracy when evaluated on Qwen2.5-VL models, and (b) significantly faster GPU execution with OpenCL kernels due to better parallelization (2-3x speedup with large token counts). By default, the splitting strategy is enabled. Advanced users can disable it by setting the environment variable CDPRUNER_SPLIT_THRESHOLD=0 to use the original approach.

**Effect:** Pruning less important visual tokens reduces memory usage and can speed up generation; extremely high pruning may degrade answer quality for complex visual queries.

## Configuration Interface
Visual Token Pruning is exposed through fields of `ov::genai::GenerationConfig`:

* `pruning_ratio` (integer, 0–99): Portion of visual tokens to prune, specified as an integer percentage. A value of 0 disables pruning. For example, `25` means prune 25% of the visual tokens (keep 75%). Out-of-range values (negative or >=100) are treated as 0 (disabled) to avoid eliminating the entire visual context.
* `relevance_weight` (float): Weighting factor applied when aggregating or scaling dominance scores. **Recommended range:** 0.0–1.0. A value of 0 disables relevance weighting (pruning is based solely on raw dominance scores), while higher values (up to 1.0) emphasize relevance, making pruning more conservative on borderline tokens. Values above 1.0 are allowed but may have diminishing or unpredictable effects; negative values are not recommended. Default in the sample is `0.5f`.

### Sample Usage (Python Benchmark Script)
[samples/python/visual_language_chat/benchmark_vlm.py](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/visual_language_chat/benchmark_vlm.py) provides a convenient way to measure performance impact of pruning.

Minimal example (prune 70% of visual tokens on GPU):
```bash
python benchmark_vlm.py \
  -m ./models/vlm \
  -i ./data/example.jpg \
  -p "What is on the image?" \
  -d GPU \
  --pruning_ratio 70 \
  --relevance_weight 0.6
```

Relevant configuration excerpt:
```python
config = ov_genai.GenerationConfig()
config.max_new_tokens = args.max_new_tokens
config.pruning_ratio = args.pruning_ratio
config.relevance_weight = args.relevance_weight
```

Pipeline creation and generation:
```python
pipe = ov_genai.VLMPipeline(models_path, device, scheduler_config=scheduler_config)
res = pipe.generate(prompt, images=images, generation_config=config)
```

The script prints performance metrics (time-to-first-token TTFT, throughput, per-stage durations). Compare runs with different `--pruning_ratio` to quantify latency improvements and memory savings.

## Performance & Benefits
* Reduced KV cache memory for visual tokens -> enables larger batch sizes or longer text generation within same memory budget.
* Lower per-step attention computations involving image tokens -> improved latency.
* Helpful for edge or GPU memory-constrained deployments (e.g., running VLM on integrated GPU with limited VRAM).

## Current Limitations
* Current implementation assumes a standard image encoder output; exotic hierarchical or sparse encoders might require adjusted scoring strategies.
* Pruning is applied only after the initial image encoding; does not dynamically re-introduce pruned tokens later.
* Score computation details are internal; no per-token debug API is exposed yet.
* The current implementation supports Qwen-VL models only; support for other models will be added in a subsequent release.