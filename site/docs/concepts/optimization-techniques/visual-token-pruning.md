---
sidebar_position: 5
---

# Visual Token Pruning (CDPruner)

## Overview
Visual Token Pruning is a context compression technique for Multimodal / Visual Language Models (VLMs) that reduces the number of vision tokens kept for subsequent decoding steps. It is based on the "CDPruner" approach described in the paper [Contextual Dominance Pruning for Long-Context Vision-Language Models](https://arxiv.org/pdf/2506.10967). The main goal is to decrease memory usage and latency during generation while preserving the most semantically relevant visual information for answering downstream questions.

During the first (prefill) pass, the model ingests the image and produces a sequence of visual tokens. Instead of keeping all of them, Visual Token Pruning selects a subset according to learned dominance scores. Pruned tokens are removed from further attention computations, shrinking KV cache footprint, reducing TTFT and improving throughput. A relevance re-weighting factor lets the user balance aggressiveness of pruning versus retention of fine-grained details.

## Conceptual Model
The visual token sequence extracted from the image encoder can be partitioned into:

* Retained Tokens: Subset judged most relevant by dominance scoring.
* Pruned Tokens: Dropped from future decoding (no longer participate in cross-attention or self-attention depending on architecture).

Pruning is controlled by a ratio (percentage of tokens to remove) and a relevance weight scaling that influences importance estimation.

High-level flow:
1. Encode image producing N visual tokens (embeddings).
2. Compute per-token dominance / relevance scores (implementation detail hidden inside the OpenVINO GenAI pipeline CDPruner module).
3. Sort / threshold to identify least important tokens according to `pruning_ratio`.
4. Optionally adjust scores using `relevance_weight` before selecting final kept set.
5. Build reduced token set; subsequent generation attends only to retained tokens.

Effect: Smaller effective visual context reduces memory and can speed up generation; extremely high pruning may degrade answer quality for complex visual queries.

## Configuration Interface
Visual Token Pruning is exposed through fields of `ov::genai::GenerationConfig`:

* `pruning_ratio` (size_t / integer percentage): Portion of visual tokens to prune. Value range: 0–99. A value of 0 disables pruning. For example, `25` means prune 25% of the visual tokens (keep 75%). Values >= 100 are ignored (treated as disabled) to avoid eliminating the entire visual context.
* `relevance_weight` (float): Weighting factor applied when aggregating or scaling dominance scores. Higher values emphasize relevance (making pruning slightly more conservative on borderline tokens); lower values increase pruning aggressiveness. Default in the sample is `0.5f`.

### Sample Usage (Python Benchmark Script)
`samples/python/visual_language_chat/benchmark_vlm.py` provides a convenient way to measure performance impact of pruning.

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
config.pruning_ratio = args.pruning_ratio if args.pruning_ratio is not None else 0
if config.pruning_ratio > 0 and config.pruning_ratio < 100:
    print(f"[CDPruner] Enabling CDPruner with {config.pruning_ratio}% visual token pruning")
    if args.relevance_weight is not None:
        config.relevance_weight = args.relevance_weight
        print(f"[CDPruner] Setting relevance weight to {config.relevance_weight}")
else:
    config.pruning_ratio = 0
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

## Limitations
* Current implementation assumes a standard image encoder output; exotic hierarchical or sparse encoders might require adjusted scoring strategies.
* Pruning applied only after the initial image encoding; does not dynamically re-introduce pruned tokens later.
* Score computation details are internal; no per-token debug API exposed yet.