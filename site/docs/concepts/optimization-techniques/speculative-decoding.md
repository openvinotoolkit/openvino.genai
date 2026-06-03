---
sidebar_position: 1
---

# Speculative Decoding

## Overview
Speculative decoding (also referred to as [assisted-generation](https://huggingface.co/blog/assisted-generation#understanding-text-generation-latency)) is a latency-oriented optimization for autoregressive text generation. A small, cheap *drafter* proposes one or more candidate continuations, and the larger *target* (main) model validates them in a single forward pass. Tokens that the target would have produced anyway are accepted; the first divergent token is replaced by the target's own sample, and drafting resumes from there.

Because validation is parallel over the candidate length, each accepted draft token replaces a sequential decode step of the target model with (a fraction of) one batched step. When acceptance is high, this can yield end-to-end speedups on memory-bandwidth-bound decoding without changing the model output distribution in theory(the standard verification scheme is statistically equivalent to sampling from the target).

`openvino.genai` implements three drafting strategies, sharing the same target-side validation step and the same `GenerationConfig` knobs (`num_assistant_tokens`, `assistant_confidence_threshold`):

| Strategy | Drafter | Strength | Configured by |
|---|---|---|---|
| Speculative Decoding (Fast Draft) | Smaller LLM | General-purpose, robust speedup | `draft_model(...)` + `num_assistant_tokens` or `assistant_confidence_threshold` (CB only) |
| Prompt Lookup Decoding | n-gram match against prompt | Best when output copies large spans of input (RAG, summarization, code edit) | `prompt_lookup=True` + `num_assistant_tokens` + `max_ngram_size` |
| EAGLE3 | Custom 1-layer draft head trained on the target's hidden states; supports tree drafting | Highest acceptance rate, tightest target/draft alignment | `draft_model(...)` with an EAGLE3 head + `num_assistant_tokens` + `branching_factor`/`tree_depth` |

All three are usable with the `LLMPipeline` Python and C++ APIs and are supported on the Continuous Batching backend; speculative decoding and prompt lookup are also supported on the Stateful backend.

## Common Configuration

The drafter is selected when constructing the `LLMPipeline`; the per-request behavior is controlled through fields of `GenerationConfig`:

* `num_assistant_tokens` (size_t) — number of candidate tokens the drafter proposes per iteration. For tree drafting (EAGLE3), this is the total number of non-root candidate tokens submitted to the target for verification (top-k of the tree). Defaults to 5 if unset.
* `assistant_confidence_threshold` (float) — dynamic-length stopping criterion: the drafter keeps proposing while the candidate token probability exceeds this threshold, then hands off to the target. Mutually exclusive with `num_assistant_tokens` when both would be active. **Continuous Batching backend only.**
* `max_ngram_size` (size_t) — maximum n-gram length to match against the prompt for Prompt Lookup. Setting this to a non-zero value together with `num_assistant_tokens` selects Prompt Lookup for that request.

EAGLE3 tree drafting adds two more fields:

* `branching_factor` (size_t) — number of top-k expansions retained per tree node and per tree layer.
* `tree_depth` (size_t) — number of draft iterations (tree layers). Setting `tree_depth > 0` switches the request from sequential drafting to tree drafting. The total candidate count is `branching_factor^2 * (tree_depth - 1) + branching_factor`, which must be ≥ `num_assistant_tokens`; the top `num_assistant_tokens` candidates by score are then sent to the target for verification.

Both backends additionally accept the standard `SchedulerConfig` (cache size, paged-attention block size, etc.). Note that on the Continuous Batching backend the cache is split between target and drafter, so users running large drafters should size `cache_size` accordingly.

Performance can be inspected via `result.extended_perf_metrics`, which exposes `main_model_metrics`, `draft_model_metrics`, and `get_num_accepted_tokens()` — the latter is the most direct measure of drafting effectiveness and the right number to optimize.

## Speculative Decoding (Fast Draft)

Fast Draft is the classic two-model setup: a smaller off-the-shelf LLM that shares the target's tokenizer drafts tokens autoregressively, and the target verifies them. It works for any target/draft pair the user has, without retraining; the speedup is bounded by how often the small model's distribution agrees with the large one.

### Usage with openvino.genai

Python:
```python
import openvino_genai

draft_model = openvino_genai.draft_model(draft_model_dir, "CPU")
pipe = openvino_genai.LLMPipeline(model_dir, "CPU", draft_model=draft_model)

config = openvino_genai.GenerationConfig()
config.max_new_tokens = 100
config.num_assistant_tokens = 5
# Or, on the Continuous Batching backend, dynamic-length drafting:
# config.assistant_confidence_threshold = 0.4

result = pipe.generate("The Sun is yellow because", config)
```

C++:
```cpp
ov::genai::LLMPipeline pipe(
    main_model_path, main_device,
    ov::genai::draft_model(draft_model_path, draft_device),
    ov::genai::scheduler_config(scheduler_config));

ov::genai::GenerationConfig config;
config.max_new_tokens = 100;
config.num_assistant_tokens = 5;
pipe.generate(prompt, config);
```

End-to-end samples: [Python](https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/python/text_generation/speculative_decoding_lm.py), [C++](https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/cpp/text_generation/speculative_decoding_lm.cpp).

### Backend differences
* **Continuous Batching** uses `num_assistant_tokens` as-is, supports `assistant_confidence_threshold`, and runs the drafter and target as fully scheduled paged-attention pipelines (each with its own KV cache). Multi-request batching is supported.
* **Stateful** uses `num_assistant_tokens` as the *initial* value and adapts it per step based on recent acceptance, falling back gracefully when acceptance is low. `assistant_confidence_threshold` is not supported. Single-request only.

For NPU deployments, the recommended configuration is to place both the target and draft models on the NPU.

## Prompt Lookup Decoding

[Prompt Lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) replaces the draft model with a string match: the most recently emitted suffix is searched as an n-gram inside the prompt, and the tokens that follow each match in the prompt are proposed as candidates. No second model is loaded.

This is highly effective for *input-grounded* generation — RAG / document QA, summarization, code editing, multi-turn chat — where the output frequently copies entity names, phrases, or code chunks verbatim from the input. On these workloads it produces speculative-decoding-class speedups at near-zero drafter cost, and on workloads with low prompt/output overlap it degrades back to plain decoding (the proposals are simply rejected).

### Usage with openvino.genai

Python:
```python
import openvino_genai

pipe = openvino_genai.LLMPipeline(model_dir, "CPU", prompt_lookup=True)

config = openvino_genai.GenerationConfig()
config.max_new_tokens = 100
config.num_assistant_tokens = 5
config.max_ngram_size = 3
pipe.generate(prompt, config)
```

End-to-end samples: [Python](https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/python/text_generation/prompt_lookup_decoding_lm.py), [C++](https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/cpp/text_generation/prompt_lookup_decoding_lm.cpp).

`max_ngram_size` controls how aggressively the lookup matches: larger values produce longer, more confident continuations when they hit, but match less often.

## EAGLE3

[EAGLE-3](https://arxiv.org/abs/2503.01840) (Extrapolation Algorithm for Greater Language-model Efficiency) replaces the generic draft model with a small head — typically one transformer layer — that is trained to mimic the target's distribution conditioned on the target's *hidden states*, not just on tokens. This much tighter coupling raises the per-token acceptance rate substantially over Fast Draft on the same target, while keeping the drafter cheap.

OpenVINO GenAI's EAGLE3 path additionally supports **tree drafting with top-k dynamic tree search**: instead of a single linear chain of `num_assistant_tokens` candidates, the drafter expands the top `branching_factor` continuations at each of `tree_depth` layers, scores the resulting tree, and submits the top `num_assistant_tokens` paths to the target for parallel verification. With one target forward pass the pipeline can therefore validate multiple alternative continuations and accept the longest matching one — directly compounding EAGLE3's already-high acceptance rate into longer accepted runs.

The Continuous Batching pipeline keeps both models' KV caches consistent across speculation/verification cycles by running an asynchronous KV-update step that propagates accepted blocks from the target back into the draft cache between iterations.

### Model preparation

Export both the target and the matching EAGLE3 draft head with `optimum-cli`. The draft repository must be an EAGLE3 head trained against the same target family (the head exposes a `d2t` mapping table that openvino.genai picks up automatically):

```bash
optimum-cli export openvino --weight-format int4 --trust-remote-code \
    --task text-generation-with-past -m Qwen/Qwen3-8B Qwen3-8B-ov-int4

optimum-cli export openvino --weight-format int4 --trust-remote-code \
    --task text-generation-with-past -m AngelSlim/Qwen3-8B_eagle3 Qwen3-8B_eagle3-ov-int4
```

### Usage with openvino.genai

Python — chain drafting (linear, EAGLE3 head only):
```python
import openvino_genai

draft_model = openvino_genai.draft_model("Qwen3-8B_eagle3-ov-int4", "GPU")
pipe = openvino_genai.LLMPipeline("Qwen3-8B-ov-int4", "GPU", draft_model=draft_model)

config = openvino_genai.GenerationConfig()
config.max_new_tokens = 100
config.num_assistant_tokens = 5
pipe.generate("What is OpenVINO?", config)
```

Python — tree drafting (top-k dynamic tree):
```python
config = openvino_genai.GenerationConfig()
config.max_new_tokens = 100
config.num_assistant_tokens = 15  # top-K verified per step
config.branching_factor = 8       # top-k per tree node
config.tree_depth = 4              # number of draft layers
pipe.generate("What is OpenVINO?", config)
```

EAGLE3 streaming is currently restricted to batch size 1 with greedy or tree-search sampling.

### How tree drafting works
1. The EAGLE3 head runs `tree_depth` iterations. At each layer it expands the top `branching_factor` children of every surviving node, producing up to `branching_factor^2 * (tree_depth - 1) + branching_factor` candidate tokens organized as a tree.
2. Candidates are scored cumulatively along their tree paths; the top `num_assistant_tokens` are flattened into a single packed input together with a tree attention mask and tree position ids.
3. The target processes the packed input in one forward pass and verifies all paths simultaneously. The longest path whose tokens match the target's own samples is accepted.
4. Accepted tokens become the new prefix; the target's KV cache for those tokens is propagated into the EAGLE3 draft cache asynchronously, and the next round begins.

Compared to chain drafting, tree drafting trades a larger validation batch for a higher per-step accepted-token count. The right operating point depends on the target's batch-size scaling: as long as the target step time grows sublinearly with the candidate count, more aggressive trees pay off.

## Performance & Tuning Notes

* The single most important diagnostic is `result.extended_perf_metrics.get_num_accepted_tokens()` divided by the number of target iterations — i.e. the average accepted-tokens-per-step. Aim for this to be well above 1; values close to 1 indicate that drafting is not paying for itself.
* Increase `num_assistant_tokens` until acceptance per step plateaus, then back off — past the plateau, rejected draft tokens are pure overhead.
* For EAGLE3 tree drafting, the trio `(branching_factor, tree_depth, num_assistant_tokens)` is the main tuning surface. A reasonable starting point on a high-acceptance target is `branching_factor=4..8`, `tree_depth=3..4`, `num_assistant_tokens` needs to be set to take hardware capability into consideration.
* On the Continuous Batching backend the target and draft KV caches share the configured `cache_size`. With a large drafter (Fast Draft), increase `SchedulerConfig.cache_size`; with EAGLE3 the draft head is small, so the target dominates as usual.

## Current Limitations

* Speculative decoding is supported on Continuous Batching (CPU/GPU) and on the Stateful backend (CPU/GPU/NPU). `assistant_confidence_threshold` and multi-request batching are Continuous Batching only for fast draft.
* EAGLE3 streaming requires with greedy or tree-search sampling for now, beam searching or multi-nomial is not supported.
* Tree drafting (`tree_depth > 0`) is supported only by the EAGLE3 path.
* Prompt Lookup is single-model and uses no draft model; the `draft_model` and `prompt_lookup` constructor options are mutually exclusive.
