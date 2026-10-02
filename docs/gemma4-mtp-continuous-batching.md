# Gemma4 MTP on the Continuous Batching Backend

## Purpose

Gemma4 MTP was initially implemented for a stateful SDPA target because Gemma4
could not use paged attention (PA) at the time. The PA implementation lets a
decomposed Gemma4 VLM target serve multiple requests through Continuous Batching
while using a Gemma4 assistant to propose tokens. Unlike the existing
continuous-batching MTP strategy, this assistant has **no KV cache**: each
assistant inference reads the target's paged key/value (KV) cache and consumes
a hidden state produced by the target.

The implementation was exercised with
`temp/gemma4_mtp/gemma-4-E2B-it` as the VLM target and
`temp/gemma4_mtp/draft` as the assistant. The E4B export is not compatible with
this assistant: its text hidden size is 2560 rather than the assistant's
expected 1536. The older `temp/gemma4_mtp/main` export remains an SDPA
reference, not the PA target.

## What changed

### Routing and model preparation

- `draft_model()` recognizes the assistant by its `full_attention_key` input
  when loading `openvino_model.xml` and enables MTP routing. Continuous
  Batching selects `Gemma4MtpDecodingImpl` for this draft interface, retaining
  the existing `MtpDecodingImpl` for other MTP models.
- `expose_last_hidden_state` exports the input to the target LM-head MatMul
  before PA conversion; it can locate that MatMul beneath Gemma4's logits
  postprocessing. The last full- and sliding-attention PA layers are identified
  from their cache widths and connected to the draft's corresponding layers.
- The assistant is also transformed with `SDPAToPagedAttention` using its
  draft-specific flag. All four assistant SDPA layers become read-only PA
  layers. The assistant is compiled as a separate infer request, without its
  own scheduler or persistent KV state; its PA cache inputs share the target's
  physical cache tensors. The text embedding model is shared with
  the target's inputs embedder.

### Request and inference flow

1. The target prefill runs through Continuous Batching using VLM embeddings,
   positions and any extra language-model inputs (including per-layer inputs).
   This preserves the existing text and image input preparation.
2. The target's CB scheduler owns the physical KV cache and each sequence's
   block table. The assistant binds the target's full/sliding cache tensors and
   uses the requesting sequence's physical block indices for its accepted
   prefix; it does not copy or re-quantize KV. Each request retains only the
   selected target hidden state.
3. For each draft position, the assistant receives the embedding of the last
   token concatenated with the current hidden state and a sequential position
   ID. The assistant's compiled cache layout and precision must match the
   target's. The draft PA transformation reads the full accepted prefix from
   the cache; its unused new-token KV inputs do not decode quantized cache
   bytes or write a draft cache. The assistant
   greedily selects the next token and feeds its output hidden state into the
   following draft call.
4. Draft candidates are submitted to the target's validation-mode pipeline.
   The target decides which candidates match and supplies the replacement or
   bonus token. Before the next draft round, rejected KV suffixes are removed
   from the next draft's `past_lens` and the hidden state for the accepted
   position is selected. Per-request
   draft and acceptance metrics are updated.

The implementation uses a single assistant infer request under the speculative
pipeline's step lock. Multiple target requests can be scheduled together, but
the assistant calls are made per request, not as one assistant batch.

### Configuration and compatibility

- Supported generation is greedy with a fixed positive
  `num_assistant_tokens`, one return sequence per request, and no tree search,
  confidence-threshold drafting or adapters.
- `SchedulerConfig.max_num_batched_tokens` must exceed
  `num_assistant_tokens` so the target can schedule the verification window.
  Requests with an insufficient budget are rejected rather than silently
  changing their draft length.
- Prefix caching and cache eviction remain disabled for this target path;
  sharing cached prefixes or evicting blocks requires explicit lifetime and
  mapping support in the draft. PA-oriented `LLMPipeline` and `VLMPipeline` constructors
  disable prefix caching by default when this assistant is selected; an
  explicitly incompatible scheduler configuration is rejected.
- Target and assistant must execute on the same device(s) and use compatible
  PA cache layouts and precisions. On CPU, the assistant cache precision is
  compiled to match the target, including the default quantized cache. The
  OpenVINO draft PA transformation must support reading a quantized cache
  without interpreting the cache's scale metadata as token values.
- The implementation requires a compatible decomposed Gemma4 VLM export and
  assistant. The SDPA stateful Gemma4 strategy is not removed. Existing MTP
  models with their own draft cache continue through the original strategy.
- The decomposed VLM-to-`LLMPipeline` adapter now propagates completed status
  from VLM finish reasons so PA generation does not appear unfinished.

## Usage

```python
import openvino_genai as genai

target = "temp/gemma4_mtp/gemma-4-E2B-it"
assistant = "temp/gemma4_mtp/draft"

pipe = genai.LLMPipeline(
    target,
    "CPU",
    draft_model=genai.draft_model(assistant, "CPU"),
    ATTENTION_BACKEND="PA",
)
config = genai.GenerationConfig(
    do_sample=False,
    max_new_tokens=32,
    num_assistant_tokens=3,
)
result = pipe.generate(["OpenVINO is"], config)
```

Use `VLMPipeline` with the same PA and `draft_model` settings for image inputs,
or `ContinuousBatchingPipeline` to submit multiple text requests directly.
For user-facing configuration guidance, see
[Speculative Decoding](../site/docs/concepts/optimization-techniques/speculative-decoding.mdx).

## Validation and remaining coverage

- The project was rebuilt and installed in `venv` with
  `pip install --pre -U . --no-deps --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly`.
- The supplied draft's four SDPA layers were converted to PA, with no SDPA
  remaining, and the transformed model compiled on CPU. Direct cache sharing
  requires the OpenVINO stateless-draft fix and the read-only quantized-cache
  draft transformation described above.
- `cmake --build build --target tests_continuous_batching --parallel 18`
  succeeded with `venv` active. The focused
  `MtpModelTransforms.*:MtpDraftUpdatePlan.*:CBForSDTest.Mtp*` filter passed
  **15 C++ tests**, including coverage for Gemma4 logits postprocessing. The
  OpenVINO draft-pass regression test was added, but the local OpenVINO build
  has `ENABLE_TESTS=OFF`; the rebuilt OpenVINO runtime passed real-model inference.
- Manual CPU comparisons using the E2B target and PA-transformed assistant
  matched non-MTP PA output for two concurrent 12-token text requests and
  synthetic-image input with the default quantized cache. With f16 caches,
  the comparison also matched for two concurrent 40-token text requests
  crossing a PA block boundary. `LLMPipeline` returned a completed result with
  draft/acceptance metrics; an undersized verification budget was rejected.
  These are correctness checks, **not** latency or throughput measurements.
- The default quantized cache is directly shared, but greedy output can differ
  from target-only PA on longer prompts: in the 40-token comparison, both text
  requests changed one word. The source of this difference has not been
  isolated; longer quantized-cache generation remains an accuracy validation
  gap.
- Three tiny-random Gemma4 Python tests cover multi-request text, f16
  cross-block generation, and image generation. In the current `venv`, all
  **skip** because its
  `optimum.intel` does not export `OVAssistantForCausalLM`, which the fixture
  needs to create the tiny assistant. They have not passed in this
  environment; the supplied E2B/draft exports were validated directly.
