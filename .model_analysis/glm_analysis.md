# Model Analysis: zai-org/glm-edge-v-2b (glm)

## Identity
- model_id: /workspace/tiny-glm-edge-v-2b (tiny-random of zai-org/glm-edge-v-2b)
- model_type: `glm` (top-level), vision_config.model_type: `siglip_vision_model`
- architecture: `GlmForCausalLM` (with a SigLIP vision tower bridged in for VLM)
- task / modality: image-text-to-text (vision-text)
- transformers version: 4.57.6   optimum-intel: local checkout (workspace/optimum-intel)

## Exported IR
| File | Role | Inputs (name: shape, dtype) | Outputs |
|------|------|-----------------------------|---------|
| openvino_language_model.xml | GLM decoder | attention_mask [?,?] i64; position_ids [?,?] i64; inputs_embeds [?,?,128] f32; beam_idx [?] i32 | logits [?,?,59264] f32 |
| openvino_text_embeddings_model.xml | token embedding | input [?,?] i64 | inputs_embeds [?,?,128] f32 |
| openvino_vision_embeddings_model.xml | SigLIP vision tower | pixel_values [?,3,?,?] f32 | last_hidden_state [1,6,128] f32 |
| openvino_tokenizer / openvino_detokenizer | tokenizer | - | - |

Note: vision tower emits `last_hidden_state` shaped `[1, num_image_tokens, hidden]`
where num_image_tokens already includes the begin/end-of-image markers baked into
the vision embeddings. For the tiny model num_image_tokens = 6.

## Transformers
- module path: remote-code `modeling_glm.py` / `configuration_glm.py` in the model dir.
- GLM decoder: standard causal LM with GlmRotaryEmbedding using **1D sequential position_ids**
  (no interleaved GLM-4V multimodal position ids). partial_rotary_factor=1.0.
- Vision tower: SigLIP; preprocessing via `MllamaImageProcessor` (do_resize to size
  {h:56, w:56}, resample=3 bicubic, do_rescale 1/255, do_normalize mean/std=0.5, do_pad,
  max_image_tiles=1). pixel_values emitted 6D `[batch, media, tiles, 3, H, W]`.
- special tokens: `<|begin_of_image|>`=59256 (boi_token_id), `<|end_of_image|>`=59257
  (eoi_token_id), `<|user|>`=59253, `<|assistant|>`=59254, eos ids [59246,59253,59255],
  pad=59246.
- chat template: image content type renders `range(N)` of `<|begin_of_image|>` tokens
  (N=6 for tiny), text content renders text verbatim. The boi placeholders are what the
  vision bridge replaces with image features.
- position ids / RoPE: standard sequential.

## Optimum-Intel
- module path: workspace/optimum-intel/optimum/intel/openvino/modeling_visual_language.py
- class: `_OVGLMEdgeVForCausalLM`, registered `"glm": _OVGLMEdgeVForCausalLM`.
- IR <-> logical mapping: `vision_embeddings` -> SigLIP tower (`last_hidden_state`);
  `text_embeddings` -> token embed; `language_model` -> GLM decoder.
- get_vision_embeddings: flattens pixel_values >4D to `[-1,3,H,W]`, runs vision tower,
  returns `last_hidden_state`.
- merge_vision_text_embeddings: replaces every `boi_token_id` position in the flattened
  text embeddings with the vision embedding rows, in order (simple positional replace).
- preprocess_inputs: applies chat template (inserts boi placeholders), then image processor
  keeps only 6D `pixel_values`.

## Notes
- Vision output row count == number of boi placeholder tokens in the prompt (model-driven).
- OpenVINO tokenizer preserves `<|begin_of_image|>` as a single id 59256 (verified),
  identical to HF. No special-token recovery needed.
- No resampler, no token_type_ids, no custom position ids -> merge is llava-style,
  preprocessing is gemma3-style (fixed resize + normalize).

## GenAI Enablement Design
- Closest GenAI model: **LLaVA** for embedding merge (positional replacement of an image
  placeholder token via `merge_text_and_image_embeddings_llava`) combined with **Gemma3**
  for vision preprocessing (fixed `size_width`/`size_height` resize + normalize by
  image_mean/image_std). GLM decoder uses default (sequential) position ids.
- Required changes:
  - `vlm_config.hpp`: add `VLMModelType::GLM_EDGE_V`; add `begin_of_image` placeholder
    string member.
  - `vlm_config.cpp`: map `"glm"` -> `VLMModelType::GLM_EDGE_V`.
  - `visual_language/glm_edge_v/classes.{hpp,cpp}`: `VisionEncoderGLMEdgeV` (bicubic resize
    to size_w/size_h + normalize) and `InputsEmbedderGLMEdgeV` (normalize_prompt expands a
    single boi tag to N boi tokens; get_inputs_embeds merges via llava util keyed on
    boi_token_id).
  - `vision_encoder.cpp` / `inputs_embedder.cpp`: register factory branches + include header.
  - `inputs_embedder.hpp`: add `friend class InputsEmbedderGLMEdgeV`.
- Gaps: none requiring new infrastructure. Image placeholder token id resolved from the
  tokenizer at runtime (encode `<|begin_of_image|>`), matching llava's approach.
