# Model Analysis: zai-org/glm-edge-v-2b (glm)

## Identity
- model_id: zai-org/glm-edge-v-2b
- model_type: glm  (config.json `model_type == "glm"`, architecture `GlmForCausalLM`, remote code)
- architecture: GLM-Edge-V — SigLIP vision tower + conv adapter + GLM language model
- task / modality: image-text-to-text (vision-text)
- transformers version: 5.5.4 (primary); model-native remote code targets 4.47.1
- optimum-intel: editable checkout at workspace/optimum-intel (has `_OVGlmEdgeVForCausalLM`, model_type "glm")

## Exported IR (workspace/ov_glm_edge_v, fp16)
| File | Role | Inputs (name: shape, dtype) | Outputs |
|------|------|-----------------------------|---------|
| openvino_language_model.xml | GLM decoder | attention_mask [?,?] i64; position_ids [?,?] i64; inputs_embeds [?,?,2048] f32; beam_idx [?] i32 | logits [?,?,59264] f32 |
| openvino_text_embeddings_model.xml | token embed | input [?,?] i64 | inputs_embeds [?,?,2048] f32 |
| openvino_vision_embeddings_model.xml | SigLIP+adapter | pixel_values [?,3,?,?] f32 | last_hidden_state [?,578,2048] **f16** |

Key facts:
- Vision output is **already at LM hidden dim 2048** (adapter projects). No resampler / projector at runtime.
- Fixed **578 image embeddings per image tile**; single tile (max_image_tiles=1).
- Vision output dtype is **float16**; must be converted to float32 before merging with f32 text embeds.

## Transformers / processor
- Text: plain tokenizer. Image: separate `MllamaImageProcessor` (no combined AutoProcessor).
- preprocessing (preprocessor_config.json): size 672x672, resample=3 (BICUBIC), do_resize+do_pad,
  do_rescale (1/255), do_normalize mean=[0.5,0.5,0.5] std=[0.5,0.5,0.5], max_image_tiles=1, do_convert_rgb.
  Mllama: aspect-preserving resize to fit 672x672 canvas, then pad bottom-right with 0 (after normalize).
- special tokens: boi `<|begin_of_image|>`=59256, eoi `<|end_of_image|>`=59257, eos {59246,59253,59255}, pad 59246.
- chat template expands one image into exactly 578 `<|begin_of_image|>` (boi) tokens, then vision embeddings
  replace those boi placeholder positions. Text side wrapped as `<|user|>\n...<|assistant|>\n`.

## Optimum-Intel (reference behavior)
- module: workspace/optimum-intel/optimum/intel/openvino/modeling_visual_language.py `_OVGlmEdgeVForCausalLM`
- get_vision_embeddings: reshape 6D pixel_values (B,media,tiles,C,H,W) -> (N,C,H,W); run vision -> last_hidden_state.
- merge_vision_text_embeddings: `input_ids == boi_token_id` mask; scatter vision embeds (flattened) into those rows.
- IR ↔ component: language_model / text_embeddings_model / vision_embeddings_model as above.

## GenAI Enablement Design
- Closest GenAI model: **LLaVA** — because it is the simplest placeholder-replacement VLM: vision encoder emits
  embeddings already in LM space and `utils::merge_text_and_image_embeddings_llava` scatters them into the
  positions of a single image placeholder token. GLM-Edge-V matches this exactly, with boi(59256) as the
  placeholder and 578 embeds/image. Only preprocessing differs (Mllama square-resize+pad vs CLIP resize+crop).
- Required changes:
  - `vlm_config.hpp`: add `GLM_EDGE_V` enum.
  - `vlm_config.cpp`: map `"glm"` -> `GLM_EDGE_V`.
  - `glm_edge_v/classes.{hpp,cpp}`: `VisionEncoderGLMEdgeV` (Mllama preprocess, f16->f32 output) and
    `InputsEmbedderGLMEdgeV` (reuse `merge_text_and_image_embeddings_llava` with boi token id, expand image
    placeholder to 578 boi tokens).
  - `vision_encoder.cpp` + `inputs_embedder.cpp`: register factory branches.
  - `inputs_embedder.hpp`: `friend class InputsEmbedderGLMEdgeV`.
  - CMake glob picks up the new dir automatically (visual_language/*/*.cpp).
- Gaps: none requiring new infrastructure. Vision output f16->f32 conversion handled in encoder. Image
  placeholder token is `<|begin_of_image|>`, resolved by tokenizing the boi string at runtime.
