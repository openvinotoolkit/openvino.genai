# Model Analysis: lfm3-vl-1.9b-rl-Intel-int8 (lfm3_vl)

## Identity
- Local IR: sandbox/lfm3-vl-1.9b-rl-Intel-int8
- Architecture: Lfm3VlForConditionalGeneration (remote code).
- Task / modality: image-text-to-text; upstream also supports video.
- Transformers 5.20.2; optimum-intel 2.3.0.dev0+32373d72 in py312-lfm3.

## Exported IR
| File | Role | Inputs | Outputs |
|------|------|--------|---------|
| openvino_language_model.xml | Hybrid decoder | inputs_embeds: [?,?,2048] f32; attention_mask: [?,?] i64; beam_idx: [?] i32 | logits: [?,?,128000] f32 |
| openvino_text_embeddings_model.xml | Token lookup | input: [?,?] i64 | inputs_embeds: [?,?,2048] f32 |
| openvino_vision_embeddings_model.xml | Vision encoder + projector | pixel_values: [?,?,?,1536] f32 | last_hidden_state: [?,?,2048] f32 |

The decoder has 35 state variables, including fixed-length convolution states.
The exported decoder computes positions internally: there is no position_ids input.

## Transformers
- Remote modeling source inspected in the existing transformers_modules cache;
  processor and image processor sources are included in the IR directory.
- Image processor subclasses Qwen2VLImageProcessor. Bicubic resize to multiples
  of patch_size * merge_size (16 * 2), area range [65536, 16777216].
- Rescale 1/255, mean and std [0.5, 0.5, 0.5].
- Spatial raster patches contain channel, temporal patch, patch row, patch column.
  Images repeat the same frame twice (temporal_patch_size=2).
- Vision model: 2D convolution / attention layers, then spatial 2x2 merging and MLP projection.
- Tokens: image=125030, video=125029, vision_start=125009, vision_end=125010.
  HF and exported OpenVINO token IDs match for all four and ChatML boundaries.
- Decoder uses ordinary 1D positions, not Qwen multimodal RoPE.

## Optimum-Intel
- Installed optimum/intel/openvino/modeling_visual_language.py,
  _OVLfm3VlForCausalLM.
- get_vision_embeddings converts Qwen block-major flattened patches to
  [temporal, grid_h, grid_w, patch_dim] raster order before invoking vision IR.
- get_multimodal_embeddings replaces image/video placeholder embeddings and
  checks feature/token count equality.
- preprocess_inputs applies the processor chat template with enable_thinking=False.
- Vision/text IR inputs and outputs are f32 despite config dtype=bfloat16.

## Notes
- Existing Optimum tests name optimum-intel-internal-testing/tiny-random-lfm3-vl,
  but unauthenticated config retrieval returned HTTP 401. Do not add an
  inaccessible hosted fixture to the test matrix.

## GenAI Enablement Design
- Closest GenAI model: LLaVA embedding merge plus Qwen2-VL dynamic image resize.
- Required changes:
  - vlm_config.hpp/.cpp: model enum and string mapping.
  - lfm3_vl/classes.hpp/.cpp: vision encoder with raster patch layout and
    image placeholder expansion with vision boundary tokens.
  - inputs_embedder.cpp / vision_encoder.cpp: both directory and ModelsMap factories.
  - inputs_embedder.hpp: friend declaration.
  - pipeline.cpp: pass position_ids only when the exported decoder accepts them.
  - tests/python_tests: deterministic local IR fixture if public tiny fixture remains inaccessible.
- Gaps: hybrid cache is already handled by generic cache-type detection and
  full-history replay. Video needs timestamp-aware prompt expansion and temporal
  frame grouping; do not alias Qwen multimodal positional logic.

## Validation
- Build: cmake --build build-Debug --target openvino_genai --parallel $(nproc), passed.
- Source module: build-Debug/openvino_genai/__init__.py.
- Text-only comparison: exact output on all 3 prompts with greedy decoding and f32 precision.
- Image comparison: exact output on all 3 synthetic image prompts with thinking disabled.
- Offline repository regression suite: 14 passed, including the Python sample,
  directory/ModelsMap constructors, image order, batched images, legacy and
  ChatHistory chat, missing images, and half-factor resize rounding.
- Full WWB, 24 images, aligned non-thinking templates: similarity 0.986449,
  20/24 exact answers. All generated answers are non-empty.
- First observed difference for remaining benchmark answers is preprocessing:
  input token IDs and raster shapes agree; Pillow bicubic versus HF's backend
  differs by up to 0.007843196 in normalized values (RMSE about 0.00026).
  Even without a resizing difference, float normalization differs by about 6e-8.
- Default thinking mode is preserved. Comparing it to Optimum's non-thinking
  baseline is not a valid accuracy comparison (observed similarity 0.493485).
- Documentation lint passed. Site build is blocked by an existing generated
  JavaScript Whisper sample import whose source README is absent.
- Upstream tiny-random fixture remains inaccessible without authentication.
  Offline deterministic local IR tests cover the GenAI integration contract;
  they are not a replacement for an upstream random-weights architecture export.
