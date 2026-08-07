# Model Analysis: jinaai/jina-vlm (jvlm)

## Identity
- model_id: /home/openvino_bot/.../workspace/tiny_jina_vlm (tiny-random of jinaai/jina-vlm)
- model_type: jvlm
- architecture: JinaVLMForConditionalGeneration (registered via remote code under AutoModelForCausalLM)
- task / modality: image-text-to-text (vision-text)
- transformers version: 4.57.6   optimum-intel version: 2.1.0.dev0+fd94990
- Remote code: configuration_jvlm.py, modeling_jvlm.py, processing_jvlm.py, image_processing_jvlm.py, blocks_jvlm.py

## Exported IR (5 submodels)
| File | Role | Inputs (name: shape, dtype) | Outputs |
|------|------|-----------------------------|---------|
| openvino_text_embeddings_model.xml | text token -> embedding | input: [?,?] i64 | inputs_embeds: [?,?,64] f32 |
| openvino_vision_embeddings_model.xml | vision encoder + connector | image_patches: [?,?,?,588] f32; image_masks: [?,?,?] i64 | last_hidden_state: [?,?,64] f32 |
| openvino_language_model.xml | LM (decoder, with past) | inputs_embeds: [?,?,64] f32; attention_mask: [?,?] i64; position_ids: [?,?] i64; beam_idx: [?] i32 | logits: [?,?,152064] f32 |
| openvino_tokenizer.xml | tokenizer | - | - |
| openvino_detokenizer.xml | detokenizer | - | - |

Notes:
- vision input `image_patches` last dim 588 = patch_size(14)^2 * 3 channels = 196*3.
- LM uses standard inputs_embeds + attention_mask + position_ids + beam_idx (Qwen-style decoder). hidden_size=64 (tiny). Real model hidden_size larger.
- text_config.vocab_size=152064; additional_vocab_size=128 (special image tokens live in additional vocab).

## Transformers (remote code)
- Processor: JinaVLMProcessor (processing_jvlm.py). Image special tokens:
  - `<im_patch>` (patch_token_id, in-text placeholder for each image patch/token)
  - `<im_start>` / `<im_end>` (image boundary)
  - `<im_col>` (column separator token)
  - `<|image|>` (image prompt token, chat template placeholder)
  - `<im_slice>`
- Image preprocessing: JinaVLMImageProcessor (image_processing_jvlm.py), Molmo-derived.
  - cropping_method = "overlap-and-resize" (Molmo), max_crops=12, overlap_margins=[4,4]
  - base_input_size=[378,378], patch_size=14, pooling 2x2, tokens_per_image=196
  - normalization: minmax (image_min=-1, image_max=1), image_mean/std = OPENAI_CLIP
  - Produces: image_patches [n_crops, n_patches, 588], image_masks [n_crops, n_patches], image_input_idx [n_crops, n_tokens]
- Merge (modeling_jvlm / optimum-intel _OVJinaVLMForCausalLM.merge_vision_text_embeddings):
  - Index-based scatter: inputs_embeds[batch_idx[valid], image_input_idx[valid]] = image_embeds[valid]
  - image_input_idx gives absolute positions in the token sequence where each image embedding row is placed (>=0 valid, <0 skip).
- LM: Qwen-style RMSNorm decoder, RoPE theta 1e6, n_kv_heads=2, head_dim=16 (tiny). standard position_ids.

## Optimum-Intel
- module path: optimum/intel/openvino/modeling_visual_language.py : class _OVJinaVLMForCausalLM (line 5646), registered key "jvlm" (line 7509).
- IR <-> logical mapping:
  - vision_embeddings(image_patches, image_masks) -> last_hidden_state (already projected to hidden_size)
  - text_embeddings(input_ids) -> inputs_embeds
  - language_model(inputs_embeds, attention_mask, position_ids, beam_idx) -> logits
- get_vision_embeddings: skips vision when decoding a single new token (input_ids.shape[1]==1).
- merge_vision_text_embeddings: scatter by image_input_idx (see above). image_input_idx carried only on prefill step.
- preprocess_inputs: uses chat template + processor(images, text) -> {input_ids, image_patches, image_masks, image_input_idx}.

## GenAI baseline
- openvino_genai.VLMPipeline(ov_model, "CPU") raises: "Unsupported 'jvlm' VLM model type" (vlm_config.cpp:43). jvlm not enabled in GenAI.

## Notes / gaps for GenAI enablement
- No existing GenAI model uses image_input_idx scatter; needs a new merge strategy.
- Vision encoder IR consumes pre-patchified image_patches + image_masks (Molmo overlap-and-resize preprocessing must be reproduced in C++), plus image_input_idx computed on host.
- Special image tokens are in the additional vocab; the OV tokenizer must be checked to preserve their IDs.

## GenAI Enablement Design
- Closest GenAI model: LLaVA — because jvlm merges vision embeddings into text at
  placeholder-token positions. Validated (proto_merge.py) that scatter by
  `image_input_idx` is IDENTICAL to sequential scatter onto `<im_patch>` (id 151938)
  positions, so a simple placeholder scatter (llava-style, but per-token not per-block
  because im_patch tokens are interrupted by im_col/im_start/im_end) reproduces the
  optimum output exactly.
- Validated facts (Python prototypes under .model_enabler/jvlm/):
  - OV tokenizer encodes special image tokens as single ids: <im_start>=151936,
    <im_end>=151937, <im_patch>=151938, <im_col>=151939, <|image|>=151940.
  - Expanding `<|image|>` into the joint token string
    [im_start (patch*TL + col)*TL im_end] per crop (global thumbnail first, then crops)
    reproduces the processor input_ids EXACTLY (439 tokens for a 64x64 image).
  - Vision IR output is [1, n_crops*196, hidden]; merge = fill <im_patch> positions in
    order with these rows.
  - Pure-numpy Molmo preprocessing (bilinear align_corners=False, minmax normalize)
    reproduces image_masks exactly and image_patches within 2/255; LM tokens are
    token-identical to the reference/optimum output.
- Required changes:
  - vlm_config.hpp/.cpp: add VLMModelType::JVLM + "jvlm" mapping; add jvlm token strings.
  - visual_language/jvlm/classes.{hpp,cpp}: VisionEncoderJVLM (Molmo preprocessing +
    vision IR: inputs image_patches, image_masks) and InputsEmbedderJVLM
    (normalize_prompt expands <|image|>, get_inputs_embeds scatters onto <im_patch>).
  - vision_encoder.cpp / inputs_embedder.cpp: factory registration; inputs_embedder.hpp friend.
- Gaps: no existing GenAI model uses image_input_idx or Molmo overlap-and-resize; both
  implemented fresh. Vision IR takes pre-patchified image_patches [n_crops, n_patches, 588]
  and image_masks [n_crops, n_patches]; EncodedImage carries these plus per-crop token
  layout for prompt expansion.
