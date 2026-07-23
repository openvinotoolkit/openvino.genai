# Model Analysis: zai-org/glm-edge-v-2b (glm) — local tiny: glm-edge-v-2b-tiny

Package versions: transformers 4.57.6; optimum.intel from
/home/panas/git/auto-openvino-bot/workspace/optimum-intel.

## Identity
- model_id (tiny/local): /home/panas/git/auto-openvino-bot/workspace/glm-edge-v-2b-tiny
- original_model_id: zai-org/glm-edge-v-2b
- config.json: model_type = "glm", architectures = ["GlmForCausalLM"]
- auto_map: configuration_glm.GlmConfig / modeling_glm.GlmForCausalLM (remote code)
- Nested vision_config: model_type = "siglip_vision_model", image_size 672, patch_size 14
- boi_token_id = 59256 (<|begin_of_image|>), eoi_token_id = 59257
- eos_token_id = [59246, 59253, 59255], pad_token_id = 59246
- hidden_size (text) = 256 (tiny); real model larger
- Modality: image-text-to-text (VLM). model_type "glm" is shared with text-only
  GLM decoder; GLM-Edge-V is distinguished by presence of nested vision_config.

Real vs tiny: identical architecture identity (model_type glm + siglip
vision_config + GlmForCausalLM + MllamaImageProcessor + boi_token_id merge).
Only layer counts / hidden sizes differ. Safe to develop on the tiny model.

## Exported IR (ov_model_glm)
- openvino_language_model.xml
  - IN attention_mask [?,?] i64, position_ids [?,?] i64,
       inputs_embeds [?,?,256] f32, beam_idx [?] i32
  - OUT logits [?,?,59264] f32
- openvino_text_embeddings_model.xml
  - IN input [?,?] i64 -> OUT inputs_embeds [?,?,256] f32
- openvino_vision_embeddings_model.xml
  - IN pixel_values [?,3,672,672] f32 -> OUT last_hidden_state [1,578,256] f32
  (vision tower already includes projection to text hidden size; 578 tokens/image)
- Standard stateful LM: inputs_embeds + position_ids + attention_mask + beam_idx.

## Transformers (modeling_glm.py, remote code)
- Prefill merge (GlmModel.forward, ~L772-810):
  - inputs_embeds = embed_tokens(input_ids)
  - images_features = self.vision(imgs)  # [num_images, 578, hidden]
  - For each sample with boi_token_id present: find first boi_token_id position,
    count num_image_padding_tokens (== 578), replace that contiguous span with
    image features. i.e. simple placeholder replacement at boi_token_id.
- Image placeholder token: boi_token_id = 59256 = "<|begin_of_image|>".
  The processor/chat template emits 578 copies of <|begin_of_image|>.
- Position IDs: standard sequential (cache_position). No custom RoPE for images.
- Cache: standard stateful KV cache.

## Optimum-Intel (modeling_visual_language.py :: _OVGlmEdgeVForCausalLM)
- Registered as "glm" -> _OVGlmEdgeVForCausalLM (L7426).
- get_vision_embeddings: flattens 6D MllamaImageProcessor output
  (batch, media, tiles, C, H, W) -> (num_images, C, H, W), runs vision_embeddings
  -> last_hidden_state.
- merge_vision_text_embeddings: masked_scatter of image features into
  inputs_embeds at positions where input_ids == boi_token_id (59256).
- preprocess_inputs: chat template with {'type':'image'} content, then
  pixel_values = processor(image).pixel_values.

## Preprocessing (preprocessor_config.json)
- image_processor_type: MllamaImageProcessor
- size 672x672, resample=3 (bicubic), do_resize + do_pad + do_rescale + do_normalize
- image_mean=[0.5,0.5,0.5], image_std=[0.5,0.5,0.5], rescale_factor=1/255
- max_image_tiles=1 (single tile; effectively resize-to-672 + normalize)

## Notes
- Vision output token count is fixed (578) per image, read dynamically from the
  vision IR output shape [1,578,256] (rows).
- Chat template (chat_template.jinja) expands {'type':'image'} into 578
  <|begin_of_image|> tokens. In GenAI the prompt is normalized BEFORE the chat
  template is applied and the template receives a flat string content, so GenAI
  must expand the native image tag into 578 <|begin_of_image|> tokens itself.

## GenAI Enablement Design
- Closest GenAI model: InternVLChat — because both use a single repeated image
  placeholder token (image_context_token / boi_token) with a fixed per-image
  token count, simple contiguous placeholder replacement (no custom position
  IDs, standard stateful cache), and a vision IR whose output already projects
  to text hidden size. LLaVA is also close but expands with a trailing newline
  and single-token count from vision shape.
- Required changes:
  - src/cpp/src/visual_language/vlm_config.hpp: add VLMModelType::GLM_EDGE_V and
    an image_pad_token default "<|begin_of_image|>" reuse (use existing member).
  - src/cpp/src/visual_language/vlm_config.cpp: map "glm" -> GLM_EDGE_V.
  - src/cpp/src/visual_language/glm_edge_v/classes.{hpp,cpp}: VisionEncoderGLMEdgeV
    (MllamaImageProcessor-equivalent preprocessing: bicubic resize to 672x672 +
    normalize mean/std 0.5) and InputsEmbedderGLMEdgeV (normalize_prompt expands
    the native <image> tag into N copies of <|begin_of_image|>; get_inputs_embeds
    merges vision features at boi_token positions like InternVL).
  - Register in vision_encoder.cpp (both create overloads) and inputs_embedder.cpp
    (both constructors) factories.
  - Add friend class to inputs_embedder.hpp.
- Gaps: none significant. No custom position IDs, no extra LM inputs, no dynamic
  tiling (max_image_tiles=1). Preprocessing is plain resize+normalize.
