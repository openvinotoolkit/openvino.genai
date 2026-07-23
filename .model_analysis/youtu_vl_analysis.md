# Model Analysis: tencent/Youtu-VL-4B-Instruct (youtu_vl)

## Identity
- model_id: /home/panas/git/auto-openvino-bot/workspace/tiny-youtu-vl (tiny-random of tencent/Youtu-VL-4B-Instruct)
- model_type: youtu_vl
- architecture: YoutuVLForConditionalGeneration
- task / modality: image-text-to-text (vision-text)
- transformers version: 4.57.6   optimum-intel: local checkout (workspace/optimum-intel)

## Exported IR
| File | Role | Inputs (name: shape, dtype) | Outputs |
|------|------|-----------------------------|---------|
| openvino_language_model.xml | DeepSeek-style MLA/MoE LLM | attention_mask [?,?] i64; position_ids [?,?] i64; inputs_embeds [?,?,32] f32; beam_idx [?] i32 | logits [?,?,283386] f32 |
| openvino_text_embeddings_model.xml | token embedding | input [?,?] i64 | inputs_embeds [?,?,32] f32 |
| openvino_vision_embeddings_model.xml | SigLIP2 patch encoder | pixel_values [?,?,768] f32 | last_hidden_state [?,32] f32 |
| openvino_vision_embeddings_merger_model.xml | window-attn + 2x2 patch merger | hidden_states [?,32] f32; attention_mask [1,?,?] f32; window_attention_mask [1,?,?] f32; window_index [?] i64; rotary_pos_emb [?,4] f32 | last_hidden_state [?,32] f32 |

Note: LM `position_ids` is 2D `[?,?]` (plain 1D position ids broadcast over batch), NOT 3D mRoPE.

## Transformers
- module path: tiny-youtu-vl/ (remote code: modeling_youtu_vl.py, modeling_siglip2.py, processing_youtu_vl.py, image_processing_siglip2_fast.py)
- LLM: DeepSeek-V3-like — MLA attention (q_lora_rank=32, kv_lora_rank=32, qk_rope_head_dim=16, qk_nope_head_dim=16, v_head_dim=16), MoE (n_routed_experts=256, n_shared_experts=1, num_experts_per_tok=8), rope_interleave=True, plain 1D position_ids.
- Vision: SigLIP2 naflex packed encoder (siglip2_vision_model), patch_size=16, spatial_merge_size=2, window attention with fullatt_block_indexes=[1,3], window_size=256 (=patch_size*2*8=256), vision_use_head=False. Merger = VLPatchMerger(RMSNorm + Linear->GELU->Linear), 2x2 merge (hidden = context_dim*4).
- preprocessing (image_processing_siglip2_fast.py):
  - resize: `get_image_size_for_patches` scales so num_patches <= max_num_patches, dims rounded to multiples of patch_size*2 (=32). BILINEAR.
  - rescale 1/255, normalize mean=[0.5,0.5,0.5] std=[0.5,0.5,0.5].
  - `convert_image_to_patches`: reshape to (C, nh/2, 2, 16, nw/2, 2, 16) permute(1,4,2,5,3,6,0) reshape(nh*nw, 768). 768 = 3*16*16 per patch. Produces `pixel_values [num_patches, 768]`, `pixel_attention_mask [num_patches]`, `spatial_shapes [h,w]` (in patches).
  - processor replaces each `<|image_pad|>` (id 128264) with h*w//4 copies (merge_length=4).
- special tokens: image_token_id=128264 (`<|image_pad|>`), vision_start_token_id=128262, vision_end_token_id=128263, video_token_id=128265. bos=128000, eos=128001, pad=128001.
- position ids / RoPE: LLM uses plain 1D position ids (arange). Vision uses 2D rotary_pos_emb (h/w pos) like Qwen2.5-VL, dim 4 in tiny model (= head_dim//2 per axis *2).

## Optimum-Intel
- module path: workspace/optimum-intel/optimum/intel/openvino/modeling_visual_language.py
- class: `_OVYoutuVLForCausalLM` (registered as `"youtu_vl"`), additional_parts=["vision_embeddings_merger"].
- IR <-> logical mapping:
  - text_embeddings -> openvino_text_embeddings_model
  - vision_embeddings -> openvino_vision_embeddings_model (in: pixel_values [N,768])
  - vision_embeddings_merger -> openvino_vision_embeddings_merger_model (in: hidden_states, attention_mask, window_attention_mask, window_index, rotary_pos_emb)
  - language_model -> openvino_language_model
- vision flow (`get_vision_embeddings`): run vision_embeddings(pixel_values) -> hidden_states; compute rot_pos_emb(spatial_shapes) and window_index/cu_window_seqlens (Qwen2.5-VL algorithm with spatial_merge_size=2, vit_merger_window_size = window_size//merge//patch); build causal attention_mask (per full-image block) and window_attention_mask (per window block); call merger.
- merge (`get_multimodal_embeddings`): text embeds via masked_scatter at image_token_id positions. position_ids plain 1D (arange or cache_position).
- per-model overrides: `_spatial_merge_size=2`, `_patch_size=vision_config.patch_size`, `_window_size = patch_size*2*8`, VisionRotaryEmbedding(head_dim//2).

## Notes
- Vision pipeline is Qwen2.5-VL-like for the merger (window_index, rotary_pos_emb, dual masks) BUT the front-end is SigLIP2 naflex packing: pre-patchified `pixel_values [N,768]` + `spatial_shapes` (NOT Qwen's temporal_patch_size=2 raw-pixel flattening). No temporal patch, images only in scope here.
- LLM is DeepSeek-style (MLA+MoE) but exported LM is a standard inputs_embeds->logits IR with plain 1D position_ids; GenAI treats it as an opaque stateful LM, so MLA/MoE need no special GenAI handling.
- Key GenAI-relevant divergence from Qwen2.5-VL: (1) SigLIP2 patch preprocessing producing [N,768] pixel_values; (2) plain 1D position_ids on the LM (no 3D mRoPE); (3) merger input tensor is `hidden_states` (already matches Qwen2.5-VL name) fed from a separate vision_embeddings model output.

## GenAI Enablement Design
- Closest GenAI model: Qwen2.5-VL — because merger uses identical window_index / rotary_pos_emb / dual-mask scheme and 2x2 spatial merge. Reuse `qwen2_5_vl_utils::get_window_index`, `get_window_attention_mask`, and Qwen2VL `get_rotary_pos_emb`.
- Required changes:
  - vlm_config.hpp/.cpp: add `VLMModelType::YOUTU_VL` + `{"youtu_vl", VLMModelType::YOUTU_VL}` mapping.
  - src/cpp/src/visual_language/youtu_vl/classes.{hpp,cpp}: new `VisionEncoderYoutuVL` (SigLIP2 patch preprocessing -> pixel_values [N,768], spatial_shapes) and `InputsEmbedderYoutuVL` (runs vision_embeddings + merger, merges at image_token_id, plain 1D position_ids).
  - vision_encoder.cpp + inputs_embedder.cpp factories: register YOUTU_VL.
  - inputs_embedder.hpp: add `friend class InputsEmbedderYoutuVL`.
  - vision_registry.cpp if needed for vision model file discovery (vision_embeddings + vision_embeddings_merger).
- Gaps vs existing infra:
  - SigLIP2 naflex patch preprocessing (resize to multiple of 32, patchify to [N,768], spatial_shapes) — NEW; not covered by Qwen2VL raw-pixel path. Must be implemented in VisionEncoderYoutuVL.
  - Plain 1D position_ids for the LM (simpler than Qwen mRoPE) — override get_position_ids to return arange.
  - EncodedImage must carry grid_thw = {1, h, w} (from spatial_shapes) so merger utils (window_index/rotary_pos_emb) work unchanged.
</content>
