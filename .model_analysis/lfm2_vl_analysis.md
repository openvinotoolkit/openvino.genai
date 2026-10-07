# Model Analysis: LiquidAI/LFM2.5-VL-3B (lfm2_vl)

## Identity
- model_id: LiquidAI/LFM2.5-VL-3B
- model_type: lfm2_vl
- architecture: Lfm2VlForConditionalGeneration
- task / modality: image-text-to-text (vision-text)
- transformers: 5.10.4   optimum-intel: 2.3.0.dev0 (editable)

## Exported IR
| File | Role | Inputs (name: shape, dtype) | Outputs |
|------|------|-----------------------------|---------|
| openvino_language_model.xml | LFM2 hybrid (conv + attn) decoder, stateful | attention_mask [?,?] i64; inputs_embeds [?,?,2048] f32; beam_idx [?] i32 | logits [?,?,128000] f32 |
| openvino_text_embeddings_model.xml | token embedding | input [?,?] i64 | inputs_embeds [?,?,2048] f32 |
| openvino_vision_embeddings_model.xml | siglip2-naflex tower + multimodal projector | pixel_values [1,?,768] f32; pos_resample_kernel [?,256] f32; spatial_shapes [1,2] i64 | last_hidden_state [?,2048] f32 |

Note: LM has **no position_ids** input — LFM2 is a hybrid short-conv/attention
model whose conv cache and RoPE are internal to the stateful IR. The generic
GenAI embeds-in / logits-out path handles it (like other stateful LMs).

## Transformers
- module path: transformers/models/lfm2_vl/
- Vision: siglip2_vision_model (naflex, patch16, hidden 1152, 27 layers,
  vision_use_head=false), packed variable-resolution patches.
- Projector: pixel-unshuffle by `downsample_factor=2` (grouped by shape),
  Linear(1152*4 -> 2048) with gelu; **included inside the vision IR** (output
  hidden = 2048 = text hidden).
- Image processing (`image_processing_lfm2_vl.py`):
  - `smart_resize`: both dims divisible by patch(16)*downsample(2)=32; pixel
    budget in [min_image_tokens*16^2*2^2, max_image_tokens*16^2*2^2] =
    [64..256 image tokens].
  - Large images (`_is_image_too_large`, tolerance 2.0) with
    `do_image_splitting`: `crop_image_to_patches` picks a grid (min_tiles=2,
    max_tiles=10) via `find_closest_aspect_ratio`, resizes whole image to
    (tile_size*grid_h, tile_size*grid_w), splits into `tile_size`=512 tiles
    (each 32x32 patches), plus a `use_thumbnail` thumbnail resized by
    smart_resize.
  - Each tile/image -> patches [N,768] (768 = 16*16*3), normalized with
    IMAGENET_STANDARD_MEAN/STD, rescale 1/255.
  - Resize is torchvision `interpolation=BILINEAR, antialias=True`.
  - `spatial_shapes[i] = [num_patches_h, num_patches_w]`; tiles padded to
    max_num_patches=1024 with `pixel_attention_mask` (encoder trims to valid).
- Processor (`processing_lfm2_vl.py`) expands each `<image>` string into:
  `<|image_start|>` + (single tile) `<image>`*tokens_for_image
  or (multi tile) for each (row,col): `<|img_row_{r}_col_{c}|>` + `<image>`*tokens_per_tile,
  then `<|img_thumbnail|>` + `<image>`*tokens_for_image(thumbnail), + `<|image_end|>`.
  - tokens_per_tile = (ceil((tile_size/patch)/downsample))^2 = (ceil(32/2))^2 = 256.
  - tokens_for_image = ceil((H/patch)/downsample) * ceil((W/patch)/downsample).
- special ids: image_token_id=124907, image_start=125009, image_end=125010,
  img_thumbnail=125008, img_row_r_col_c = 124908.. (contiguous).

## Optimum-Intel
- module path: optimum-intel/optimum/intel/openvino/modeling_visual_language.py
  (`_OVLfm2VlForCausalLM`).
- IR mapping: text_embeddings -> get_text_embeddings; vision_embeddings runs
  per-image (trims padded patches to valid `H*W`), builds
  `pos_resample_kernel` on the fly (no model weight) via bilinear-antialias
  interpolation of an identity map of the learned position grid
  (num_patches=256 -> side 16); output rows scattered into image_token_id
  positions of the text embeds.
- The pos_resample_kernel K is `interpolate(eye(256).reshape(1,256,16,16),
  size=(H,W), mode="bilinear", align_corners=False, antialias=True)` reshaped
  to [256, H*W] then transposed -> [H*W, 256]. Independent of image content.

## Notes
- Vision IR consumes one image's flattened valid patches at a time
  (batch dim 1). GenAI must loop tiles+thumbnail, concatenate the per-tile
  `[tokens, 2048]` outputs in placeholder order, and scatter at image_token_id.
- The two numerically sensitive pieces are (a) torchvision antialias bilinear
  resize of the RGB image and (b) the antialias bilinear pos_resample_kernel
  from the identity grid. Both must match optimum/HF to reach WWB threshold.

## GenAI Enablement Design
- Closest GenAI model: none is a close match (naflex packed vision is unique).
  Structural skeleton borrowed from `llava` (single vision IR + projector +
  merge at image token id); prompt-expansion logic is model-specific.
- Required changes:
  - `vlm_config.hpp`: add `LFM2_VL` to `VLMModelType`; add fields
    (patch_size, downsample_factor, tile_size, min/max_image_tokens, tiling
    params, special token strings). `vlm_config.cpp`: string map + JSON.
  - `visual_language/lfm2_vl/classes.{hpp,cpp}`: `VisionEncoderLFM2VL`
    (naflex preprocessing: smart_resize, tiling, thumbnail, antialias bilinear
    resize, patch packing, per-tile pos_resample_kernel, run IR, concat) and
    `InputsEmbedderLFM2VL` (prompt expansion reproducing the processor,
    tokenize, merge at image_token_id).
  - Register in `vision_encoder.cpp` and `inputs_embedder.cpp` factories;
    `friend class` in `inputs_embedder.hpp`.
- Gaps needing new infra: antialias bilinear resize (add local helper),
  variable-resolution packed vision inputs (3-input vision IR),
  multi-tile per-image concatenation, added special-token id recovery in the
  OV tokenizer.
