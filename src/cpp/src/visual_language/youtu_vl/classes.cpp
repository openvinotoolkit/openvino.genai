// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/youtu_vl/classes.hpp"

#include <cmath>

#include "visual_language/clip.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

// SigLIP2 fast image processor resize rounding: each side is rounded up to a
// multiple of patch_size * 2 and the whole image is scaled down (in 0.02 steps)
// until (H/patch_size) * (W/patch_size) <= max_num_patches.
// Mirrors image_processing_siglip2_fast.get_image_size_for_patches.
size_t get_scaled_image_size(double scale, size_t size, size_t patch_size) {
    size_t rounding = patch_size * 2;
    double scaled = static_cast<double>(size) * scale;
    size_t rounded = static_cast<size_t>(std::ceil(scaled / static_cast<double>(rounding))) * rounding;
    return std::max(rounding, rounded);
}

std::pair<size_t, size_t> get_target_image_size(size_t height, size_t width, size_t patch_size, size_t max_num_patches) {
    double scale = 1.0;
    size_t target_height = 0;
    size_t target_width = 0;
    while (true) {
        target_height = get_scaled_image_size(scale, height, patch_size);
        target_width = get_scaled_image_size(scale, width, patch_size);
        double num_patches = (static_cast<double>(target_height) / patch_size) *
                             (static_cast<double>(target_width) / patch_size);
        if (num_patches > static_cast<double>(max_num_patches)) {
            scale -= 0.02;
        } else {
            break;
        }
    }
    return {target_height, target_width};
}

// One-dimensional anti-aliased bilinear (triangle-filter) resampling weights,
// matching torchvision.transforms.functional.resize(..., interpolation=BILINEAR,
// antialias=True), which the SigLIP2 fast image processor uses. On downscaling
// the triangle kernel is stretched by the scale factor so that source pixels are
// averaged (anti-aliasing); on upscaling it degrades to plain bilinear. Each
// output pixel gets a contiguous window [start, start + weights.size()).
struct ResampleDim {
    std::vector<size_t> starts;               // per output index: first source index
    std::vector<std::vector<double>> weights; // per output index: normalized weights
};

ResampleDim compute_resample_weights(size_t in_size, size_t out_size) {
    const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
    const double filterscale = scale > 1.0 ? scale : 1.0;
    const double support = filterscale;  // bilinear support radius (== 1) times filterscale

    ResampleDim dim;
    dim.starts.resize(out_size);
    dim.weights.resize(out_size);

    for (size_t ox = 0; ox < out_size; ++ox) {
        const double center = (static_cast<double>(ox) + 0.5) * scale;
        long xmin = static_cast<long>(std::floor(center - support));
        long xmax = static_cast<long>(std::ceil(center + support));
        if (xmin < 0) {
            xmin = 0;
        }
        if (xmax > static_cast<long>(in_size)) {
            xmax = static_cast<long>(in_size);
        }

        std::vector<double> ws;
        ws.reserve(static_cast<size_t>(xmax - xmin));
        double sum = 0.0;
        for (long x = xmin; x < xmax; ++x) {
            const double arg = std::abs((static_cast<double>(x) + 0.5 - center) / filterscale);
            const double w = arg < 1.0 ? (1.0 - arg) : 0.0;
            ws.push_back(w);
            sum += w;
        }
        if (sum != 0.0) {
            for (double& w : ws) {
                w /= sum;
            }
        }
        dim.starts[ox] = static_cast<size_t>(xmin);
        dim.weights[ox] = std::move(ws);
    }
    return dim;
}

// Anti-aliased bilinear resize of an HWC uint8 image, producing a normalized CHW
// float buffer: value = ((pixel / 255) - mean) / std. Uses a separable two-pass
// resample (horizontal then vertical) to mirror torchvision's implementation.
std::vector<float> antialias_resize_normalize_chw(const clip_image_u8& src,
                                                  size_t target_h,
                                                  size_t target_w,
                                                  const float* image_mean,
                                                  const float* image_std) {
    const size_t src_h = static_cast<size_t>(src.ny);
    const size_t src_w = static_cast<size_t>(src.nx);
    const size_t channels = 3;

    const ResampleDim wx = compute_resample_weights(src_w, target_w);
    const ResampleDim wy = compute_resample_weights(src_h, target_h);

    // Horizontal pass: [src_h, target_w, C] in double precision.
    std::vector<double> tmp(src_h * target_w * channels, 0.0);
    const uint8_t* src_buf = src.buf.data();
    for (size_t y = 0; y < src_h; ++y) {
        for (size_t ox = 0; ox < target_w; ++ox) {
            const size_t xstart = wx.starts[ox];
            const std::vector<double>& ws = wx.weights[ox];
            double acc[3] = {0.0, 0.0, 0.0};
            for (size_t k = 0; k < ws.size(); ++k) {
                const size_t sx = xstart + k;
                const uint8_t* px = src_buf + (y * src_w + sx) * channels;
                acc[0] += ws[k] * px[0];
                acc[1] += ws[k] * px[1];
                acc[2] += ws[k] * px[2];
            }
            double* dst = tmp.data() + (y * target_w + ox) * channels;
            dst[0] = acc[0];
            dst[1] = acc[1];
            dst[2] = acc[2];
        }
    }

    // Vertical pass + normalize, writing directly to CHW float layout.
    std::vector<float> chw(channels * target_h * target_w);
    for (size_t oy = 0; oy < target_h; ++oy) {
        const size_t ystart = wy.starts[oy];
        const std::vector<double>& ws = wy.weights[oy];
        for (size_t ox = 0; ox < target_w; ++ox) {
            double acc[3] = {0.0, 0.0, 0.0};
            for (size_t k = 0; k < ws.size(); ++k) {
                const size_t sy = ystart + k;
                const double* px = tmp.data() + (sy * target_w + ox) * channels;
                acc[0] += ws[k] * px[0];
                acc[1] += ws[k] * px[1];
                acc[2] += ws[k] * px[2];
            }
            for (size_t c = 0; c < channels; ++c) {
                const double normalized = (acc[c] / 255.0 - image_mean[c]) / image_std[c];
                chw[(c * target_h + oy) * target_w + ox] = static_cast<float>(normalized);
            }
        }
    }
    return chw;
}

// Convert a normalized CHW float image into flattened patches with the SigLIP2
// merge-block ordering, producing a [num_patches_h * num_patches_w, C * patch_size^2] tensor.
// Mirrors image_processing_siglip2_fast.convert_image_to_patches with merge_size = 2.
ov::Tensor convert_image_to_patches(const std::vector<float>& chw_image,
                                    size_t channels,
                                    size_t height,
                                    size_t width,
                                    size_t patch_size,
                                    size_t merge_size) {
    const size_t nph = height / patch_size;
    const size_t npw = width / patch_size;
    const size_t feat = channels * patch_size * patch_size;
    const size_t num_patches = nph * npw;

    ov::Tensor patches(ov::element::f32, ov::Shape{num_patches, feat});
    float* dst = patches.data<float>();
    const float* src = chw_image.data();  // CHW layout: [c][y][x]

    // Iterate patches in merge-block order: outer 2x2 block grid, inner 2x2 within block.
    // Original numpy: reshape [C, nph/m, m, ps, npw/m, m, ps] -> permute(1,4,2,5,3,6,0)
    // -> [nph/m, npw/m, m_h, m_w, ps_h, ps_w, C] -> reshape [num_patches, C*ps*ps].
    size_t patch_idx = 0;
    for (size_t bh = 0; bh < nph / merge_size; ++bh) {
        for (size_t bw = 0; bw < npw / merge_size; ++bw) {
            for (size_t mh = 0; mh < merge_size; ++mh) {
                for (size_t mw = 0; mw < merge_size; ++mw) {
                    const size_t patch_row = bh * merge_size + mh;
                    const size_t patch_col = bw * merge_size + mw;
                    float* row = dst + patch_idx * feat;
                    // Within-patch layout is [ps_h][ps_w][C] (row-major HWC).
                    size_t f = 0;
                    for (size_t py = 0; py < patch_size; ++py) {
                        for (size_t px = 0; px < patch_size; ++px) {
                            const size_t y = patch_row * patch_size + py;
                            const size_t x = patch_col * patch_size + px;
                            for (size_t c = 0; c < channels; ++c) {
                                row[f++] = src[(c * height + y) * width + x];
                            }
                        }
                    }
                    ++patch_idx;
                }
            }
        }
    }
    return patches;
}

// Per-patch 2-D (h, w) rotary position embedding, honoring the spatial-merge block
// ordering, computed with theta = 10000. Output shape [num_patches, head_dim].
// Mirrors _OVYoutuVLForCausalLM.rot_pos_emb.
ov::Tensor compute_rotary_pos_emb(size_t grid_h, size_t grid_w, size_t merge_size, size_t rope_dim) {
    // rope_dim is the vision head_dim // 2, and the inv_freq has rope_dim // 2 entries.
    const size_t half = rope_dim / 2;
    std::vector<float> inv_freq(half);
    for (size_t i = 0; i < half; ++i) {
        inv_freq[i] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / static_cast<float>(rope_dim));
    }

    // hpos_ids / wpos_ids in merge-block order (see rot_pos_emb).
    const size_t num_patches = grid_h * grid_w;
    std::vector<size_t> hpos(num_patches);
    std::vector<size_t> wpos(num_patches);
    size_t idx = 0;
    for (size_t bh = 0; bh < grid_h / merge_size; ++bh) {
        for (size_t bw = 0; bw < grid_w / merge_size; ++bw) {
            for (size_t mh = 0; mh < merge_size; ++mh) {
                for (size_t mw = 0; mw < merge_size; ++mw) {
                    hpos[idx] = bh * merge_size + mh;
                    wpos[idx] = bw * merge_size + mw;
                    ++idx;
                }
            }
        }
    }

    ov::Tensor rotary(ov::element::f32, ov::Shape{num_patches, rope_dim});
    float* out = rotary.data<float>();
    for (size_t p = 0; p < num_patches; ++p) {
        float* row = out + p * rope_dim;
        // First half from height position, second half from width position (flatten(1) of stack).
        for (size_t i = 0; i < half; ++i) {
            row[i] = static_cast<float>(hpos[p]) * inv_freq[i];
        }
        for (size_t i = 0; i < half; ++i) {
            row[half + i] = static_cast<float>(wpos[p]) * inv_freq[i];
        }
    }
    return rotary;
}

// Window index and cumulative window sequence lengths for the SigLIP2 windowed
// attention. Mirrors _OVYoutuVLForCausalLM.get_window_index.
std::pair<std::vector<int64_t>, std::vector<int32_t>> compute_window_index(size_t grid_h,
                                                                           size_t grid_w,
                                                                           size_t merge_size,
                                                                           size_t patch_size,
                                                                           size_t window_size) {
    const size_t spatial_merge_unit = merge_size * merge_size;
    const size_t vit_merger_window_size = window_size / merge_size / patch_size;

    const size_t llm_grid_h = grid_h / merge_size;
    const size_t llm_grid_w = grid_w / merge_size;

    const size_t pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size) % vit_merger_window_size;
    const size_t pad_w = (vit_merger_window_size - llm_grid_w % vit_merger_window_size) % vit_merger_window_size;
    const size_t num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
    const size_t num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;

    std::vector<int64_t> window_index;
    window_index.reserve(llm_grid_h * llm_grid_w);
    std::vector<int32_t> cu_window_seqlens;
    cu_window_seqlens.push_back(0);

    for (size_t wh = 0; wh < num_windows_h; ++wh) {
        for (size_t ww = 0; ww < num_windows_w; ++ww) {
            size_t seqlen = 0;
            for (size_t ih = 0; ih < vit_merger_window_size; ++ih) {
                for (size_t iw = 0; iw < vit_merger_window_size; ++iw) {
                    const size_t gh = wh * vit_merger_window_size + ih;
                    const size_t gw = ww * vit_merger_window_size + iw;
                    if (gh < llm_grid_h && gw < llm_grid_w) {
                        window_index.push_back(static_cast<int64_t>(gh * llm_grid_w + gw));
                        ++seqlen;
                    }
                }
            }
            const int32_t prev = cu_window_seqlens.back();
            cu_window_seqlens.push_back(prev + static_cast<int32_t>(seqlen * spatial_merge_unit));
        }
    }

    // torch.unique_consecutive on cu_window_seqlens.
    std::vector<int32_t> unique_cws;
    for (int32_t v : cu_window_seqlens) {
        if (unique_cws.empty() || unique_cws.back() != v) {
            unique_cws.push_back(v);
        }
    }
    return {std::move(window_index), std::move(unique_cws)};
}

// Block-diagonal float attention mask: 0 inside a block, -inf outside.
ov::Tensor make_block_diag_mask(size_t seq_len, const std::vector<int32_t>& cu_seqlens) {
    ov::Tensor mask(ov::element::f32, ov::Shape{1, seq_len, seq_len});
    float* data = mask.data<float>();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::fill(data, data + seq_len * seq_len, neg_inf);
    for (size_t b = 1; b < cu_seqlens.size(); ++b) {
        const size_t start = static_cast<size_t>(cu_seqlens[b - 1]);
        const size_t end = static_cast<size_t>(cu_seqlens[b]);
        for (size_t i = start; i < end; ++i) {
            for (size_t j = start; j < end; ++j) {
                data[i * seq_len + j] = 0.0f;
            }
        }
    }
    return mask;
}

} // namespace

EncodedImage VisionEncoderYoutuVL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    const size_t patch_size = config.patch_size;
    const size_t merge_size = 2;                 // spatial_merge_size for Youtu-VL / SigLIP2
    // YoutuVLProcessor.__call__ forwards max_image_patches=36864 to the SigLIP2
    // fast image processor (its own class attribute default of 256 is overridden
    // by the processor), so images are kept at near-native resolution and only
    // downscaled when they exceed 36864 patches. Using the un-overridden 256 here
    // aggressively downscales large images, yielding far fewer vision tokens and
    // materially worse answers, so the processor-level value must be used.
    const size_t max_num_patches = 36864;        // YoutuVLProcessor default max_image_patches
    const size_t window_size = 256;              // patch_size * 2 * 8

    // 1. Convert input tensor (NHWC uint8) to clip_image_u8 (HWC uint8).
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    const size_t orig_h = static_cast<size_t>(input_image.ny);
    const size_t orig_w = static_cast<size_t>(input_image.nx);

    // 2. Resize to a patch-aligned target size, then rescale (1/255) + normalize.
    // The SigLIP2 fast image processor resizes with anti-aliased bilinear
    // interpolation (torchvision resize, antialias=True). Using a plain
    // (non-anti-aliased) bilinear resize here produces visibly different pixel
    // values on large downscales and shifts the vision features enough to change
    // fine-grained answers, so the anti-aliased path is required to match
    // optimum-intel. The helper returns the normalized image directly in CHW.
    auto [target_h, target_w] = get_target_image_size(orig_h, orig_w, patch_size, max_num_patches);
    std::vector<float> normalized_chw = antialias_resize_normalize_chw(
        input_image, target_h, target_w, config.image_mean.data(), config.image_std.data());

    const size_t grid_h = target_h / patch_size;
    const size_t grid_w = target_w / patch_size;

    // 3. Patchify into [num_patches, 3 * patch_size^2] in merge-block order.
    ov::Tensor pixel_values = convert_image_to_patches(normalized_chw, 3, target_h, target_w, patch_size, merge_size);
    const size_t seq_len = grid_h * grid_w;

    // 4. Recompute the auxiliary tensors expected by the fused vision model.
    // rope_dim is inferred from the model's rotary_pos_emb input (last dim).
    size_t rope_dim = 0;
    for (const auto& port : encoder.get_compiled_model().inputs()) {
        if (port.get_any_name() == "rotary_pos_emb") {
            const auto& pshape = port.get_partial_shape();
            OPENVINO_ASSERT(pshape.rank().is_static() && pshape.size() == 2 && pshape[1].is_static(),
                            "Youtu-VL vision model rotary_pos_emb must have a static last dimension.");
            rope_dim = static_cast<size_t>(pshape[1].get_length());
        }
    }
    OPENVINO_ASSERT(rope_dim > 0, "Youtu-VL vision model is missing the 'rotary_pos_emb' input.");
    ov::Tensor rotary_pos_emb = compute_rotary_pos_emb(grid_h, grid_w, merge_size, rope_dim);

    auto [window_index_vec, cu_window_seqlens] = compute_window_index(grid_h, grid_w, merge_size, patch_size, window_size);

    // cu_seqlens: cumulative patch counts per image (single image here).
    std::vector<int32_t> cu_seqlens = {0, static_cast<int32_t>(seq_len)};

    ov::Tensor attention_mask = make_block_diag_mask(seq_len, cu_seqlens);
    ov::Tensor window_attention_mask = make_block_diag_mask(seq_len, cu_window_seqlens);

    ov::Tensor window_index(ov::element::i64, ov::Shape{window_index_vec.size()});
    std::copy(window_index_vec.begin(), window_index_vec.end(), window_index.data<int64_t>());

    // 5. Run the fused vision tower + merger.
    encoder.set_tensor("pixel_values", pixel_values);
    encoder.set_tensor("attention_mask", attention_mask);
    encoder.set_tensor("window_attention_mask", window_attention_mask);
    encoder.set_tensor("window_index", window_index);
    encoder.set_tensor("rotary_pos_emb", rotary_pos_emb);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    EncodedImage encoded_image;
    encoded_image.resized_source = std::move(image_features);
    encoded_image.resized_source_size = ImageSize{grid_h, grid_w};
    encoded_image.original_image_size = ImageSize{orig_h, orig_w};
    return encoded_image;
}

InputsEmbedderYoutuVL::InputsEmbedderYoutuVL(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {
    encode_vision_placeholder_tokens();
}

InputsEmbedderYoutuVL::InputsEmbedderYoutuVL(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    encode_vision_placeholder_tokens();
}

void InputsEmbedderYoutuVL::encode_vision_placeholder_tokens() {
    auto encoded_vision_tokens = m_tokenizer.encode(m_vlm_config.vision_start_token +
                                                    m_vlm_config.vision_end_token +
                                                    m_vlm_config.image_pad_token,
                                                    ov::genai::add_special_tokens(false));
    const int64_t* ids = encoded_vision_tokens.input_ids.data<int64_t>();
    OPENVINO_ASSERT(encoded_vision_tokens.input_ids.get_size() >= 3,
                    "Failed to encode Youtu-VL vision placeholder tokens.");
    m_vision_token_ids["vision_start"] = ids[0];
    m_vision_token_ids["vision_end"] = ids[1];
    m_vision_token_ids["image_pad"] = ids[2];
}

size_t InputsEmbedderYoutuVL::calc_tokens_num(size_t grid_h, size_t grid_w) const {
    return grid_h * grid_w / m_merge_length;
}

NormalizedPrompt InputsEmbedderYoutuVL::normalize_prompt(const std::string& prompt,
                                                         size_t base_id,
                                                         const std::vector<EncodedImage>& images) const {
    auto [unified_prompt, images_sequence] =
        normalize(prompt, NATIVE_TAG, NATIVE_TAG, base_id, images.size(), VisionType::IMAGE);

    for (size_t new_image_id : images_sequence) {
        const auto& encoded_image = images.at(new_image_id - base_id);
        const size_t grid_h = encoded_image.resized_source_size.height;
        const size_t grid_w = encoded_image.resized_source_size.width;
        const size_t num_image_pad_tokens = calc_tokens_num(grid_h, grid_w);

        std::string expanded_tag;
        expanded_tag.reserve(m_vlm_config.vision_start_token.length() +
                             m_vlm_config.image_pad_token.length() * num_image_pad_tokens +
                             m_vlm_config.vision_end_token.length());
        expanded_tag.append(m_vlm_config.vision_start_token);
        for (size_t i = 0; i < num_image_pad_tokens; ++i) {
            expanded_tag.append(m_vlm_config.image_pad_token);
        }
        expanded_tag.append(m_vlm_config.vision_end_token);

        const auto pos = unified_prompt.find(NATIVE_TAG);
        OPENVINO_ASSERT(pos != std::string::npos, "Failed to locate Youtu-VL image tag in prompt.");
        unified_prompt.replace(pos, NATIVE_TAG.length(), expanded_tag);
    }

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderYoutuVL::get_inputs_embeds(const std::string& unified_prompt,
                                                    const std::vector<ov::genai::EncodedImage>& images,
                                                    ov::genai::VLMPerfMetrics& metrics,
                                                    bool recalculate_merged_embeddings,
                                                    const std::vector<size_t>& images_sequence) {
    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    const int64_t image_pad_token_id = m_vision_token_ids.at("image_pad");

    // Merge image embeddings into text embeddings at <|image_pad|> token positions,
    // consuming per-image embedding rows in prompt order.
    const ov::Shape text_shape = text_embeds.get_shape();  // [1, seq_len, hidden]
    const size_t seq_len = text_shape[1];
    const size_t hidden_size = text_shape[2];

    ov::Tensor merged_embeds(text_embeds.get_element_type(), text_shape);
    std::memcpy(merged_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());

    const int64_t* input_ids_data = input_ids.data<int64_t>();
    float* merged_data = merged_embeds.data<float>();

    // Order the image embeddings by their appearance in the prompt.
    std::vector<const float*> image_embed_ptrs;
    std::vector<size_t> image_embed_rows;
    size_t total_image_rows = 0;
    for (size_t new_image_id : images_sequence) {
        const ov::Tensor& src = images.at(new_image_id).resized_source;
        const size_t rows = src.get_shape().at(0);
        image_embed_ptrs.push_back(src.data<float>());
        image_embed_rows.push_back(rows);
        total_image_rows += rows;
    }

    size_t cur_image = 0;
    size_t cur_row = 0;
    size_t consumed_rows = 0;
    for (size_t s = 0; s < seq_len; ++s) {
        if (input_ids_data[s] == image_pad_token_id) {
            OPENVINO_ASSERT(cur_image < image_embed_ptrs.size(),
                            "Youtu-VL: more <|image_pad|> tokens than available image embeddings.");
            const float* src_row = image_embed_ptrs[cur_image] + cur_row * hidden_size;
            std::copy_n(src_row, hidden_size, merged_data + s * hidden_size);
            ++cur_row;
            ++consumed_rows;
            if (cur_row == image_embed_rows[cur_image]) {
                ++cur_image;
                cur_row = 0;
            }
        }
    }
    OPENVINO_ASSERT(consumed_rows == total_image_rows,
                    "Youtu-VL: number of <|image_pad|> tokens (", consumed_rows,
                    ") does not match total image embedding rows (", total_image_rows, ").");

    return merged_embeds;
}

} // namespace ov::genai
