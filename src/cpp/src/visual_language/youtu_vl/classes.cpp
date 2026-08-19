// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/youtu_vl/classes.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {

namespace {

// Antialiased separable bilinear resize matching torchvision's
// F.resize(..., interpolation=BILINEAR, antialias=True), which the Siglip2 fast image processor
// uses. Unlike the plain bilinear_resize in clip.cpp, this applies a triangle (bilinear) kernel
// whose support is widened by the downsampling ratio, so large downscales average the correct
// neighborhood of source pixels. Kept local to youtu_vl to avoid changing shared preprocessing.
//
// Weight computation follows the PIL/torchvision precompute_coeffs algorithm:
//   scale  = in_size / out_size
//   filterscale = max(scale, 1.0)
//   support = 1.0 * filterscale   (triangle filter support = 1.0)
//   for each output index xx, center = (xx + 0.5) * scale, window [center - support, center + support)
//   weight(x) = triangle((x - center + 0.5) / filterscale), triangle(t) = max(0, 1 - |t|)
//   weights normalized to sum to 1.
void compute_coeffs(size_t in_size,
                    size_t out_size,
                    std::vector<int>& bounds,      // 2 per out index: [start, length]
                    std::vector<double>& weights,   // ksize per out index
                    int& ksize) {
    const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
    const double filterscale = std::max(scale, 1.0);
    const double support = 1.0 * filterscale;  // triangle filter support == 1.0
    ksize = static_cast<int>(std::ceil(support)) * 2 + 1;

    bounds.assign(out_size * 2, 0);
    weights.assign(out_size * static_cast<size_t>(ksize), 0.0);

    for (size_t xx = 0; xx < out_size; ++xx) {
        const double center = (static_cast<double>(xx) + 0.5) * scale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) {
            xmin = 0;
        }
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > static_cast<int>(in_size)) {
            xmax = static_cast<int>(in_size);
        }
        const int length = xmax - xmin;
        double ww = 0.0;
        double* w = &weights[xx * static_cast<size_t>(ksize)];
        for (int x = 0; x < length; ++x) {
            double t = (static_cast<double>(x + xmin) - center + 0.5) / filterscale;
            t = std::abs(t);
            double weight = t < 1.0 ? (1.0 - t) : 0.0;  // triangle kernel
            w[x] = weight;
            ww += weight;
        }
        if (ww != 0.0) {
            for (int x = 0; x < length; ++x) {
                w[x] /= ww;
            }
        }
        bounds[xx * 2 + 0] = xmin;
        bounds[xx * 2 + 1] = length;
    }
}

clip_image_u8 antialias_bilinear_resize(const clip_image_u8& src, int target_width, int target_height) {
    clip_image_u8 dst;
    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(static_cast<size_t>(target_width) * target_height * 3);

    const int in_w = src.nx;
    const int in_h = src.ny;

    // Horizontal pass: (in_h, in_w) -> (in_h, target_width), keep float precision.
    std::vector<int> hbounds;
    std::vector<double> hweights;
    int hksize = 0;
    compute_coeffs(static_cast<size_t>(in_w), static_cast<size_t>(target_width), hbounds, hweights, hksize);

    std::vector<float> horiz(static_cast<size_t>(in_h) * target_width * 3);
    for (int y = 0; y < in_h; ++y) {
        for (int xx = 0; xx < target_width; ++xx) {
            const int xmin = hbounds[xx * 2 + 0];
            const int length = hbounds[xx * 2 + 1];
            const double* w = &hweights[static_cast<size_t>(xx) * hksize];
            double acc[3] = {0.0, 0.0, 0.0};
            for (int k = 0; k < length; ++k) {
                const size_t sidx = (static_cast<size_t>(y) * in_w + (xmin + k)) * 3;
                acc[0] += w[k] * src.buf[sidx + 0];
                acc[1] += w[k] * src.buf[sidx + 1];
                acc[2] += w[k] * src.buf[sidx + 2];
            }
            const size_t didx = (static_cast<size_t>(y) * target_width + xx) * 3;
            horiz[didx + 0] = static_cast<float>(acc[0]);
            horiz[didx + 1] = static_cast<float>(acc[1]);
            horiz[didx + 2] = static_cast<float>(acc[2]);
        }
    }

    // Vertical pass: (in_h, target_width) -> (target_height, target_width).
    std::vector<int> vbounds;
    std::vector<double> vweights;
    int vksize = 0;
    compute_coeffs(static_cast<size_t>(in_h), static_cast<size_t>(target_height), vbounds, vweights, vksize);

    auto clamp_u8 = [](double v) -> uint8_t {
        double r = std::round(v);
        if (r < 0.0) {
            r = 0.0;
        }
        if (r > 255.0) {
            r = 255.0;
        }
        return static_cast<uint8_t>(r);
    };

    for (int yy = 0; yy < target_height; ++yy) {
        const int ymin = vbounds[yy * 2 + 0];
        const int length = vbounds[yy * 2 + 1];
        const double* w = &vweights[static_cast<size_t>(yy) * vksize];
        for (int x = 0; x < target_width; ++x) {
            double acc[3] = {0.0, 0.0, 0.0};
            for (int k = 0; k < length; ++k) {
                const size_t sidx = (static_cast<size_t>(ymin + k) * target_width + x) * 3;
                acc[0] += w[k] * horiz[sidx + 0];
                acc[1] += w[k] * horiz[sidx + 1];
                acc[2] += w[k] * horiz[sidx + 2];
            }
            const size_t didx = (static_cast<size_t>(yy) * target_width + x) * 3;
            dst.buf[didx + 0] = clamp_u8(acc[0]);
            dst.buf[didx + 1] = clamp_u8(acc[1]);
            dst.buf[didx + 2] = clamp_u8(acc[2]);
        }
    }
    return dst;
}

// Mirrors image_processing_siglip2_fast.get_image_size_for_patches: scale the image so that
// (H / patch_size) * (W / patch_size) <= max_num_patches, with each side rounded up to a multiple
// of patch_size * 2 (and at least patch_size * 2).
std::pair<size_t, size_t> get_image_size_for_patches(size_t image_height,
                                                     size_t image_width,
                                                     size_t patch_size,
                                                     size_t max_num_patches) {
    auto scaled = [patch_size](double scale, size_t size) -> size_t {
        size_t step = patch_size * 2;
        double scaled_size = static_cast<double>(size) * scale;
        size_t rounded = static_cast<size_t>(std::ceil(scaled_size / static_cast<double>(step))) * step;
        return std::max(step, rounded);
    };

    double scale = 1.0;
    size_t target_height = 0;
    size_t target_width = 0;
    while (true) {
        target_height = scaled(scale, image_height);
        target_width = scaled(scale, image_width);
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

// Reproduces image_processing_siglip2_fast.convert_image_to_patches followed by optimum-intel's
// reshape(-1, channels * patch_size^2). The Siglip2 patch embedding consumes flattened patches of
// shape [grid_h * grid_w, channels * patch_size^2]. Input is a normalized CHW float image.
// convert_image_to_patches groups patches into merge blocks and flattens as
// permute(1, 4, 2, 5, 3, 6, 0) -> [num_ph * num_pw, channels * merge^2 * patch^2]; optimum then
// reshapes to (-1, channels * patch^2). The net element order is exactly the same as iterating
// each merge block, then each patch in the block (row-major inside the merge block), then, per
// patch, (py, px, channel). We build it directly in that order.
ov::Tensor build_pixel_values(const clip_image_f32& image_chw,
                              size_t channels,
                              size_t patch_size,
                              size_t merge_size,
                              size_t grid_h,
                              size_t grid_w) {
    const size_t H = static_cast<size_t>(image_chw.ny);
    const size_t W = static_cast<size_t>(image_chw.nx);
    OPENVINO_ASSERT(H == grid_h * patch_size && W == grid_w * patch_size,
                    "Youtu-VL: normalized image size does not match patch grid.");
    const size_t feature_dim = channels * patch_size * patch_size;      // 768
    const size_t num_rows = grid_h * grid_w;                            // one row per patch

    ov::Tensor pixel_values(ov::element::f32, ov::Shape{num_rows, feature_dim});
    float* dst = pixel_values.data<float>();
    // image_chw.buf is CHW: index = c * H * W + y * W + x
    const float* src = image_chw.buf.data();

    const size_t num_ph_blocks = grid_h / merge_size;
    const size_t num_pw_blocks = grid_w / merge_size;

    size_t row = 0;
    // Order: for each merge block (bh, bw), for each patch inside block (mh, mw) -> a patch row.
    for (size_t bh = 0; bh < num_ph_blocks; ++bh) {
        for (size_t bw = 0; bw < num_pw_blocks; ++bw) {
            for (size_t mh = 0; mh < merge_size; ++mh) {
                for (size_t mw = 0; mw < merge_size; ++mw) {
                    const size_t patch_row = bh * merge_size + mh;
                    const size_t patch_col = bw * merge_size + mw;
                    float* row_ptr = dst + row * feature_dim;
                    size_t k = 0;
                    // Per-patch layout: (patch_y, patch_x, channel), matching the reshape/permute.
                    for (size_t py = 0; py < patch_size; ++py) {
                        const size_t y = patch_row * patch_size + py;
                        for (size_t px = 0; px < patch_size; ++px) {
                            const size_t x = patch_col * patch_size + px;
                            for (size_t c = 0; c < channels; ++c) {
                                row_ptr[k++] = src[c * H * W + y * W + x];
                            }
                        }
                    }
                    ++row;
                }
            }
        }
    }
    return pixel_values;
}

// Mirrors _OVYoutuVLForCausalLM.rot_pos_emb: builds per-patch (h_pos, w_pos) ids respecting the
// spatial-merge block layout, then gathers rotary theta values. Output shape [seq_len, dim] where
// dim = 2 * (head_dim / 2 / 2). For Youtu-VL head_dim = 72 -> rot dim per axis = 18 -> total 36.
ov::Tensor rot_pos_emb(size_t grid_h,
                       size_t grid_w,
                       size_t merge_size,
                       size_t rope_dim_half /* = head_dim / 2 / 2, per-axis freq count */) {
    const size_t seq_len = grid_h * grid_w;
    // inv_freq for _YoutuVisionRope(dim = head_dim // 2). torch: arange(0, dim, 2) / dim, dim = 2 * rope_dim_half.
    const size_t dim = rope_dim_half * 2;
    std::vector<float> inv_freq(rope_dim_half);
    for (size_t i = 0; i < rope_dim_half; ++i) {
        inv_freq[i] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / static_cast<float>(dim));
    }

    // pos ids: for each merge block, list (h, w) with h/w derived from block layout.
    std::vector<int64_t> hpos(seq_len);
    std::vector<int64_t> wpos(seq_len);
    const size_t num_ph_blocks = grid_h / merge_size;
    const size_t num_pw_blocks = grid_w / merge_size;
    size_t idx = 0;
    for (size_t bh = 0; bh < num_ph_blocks; ++bh) {
        for (size_t bw = 0; bw < num_pw_blocks; ++bw) {
            for (size_t mh = 0; mh < merge_size; ++mh) {
                for (size_t mw = 0; mw < merge_size; ++mw) {
                    hpos[idx] = static_cast<int64_t>(bh * merge_size + mh);
                    wpos[idx] = static_cast<int64_t>(bw * merge_size + mw);
                    ++idx;
                }
            }
        }
    }

    // rotary_pos_emb_full[pos_ids].flatten(1): concat h-freqs then w-freqs -> [seq_len, 2 * rope_dim_half].
    const size_t out_dim = rope_dim_half * 2;
    ov::Tensor rotary(ov::element::f32, ov::Shape{seq_len, out_dim});
    float* r = rotary.data<float>();
    for (size_t s = 0; s < seq_len; ++s) {
        for (size_t j = 0; j < rope_dim_half; ++j) {
            r[s * out_dim + j] = static_cast<float>(hpos[s]) * inv_freq[j];
            r[s * out_dim + rope_dim_half + j] = static_cast<float>(wpos[s]) * inv_freq[j];
        }
    }
    return rotary;
}

// Mirrors _OVYoutuVLForCausalLM.get_window_index (single image; grid_t = 1).
std::pair<std::vector<int64_t>, std::vector<int32_t>> get_window_index(size_t grid_h,
                                                                       size_t grid_w,
                                                                       size_t merge_size,
                                                                       size_t patch_size,
                                                                       size_t window_size) {
    std::vector<int64_t> window_index;
    std::vector<int32_t> cu_window_seqlens{0};
    const size_t spatial_merge_unit = merge_size * merge_size;
    const size_t vit_merger_window_size = window_size / merge_size / patch_size;

    const size_t llm_grid_h = grid_h / merge_size;
    const size_t llm_grid_w = grid_w / merge_size;

    const size_t pad_h =
        (vit_merger_window_size - llm_grid_h % vit_merger_window_size) % vit_merger_window_size;
    const size_t pad_w =
        (vit_merger_window_size - llm_grid_w % vit_merger_window_size) % vit_merger_window_size;
    const size_t num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
    const size_t num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;

    // index grid [llm_grid_h, llm_grid_w] padded with -100, reshaped into windows.
    const size_t padded_h = llm_grid_h + pad_h;
    const size_t padded_w = llm_grid_w + pad_w;
    auto index_at = [&](size_t r, size_t c) -> int64_t {
        if (r < llm_grid_h && c < llm_grid_w) {
            return static_cast<int64_t>(r * llm_grid_w + c);
        }
        return -100;
    };

    int64_t window_index_id = 0;  // grid_t * llm_grid_h * llm_grid_w accumulated; single image => 0
    // Iterate windows in (num_windows_h, num_windows_w) order, and within a window in
    // (vit_merger_window_size, vit_merger_window_size) order, matching the permute(0,1,3,2,4).
    for (size_t wh = 0; wh < num_windows_h; ++wh) {
        for (size_t ww = 0; ww < num_windows_w; ++ww) {
            size_t seqlen = 0;
            for (size_t ih = 0; ih < vit_merger_window_size; ++ih) {
                for (size_t iw = 0; iw < vit_merger_window_size; ++iw) {
                    const size_t rr = wh * vit_merger_window_size + ih;
                    const size_t cc = ww * vit_merger_window_size + iw;
                    int64_t v = index_at(rr, cc);
                    if (v != -100) {
                        window_index.push_back(v + window_index_id);
                        ++seqlen;
                    }
                }
            }
            const int32_t prev = cu_window_seqlens.back();
            cu_window_seqlens.push_back(prev + static_cast<int32_t>(seqlen * spatial_merge_unit));
        }
    }
    (void)padded_h;
    (void)padded_w;

    // torch.unique_consecutive on cu_window_seqlens.
    std::vector<int32_t> unique_cu;
    for (int32_t v : cu_window_seqlens) {
        if (unique_cu.empty() || unique_cu.back() != v) {
            unique_cu.push_back(v);
        }
    }
    return {window_index, unique_cu};
}

// Builds a [1, seq, seq] float mask: 0 inside each cumulative-length block, -inf elsewhere.
ov::Tensor build_block_mask(const std::vector<int32_t>& cu_seqlens, size_t seq_len) {
    ov::Tensor mask(ov::element::f32, ov::Shape{1, seq_len, seq_len});
    float* m = mask.data<float>();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::fill(m, m + seq_len * seq_len, neg_inf);
    for (size_t b = 1; b < cu_seqlens.size(); ++b) {
        const size_t start = static_cast<size_t>(cu_seqlens[b - 1]);
        const size_t end = static_cast<size_t>(cu_seqlens[b]);
        for (size_t i = start; i < end; ++i) {
            for (size_t j = start; j < end; ++j) {
                m[i * seq_len + j] = 0.0f;
            }
        }
    }
    return mask;
}

}  // namespace

EncodedImage VisionEncoderYoutuVL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    const size_t patch_size = config.patch_size;
    size_t merge_size = config.merge_size;
    size_t window_size = 256;
    // The remote-code YoutuVLProcessor.__call__ overrides preprocessor_config.json's
    // max_num_patches (256) with a much larger default (36864). optimum-intel uses that same
    // higher-resolution preprocessing, so GenAI must too or vision quality degrades. The value is
    // passed in via the InputsEmbedder (from VLMConfig); fall back to the processor default here.
    size_t max_num_patches = 36864;
    utils::read_anymap_param(config_map, "merge_size", merge_size);
    utils::read_anymap_param(config_map, "window_size", window_size);
    utils::read_anymap_param(config_map, "max_num_patches", max_num_patches);
    const size_t channels = 3;

    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    auto [target_h, target_w] = get_image_size_for_patches(static_cast<size_t>(input_image.ny),
                                                           static_cast<size_t>(input_image.nx),
                                                           patch_size,
                                                           max_num_patches);

    // Siglip2 fast image processor uses torchvision BILINEAR with antialias=True.
    clip_image_u8 resized = antialias_bilinear_resize(input_image, static_cast<int>(target_w), static_cast<int>(target_h));

    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
    // clip_image_preprocess rescales by 1/255, subtracts mean, divides by std, and lays out CHW.
    clip_image_f32 normalized = clip_image_preprocess(ctx, resized);

    const size_t grid_h = target_h / patch_size;
    const size_t grid_w = target_w / patch_size;
    const size_t seq_len = grid_h * grid_w;

    ov::Tensor pixel_values = build_pixel_values(normalized, channels, patch_size, merge_size, grid_h, grid_w);

    // Host-side vision bookkeeping (matches optimum-intel _OVYoutuVLForCausalLM.get_vision_embeddings).
    // rope per-axis freq count derived from the vision IR rotary_pos_emb input width (dim / 2).
    ov::PartialShape rotary_shape = encoder.get_compiled_model().input("rotary_pos_emb").get_partial_shape();
    OPENVINO_ASSERT(rotary_shape.rank().get_length() == 2 && rotary_shape[1].is_static(),
                    "Youtu-VL: unexpected rotary_pos_emb input shape.");
    const size_t rope_total_dim = static_cast<size_t>(rotary_shape[1].get_length());
    OPENVINO_ASSERT(rope_total_dim > 0 && rope_total_dim % 2 == 0,
                    "Youtu-VL: could not determine rotary_pos_emb dimension.");
    const size_t rope_dim_half = rope_total_dim / 2;  // per-axis freq count

    ov::Tensor rotary_pos_emb = rot_pos_emb(grid_h, grid_w, merge_size, rope_dim_half);

    auto [window_index_vec, cu_window_seqlens] =
        get_window_index(grid_h, grid_w, merge_size, patch_size, window_size);

    // full-attention cu_seqlens for a single image: [0, seq_len].
    std::vector<int32_t> cu_seqlens{0, static_cast<int32_t>(seq_len)};

    ov::Tensor attention_mask = build_block_mask(cu_seqlens, seq_len);
    ov::Tensor window_attention_mask = build_block_mask(cu_window_seqlens, seq_len);

    ov::Tensor window_index(ov::element::i64, ov::Shape{window_index_vec.size()});
    std::copy(window_index_vec.begin(), window_index_vec.end(), window_index.data<int64_t>());

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.set_tensor("attention_mask", attention_mask);
    encoder.set_tensor("window_attention_mask", window_attention_mask);
    encoder.set_tensor("window_index", window_index);
    encoder.set_tensor("rotary_pos_emb", rotary_pos_emb);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    const ov::Shape out_shape = infer_output.get_shape();  // [num_image_tokens, hidden_size]
    OPENVINO_ASSERT(out_shape.size() == 2, "Youtu-VL: unexpected vision output rank.");
    const size_t num_image_tokens = out_shape[0];
    const size_t hidden_size = out_shape[1];

    // Store as [1, num_image_tokens, hidden_size] to align with other embedders.
    ov::Tensor image_features(infer_output.get_element_type(), ov::Shape{1, num_image_tokens, hidden_size});
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    EncodedImage encoded_image;
    encoded_image.resized_source = std::move(image_features);
    encoded_image.resized_source_size = ImageSize{grid_h, grid_w};
    encoded_image.num_image_tokens = num_image_tokens;
    return encoded_image;
}

InputsEmbedderYoutuVL::InputsEmbedderYoutuVL(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) { }

InputsEmbedderYoutuVL::InputsEmbedderYoutuVL(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

std::vector<ov::genai::EncodedImage> InputsEmbedderYoutuVL::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    ov::AnyMap vision_config = {
        {"patch_size", m_vlm_config.vision_config_patch_size},
        {"merge_size", m_vlm_config.vision_config_spatial_merge_size},
        {"window_size", m_vlm_config.vision_config_window_size},
        {"max_num_patches", m_vlm_config.youtu_vl_max_image_patches},
    };
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
    }
    return embeds;
}

NormalizedPrompt InputsEmbedderYoutuVL::normalize_prompt(
    const std::string& prompt,
    size_t base_id,
    const std::vector<EncodedImage>& images) const {
    auto [unified_prompt, images_sequence] = normalize(prompt, NATIVE_TAG, NATIVE_TAG, base_id, images.size());

    const std::string image_pad = m_vlm_config.image_pad_token;
    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& enc = images.at(new_image_id - base_id);
        const size_t num_image_tokens = enc.num_image_tokens;
        std::string expanded;
        expanded.reserve(image_pad.size() * num_image_tokens);
        for (size_t i = 0; i < num_image_tokens; ++i) {
            expanded += image_pad;
        }
        // Replace the single <|image_pad|> inside the next NATIVE_TAG occurrence.
        searched_pos = unified_prompt.find(image_pad, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos,
                        "Youtu-VL: image placeholder token not found in prompt during normalization.");
        unified_prompt.replace(searched_pos, image_pad.length(), expanded);
        searched_pos += expanded.length();
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderYoutuVL::get_inputs_embeds(
    const std::string& unified_prompt,
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

    const int64_t image_token_id = m_vlm_config.image_token_id;
    OPENVINO_ASSERT(image_token_id >= 0, "Youtu-VL: image_token_id is not set in config.");

    ov::Tensor merged_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
    std::memcpy(merged_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());

    const ov::Shape shape = text_embeds.get_shape();  // [batch, seq, hidden]
    const size_t batch_size = shape.at(0);
    const size_t seq_length = shape.at(1);
    const size_t hidden_size = shape.at(2);

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    float* merged_data = merged_embeds.data<float>();

    // masked_scatter: fill positions of image_token_id, in order, from the image embeddings
    // (ordered by images_sequence), mirroring merge_vision_text_embeddings in optimum-intel.
    size_t image_embed_row = 0;
    size_t cur_image_in_seq = 0;
    const EncodedImage* cur_image = nullptr;
    size_t cur_image_rows = 0;
    const float* cur_image_data = nullptr;

    auto advance_image = [&]() {
        OPENVINO_ASSERT(cur_image_in_seq < images_sequence.size(),
                        "Youtu-VL: more image tokens in prompt than provided image embeddings.");
        cur_image = &images.at(images_sequence.at(cur_image_in_seq));
        cur_image_rows = cur_image->num_image_tokens;
        cur_image_data = cur_image->resized_source.data<const float>();
        image_embed_row = 0;
        ++cur_image_in_seq;
    };

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_length; ++s) {
            const size_t flat = b * seq_length + s;
            if (input_ids_data[flat] == image_token_id) {
                if (cur_image == nullptr || image_embed_row >= cur_image_rows) {
                    advance_image();
                }
                std::copy_n(cur_image_data + image_embed_row * hidden_size,
                            hidden_size,
                            merged_data + flat * hidden_size);
                ++image_embed_row;
            }
        }
    }

    OPENVINO_ASSERT(cur_image == nullptr || image_embed_row == cur_image_rows,
                    "Youtu-VL: image embeddings count does not match image placeholder tokens in prompt.");
    return merged_embeds;
}

} // namespace ov::genai
