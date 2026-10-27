// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/minicpmv4_6/classes.hpp"

#include <algorithm>
#include <numeric>

#include "visual_language/clip.hpp"
#include "visual_language/vlm_utils.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

// The chat template renders an image as a single "<|image_pad|>" token. GenAI's
// universal/native tag machinery inserts this native tag; normalize_prompt then
// expands it into the full MiniCPM-V-4.6 placeholder structure.
const std::string NATIVE_TAG = "<|image_pad|>";

// Crop [y0:y0+ph, x0:x0+pw] from a u8 RGB (HWC) image.
clip_image_u8 crop_patch(const clip_image_u8& img, int y0, int x0, int ph, int pw) {
    clip_image_u8 patch;
    patch.nx = pw;
    patch.ny = ph;
    patch.buf.resize(3 * ph * pw);
    for (int y = 0; y < ph; ++y) {
        for (int x = 0; x < pw; ++x) {
            int src = 3 * ((y0 + y) * img.nx + (x0 + x));
            int dst = 3 * (y * pw + x);
            patch.buf[dst] = img.buf[src];
            patch.buf[dst + 1] = img.buf[src + 1];
            patch.buf[dst + 2] = img.buf[src + 2];
        }
    }
    return patch;
}

// Mirrors MiniCPMV4_6ImageProcessor.reshape_by_patch.
// Input: normalized CHW f32 image (H, W). Output: pixel_values tensor
// [1, C, patch_size, (H/patch_size)*(W/patch_size)*patch_size].
ov::Tensor reshape_by_patch(const clip_image_f32& img, int patch_size) {
    const int C = 3;
    const int H = img.ny;
    const int W = img.nx;
    const int grid_h = H / patch_size;
    const int grid_w = W / patch_size;
    const int L = grid_h * grid_w;  // number of patches
    ov::Tensor pixel_values(ov::element::f32, ov::Shape{1, static_cast<size_t>(C), static_cast<size_t>(patch_size), static_cast<size_t>(L) * patch_size});
    float* dst = pixel_values.data<float>();
    const float* src = img.buf.data();  // CHW: src[c*H*W + y*W + x]
    const size_t HW = static_cast<size_t>(H) * W;
    // reshape_by_patch semantics: patches[c, kh, l*patch_size + kw] where
    // l = row*grid_w + col enumerates patches row-major and (kh, kw) is the
    // intra-patch coordinate.
    const size_t out_last = static_cast<size_t>(L) * patch_size;
    for (int c = 0; c < C; ++c) {
        for (int row = 0; row < grid_h; ++row) {
            for (int col = 0; col < grid_w; ++col) {
                int l = row * grid_w + col;
                for (int kh = 0; kh < patch_size; ++kh) {
                    for (int kw = 0; kw < patch_size; ++kw) {
                        int y = row * patch_size + kh;
                        int x = col * patch_size + kw;
                        float v = src[c * HW + static_cast<size_t>(y) * W + x];
                        size_t out_idx = (static_cast<size_t>(c) * patch_size + kh) * out_last + static_cast<size_t>(l) * patch_size + kw;
                        dst[out_idx] = v;
                    }
                }
            }
        }
    }
    return pixel_values;
}

// Mirrors _OVMiniCPMV4_6ForCausalLM._position_ids (bucketized interpolated positions).
std::vector<int64_t> compute_position_ids(int grid_h, int grid_w, int num_patches_per_side) {
    std::vector<float> boundaries;
    for (int i = 1; i < num_patches_per_side; ++i) {
        boundaries.push_back(static_cast<float>(i) / num_patches_per_side);
    }
    auto bucketize = [&](int n) {
        std::vector<int64_t> buckets;
        buckets.reserve(n);
        for (int i = 0; i < n; ++i) {
            const float fraction = static_cast<float>(i) / n;
            buckets.push_back(std::upper_bound(boundaries.begin(), boundaries.end(), fraction) - boundaries.begin());
        }
        return buckets;
    };
    std::vector<int64_t> bh = bucketize(grid_h);
    std::vector<int64_t> bw = bucketize(grid_w);
    std::vector<int64_t> pos_ids;
    pos_ids.reserve(static_cast<size_t>(grid_h) * grid_w);
    for (int64_t h : bh) {
        for (int64_t w : bw) {
            pos_ids.push_back(h * num_patches_per_side + w);
        }
    }
    return pos_ids;
}

// Mirrors _OVMiniCPMV4_6ForCausalLM._grouping_index (permutation grouping a h x w grid
// into row-major kernel_h x kernel_w blocks).
std::vector<int64_t> grouping_index(int h, int w, int kernel_h, int kernel_w) {
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(h) * w);
    int H = h / kernel_h;
    int W = w / kernel_w;
    // index.reshape(H, kh, W, kw).permute(0, 2, 1, 3).reshape(-1)
    for (int a = 0; a < H; ++a) {
        for (int b = 0; b < W; ++b) {
            for (int i = 0; i < kernel_h; ++i) {
                for (int j = 0; j < kernel_w; ++j) {
                    int row = a * kernel_h + i;
                    int col = b * kernel_w + j;
                    out.push_back(static_cast<int64_t>(row) * w + col);
                }
            }
        }
    }
    return out;
}

} // namespace

VisionEncoderMiniCPMV4_6::VisionEncoderMiniCPMV4_6(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap properties) :
    VisionEncoder(model_dir, device, properties) {}

VisionEncoderMiniCPMV4_6::VisionEncoderMiniCPMV4_6(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    VisionEncoder(models_map, config_dir_path, device, device_config) {}

void VisionEncoderMiniCPMV4_6::update_vision_config(const VLMConfig& vlm_config) {
    m_processor_config.image_size = vlm_config.vision_config_image_size;
    m_processor_config.window_kernel_size = vlm_config.vision_config_window_kernel_size;
    m_processor_config.merge_kernel_size = vlm_config.merge_kernel_size;
}

EncodedImage VisionEncoderMiniCPMV4_6::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);
    const int patch_size = static_cast<int>(config.patch_size);
    const int scale_resolution = static_cast<int>(config.scale_resolution);
    const int max_slice_nums = static_cast<int>(config.max_slice_nums);
    OPENVINO_ASSERT(patch_size > 0 && scale_resolution > 0 &&
                        config.window_kernel_size > 0 && config.merge_kernel_size > 0,
                    "MiniCPM vision patch size, resolution and merge kernels must be positive");
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    const auto encoder_inputs = encoder.get_compiled_model().inputs();
    const bool has_window_index = std::any_of(encoder_inputs.begin(), encoder_inputs.end(), [](const auto& input) {
        return input.get_names().count("window_index") > 0;
    });
    const size_t downsample_factor = config.merge_kernel_size * (has_window_index ? config.window_kernel_size : 1);

    clip_image_u8 source = tensor_to_clip_image_u8(image);
    const SlicesGrid slices_grid = vlm_utils::get_slices_grid(source.ny, source.nx, max_slice_nums, scale_resolution);

    // Build the ordered list of resized u8 patches: [source, *slices] and their
    // patch-grid sizes {grid_h, grid_w}.
    std::vector<clip_image_u8> patches_u8;
    std::vector<std::pair<int, int>> patch_grids;  // {grid_h, grid_w} in patches

    // Source image (always present).
    const ImageSize src_size = vlm_utils::find_best_resize(
        source.ny,
        source.nx,
        scale_resolution,
        patch_size,
        static_cast<int>(downsample_factor),
        /*allow_upscale=*/!slices_grid.has_slices()
    );
    clip_image_u8 source_resized;
    bicubic_resize(source, source_resized, static_cast<int>(src_size.width), static_cast<int>(src_size.height));
    patches_u8.push_back(std::move(source_resized));
    patch_grids.push_back({static_cast<int>(src_size.height) / patch_size, static_cast<int>(src_size.width) / patch_size});

    int slice_grid_h = 0;
    int slice_grid_w = 0;
    if (slices_grid.has_slices()) {
        const ImageSize refine = vlm_utils::get_refine_size(
            source.ny,
            source.nx,
            slices_grid,
            scale_resolution,
            patch_size,
            static_cast<int>(downsample_factor)
        );
        clip_image_u8 refine_img;
        const int refine_height = static_cast<int>(refine.height);
        const int refine_width = static_cast<int>(refine.width);
        bicubic_resize(source, refine_img, refine_width, refine_height);
        const int patch_h = static_cast<int>(refine.height / slices_grid.rows);
        const int patch_w = static_cast<int>(refine.width / slices_grid.cols);
        slice_grid_h = patch_h / patch_size;
        slice_grid_w = patch_w / patch_size;
        for (int y = 0; y + patch_h <= refine_height; y += patch_h) {
            for (int x = 0; x + patch_w <= refine_width; x += patch_w) {
                patches_u8.push_back(crop_patch(refine_img, y, x, patch_h, patch_w));
                patch_grids.push_back({slice_grid_h, slice_grid_w});
            }
        }
    }

    // Run the vision encoder for each patch and concatenate the visual tokens.
    clip_ctx_double mean_std;
    for (int c = 0; c < 3; ++c) {
        mean_std.image_mean[c] = static_cast<double>(config.image_mean[c]) * 255.0;
        mean_std.image_std[c] = static_cast<double>(config.image_std[c]) * 255.0;
    }

    const auto position_ids_rank = encoder.get_compiled_model().input("position_ids").get_partial_shape().rank();
    OPENVINO_ASSERT(position_ids_rank == 1 || position_ids_rank == 2,
                    "MiniCPM vision position_ids must have rank 1 or 2, got ", position_ids_rank);

    std::vector<ov::Tensor> patch_features;
    size_t total_tokens = 0;
    size_t hidden_size = 0;
    for (size_t p = 0; p < patches_u8.size(); ++p) {
        int grid_h = patch_grids[p].first;
        int grid_w = patch_grids[p].second;
        clip_image_f32 normalized = normalize_and_convert_to_chw(patches_u8[p], mean_std);
        ov::Tensor pixel_values = reshape_by_patch(normalized, patch_size);

        const int window_kernel = static_cast<int>(config.window_kernel_size);
        const int merge_kernel = static_cast<int>(config.merge_kernel_size);
        std::vector<int64_t> position_ids = compute_position_ids(grid_h, grid_w, static_cast<int>(config.image_size / config.patch_size));
        const int merge_grid_h = has_window_index ? grid_h / window_kernel : grid_h;
        const int merge_grid_w = has_window_index ? grid_w / window_kernel : grid_w;
        std::vector<int64_t> merge_index = grouping_index(merge_grid_h, merge_grid_w, merge_kernel, merge_kernel);

        const ov::Shape position_ids_shape = position_ids_rank == 2
                                                ? ov::Shape{1, position_ids.size()}
                                                : ov::Shape{position_ids.size()};
        ov::Tensor position_ids_t(ov::element::i64, position_ids_shape);
        std::copy(position_ids.begin(), position_ids.end(), position_ids_t.data<int64_t>());
        ov::Tensor merge_index_t(ov::element::i64, ov::Shape{merge_index.size()});
        std::copy(merge_index.begin(), merge_index.end(), merge_index_t.data<int64_t>());

        encoder.set_tensor("pixel_values", pixel_values);
        encoder.set_tensor("position_ids", position_ids_t);
        if (has_window_index) {
            const auto window_index = grouping_index(grid_h, grid_w, window_kernel, window_kernel);
            ov::Tensor window_index_t(ov::element::i64, ov::Shape{window_index.size()});
            std::copy(window_index.begin(), window_index.end(), window_index_t.data<int64_t>());
            encoder.set_tensor("window_index", window_index_t);
        }
        encoder.set_tensor("merge_index", merge_index_t);
        encoder.infer();

        const ov::Tensor& out = encoder.get_output_tensor();  // [num_tokens, hidden]
        ov::Tensor feat(out.get_element_type(), out.get_shape());
        std::memcpy(feat.data(), out.data(), out.get_byte_size());
        total_tokens += feat.get_shape().at(0);
        hidden_size = feat.get_shape().at(1);
        patch_features.push_back(std::move(feat));
    }

    // Concatenate along the token dimension -> [total_tokens, hidden_size].
    ov::Tensor resized_source(ov::element::f32, ov::Shape{total_tokens, hidden_size});
    float* dst = resized_source.data<float>();
    size_t offset = 0;
    for (const auto& feat : patch_features) {
        std::memcpy(dst + offset, feat.data(), feat.get_byte_size());
        offset += feat.get_size();
    }

    EncodedImage encoded;
    encoded.resized_source = std::move(resized_source);
    encoded.resized_source_size = ImageSize{static_cast<size_t>(patch_grids[0].first), static_cast<size_t>(patch_grids[0].second)};
    encoded.original_image_size = ImageSize{static_cast<size_t>(source.ny), static_cast<size_t>(source.nx)};
    encoded.num_image_tokens = total_tokens;
    if (slices_grid.has_slices()) {
        encoded.slices_grid = slices_grid;
        encoded.slices_shape = {patch_features.front().get_shape()[0], patch_features.at(1).get_shape()[0]};
    }
    return encoded;
}

InputsEmbedderMiniCPMV4_6::InputsEmbedderMiniCPMV4_6(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {
    static_cast<VisionEncoderMiniCPMV4_6&>(*m_vision_encoder).update_vision_config(m_vlm_config);
}

InputsEmbedderMiniCPMV4_6::InputsEmbedderMiniCPMV4_6(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    static_cast<VisionEncoderMiniCPMV4_6&>(*m_vision_encoder).update_vision_config(m_vlm_config);
}

void InputsEmbedderMiniCPMV4_6::configure_position_ids(const ov::Model& language_model, bool is_paged_attention) {
    const auto& shape = language_model.input("position_ids").get_partial_shape();
    const int64_t sequential_rank = is_paged_attention ? 1 : 2;
    OPENVINO_ASSERT(shape.rank() == sequential_rank || shape.rank() == sequential_rank + 1,
                    "Unexpected MiniCPM language position_ids shape: ", shape);
    if (shape.rank() == sequential_rank) {
        m_position_ids_axes = 1;
    } else {
        OPENVINO_ASSERT(shape[0].is_static() && shape[0].get_length() > 1,
                        "MiniCPM multi-axis position_ids require a static axis count greater than one, got ", shape);
        m_position_ids_axes = static_cast<size_t>(shape[0].get_length());
    }
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderMiniCPMV4_6::get_position_ids(
    const size_t inputs_embeds_size,
    const size_t history_size
) {
    if (m_position_ids_axes == 1) {
        return IInputsEmbedder::get_position_ids(inputs_embeds_size, history_size);
    }
    ov::Tensor position_ids{ov::element::i64, {m_position_ids_axes, 1, inputs_embeds_size}};
    for (size_t axis = 0; axis < m_position_ids_axes; ++axis) {
        int64_t* axis_position_ids = position_ids.data<int64_t>() + axis * inputs_embeds_size;
        std::iota(axis_position_ids, axis_position_ids + inputs_embeds_size, static_cast<int64_t>(history_size));
    }
    return {std::move(position_ids), 0};
}

std::string InputsEmbedderMiniCPMV4_6::build_image_placeholder(const EncodedImage& image, size_t image_id) const {
    const std::string& image_pad = m_vlm_config.image_pad_token;
    size_t num_source_tokens = image.slices_shape.size() > 0 ? image.slices_shape[0] : image.num_image_tokens;
    size_t per_slice_tokens = image.slices_shape.size() > 1 ? image.slices_shape[1] : 0;
    const size_t num_rows = image.slices_grid.rows;
    const size_t num_cols = image.slices_grid.cols;

    std::string placeholder;
    if (m_vlm_config.use_image_id) {
        placeholder += m_vlm_config.im_id_start + std::to_string(image_id) + m_vlm_config.im_id_end;
    }
    placeholder += m_vlm_config.im_start;
    for (size_t i = 0; i < num_source_tokens; ++i) {
        placeholder += image_pad;
    }
    placeholder += m_vlm_config.im_end;

    if (num_rows > 0 && num_cols > 0) {
        std::string slice_placeholder = m_vlm_config.slice_start;
        for (size_t i = 0; i < per_slice_tokens; ++i) {
            slice_placeholder += image_pad;
        }
        slice_placeholder += m_vlm_config.slice_end;
        std::string slices;
        for (size_t r = 0; r < num_rows; ++r) {
            if (r > 0) {
                slices += "\n";
            }
            for (size_t c = 0; c < num_cols; ++c) {
                slices += slice_placeholder;
            }
        }
        placeholder += slices;
    }
    return placeholder;
}

NormalizedPrompt InputsEmbedderMiniCPMV4_6::normalize_prompt(
    const std::string& prompt,
    size_t base_id,
    const std::vector<EncodedImage>& images
) const {
    auto [unified_prompt, images_sequence] = normalize(prompt, NATIVE_TAG, NATIVE_TAG + '\n', base_id, images.size());

    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& enc = images.at(new_image_id - base_id);
        std::string expanded = build_image_placeholder(enc, new_image_id);
        searched_pos = unified_prompt.find(NATIVE_TAG, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos, "Image placeholder not found in prompt.");
        unified_prompt.replace(searched_pos, NATIVE_TAG.length(), expanded);
        searched_pos += expanded.length();
    }

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

void InputsEmbedderMiniCPMV4_6::encode_vision_token_ids() {
    std::call_once(m_vision_token_ids_once_flag, [this]() {
        const auto encoded_pad = m_tokenizer.encode(
            m_vlm_config.image_pad_token, ov::genai::add_special_tokens(false)).input_ids;
        OPENVINO_ASSERT(encoded_pad.get_size() == 1, "MiniCPM image pad must encode to exactly one token");
        m_image_token_id = encoded_pad.data<int64_t>()[0];
    });
}

ov::Tensor InputsEmbedderMiniCPMV4_6::get_inputs_embeds(
    const std::string& unified_prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& images_sequence) {
    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = get_text_embedding(req, input_ids, metrics);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    encode_vision_token_ids();

    // Each crop has a separate placeholder run; the shared merger consumes one view per run.
    std::vector<ov::Tensor> image_embeds;
    size_t expected_image_tokens = 0;
    for (size_t new_image_id : images_sequence) {
        const auto& image = images.at(new_image_id);
        const auto& source = image.resized_source;
        const size_t hidden_size = source.get_shape().at(1);
        const size_t source_tokens = image.slices_shape.empty() ? image.num_image_tokens : image.slices_shape[0];
        const size_t crop_count = 1 + image.slices_grid.rows * image.slices_grid.cols;
        size_t offset = 0;
        for (size_t crop = 0; crop < crop_count; ++crop) {
            const size_t tokens = crop == 0 ? source_tokens : image.slices_shape.at(1);
            image_embeds.emplace_back(source.get_element_type(), ov::Shape{1, tokens, hidden_size},
                                      const_cast<float*>(source.data<float>()) + offset * hidden_size);
            offset += tokens;
        }
        OPENVINO_ASSERT(offset == source.get_shape()[0], "MiniCPM crop token counts do not match vision embeddings");
        expected_image_tokens += offset;
    }

    const int64_t* ids = input_ids.data<int64_t>();
    const size_t actual_image_tokens = std::count(ids, ids + input_ids.get_size(), m_image_token_id);
    OPENVINO_ASSERT(actual_image_tokens == expected_image_tokens,
                    "The number of image placeholders (", actual_image_tokens,
                    ") does not match the number of image embeddings (", expected_image_tokens, ").");
    return utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, m_image_token_id);
}

} // namespace ov::genai
