// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/paddleocr_vl/classes.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

#include "visual_language/clip.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

// Rescale (1/255) + CLIP-style normalization ((x - mean) / std) applied to a bicubic-resized
// image, extracting raster-order patch_size x patch_size patches into pixel_values [N, 3, P, P].
ov::Tensor extract_normalized_patches(const clip_image_u8& resized,
                                      const std::array<float, 3>& image_mean,
                                      const std::array<float, 3>& image_std,
                                      size_t patch_size,
                                      size_t grid_h,
                                      size_t grid_w) {
    const size_t channels = 3;
    const size_t num_patches = grid_h * grid_w;
    ov::Tensor pixel_values{ov::element::f32, {num_patches, channels, patch_size, patch_size}};
    float* dst = pixel_values.data<float>();

    const int width = resized.nx;
    const uint8_t* src = resized.buf.data();

    for (size_t bh = 0; bh < grid_h; ++bh) {
        for (size_t bw = 0; bw < grid_w; ++bw) {
            const size_t patch_idx = bh * grid_w + bw;
            float* patch_dst = dst + patch_idx * channels * patch_size * patch_size;
            for (size_t c = 0; c < channels; ++c) {
                const float mean = image_mean[c];
                const float inv_std = 1.0f / image_std[c];
                for (size_t py = 0; py < patch_size; ++py) {
                    const size_t src_y = bh * patch_size + py;
                    for (size_t px = 0; px < patch_size; ++px) {
                        const size_t src_x = bw * patch_size + px;
                        const uint8_t pixel = src[(src_y * width + src_x) * channels + c];
                        const float normalized = (pixel / 255.0f - mean) * inv_std;
                        patch_dst[c * patch_size * patch_size + py * patch_size + px] = normalized;
                    }
                }
            }
        }
    }
    return pixel_values;
}

}  // namespace

VisionEncoderPaddleOCRVL::VisionEncoderPaddleOCRVL(const std::filesystem::path& model_dir,
                                                   const std::string& device,
                                                   const ov::AnyMap properties)
    : VisionEncoder(model_dir, ConfigOnlyTag{}) {
    auto model = utils::singleton_core().read_model(model_dir / "openvino_vision_embeddings_model.xml");
    auto compiled_model = utils::singleton_core().compile_model(
        model, device, utils::get_model_properties(properties, "vision_embeddings", device));
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings model");
    m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
}

VisionEncoderPaddleOCRVL::VisionEncoderPaddleOCRVL(const ModelsMap& models_map,
                                                   const std::filesystem::path& config_dir_path,
                                                   const std::string& device,
                                                   const ov::AnyMap properties)
    : VisionEncoder(models_map, config_dir_path, ConfigOnlyTag{}) {
    const auto& [vision_encoder_model, vision_encoder_weights] =
        utils::get_model_weights_pair(models_map, "vision_embeddings");
    auto model = utils::singleton_core().read_model(vision_encoder_model, vision_encoder_weights);
    auto compiled_model = utils::singleton_core().compile_model(
        model, device, utils::get_model_properties(properties, "vision_embeddings", device));
    m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
}

EncodedImage VisionEncoderPaddleOCRVL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    const ProcessorConfig& config = m_processor_config;

    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    // factor = patch_size * merge_size; min/max pixels come from size.shortest_edge/longest_edge.
    ImageSize target = qwen2_vl_utils::smart_resize(static_cast<size_t>(input_image.ny),
                                                    static_cast<size_t>(input_image.nx),
                                                    config.patch_size * config.merge_size,
                                                    config.min_pixels,
                                                    config.max_pixels);

    clip_image_u8 resized;
    bicubic_resize(input_image, resized, static_cast<int>(target.width), static_cast<int>(target.height));

    const size_t grid_h = target.height / config.patch_size;
    const size_t grid_w = target.width / config.patch_size;

    ov::Tensor pixel_values = extract_normalized_patches(
        resized, config.image_mean, config.image_std, config.patch_size, grid_h, grid_w);

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();
    ov::Tensor hidden_states = encoder.get_output_tensor();

    EncodedImage encoded_img;
    encoded_img.resized_source = ov::Tensor(hidden_states.get_element_type(), hidden_states.get_shape());
    hidden_states.copy_to(encoded_img.resized_source);
    encoded_img.resized_source_size = ImageSize{grid_h, grid_w};
    return encoded_img;
}

InputsEmbedderPaddleOCRVL::InputsEmbedderPaddleOCRVL(const VLMConfig& vlm_config,
                                                     const std::filesystem::path& model_dir,
                                                     const Tokenizer& tokenizer,
                                                     const std::string& device,
                                                     const ov::AnyMap device_config)
    : InputsEmbedderQwen2VL(vlm_config, model_dir, tokenizer, device, device_config) {
    m_num_grid_per_side = m_vision_encoder->get_processor_config().patch_size == 0
                              ? 0
                              : (384 / m_vision_encoder->get_processor_config().patch_size);
    auto pos_model = utils::singleton_core().read_model(model_dir / "openvino_vision_embeddings_pos_model.xml");
    init_pos_model(pos_model, device, device_config);
}

InputsEmbedderPaddleOCRVL::InputsEmbedderPaddleOCRVL(const VLMConfig& vlm_config,
                                                     const ModelsMap& models_map,
                                                     const Tokenizer& tokenizer,
                                                     const std::filesystem::path& config_dir_path,
                                                     const std::string& device,
                                                     const ov::AnyMap device_config)
    : InputsEmbedderQwen2VL(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    m_num_grid_per_side = m_vision_encoder->get_processor_config().patch_size == 0
                              ? 0
                              : (384 / m_vision_encoder->get_processor_config().patch_size);
    const auto& [pos_model_str, pos_weights] = utils::get_model_weights_pair(models_map, "vision_embeddings_pos");
    auto pos_model = utils::singleton_core().read_model(pos_model_str, pos_weights);
    init_pos_model(pos_model, device, device_config);
}

void InputsEmbedderPaddleOCRVL::init_pos_model(const std::shared_ptr<ov::Model>& pos_model,
                                               const std::string& device,
                                               const ov::AnyMap& device_config) {
    auto pos_compiled = utils::singleton_core().compile_model(
        pos_model, device, utils::get_model_properties(device_config, "vision_embeddings_pos", device));
    m_ireq_queue_vision_embeddings_pos = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        pos_compiled.get_property(ov::optimal_number_of_infer_requests),
        [&pos_compiled]() -> ov::InferRequest {
            return pos_compiled.create_infer_request();
        });
}

NormalizedPrompt InputsEmbedderPaddleOCRVL::normalize_prompt(
    const std::string& prompt,
    size_t base_id,
    const std::vector<EncodedImage>& images) const {
    auto norm_prompt = normalize_prompt(prompt, base_id, 0, images, {});
    return {norm_prompt.unified_prompt, norm_prompt.images_sequence};
}

NormalizedPrompt InputsEmbedderPaddleOCRVL::normalize_prompt(
    const std::string& prompt,
    size_t image_base_id,
    size_t /*video_base_id*/,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& /*videos*/) const {
    auto [unified_prompt, images_sequence] =
        normalize(prompt, NATIVE_TAG_PADDLE, NATIVE_TAG_PADDLE, image_base_id, images.size(), VisionType::IMAGE);

    std::vector<std::array<size_t, 3>> images_grid_thw;
    images_grid_thw.reserve(images.size());
    for (const auto& encoded_image : images) {
        images_grid_thw.push_back({1, encoded_image.resized_source_size.height, encoded_image.resized_source_size.width});
    }

    for (size_t new_image_id : images_sequence) {
        auto [grid_t, grid_h, grid_w] = images_grid_thw.at(new_image_id - image_base_id);
        const size_t num_image_pad_tokens = calc_tokens_num(grid_t, grid_h, grid_w);

        std::string expanded_tag;
        expanded_tag.reserve(m_vlm_config.vision_start_token.length() +
                             m_vlm_config.image_pad_token.length() * num_image_pad_tokens +
                             m_vlm_config.vision_end_token.length());
        expanded_tag.append(m_vlm_config.vision_start_token);
        for (size_t i = 0; i < num_image_pad_tokens; ++i) {
            expanded_tag.append(m_vlm_config.image_pad_token);
        }
        expanded_tag.append(m_vlm_config.vision_end_token);

        unified_prompt.replace(unified_prompt.find(NATIVE_TAG_PADDLE), NATIVE_TAG_PADDLE.length(), expanded_tag);
    }

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

const ov::Tensor& InputsEmbedderPaddleOCRVL::get_pos_embed_table() const {
    if (m_pos_embed_table) {
        return m_pos_embed_table;
    }
    const size_t num_positions = m_num_grid_per_side * m_num_grid_per_side;
    ov::Tensor idx{ov::element::i64, {num_positions}};
    int64_t* idx_data = idx.data<int64_t>();
    std::iota(idx_data, idx_data + num_positions, static_cast<int64_t>(0));

    CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_vision_embeddings_pos.get());
    ov::InferRequest& pos_req = guard.get();
    pos_req.set_tensor("input", idx);
    pos_req.infer();
    ov::Tensor table = pos_req.get_output_tensor();
    m_pos_embed_table = ov::Tensor(table.get_element_type(), table.get_shape());
    table.copy_to(m_pos_embed_table);
    return m_pos_embed_table;
}

void InputsEmbedderPaddleOCRVL::add_interpolated_pos_embeds(
    const std::vector<std::array<size_t, 3>>& grids_thw,
    ov::Tensor& hidden_states) const {
    const ov::Tensor& table = get_pos_embed_table();  // [num_positions, dim]
    const size_t dim = table.get_shape().at(1);
    const size_t side = m_num_grid_per_side;
    const float* table_data = table.data<const float>();
    float* dst = hidden_states.data<float>();

    size_t offset = 0;
    for (const auto& grid_thw : grids_thw) {
        const size_t t = grid_thw.at(0);
        const size_t h = grid_thw.at(1);
        const size_t w = grid_thw.at(2);

        // Bilinear interpolation of the [side, side, dim] table to [h, w, dim], align_corners=False.
        std::vector<float> interp(h * w * dim);
        const float scale_h = static_cast<float>(side) / static_cast<float>(h);
        const float scale_w = static_cast<float>(side) / static_cast<float>(w);
        for (size_t oy = 0; oy < h; ++oy) {
            float sy = (static_cast<float>(oy) + 0.5f) * scale_h - 0.5f;
            if (sy < 0.0f) {
                sy = 0.0f;
            }
            const size_t y0 = static_cast<size_t>(std::floor(sy));
            const size_t y1 = std::min(y0 + 1, side - 1);
            const float ly = sy - static_cast<float>(y0);
            for (size_t ox = 0; ox < w; ++ox) {
                float sx = (static_cast<float>(ox) + 0.5f) * scale_w - 0.5f;
                if (sx < 0.0f) {
                    sx = 0.0f;
                }
                const size_t x0 = static_cast<size_t>(std::floor(sx));
                const size_t x1 = std::min(x0 + 1, side - 1);
                const float lx = sx - static_cast<float>(x0);

                const float w00 = (1.0f - ly) * (1.0f - lx);
                const float w01 = (1.0f - ly) * lx;
                const float w10 = ly * (1.0f - lx);
                const float w11 = ly * lx;

                const float* c00 = table_data + (y0 * side + x0) * dim;
                const float* c01 = table_data + (y0 * side + x1) * dim;
                const float* c10 = table_data + (y1 * side + x0) * dim;
                const float* c11 = table_data + (y1 * side + x1) * dim;
                float* out = interp.data() + (oy * w + ox) * dim;
                for (size_t d = 0; d < dim; ++d) {
                    out[d] = w00 * c00[d] + w01 * c01[d] + w10 * c10[d] + w11 * c11[d];
                }
            }
        }

        // Repeat the interpolated grid over the temporal dimension and add into hidden_states.
        for (size_t ti = 0; ti < t; ++ti) {
            for (size_t p = 0; p < h * w; ++p) {
                float* dst_row = dst + (offset + ti * h * w + p) * dim;
                const float* src_row = interp.data() + p * dim;
                for (size_t d = 0; d < dim; ++d) {
                    dst_row[d] += src_row[d];
                }
            }
        }
        offset += t * h * w;
    }
}

ov::Tensor InputsEmbedderPaddleOCRVL::get_rotary_pos_emb(
    const std::vector<std::array<size_t, 3>>& grids_thw) const {
    // merge_size = 1: raster (row, col) positions. dim taken from merger rotary_pos_emb input.
    CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_vision_embeddings_merger.get());
    ov::InferRequest& merger = guard.get();
    const size_t dim = merger.get_tensor("rotary_pos_emb").get_shape().at(1);
    const size_t half = dim / 2;
    const float theta = 10000.0f;

    std::vector<float> inv_freq(half);
    for (size_t i = 0; i < half; ++i) {
        inv_freq[i] = 1.0f / std::pow(theta, static_cast<float>(i) / static_cast<float>(half));
    }

    size_t total_tokens = 0;
    for (const auto& grid_thw : grids_thw) {
        total_tokens += grid_thw.at(0) * grid_thw.at(1) * grid_thw.at(2);
    }

    ov::Tensor rotary_pos_emb{ov::element::f32, {total_tokens, dim}};
    float* out = rotary_pos_emb.data<float>();

    size_t token = 0;
    for (const auto& grid_thw : grids_thw) {
        const size_t t = grid_thw.at(0);
        const size_t h = grid_thw.at(1);
        const size_t w = grid_thw.at(2);
        for (size_t ti = 0; ti < t; ++ti) {
            for (size_t r = 0; r < h; ++r) {
                for (size_t c = 0; c < w; ++c) {
                    float* row = out + token * dim;
                    for (size_t i = 0; i < half; ++i) {
                        row[i] = static_cast<float>(r) * inv_freq[i];
                        row[half + i] = static_cast<float>(c) * inv_freq[i];
                    }
                    ++token;
                }
            }
        }
    }
    return rotary_pos_emb;
}

ov::Tensor InputsEmbedderPaddleOCRVL::get_merge_index(
    const std::vector<std::array<size_t, 3>>& grids_thw) const {
    const size_t merge = m_vision_encoder->get_processor_config().merge_size;

    size_t total = 0;
    for (const auto& grid_thw : grids_thw) {
        total += grid_thw.at(0) * grid_thw.at(1) * grid_thw.at(2);
    }
    ov::Tensor merge_index{ov::element::i64, {total}};
    int64_t* data = merge_index.data<int64_t>();

    size_t out_pos = 0;
    size_t base = 0;
    for (const auto& grid_thw : grids_thw) {
        const size_t t = grid_thw.at(0);
        const size_t h = grid_thw.at(1);
        const size_t w = grid_thw.at(2);
        for (size_t ti = 0; ti < t; ++ti) {
            for (size_t bh = 0; bh < h / merge; ++bh) {
                for (size_t bw = 0; bw < w / merge; ++bw) {
                    for (size_t i = 0; i < merge; ++i) {
                        for (size_t j = 0; j < merge; ++j) {
                            data[out_pos++] = static_cast<int64_t>(
                                base + ti * h * w + (bh * merge + i) * w + (bw * merge + j));
                        }
                    }
                }
            }
        }
        base += t * h * w;
    }
    return merge_index;
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderPaddleOCRVL::run_video_image_embeddings_merger(
    const std::vector<EncodedImage>& images,
    const std::vector<size_t>& images_sequence,
    const std::vector<EncodedVideo>& /*videos*/,
    const std::vector<size_t>& /*videos_sequence*/) {
    auto [reordered_image_embeds, reordered_images_grid_thw] =
        qwen2_vl_utils::reorder_image_embeds_and_grid_thw(images, images_sequence);

    ov::Tensor hidden_states = qwen2_vl_utils::concatenate_video_image_embeds({}, reordered_image_embeds);

    OPENVINO_ASSERT(hidden_states, "PaddleOCR-VL merger received no image embeddings.");

    // Add interpolated position embeddings into the patch hidden states (in place).
    add_interpolated_pos_embeds(reordered_images_grid_thw, hidden_states);

    ov::Tensor rotary_pos_emb = get_rotary_pos_emb(reordered_images_grid_thw);
    ov::Tensor attention_mask = qwen2_vl_utils::get_attention_mask(reordered_images_grid_thw, {});
    ov::Tensor merge_index = get_merge_index(reordered_images_grid_thw);

    CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_vision_embeddings_merger.get());
    ov::InferRequest& merger = guard.get();
    merger.set_tensor("hidden_states", hidden_states);
    merger.set_tensor("attention_mask", attention_mask);
    merger.set_tensor("rotary_pos_emb", rotary_pos_emb);
    merger.set_tensor("merge_index", merge_index);
    merger.infer();
    ov::Tensor merged = merger.get_tensor("last_hidden_state");

    ov::Tensor image_embeds(merged.get_element_type(), merged.get_shape());
    merged.copy_to(image_embeds);

    // Empty (but initialized) video embeddings: PaddleOCR-VL image path only.
    ov::Tensor empty_video{merged.get_element_type(), ov::Shape{0, merged.get_shape().at(1)}};

    return {empty_video, image_embeds};
}

}  // namespace ov::genai
