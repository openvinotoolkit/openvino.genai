// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/lfm2_vl/classes.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "utils.hpp"
#include "visual_language/clip.hpp"

namespace ov::genai {

namespace {

constexpr size_t NUM_CHANNELS = 3;

size_t round_by_factor(double value, size_t factor) {
    return static_cast<size_t>(std::llround(value / static_cast<double>(factor))) * factor;
}

// Mirrors Lfm2VlImageProcessor._is_image_too_large: an image whose smart-resized single-tile patch
// grid would exceed max_image_tokens (with max_pixels_tolerance headroom) requires multi-tile
// grid splitting, which VisionEncoderLfm2Vl does not implement yet.
bool exceeds_single_tile_budget(size_t height, size_t width, const ProcessorConfig& config) {
    const size_t total_factor = config.encoder_patch_size * config.downsample_factor;
    const size_t h_bar =
        std::max(config.encoder_patch_size, round_by_factor(static_cast<double>(height), total_factor));
    const size_t w_bar = std::max(config.encoder_patch_size, round_by_factor(static_cast<double>(width), total_factor));
    const double max_pixels = static_cast<double>(config.max_image_tokens) *
                              static_cast<double>(config.encoder_patch_size * config.encoder_patch_size) *
                              static_cast<double>(config.downsample_factor * config.downsample_factor) *
                              static_cast<double>(config.max_pixels_tolerance);
    return static_cast<double>(h_bar * w_bar) > max_pixels;
}

// Mirrors Lfm2VlImageProcessor.smart_resize: rescales height/width so both dimensions are divisible
// by encoder_patch_size * downsample_factor (no padding needed before downsampling) and the resulting
// patch count falls within [min_image_tokens, max_image_tokens] after downsampling.
std::pair<size_t, size_t> smart_resize(size_t height, size_t width, const ProcessorConfig& config) {
    const size_t total_factor = config.encoder_patch_size * config.downsample_factor;
    const size_t patch_area =
        config.encoder_patch_size * config.encoder_patch_size * config.downsample_factor * config.downsample_factor;
    const size_t min_pixels = config.min_image_tokens * patch_area;
    const size_t max_pixels = config.max_image_tokens * patch_area;

    size_t h_bar = std::max(total_factor, round_by_factor(static_cast<double>(height), total_factor));
    size_t w_bar = std::max(total_factor, round_by_factor(static_cast<double>(width), total_factor));

    if (h_bar * w_bar > max_pixels) {
        const double beta = std::sqrt(static_cast<double>(height * width) / static_cast<double>(max_pixels));
        h_bar =
            std::max(total_factor,
                     static_cast<size_t>(std::floor(static_cast<double>(height) / beta / total_factor)) * total_factor);
        w_bar =
            std::max(total_factor,
                     static_cast<size_t>(std::floor(static_cast<double>(width) / beta / total_factor)) * total_factor);
    } else if (h_bar * w_bar < min_pixels) {
        const double beta = std::sqrt(static_cast<double>(min_pixels) / static_cast<double>(height * width));
        h_bar = static_cast<size_t>(std::ceil(static_cast<double>(height) * beta / total_factor)) * total_factor;
        w_bar = static_cast<size_t>(std::ceil(static_cast<double>(width) * beta / total_factor)) * total_factor;
    }
    return {h_bar, w_bar};
}

// Mirrors Lfm2VlImageProcessor.convert_image_to_patches: flattens a resized, normalized image into a
// NaFlex-style patch sequence with a channel-last patch layout, patches ordered row-major over the
// patch grid. resized is expected to already be sized to whole patch_size multiples.
ov::Tensor image_to_patches(const clip_image_u8& resized,
                            size_t patch_size,
                            const std::array<float, 3>& mean,
                            const std::array<float, 3>& std_dev) {
    const size_t num_patches_height = static_cast<size_t>(resized.ny) / patch_size;
    const size_t num_patches_width = static_cast<size_t>(resized.nx) / patch_size;
    const size_t num_patches = num_patches_height * num_patches_width;
    const size_t patch_dim = patch_size * patch_size * NUM_CHANNELS;
    const size_t image_width = static_cast<size_t>(resized.nx);

    ov::Tensor pixel_values{ov::element::f32, {1, num_patches, patch_dim}};
    float* pixel_values_data = pixel_values.data<float>();

    for (size_t patch_row = 0; patch_row < num_patches_height; ++patch_row) {
        for (size_t patch_col = 0; patch_col < num_patches_width; ++patch_col) {
            float* patch_data = pixel_values_data + (patch_row * num_patches_width + patch_col) * patch_dim;
            for (size_t py = 0; py < patch_size; ++py) {
                const size_t src_y = patch_row * patch_size + py;
                for (size_t px = 0; px < patch_size; ++px) {
                    const size_t src_x = patch_col * patch_size + px;
                    const uint8_t* src_pixel = &resized.buf[(src_y * image_width + src_x) * NUM_CHANNELS];
                    float* dst_pixel = patch_data + (py * patch_size + px) * NUM_CHANNELS;
                    for (size_t c = 0; c < NUM_CHANNELS; ++c) {
                        dst_pixel[c] = (static_cast<float>(src_pixel[c]) / 255.0f - mean[c]) / std_dev[c];
                    }
                }
            }
        }
    }
    return pixel_values;
}

}  // namespace

EncodedImage VisionEncoderLfm2Vl::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);

    OPENVINO_ASSERT(
        !exceeds_single_tile_budget(static_cast<size_t>(input_image.ny), static_cast<size_t>(input_image.nx), config),
        "LFM2-VL multi-tile image splitting isn't implemented: the input image (",
        input_image.nx,
        "x",
        input_image.ny,
        ") exceeds the single-tile pixel budget (max_image_tokens=",
        config.max_image_tokens,
        ", tile_size=",
        config.tile_size,
        "). Resize the image below the single-tile threshold before passing it to VLMPipeline.");

    const auto [target_height, target_width] =
        smart_resize(static_cast<size_t>(input_image.ny), static_cast<size_t>(input_image.nx), config);

    clip_image_u8 resized_image;
    bilinear_resize(input_image, resized_image, static_cast<int>(target_width), static_cast<int>(target_height));

    const size_t num_patches_height = target_height / config.encoder_patch_size;
    const size_t num_patches_width = target_width / config.encoder_patch_size;
    const size_t num_patches = num_patches_height * num_patches_width;

    ov::Tensor pixel_values =
        image_to_patches(resized_image, config.encoder_patch_size, config.image_mean, config.image_std);

    ov::Tensor spatial_shapes{ov::element::i64, {1, 2}};
    int64_t* spatial_shapes_data = spatial_shapes.data<int64_t>();
    spatial_shapes_data[0] = static_cast<int64_t>(num_patches_height);
    spatial_shapes_data[1] = static_cast<int64_t>(num_patches_width);

    ov::Tensor pixel_attention_mask{ov::element::boolean, {1, num_patches}};
    std::fill_n(pixel_attention_mask.data<bool>(), num_patches, true);

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    encoder.set_tensor("pixel_values", pixel_values);
    encoder.set_tensor("spatial_shapes", spatial_shapes);
    encoder.set_tensor("pixel_attention_mask", pixel_attention_mask);
    encoder.infer();

    // The exported openvino_vision_embeddings_model already fuses the SigLIP2-NaFlex vision tower and
    // Lfm2VlMultiModalProjector (including the pixel-unshuffle downsampling), so its output token count
    // reflects downsample_factor already applied; it isn't re-derived here.
    const ov::Tensor& infer_output = encoder.get_output_tensor();
    const size_t num_image_tokens = infer_output.get_shape().at(0);
    const size_t hidden_size = infer_output.get_shape().at(1);
    ov::Tensor image_features{ov::element::f32, {1, num_image_tokens, hidden_size}};
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    EncodedImage encoded_image;
    encoded_image.resized_source = std::move(image_features);
    encoded_image.resized_source_size =
        ImageSize{num_patches_height / config.downsample_factor, num_patches_width / config.downsample_factor};
    encoded_image.original_image_size =
        ImageSize{static_cast<size_t>(input_image.ny), static_cast<size_t>(input_image.nx)};
    encoded_image.num_image_tokens = num_image_tokens;
    return encoded_image;
}

InputsEmbedderLfm2Vl::InputsEmbedderLfm2Vl(const VLMConfig& vlm_config,
                                           const std::filesystem::path& model_dir,
                                           const Tokenizer& tokenizer,
                                           const std::string& device,
                                           const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {}

InputsEmbedderLfm2Vl::InputsEmbedderLfm2Vl(const VLMConfig& vlm_config,
                                           const ModelsMap& models_map,
                                           const Tokenizer& tokenizer,
                                           const std::filesystem::path& config_dir_path,
                                           const std::string& device,
                                           const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

NormalizedPrompt InputsEmbedderLfm2Vl::normalize_prompt(const std::string& prompt,
                                                        size_t base_id,
                                                        const std::vector<EncodedImage>& images) const {
    const std::string& image_token = m_vlm_config.im_start;
    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size());

    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& image = images.at(new_image_id - base_id);
        std::string expanded_tag = m_vlm_config.lfm2_vl_image_start_token;
        for (size_t idx = 0; idx < image.num_image_tokens; ++idx) {
            expanded_tag += image_token;
        }
        expanded_tag += m_vlm_config.lfm2_vl_image_end_token;
        OPENVINO_ASSERT(searched_pos < unified_prompt.length());
        searched_pos = unified_prompt.find(image_token, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos);
        unified_prompt.replace(searched_pos, image_token.length(), expanded_tag);
        searched_pos += expanded_tag.length();
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderLfm2Vl::get_inputs_embeds(const std::string& unified_prompt,
                                                   const std::vector<ov::genai::EncodedImage>& images,
                                                   ov::genai::VLMPerfMetrics& metrics,
                                                   bool recalculate_merged_embeddings,
                                                   const std::vector<size_t>& images_sequence) {
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id).resized_source);
    }

    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = get_text_embedding(req, input_ids, metrics);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }
    return utils::merge_text_and_image_embeddings_llava(input_ids,
                                                        text_embeds,
                                                        image_embeds,
                                                        m_vlm_config.image_token_id);
}

}  // namespace ov::genai
