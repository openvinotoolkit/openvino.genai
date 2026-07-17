// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/onyx/classes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <tuple>
#include <vector>

#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "utils.hpp"
#include "visual_language/clip.hpp"

namespace {

constexpr const char* IMAGE_SENTINEL = "<|image|>";
constexpr const char* IMAGE_START = "<|image_start|>";
constexpr const char* PATCH_TOKEN = "<|patch|>";
constexpr const char* IMAGE_END = "<|image_end|>";

std::pair<int, int> compute_image_size(const int image_width,
                                       const int image_height,
                                       const size_t patch_hw,
                                       const size_t max_tokens) {
    float image_grid_height = static_cast<float>(image_height) / static_cast<float>(patch_hw);
    float image_grid_width = static_cast<float>(image_width) / static_cast<float>(patch_hw);
    const float ratio = image_grid_height > 0.0f ? image_grid_width / image_grid_height : 1.0f;
    if (image_grid_height * image_grid_width > static_cast<float>(max_tokens)) {
        image_grid_height = std::sqrt(static_cast<float>(max_tokens) / ratio);
        image_grid_width = image_grid_height * ratio;
    }

    const std::array<int, 2> height_candidates{static_cast<int>(std::floor(image_grid_height)),
                                               static_cast<int>(std::ceil(image_grid_height))};
    const std::array<int, 2> width_candidates{static_cast<int>(std::floor(image_grid_width)),
                                              static_cast<int>(std::ceil(image_grid_width))};

    std::vector<std::pair<int, int>> candidates;
    for (const int grid_height : height_candidates) {
        for (const int grid_width : width_candidates) {
            if (grid_height >= 1 && grid_width >= 1 && static_cast<size_t>(grid_height * grid_width) <= max_tokens) {
                candidates.emplace_back(grid_height, grid_width);
            }
        }
    }
    if (candidates.empty()) {
        candidates.emplace_back(std::max(1, static_cast<int>(std::round(image_grid_height))),
                                std::max(1, static_cast<int>(std::round(image_grid_width))));
    }

    const float source_ratio = static_cast<float>(image_height) / static_cast<float>(image_width);
    const auto best = std::min_element(
        candidates.cbegin(),
        candidates.cend(),
        [source_ratio](const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) {
            const float lhs_delta =
                std::abs(static_cast<float>(lhs.first) / static_cast<float>(lhs.second) - source_ratio);
            const float rhs_delta =
                std::abs(static_cast<float>(rhs.first) / static_cast<float>(rhs.second) - source_ratio);
            return lhs_delta < rhs_delta;
        });

    return {best->first * static_cast<int>(patch_hw), best->second * static_cast<int>(patch_hw)};
}

ov::Tensor get_pixel_values_onyx(const ov::Tensor& image, const ov::genai::ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    const size_t patch_hw = config.patch_size * config.downsample_factor;
    const auto [target_height, target_width] =
        compute_image_size(input_image.nx, input_image.ny, patch_hw, config.max_image_tokens);

    clip_image_u8 resized_image;
    lanczos_resize(input_image, resized_image, target_width, target_height);

    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
    clip_image_f32 normalized_image = clip_image_preprocess(ctx, resized_image);

    ov::Tensor pixel_values(ov::element::f32,
                            {2 * 3, static_cast<size_t>(target_height), static_cast<size_t>(target_width)});
    float* output_data = pixel_values.data<float>();
    const size_t plane_size = static_cast<size_t>(target_height) * static_cast<size_t>(target_width);
    const size_t image_size = 3 * plane_size;
    std::copy_n(normalized_image.buf.data(), image_size, output_data);
    std::copy_n(normalized_image.buf.data(), image_size, output_data + image_size);
    return pixel_values;
}

}  // namespace

namespace ov::genai {

EncodedImage VisionEncoderOnyx::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);
    ov::Tensor pixel_values = get_pixel_values_onyx(image, config);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    const ov::Shape& infer_output_shape = infer_output.get_shape();
    OPENVINO_ASSERT(infer_output_shape.size() == 2,
                    "Onyx vision embeddings output must have rank 2 [num_patches, hidden_size], got ",
                    infer_output_shape);
    const size_t num_image_tokens = infer_output_shape.at(0);
    const size_t hidden_size = infer_output_shape.at(1);

    ov::Tensor image_features(infer_output.get_element_type(), {1, num_image_tokens, hidden_size});
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    EncodedImage encoded_image{std::move(image_features)};
    encoded_image.num_image_tokens = num_image_tokens;
    return encoded_image;
}

InputsEmbedderOnyx::InputsEmbedderOnyx(const VLMConfig& vlm_config,
                                       const std::filesystem::path& model_dir,
                                       const Tokenizer& tokenizer,
                                       const std::string& device,
                                       const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {}

InputsEmbedderOnyx::InputsEmbedderOnyx(const VLMConfig& vlm_config,
                                       const ModelsMap& models_map,
                                       const Tokenizer& tokenizer,
                                       const std::filesystem::path& config_dir_path,
                                       const std::string& device,
                                       const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

std::vector<EncodedImage> InputsEmbedderOnyx::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image));
    }
    return embeds;
}

NormalizedPrompt InputsEmbedderOnyx::normalize_prompt(const std::string& prompt,
                                                      size_t base_id,
                                                      const std::vector<EncodedImage>& images) const {
    auto [unified_prompt, images_sequence] = normalize(prompt, IMAGE_SENTINEL, IMAGE_SENTINEL, base_id, images.size());

    size_t searched_pos = 0;
    for (const size_t new_image_id : images_sequence) {
        const EncodedImage& image = images.at(new_image_id - base_id);
        const ov::Shape& image_features_shape = image.resized_source.get_shape();
        OPENVINO_ASSERT(image_features_shape.size() == 3,
                        "Onyx image features must have rank 3 [1, num_patches, hidden_size], got ",
                        image_features_shape);
        const size_t num_image_tokens = image_features_shape.at(1);

        std::string expanded_tag = IMAGE_START;
        for (size_t idx = 0; idx < num_image_tokens; ++idx) {
            expanded_tag += PATCH_TOKEN;
        }
        expanded_tag += IMAGE_END;

        searched_pos = unified_prompt.find(IMAGE_SENTINEL, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos, "Onyx image sentinel is missing from normalized prompt");
        unified_prompt.replace(searched_pos, std::char_traits<char>::length(IMAGE_SENTINEL), expanded_tag);
        searched_pos += expanded_tag.length();
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderOnyx::get_inputs_embeds(const std::string& unified_prompt,
                                                 const std::vector<EncodedImage>& images,
                                                 VLMPerfMetrics& metrics,
                                                 bool recalculate_merged_embeddings,
                                                 const std::vector<size_t>& images_sequence) {
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (const size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id).resized_source);
    }

    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    const auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor encoded_image_token = m_tokenizer.encode(PATCH_TOKEN, ov::genai::add_special_tokens(false)).input_ids;
    const auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] +=
        ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    const int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];

    return utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
}

}  // namespace ov::genai
