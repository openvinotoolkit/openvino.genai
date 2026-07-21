// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/glm_edge_v/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

#include <algorithm>
#include <cmath>

namespace ov::genai {
namespace {

// Aspect-preserving fit-to-canvas resize followed by bottom/right padding to a
// fixed square, matching MllamaImageProcessor with max_image_tiles == 1.
// See transformers get_image_size_fit_to_canvas / MllamaImageProcessor.pad.
clip_image_u8 fit_to_square_and_pad(const clip_image_u8& image, int tile_size) {
    const int image_height = image.ny;
    const int image_width = image.nx;

    // canvas is a single tile of size (tile_size, tile_size)
    const int canvas = tile_size;

    const int target_width = std::min(std::max(image_width, tile_size), canvas);
    const int target_height = std::min(std::max(image_height, tile_size), canvas);

    const float scale_h = static_cast<float>(target_height) / image_height;
    const float scale_w = static_cast<float>(target_width) / image_width;

    int new_width = 0;
    int new_height = 0;
    if (scale_w < scale_h) {
        new_width = target_width;
        new_height = std::min(std::max(static_cast<int>(std::floor(image_height * scale_w)), 1), target_height);
    } else {
        new_height = target_height;
        new_width = std::min(std::max(static_cast<int>(std::floor(image_width * scale_h)), 1), target_width);
    }

    clip_image_u8 resized;
    bicubic_resize(image, resized, new_width, new_height);

    // Pad to (canvas, canvas). MllamaImageProcessor pads with zeros at the
    // bottom/right of the tile.
    clip_image_u8 padded;
    padded.nx = canvas;
    padded.ny = canvas;
    padded.buf.assign(static_cast<size_t>(canvas) * canvas * 3, 0);
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                padded.buf[3 * (y * canvas + x) + c] = resized.buf[3 * (y * new_width + x) + c];
            }
        }
    }
    return padded;
}

ov::Tensor get_pixel_values_glm_edge_v(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    // GLM-Edge-V uses a square image_size (preprocessor size.height == size.width == image_size).
    const int tile_size = static_cast<int>(config.image_size);
    clip_image_u8 fitted = fit_to_square_and_pad(input_image, tile_size);

    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);

    clip_image_f32 normalized_image = clip_image_preprocess(ctx, fitted);
    return clip_image_f32_to_tensor(normalized_image);
}

} // namespace

EncodedImage VisionEncoderGlmEdgeV::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    ov::Tensor pixel_values = get_pixel_values_glm_edge_v(image, config);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    return {std::move(image_features)};
}

InputsEmbedderGlmEdgeV::InputsEmbedderGlmEdgeV(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, device, device_config) { }

InputsEmbedderGlmEdgeV::InputsEmbedderGlmEdgeV(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

std::vector<ov::genai::EncodedImage> InputsEmbedderGlmEdgeV::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    ov::AnyMap vision_config = {{"patch_size", m_vlm_config.vision_config_patch_size}};
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
    }
    return embeds;
}

NormalizedPrompt InputsEmbedderGlmEdgeV::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    const std::string image_token = m_vlm_config.glm_edge_image_token;
    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size());

    size_t search_offset = 0;
    for (size_t new_image_id : images_sequence) {
        // The vision tower emits a fixed number of tokens per image (already
        // including the boi/eoi separator rows). Expand the placeholder into
        // exactly that many `<|begin_of_image|>` tokens so the merge replaces
        // them 1:1.
        const size_t num_image_tokens = images.at(new_image_id - base_id).resized_source.get_shape().at(1);

        std::string expanded_tag;
        expanded_tag.reserve(num_image_tokens * image_token.size());
        for (size_t i = 0; i < num_image_tokens; ++i) {
            expanded_tag += image_token;
        }

        size_t pos = unified_prompt.find(image_token, search_offset);
        OPENVINO_ASSERT(pos != std::string::npos, "Failed to find image token in prompt during normalization");
        unified_prompt.replace(pos, image_token.length(), expanded_tag);
        search_offset = pos + expanded_tag.size();
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderGlmEdgeV::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
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

    auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor encoded_image_token = m_tokenizer.encode(m_vlm_config.glm_edge_image_token, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];
    return utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
}

} // namespace ov::genai
