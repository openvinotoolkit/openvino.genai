// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/glm_edge_v/classes.hpp"

#include <algorithm>
#include <cmath>

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {
namespace {

// GLM-Edge-V uses MllamaImageProcessor with max_image_tiles == 1. The image is resized
// preserving aspect ratio so that it fits within a single size.height x size.width tile
// (bicubic, resample=3), then padded with 0 on the bottom/right to the full tile, and
// finally rescaled to [0, 1] and normalized with image_mean / image_std. Padding value 0
// therefore maps to (0 - mean) / std after normalization, matching the reference.
//
// Fit-to-canvas dimensions follow transformers
// MllamaImageProcessor.get_image_size_fit_to_canvas with canvas == tile == size:
//   target = clip(image_dim, tile, canvas) == size (for a single tile)
//   scale_h = size_h / h, scale_w = size_w / w
//   if scale_w < scale_h: new_w = size_w, new_h = floor(h * scale_w)
//   else:                 new_h = size_h, new_w = floor(w * scale_h)
std::pair<int, int> fit_to_canvas_hw(int image_h, int image_w, int size_h, int size_w) {
    const double scale_h = static_cast<double>(size_h) / image_h;
    const double scale_w = static_cast<double>(size_w) / image_w;
    int new_h;
    int new_w;
    if (scale_w < scale_h) {
        new_w = size_w;
        new_h = std::min(std::max(static_cast<int>(std::floor(image_h * scale_w)), 1), size_h);
    } else {
        new_h = size_h;
        new_w = std::min(std::max(static_cast<int>(std::floor(image_w * scale_h)), 1), size_w);
    }
    return {new_h, new_w};
}

clip_image_f32 preprocess_clip_image_glm_edge_v(const clip_image_u8& image, const ProcessorConfig& config) {
    const int size_h = static_cast<int>(config.size_height);
    const int size_w = static_cast<int>(config.size_width);

    auto [new_h, new_w] = fit_to_canvas_hw(image.ny, image.nx, size_h, size_w);

    clip_image_u8 resized_image;
    bicubic_resize(image, resized_image, new_w, new_h);

    // Pad (bottom/right, top-left aligned) to the full size_h x size_w tile with 0.
    clip_image_u8 padded_image;
    padded_image.nx = size_w;
    padded_image.ny = size_h;
    padded_image.buf.assign(static_cast<size_t>(size_w) * size_h * 3, 0);
    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            for (int c = 0; c < 3; ++c) {
                padded_image.buf[3 * (y * size_w + x) + c] =
                    resized_image.buf[3 * (y * new_w + x) + c];
            }
        }
    }

    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);

    return clip_image_preprocess(ctx, padded_image);
}

ov::Tensor get_pixel_values_glm_edge_v(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    clip_image_f32 preprocessed_image = preprocess_clip_image_glm_edge_v(input_image, config);
    return clip_image_f32_to_tensor(preprocessed_image);
}

} // namespace

EncodedImage VisionEncoderGLMEdgeV::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
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

InputsEmbedderGLMEdgeV::InputsEmbedderGLMEdgeV(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, device, device_config) { }

InputsEmbedderGLMEdgeV::InputsEmbedderGLMEdgeV(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

std::vector<ov::genai::EncodedImage> InputsEmbedderGLMEdgeV::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    ov::AnyMap vision_config = {{"patch_size", m_vlm_config.vision_config_patch_size}};
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
    }
    return embeds;
}

NormalizedPrompt InputsEmbedderGLMEdgeV::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    const std::string image_token = m_vlm_config.begin_of_image;
    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size());

    // Expand each image placeholder into one <|begin_of_image|> per vision embedding row.
    // The vision submodel already bakes the begin/end-of-image markers into its output,
    // so the number of placeholder tokens must equal last_hidden_state's sequence length.
    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        const size_t num_image_tokens = images.at(new_image_id - base_id).resized_source.get_shape().at(1);
        std::string expanded_tag;
        expanded_tag.reserve(num_image_tokens * image_token.size());
        for (size_t idx = 0; idx < num_image_tokens; ++idx) {
            expanded_tag += image_token;
        }
        searched_pos = unified_prompt.find(image_token, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos,
            "Failed to find GLM-Edge-V image placeholder token in prompt during normalization");
        unified_prompt.replace(searched_pos, image_token.length(), expanded_tag);
        searched_pos += expanded_tag.length();
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderGLMEdgeV::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
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
    ov::Tensor encoded_image_token = m_tokenizer.encode(m_vlm_config.begin_of_image, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];
    return utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
}

} // namespace ov::genai
