// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/glm_edge_v/classes.hpp"

#include <algorithm>
#include <cmath>

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {
namespace {

// GLM-Edge-V uses an MllamaImageProcessor with a single tile (max_image_tiles=1):
// the image is resized preserving aspect ratio so the longer side fits the
// target square, placed at the top-left, and the remaining area is zero-padded.
// Padding pixels (0) become -1.0 after mean/std=0.5 normalization, matching HF.
clip_image_u8 resize_and_pad_top_left_glm_edge_v(const clip_image_u8& image, int target_width, int target_height) {
    float scale_w = static_cast<float>(target_width) / image.nx;
    float scale_h = static_cast<float>(target_height) / image.ny;
    float scale = std::min(scale_w, scale_h);

    int new_width = std::min(static_cast<int>(std::lround(image.nx * scale)), target_width);
    int new_height = std::min(static_cast<int>(std::lround(image.ny * scale)), target_height);
    new_width = std::max(new_width, 1);
    new_height = std::max(new_height, 1);

    clip_image_u8 resized_image;
    bicubic_resize(image, resized_image, new_width, new_height);

    clip_image_u8 padded_image;
    padded_image.nx = target_width;
    padded_image.ny = target_height;
    padded_image.buf.assign(static_cast<size_t>(3) * target_width * target_height, 0);

    // Copy the resized image into the top-left corner of the padded buffer.
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                padded_image.buf[3 * (y * target_width + x) + c] = resized_image.buf[3 * (y * new_width + x) + c];
            }
        }
    }
    return padded_image;
}

clip_image_f32 preprocess_clip_image_glm_edge_v(const clip_image_u8& image, const ProcessorConfig& config) {
    // Aspect-preserving resize + top-left zero-pad to a fixed square
    // (preprocessor_config.json size.height/size.width, default 672x672).
    clip_image_u8 resized_padded = resize_and_pad_top_left_glm_edge_v(
        image, static_cast<int>(config.size_width), static_cast<int>(config.size_height));

    // Normalize with image_mean/image_std (0.5/0.5).
    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);

    clip_image_f32 normalized_image = clip_image_preprocess(ctx, resized_padded);
    return normalized_image;
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

    // The GLM-Edge-V vision encoder already emits one embedding per image
    // placeholder token (learned boi/eoi vectors are baked into the output),
    // so the number of image tokens equals the sequence length of the output.
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
    std::string image_token = m_vlm_config.glm_edge_v_image_token;
    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size());

    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id - base_id).resized_source);
        // Expand a single placeholder tag into as many begin-of-image tokens as
        // there are image embeddings for this image (e.g. 578 for GLM-Edge-V).
        std::string expanded_tag;
        for (size_t idx = 0; idx < image_embeds.back().get_shape().at(1); ++idx) {
            expanded_tag += image_token;
        }
        OPENVINO_ASSERT(searched_pos < unified_prompt.length());
        searched_pos = unified_prompt.find(image_token, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos);
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

    // GLM-Edge-V uses the begin-of-image token id as the image placeholder in
    // the prompt; image embeddings replace those positions in-place.
    int64_t image_token_id = m_vlm_config.boi_token_id;
    return utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
}

} // namespace ov::genai
