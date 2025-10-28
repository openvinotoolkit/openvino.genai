// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/llava/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {

clip_image_f32 preprocess_clip_image_llava(const clip_image_u8& image, const ProcessorConfig& config) {
    // Resize
    clip_image_u8 resized_image;
    int target_size = config.size_shortest_edge;
    float scale = static_cast<float>(target_size) / std::min(image.nx, image.ny);
    int new_width = static_cast<int>(image.nx * scale);
    int new_height = static_cast<int>(image.ny * scale);
    bicubic_resize(image, resized_image, new_width, new_height);

    // Center crop
    clip_image_u8 cropped_image = center_crop(resized_image, config.crop_size_height, config.crop_size_width);

    // Normalize
    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);

    clip_image_f32 normalized_image = clip_image_preprocess(ctx, cropped_image);
    return normalized_image;
}

namespace {

ov::Tensor get_pixel_values_llava(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    clip_image_f32 preprocessed_image = preprocess_clip_image_llava(input_image, config);
    return clip_image_f32_to_tensor(preprocessed_image);
}

} // namespace

EncodedImage VisionEncoderLLaVA::encode( const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    ov::Tensor pixel_values = get_pixel_values_llava(image, config);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    ImageSize resized_source_size{config.crop_size_height / config.patch_size, config.crop_size_width / config.patch_size};

    return {std::move(image_features), resized_source_size};
}

InputsEmbedderLLaVA::InputsEmbedderLLaVA(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, device, device_config) { }

InputsEmbedderLLaVA::InputsEmbedderLLaVA(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

std::vector<ov::genai::EncodedImage> InputsEmbedderLLaVA::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    ov::AnyMap vision_config = {{"patch_size", m_vlm_config.vision_config_patch_size}};
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
    }
    return embeds;
}

NormlizedPrompt InputsEmbedderLLaVA::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    std::string image_token = m_vlm_config.im_start;
    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size());

    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id - base_id).resized_source);
        std::string expanded_tag;
        for (size_t idx = 0; idx < image_embeds.back().get_shape().at(1); ++idx) {
            expanded_tag += image_token;
        }
        expanded_tag += '\n';
        OPENVINO_ASSERT(searched_pos < unified_prompt.length());
        searched_pos = unified_prompt.find(image_token, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos);
        unified_prompt.replace(searched_pos, image_token.length(), expanded_tag);
        searched_pos += expanded_tag.length();
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderLLaVA::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
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
    ov::Tensor encoded_image_token = m_tokenizer.encode(m_vlm_config.im_start, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];
    return utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
}

} // namespace ov::genai
