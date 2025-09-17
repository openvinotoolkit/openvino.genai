// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/nanollava/classes.hpp"
#include "visual_language/clip.hpp"
#include "utils.hpp"

namespace ov::genai {

const std::string NATIVE_TAG = "<image>\n";
const int64_t IMAGE_TOKEN_ID = -200;

clip_image_f32 preprocess_clip_image_nanollava(const clip_image_u8& image_orig, const ProcessorConfig& config) {
    // Resize
    auto image = image_orig;
    clip_image_u8 resized_image;
    int target_size = config.size_shortest_edge;
    resized_image = resize_and_pad_image(image, {target_size, target_size}, 127);

    // Normalize
    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
    return clip_image_preprocess(ctx, resized_image);
}

EncodedImage VisionEncoderNanoLLaVA::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    // nanollava specific preprocess params
    config.image_mean = std::array<float, 3>{0.5f, 0.5f, 0.5f};
    config.image_std = std::array<float, 3>{0.5f, 0.5f, 0.5f};
    config.crop_size_height = 384;
    config.crop_size_width = 384;
    config.size_shortest_edge = 384;

    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    clip_image_f32 preprocessed_image = preprocess_clip_image_nanollava(input_image, config);
    ov::Tensor pixel_values = clip_image_f32_to_tensor(preprocessed_image);

    encoder.set_tensor("images", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    ImageSize resized_source_size{config.crop_size_height / config.patch_size, config.crop_size_width / config.patch_size};

    return {std::move(image_features), resized_source_size};
}

InputsEmbedderNanoLLaVA::InputsEmbedderNanoLLaVA(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, device, device_config) { }

InputsEmbedderNanoLLaVA::InputsEmbedderNanoLLaVA(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

std::vector<ov::genai::EncodedImage> InputsEmbedderNanoLLaVA::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    ov::AnyMap vision_config = {{"patch_size", m_vlm_config.vision_config_patch_size}};
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
    }
    return embeds;
}

std::pair<std::string, std::vector<size_t>> InputsEmbedderNanoLLaVA::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    auto [unified_prompt, images_sequence] = normalize(prompt, NATIVE_TAG, NATIVE_TAG, base_id, images.size());
    return {std::move(unified_prompt), std::move(images_sequence)};
}

ov::Tensor InputsEmbedderNanoLLaVA::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
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
    ov::Tensor encoded_image_token = m_tokenizer.encode(NATIVE_TAG, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    return utils::merge_text_and_image_embeddings_nanollava(input_ids, text_embeds, image_embeds, IMAGE_TOKEN_ID);
}


ov::Tensor InputsEmbedderNanoLLaVA::tokenize_without_image_tag(const std::string& prompt) {
    std::string prompt_substr;
    size_t prev_pos = 0;
    std::vector<ov::Tensor> encoded_substrings;
    size_t encoded_size = 0;
    size_t n_images = 0;
    const std::string image_tag = "<image>";
    auto image_pos = prompt.find(image_tag);
    while (image_pos != std::string::npos) {
        auto substr = prompt.substr(prev_pos, image_pos);
        encoded_substrings.emplace_back(m_tokenizer.encode(substr, ov::genai::add_special_tokens(false)).input_ids);
        encoded_size += encoded_substrings[encoded_substrings.size() - 1].get_shape()[1];
        prev_pos = image_pos + image_tag.size();
        image_pos = prompt.find(image_tag, prev_pos);
        n_images++;
    }
    auto last_substr = prev_pos > 0 ? prompt.substr(prev_pos, prompt.size()) : prompt;
    encoded_substrings.emplace_back(m_tokenizer.encode(last_substr, ov::genai::add_special_tokens(false)).input_ids);
    encoded_size += encoded_substrings[encoded_substrings.size() - 1].get_shape()[1];

    if (encoded_substrings.size() == 1) {
        return encoded_substrings[0];
    }
    ov::Tensor new_chat_tokens = ov::Tensor(encoded_substrings[encoded_substrings.size() - 1].get_element_type(), {1, encoded_size + n_images});
    int64_t* new_chat_tokens_data = new_chat_tokens.data<int64_t>();
    for (size_t idx = 0; idx < encoded_substrings.size(); idx++) {
        memcpy(new_chat_tokens_data, encoded_substrings[idx].data(), encoded_substrings[idx].get_byte_size());
        new_chat_tokens_data += ov::shape_size(encoded_substrings[idx].get_shape());
        if (idx < encoded_substrings.size() - 1) {
            new_chat_tokens_data[0] = IMAGE_TOKEN_ID;
            new_chat_tokens_data ++;
        }
    }
    return new_chat_tokens;
}

ov::Tensor InputsEmbedderNanoLLaVA::apply_chat_template_tokenize(const std::string& prompt, ov::genai::VLMPerfMetrics& metrics) {
    if (m_is_chat_conversation) {
        std::string prompt_to_encode = prompt;
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        ov::Tensor new_chat_tokens = tokenize_without_image_tag(prompt);

        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        return new_chat_tokens;
    } else {
        ov::Tensor encoded_input_ids;
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        if (m_apply_chat_template) {
            std::string templated_prompt;
            ChatHistory history({{{"role", "user"}, {"content", prompt}}});
            constexpr bool add_generation_prompt = true;

            templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
            encoded_input_ids = tokenize_without_image_tag(templated_prompt);
        } else {
            encoded_input_ids = tokenize_without_image_tag(prompt);
        }
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        return encoded_input_ids;
    }
}

} // namespace ov::genai
