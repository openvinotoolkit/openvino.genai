// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/nanollava/classes.hpp"
#include "visual_language/clip.hpp"
#include "utils.hpp"

namespace ov::genai {

const std::string NATIVE_TAG = "<image>\n";
const int64_t IMAGE_PLACEHOLDER = -200;

namespace {

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

void merge_text_and_image_embeddings_nanollava(const ov::Tensor& input_ids, ov::Tensor& text_embeds, const std::vector<ov::Tensor>& image_embeds, int64_t image_tok) {
    size_t text_tokens_size = text_embeds.get_shape()[1];
    size_t hidden_size = text_embeds.get_shape()[2];
    OPENVINO_ASSERT(text_embeds.get_shape()[1] == input_ids.get_shape()[1]);

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    float* text_embeds_data = text_embeds.data<float>();
    size_t text_token_idx = 0;
    size_t image_idx = 0;
    while (text_token_idx < text_tokens_size) {
        if (input_ids_data[text_token_idx] == image_tok) {
            const auto im_embed = image_embeds[image_idx];
            image_idx++;
            std::memcpy(text_embeds_data, im_embed.data(), im_embed.get_byte_size());
            text_embeds_data += ov::shape_size(im_embed.get_shape());
            text_token_idx += im_embed.get_shape()[1];
        }
        else {
            text_embeds_data += hidden_size;
            text_token_idx++;
        }
    }
}

ov::Tensor insert_image_placeholders(const ov::Tensor& tokens, size_t image_placeholder_size) {
    size_t tokens_size = tokens.get_size();
    const int64_t* tokens_data = tokens.data<const int64_t>();
    size_t images_num = 0;
    for (size_t idx = 0; idx < tokens_size; idx++) {
        if (tokens_data[idx] == IMAGE_PLACEHOLDER)
            images_num++;
    }
    size_t new_tokens_size = tokens_size + images_num * image_placeholder_size - images_num;
    ov::Tensor res_tokens(tokens.get_element_type(), {1, new_tokens_size});

    int64_t* res_tokens_data = res_tokens.data<int64_t>();

    size_t idx = 0;
    size_t token_idx = 0;
    while (idx < new_tokens_size) {
        if (tokens_data[token_idx] == IMAGE_PLACEHOLDER) {
            for (size_t i = 0; i < image_placeholder_size; i++) {
                res_tokens_data[idx + i] = IMAGE_PLACEHOLDER;
            }
            idx += image_placeholder_size;
        }
        else {
            res_tokens_data[idx] = tokens_data[token_idx];
            idx++;
        }
        token_idx++;
    }
    return res_tokens;
}
} // namespace

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

NormlizedPrompt InputsEmbedderNanoLLaVA::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    auto norm_res = normalize(prompt, NATIVE_TAG, NATIVE_TAG, base_id, images.size());
    return {norm_res.first, norm_res.second, {}};
}

ov::Tensor InputsEmbedderNanoLLaVA::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id).resized_source);
    }
    if (image_embeds.size() > 0) {
        if (m_image_features_size > 0) {
            OPENVINO_ASSERT(m_image_features_size == image_embeds[0].get_shape()[1], "Got unexpected shape of image embeddings.");
        }
        else {
            m_image_features_size = image_embeds[0].get_shape()[1];
        }
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
    merge_text_and_image_embeddings_nanollava(input_ids, text_embeds, image_embeds, IMAGE_PLACEHOLDER);
    return text_embeds;
}


ov::Tensor InputsEmbedderNanoLLaVA::tokenize_without_image_tag(const std::string& prompt, bool add_special_tokens) {
    std::string prompt_substr;
    size_t prev_pos = 0;
    std::vector<ov::Tensor> encoded_substrings;
    size_t encoded_size = 0;
    size_t n_images = 0;
    const std::string image_tag = "<image>";
    auto image_pos = prompt.find(image_tag);
    while (image_pos != std::string::npos) {
        auto substr = prompt.substr(prev_pos, image_pos - prev_pos);
        encoded_substrings.emplace_back(m_tokenizer.encode(substr, ov::genai::add_special_tokens(add_special_tokens)).input_ids);
        encoded_size += encoded_substrings[encoded_substrings.size() - 1].get_shape()[1];
        prev_pos = image_pos + image_tag.size();
        image_pos = prompt.find(image_tag, prev_pos);
        n_images++;
    }
    auto last_substr = prev_pos > 0 ? prompt.substr(prev_pos, prompt.size()) : prompt;
    encoded_substrings.emplace_back(m_tokenizer.encode(last_substr, ov::genai::add_special_tokens(add_special_tokens)).input_ids);
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
            new_chat_tokens_data[0] = IMAGE_PLACEHOLDER;
            new_chat_tokens_data++;
        }
    }
    return new_chat_tokens;
}

ov::Tensor InputsEmbedderNanoLLaVA::apply_chat_template_tokenize(const std::string& prompt, ov::genai::VLMPerfMetrics& metrics) {
    if (m_is_chat_conversation) {
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        ov::Tensor new_chat_tokens = tokenize_without_image_tag(prompt, false);

        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        if (m_image_features_size > 0)
            new_chat_tokens = insert_image_placeholders(new_chat_tokens, m_image_features_size);
        return new_chat_tokens;
    } else {
        ov::Tensor encoded_input_ids;
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        if (m_apply_chat_template) {
            std::string templated_prompt;
            ChatHistory history({{{"role", "user"}, {"content", prompt}}});
            constexpr bool add_generation_prompt = true;

            templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
            encoded_input_ids = tokenize_without_image_tag(templated_prompt, false);
        } else {
            encoded_input_ids = tokenize_without_image_tag(prompt, true);
        }
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        if (m_image_features_size > 0)
            encoded_input_ids = insert_image_placeholders(encoded_input_ids, m_image_features_size);
        return encoded_input_ids;
    }
}


} // namespace ov::genai
