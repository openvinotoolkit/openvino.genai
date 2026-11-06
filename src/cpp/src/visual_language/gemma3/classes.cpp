// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/gemma3/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {
namespace {

clip_image_f32 preprocess_clip_image_gemma3(const clip_image_u8& image, const ProcessorConfig& config) {

    // Resize
    clip_image_u8 resized_image;
    if (config.pad_to_target)
        resized_image = resize_and_pad_image(image, {config.size_width, config.size_height}, 0, true);
    else
        bilinear_resize(image, resized_image, config.size_width, config.size_height);

    // Normalize
    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);

    clip_image_f32 normalized_image = clip_image_preprocess(ctx, resized_image);
    return normalized_image;
}

ov::Tensor get_pixel_values_gemma3(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    clip_image_f32 preprocessed_image = preprocess_clip_image_gemma3(input_image, config);
    return clip_image_f32_to_tensor(preprocessed_image);
}

} // namespace

EncodedImage VisionEncoderGemma3::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    ov::Tensor pixel_values = get_pixel_values_gemma3(image, config);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    return {std::move(image_features)};
}

InputsEmbedderGemma3::InputsEmbedderGemma3(const VLMModels& models,
                                           const Tokenizer& tokenizer,
                                           const std::string& device,
                                           const ov::AnyMap& properties)
    : IInputsEmbedder(models, tokenizer, device, properties) {
    patch_chat_template();
}

InputsEmbedderGemma3::InputsEmbedderGemma3(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {
        patch_chat_template();
    }

InputsEmbedderGemma3::InputsEmbedderGemma3(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
        patch_chat_template();
    }

std::vector<ov::genai::EncodedImage> InputsEmbedderGemma3::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;

    ov::AnyMap vision_config = {{"patch_size", m_vlm_config.vision_config_patch_size}};

    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
    }
    
    return embeds;
}

NormalizedPrompt InputsEmbedderGemma3::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    std::string start_of_image = m_vlm_config.start_of_image;
    std::string image_token = m_vlm_config.image_soft_token;
    std::string end_of_image = m_vlm_config.end_of_image;
    
    auto [unified_prompt, images_sequence] = normalize(prompt, start_of_image, start_of_image, base_id, images.size());

    size_t search_offset = 0;
    for (size_t new_image_id : images_sequence) {
        const size_t num_image_tokens = images.at(new_image_id - base_id).resized_source.get_shape().at(1);

        std::string expanded_tag;
        expanded_tag.reserve(2 + start_of_image.size() + num_image_tokens * image_token.size() + end_of_image.size() + 2);
        expanded_tag = m_vlm_config.image_separator + start_of_image;
        for (size_t i = 0; i < num_image_tokens; i++) {
            expanded_tag += image_token;
        }
        expanded_tag += end_of_image + m_vlm_config.image_separator;

        size_t pos = unified_prompt.find(start_of_image, search_offset);
        OPENVINO_ASSERT(pos != std::string::npos, "Failed to find image token in prompt during normalization");
        unified_prompt.replace(pos, start_of_image.length(), expanded_tag);
        search_offset = pos + expanded_tag.size();
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderGemma3::apply_chat_template_tokenize(const std::string& prompt, VLMPerfMetrics& metrics) {
    auto input_ids = IInputsEmbedder::apply_chat_template_tokenize(prompt, metrics);
    m_vision_token_mask = ov::Tensor(ov::element::i64, input_ids.get_shape());
    auto* mask = m_vision_token_mask.data<int64_t>();
    std::fill_n(mask, m_vision_token_mask.get_size(), 0);
    if (prompt.find(m_vlm_config.start_of_image) == std::string::npos)
        return input_ids;

    const auto start_tokenizer_time = std::chrono::steady_clock::now();
    const auto token_id = [this](const std::string& token) {
        const auto encoded = m_tokenizer.encode(token, ov::genai::add_special_tokens(false)).input_ids;
        OPENVINO_ASSERT(encoded.get_size() == 1, "Gemma3 image token must encode to one token: ", token);
        return encoded.data<const int64_t>()[0];
    };
    const auto start_of_image_id = token_id(m_vlm_config.start_of_image);
    const auto image_token_id = token_id(m_vlm_config.image_soft_token);
    const auto end_of_image_id = token_id(m_vlm_config.end_of_image);
    metrics.raw_metrics.tokenization_durations.back() +=
        ov::genai::MicroSeconds(PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start_tokenizer_time));

    const auto* ids = input_ids.data<const int64_t>();
    bool inside_image = false;
    for (size_t i = 0; i < input_ids.get_size(); ++i) {
        if (ids[i] == start_of_image_id)
            inside_image = true;
        else if (ids[i] == end_of_image_id)
            inside_image = false;
        else
            mask[i] = inside_image && ids[i] == image_token_id;
    }
    return input_ids;
}

void InputsEmbedderGemma3::patch_chat_template() {
    set_chat_template_patcher(m_tokenizer, [](const std::string& chat_template) {
        std::string patched_chat_template = chat_template;
        const std::string trim_filter = "| trim";
        size_t pos = patched_chat_template.find(trim_filter);
        while (pos != std::string::npos) {
            patched_chat_template.erase(pos, trim_filter.length());
            pos = patched_chat_template.find(trim_filter, pos);
        }
        return patched_chat_template;
    });
}

ov::Tensor InputsEmbedderGemma3::get_inputs_embeds(
    const std::string& unified_prompt,
    const std::vector<EncodedImage>& images,
    VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& images_sequence
) {
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

        const size_t seq_len = text_embeds.get_shape()[1];

        ov::Tensor token_type_ids_all_0(ov::element::i64, {1, seq_len});
        std::fill_n(token_type_ids_all_0.data<int64_t>(), token_type_ids_all_0.get_size(), 0);
        m_lm_extra_inputs["token_type_ids"] = token_type_ids_all_0;

        return inputs_embeds;
    }

    OPENVINO_ASSERT(input_ids.get_size() <= m_vision_token_mask.get_size(),
                    "Gemma3 vision mask is shorter than input IDs");
    ov::Tensor token_type_ids(ov::element::i64, input_ids.get_shape());
    std::copy_n(m_vision_token_mask.data<const int64_t>() + m_vision_token_mask.get_size() - input_ids.get_size(),
                input_ids.get_size(),
                token_type_ids.data<int64_t>());
    m_lm_extra_inputs["token_type_ids"] = token_type_ids;
    return utils::merge_text_and_image_embeddings_llava(token_type_ids, text_embeds, image_embeds, 1);
}

const std::unordered_map<std::string, ov::Tensor>& InputsEmbedderGemma3::get_lm_extra_inputs() const {
    return m_lm_extra_inputs;
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderGemma3::get_position_ids(const size_t inputs_embeds_size, const size_t history_size) {
    return IInputsEmbedder::get_position_ids(inputs_embeds_size, history_size + m_vlm_config.position_ids_offset);
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderGemma3::get_generation_phase_position_ids(const size_t inputs_embeds_size, const size_t history_size, int64_t rope_delta) {
    return IInputsEmbedder::get_position_ids(inputs_embeds_size, history_size + m_vlm_config.position_ids_offset);
}

} // namespace ov::genai
