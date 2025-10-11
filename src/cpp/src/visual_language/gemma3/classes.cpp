// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/gemma3/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {
namespace {

clip_image_f32 preprocess_clip_image_gemma3(const clip_image_u8& image, const ProcessorConfig& config) {

    // Resize
    clip_image_u8 resized_image;
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

    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    ov::Tensor pixel_values = get_pixel_values_gemma3(image, config);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    return {std::move(image_features)};
}

InputsEmbedderGemma3::InputsEmbedderGemma3(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, device, device_config) { }

InputsEmbedderGemma3::InputsEmbedderGemma3(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

bool InputsEmbedderGemma3::has_token_type_ids() const {
    return true;
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

NormlizedPrompt InputsEmbedderGemma3::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    std::string start_of_image = m_vlm_config.start_of_image;
    std::string image_token = m_vlm_config.image_soft_token;
    std::string end_of_image = m_vlm_config.end_of_image;
    
    auto [unified_prompt, images_sequence] = normalize(prompt, start_of_image, start_of_image, base_id, images.size());

    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id - base_id).resized_source);

        size_t num_image_tokens = image_embeds.back().get_shape().at(1);

        std::string expanded_tag = "\n\n" + start_of_image;
        for (size_t i = 0; i < num_image_tokens; i++) {
            expanded_tag += image_token;
        }
        expanded_tag += end_of_image + "\n\n";

        unified_prompt.replace(unified_prompt.find(start_of_image), start_of_image.length(), expanded_tag);
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderGemma3::get_inputs_embeds(const std::string& prompt, const std::vector<EncodedImage>& images, VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    OPENVINO_THROW(
        "[InputsEmbedderGemma3] The method get_inputs_embeds is not supported for Gemma3 models because token type IDs are required to distinguish between text and image tokens in the input sequence. "
        "Please use get_inputs_embeds_with_token_type_ids instead, which returns both the input embeddings and the necessary token type IDs. "
        "This is required for correct processing of multimodal inputs in Gemma3."
    );
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderGemma3::get_inputs_embeds_with_token_type_ids(const std::string& unified_prompt, const std::vector<EncodedImage>& images, VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {

    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id).resized_source);
    }

    // HF/optimum-intel first apply chat template to messages/text prompt:
    //      <bos><start_of_turn>user\n<start_of_image>text prompt example<end_of_turn>\n<start_of_turn>model
    // Then templated prompt string is modified - they replace image start token with image pad + end tokens and pad double newlines from both sides:
    //      <start_of_image> -> \n\n<start_of_image><image_soft_token>...<image_soft_token><end_of_image>\n\n
    // In GenAI we first prepare unified prompt that already includes all image tokens (start, pad, end) and new lines, 
    //  and only then apply chat template.
    // As Gemma3 original chat template has trim filter for message content (`{{ message['content'] | trim }}`),
    //  unified prompt in GenAI is affected:
    //  - left padded double newline is removed
    //  - right padded double newline transformed to single newline (Jinja2Cpp behaviour)
    // Removing trim filter in chat template fallback resolves this issue.
    // TODO Check trim filter behaviour and chat template rendering with minja.
    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);

    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());

        const size_t seq_len = text_embeds.get_shape()[1];
        std::vector<int64_t> token_type_ids_data(seq_len, 0);
        ov::Tensor token_type_ids_all_0(ov::element::i64, {1, seq_len});
        std::fill_n(token_type_ids_all_0.data<int64_t>(), seq_len, 0);

        return {inputs_embeds, token_type_ids_all_0};
    }

    auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor encoded_image_token = m_tokenizer.encode(m_vlm_config.image_soft_token, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];

    auto inputs_embeds = utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    const ov::Shape& shape = input_ids.get_shape();
    size_t num_elements = input_ids.get_size();
    ov::Tensor token_type_ids(ov::element::i64, shape);
    int64_t* token_type_data = token_type_ids.data<int64_t>();
    for (size_t i = 0; i < num_elements; ++i) {
        token_type_data[i] = (input_ids_data[i] == image_token_id) ? 1 : 0;
    }

    return {inputs_embeds, token_type_ids};
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderGemma3::get_position_ids(const size_t inputs_embeds_size, const size_t history_size) {
    // position_ids in Gemma3 are 1-indexed
    // https://github.com/huggingface/optimum-intel/blob/v1.24.0/optimum/intel/openvino/modeling_visual_language.py#L874-L876
    return IInputsEmbedder::get_position_ids(inputs_embeds_size, history_size + 1);
}

} // namespace ov::genai