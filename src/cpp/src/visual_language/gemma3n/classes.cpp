// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/gemma3n/classes.hpp"

#include <iostream>

#include "debug_utils.hpp"
#include "utils.hpp"
#include "visual_language/clip.hpp"

namespace ov::genai {
namespace {

clip_image_f32 preprocess_clip_image_gemma3n(const clip_image_u8& image, const ProcessorConfig& config) {
    // Resize using bilinear interpolation (same as Gemma3 / SigLIP)
    clip_image_u8 resized_image;
    bilinear_resize(image, resized_image, config.size_width, config.size_height);

    // Gemma3n preprocessor_config.json has do_normalize=false.
    // Only rescale pixel values to [0, 1] (divide by 255) without mean/std normalization.
    clip_ctx ctx;
    ctx.image_mean[0] = 0.0f;
    ctx.image_mean[1] = 0.0f;
    ctx.image_mean[2] = 0.0f;
    ctx.image_std[0] = 1.0f;
    ctx.image_std[1] = 1.0f;
    ctx.image_std[2] = 1.0f;

    clip_image_f32 rescaled_image = clip_image_preprocess(ctx, resized_image);
    return rescaled_image;
}

ov::Tensor get_pixel_values_gemma3n(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    clip_image_f32 preprocessed_image = preprocess_clip_image_gemma3n(input_image, config);
    return clip_image_f32_to_tensor(preprocessed_image);
}

}  // namespace

EncodedImage VisionEncoderGemma3n::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    ov::Tensor pixel_values = get_pixel_values_gemma3n(image, config);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    return {std::move(image_features)};
}

InputsEmbedderGemma3n::InputsEmbedderGemma3n(const VLMConfig& vlm_config,
                                             const std::filesystem::path& model_dir,
                                             const std::string& device,
                                             const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, device, device_config) {
    auto per_layer_model_path = model_dir / "openvino_text_embeddings_per_layer_model.xml";
    auto core = utils::singleton_core();
    auto model = core.read_model(per_layer_model_path);
    auto compiled_model = core.compile_model(model, device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "text embeddings per layer model");
    m_per_layer_embeddings_requests =
        std::make_unique<CircularBufferQueue<ov::InferRequest>>(1, [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
}

InputsEmbedderGemma3n::InputsEmbedderGemma3n(const VLMConfig& vlm_config,
                                             const ModelsMap& models_map,
                                             const Tokenizer& tokenizer,
                                             const std::filesystem::path& config_dir_path,
                                             const std::string& device,
                                             const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    auto per_layer_it = models_map.find("text_embeddings_per_layer");
    auto core = utils::singleton_core();
    std::shared_ptr<ov::Model> model;
    if (per_layer_it != models_map.end()) {
        const auto& [model_str, weights] = per_layer_it->second;
        model = core.read_model(model_str, weights);
    } else {
        model = core.read_model(config_dir_path / "openvino_text_embeddings_per_layer_model.xml");
    }
    auto compiled_model = core.compile_model(model, device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "text embeddings per layer model");
    m_per_layer_embeddings_requests =
        std::make_unique<CircularBufferQueue<ov::InferRequest>>(1, [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
}

std::vector<ov::genai::EncodedImage> InputsEmbedderGemma3n::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;

    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image));
    }

    return embeds;
}

NormalizedPrompt InputsEmbedderGemma3n::normalize_prompt(const std::string& prompt,
                                                         size_t base_id,
                                                         const std::vector<EncodedImage>& images) const {
    const std::string& image_token = m_vlm_config.image_soft_token;

    // Use <image_soft_token> as the native tag (matches what the chat template outputs).
    // Don't expand here — expansion happens in get_inputs_embeds after the chat template is applied,
    // to avoid the template's | trim stripping leading newlines.
    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size());

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderGemma3n::get_inputs_embeds(const std::string& prompt,
                                                    const std::vector<EncodedImage>& images,
                                                    VLMPerfMetrics& metrics,
                                                    bool recalculate_merged_embeddings,
                                                    const std::vector<size_t>& images_sequence) {
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id).resized_source);
    }

    // Store per-image token counts for use by apply_chat_template_tokenize.
    m_pending_image_token_counts.clear();
    for (size_t i = 0; i < images_sequence.size(); ++i) {
        m_pending_image_token_counts.push_back(images.at(images_sequence[i]).resized_source.get_shape().at(1));
    }

    ov::Tensor input_ids = get_encoded_input_ids(prompt, metrics);

    m_lm_extra_inputs["per_layer_inputs"] = get_per_layer_embeddings(input_ids);

    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor encoded_image_token =
        m_tokenizer.encode(m_vlm_config.image_soft_token, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] +=
        ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];

    auto inputs_embeds =
        utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);

    return inputs_embeds;
}

std::string InputsEmbedderGemma3n::expand_image_placeholders(const std::string& text) const {
    if (m_pending_image_token_counts.empty()) {
        return text;
    }

    const std::string& image_token = m_vlm_config.image_soft_token;
    const std::string& start_of_image = m_vlm_config.start_of_image;
    const std::string& end_of_image = m_vlm_config.end_of_image;

    std::string result = text;
    size_t search_offset = 0;
    for (size_t i = 0; i < m_pending_image_token_counts.size(); ++i) {
        const size_t num_tokens = m_pending_image_token_counts[i];

        std::string expanded_tag = "\n\n" + start_of_image;
        for (size_t t = 0; t < num_tokens; ++t) {
            expanded_tag += image_token;
        }
        expanded_tag += end_of_image + "\n\n";

        size_t pos = result.find(image_token, search_offset);
        OPENVINO_ASSERT(pos != std::string::npos, "Failed to find image_soft_token in prompt during expansion");
        result.replace(pos, image_token.length(), expanded_tag);
        search_offset = pos + expanded_tag.size();
    }
    return result;
}

ov::Tensor InputsEmbedderGemma3n::apply_chat_template_tokenize(const std::string& prompt, VLMPerfMetrics& metrics) {
    bool add_special_tokens_val =
        m_add_special_tokens_is_set ? m_add_special_tokens : !(m_is_chat_conversation || m_apply_chat_template);

    std::string text_to_tokenize = prompt;

    if (!m_is_chat_conversation && m_apply_chat_template) {
        ChatHistory history({{{"role", "user"}, {"content", prompt}}});
        text_to_tokenize = m_tokenizer.apply_chat_template(history, true);
    }

    text_to_tokenize = expand_image_placeholders(text_to_tokenize);

    auto start_time = std::chrono::steady_clock::now();
    ov::Tensor encoded =
        m_tokenizer.encode(text_to_tokenize, ov::genai::add_special_tokens(add_special_tokens_val)).input_ids;
    auto end_time = std::chrono::steady_clock::now();
    metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_time - start_time));

    return encoded;
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderGemma3n::get_position_ids(const size_t inputs_embeds_size,
                                                                                      const size_t history_size) {
    // position_ids in Gemma3n are 1-indexed (same as Gemma3)
    return IInputsEmbedder::get_position_ids(inputs_embeds_size, history_size + 1);
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderGemma3n::get_generation_phase_position_ids(
    const size_t inputs_embeds_size,
    const size_t history_size,
    int64_t rope_delta) {
    // position_ids in Gemma3n are 1-indexed (same as Gemma3)
    return IInputsEmbedder::get_position_ids(inputs_embeds_size, history_size + 1);
}

const std::unordered_map<std::string, ov::Tensor>& InputsEmbedderGemma3n::get_lm_extra_inputs() const {
    return m_lm_extra_inputs;
}

ov::Tensor InputsEmbedderGemma3n::get_per_layer_embeddings(const ov::Tensor& input_ids) {
    OPENVINO_ASSERT(m_per_layer_embeddings_requests, "Per-layer embeddings model is not loaded");

    CircularBufferQueueElementGuard<ov::InferRequest> guard(m_per_layer_embeddings_requests.get());
    ov::InferRequest& req = guard.get();
    req.set_tensor("input_ids", input_ids);
    req.infer();

    const ov::Tensor& output = req.get_output_tensor();
    ov::Tensor result(output.get_element_type(), output.get_shape());
    output.copy_to(result);
    return result;
}

}  // namespace ov::genai
