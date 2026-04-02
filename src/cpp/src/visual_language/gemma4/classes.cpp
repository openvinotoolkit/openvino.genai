// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/gemma4/classes.hpp"

#include "utils.hpp"
#include "visual_language/clip.hpp"

namespace ov::genai {
namespace {

clip_image_f32 preprocess_clip_image_gemma4(const clip_image_u8& image, const ProcessorConfig& config) {
    clip_image_u8 resized_image;
    bilinear_resize(image, resized_image, config.size_width, config.size_height);

    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);

    clip_image_f32 normalized_image = clip_image_preprocess(ctx, resized_image);
    return normalized_image;
}

ov::Tensor get_pixel_values_gemma4(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    clip_image_f32 preprocessed_image = preprocess_clip_image_gemma4(input_image, config);
    return clip_image_f32_to_tensor(preprocessed_image);
}

const std::string PER_LAYER_EMBEDDINGS_MODEL_NAME = "openvino_text_embeddings_per_layer_model.xml";

}  // namespace

EncodedImage VisionEncoderGemma4::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    ov::Tensor pixel_values = get_pixel_values_gemma4(image, config);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    return {std::move(image_features)};
}

InputsEmbedderGemma4::InputsEmbedderGemma4(const VLMConfig& vlm_config,
                                           const std::filesystem::path& model_dir,
                                           const std::string& device,
                                           const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, device, device_config) {
    // Load per-layer text embeddings model if available
    auto per_layer_model_path = model_dir / PER_LAYER_EMBEDDINGS_MODEL_NAME;
    if (std::filesystem::exists(per_layer_model_path)) {
        auto compiled = utils::singleton_core().compile_model(per_layer_model_path, device, device_config);
        ov::genai::utils::print_compiled_model_properties(compiled, "VLM per-layer text embeddings model");
        m_per_layer_embeddings_requests = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
            compiled.get_property(ov::optimal_number_of_infer_requests),
            [&compiled]() -> ov::InferRequest {
                return compiled.create_infer_request();
            });
    }
}

InputsEmbedderGemma4::InputsEmbedderGemma4(const VLMConfig& vlm_config,
                                           const ModelsMap& models_map,
                                           const Tokenizer& tokenizer,
                                           const std::filesystem::path& config_dir_path,
                                           const std::string& device,
                                           const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    // Per-layer embeddings model may be in models_map
    auto it = models_map.find("text_embeddings_per_layer");
    if (it != models_map.end()) {
        const auto& [model_str, weights] = it->second;
        auto compiled = utils::singleton_core().compile_model(model_str, weights, device, device_config);
        ov::genai::utils::print_compiled_model_properties(compiled, "VLM per-layer text embeddings model");
        m_per_layer_embeddings_requests = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
            compiled.get_property(ov::optimal_number_of_infer_requests),
            [&compiled]() -> ov::InferRequest {
                return compiled.create_infer_request();
            });
    }
}

std::vector<ov::genai::EncodedImage> InputsEmbedderGemma4::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;

    ov::AnyMap vision_config = {{"patch_size", m_vlm_config.vision_config_patch_size}};

    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
    }

    return embeds;
}

NormalizedPrompt InputsEmbedderGemma4::normalize_prompt(const std::string& prompt,
                                                        size_t base_id,
                                                        const std::vector<EncodedImage>& images) const {
    auto [unified_prompt, images_sequence] = normalize(prompt, BOI_TOKEN, BOI_TOKEN, base_id, images.size());

    for (size_t new_image_id : images_sequence) {
        const ov::Tensor& image_embed = images.at(new_image_id - base_id).resized_source;
        size_t num_image_tokens = image_embed.get_shape().at(1);

        std::string expanded_tag = std::string(BOI_TOKEN);
        for (size_t i = 0; i < num_image_tokens; i++) {
            expanded_tag += IMAGE_TOKEN;
        }
        expanded_tag += EOI_TOKEN;

        size_t pos = unified_prompt.find(BOI_TOKEN);
        OPENVINO_ASSERT(pos != std::string::npos, "Failed to find BOI token in prompt during normalization");
        unified_prompt.replace(pos, std::string(BOI_TOKEN).length(), expanded_tag);
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderGemma4::compute_per_layer_embeddings(const ov::Tensor& input_ids) {
    OPENVINO_ASSERT(m_per_layer_embeddings_requests, "Per-layer embeddings model is not loaded");

    CircularBufferQueueElementGuard<ov::InferRequest> guard(m_per_layer_embeddings_requests.get());
    ov::InferRequest& req = guard.get();
    req.set_tensor("input_ids", input_ids);
    req.infer();

    const ov::Tensor& output = req.get_output_tensor();
    ov::Tensor result(output.get_element_type(), output.get_shape());
    std::memcpy(result.data(), output.data(), output.get_byte_size());
    return result;
}

ov::Tensor InputsEmbedderGemma4::get_inputs_embeds(const std::string& prompt,
                                                   const std::vector<EncodedImage>& images,
                                                   VLMPerfMetrics& metrics,
                                                   bool recalculate_merged_embeddings,
                                                   const std::vector<size_t>& images_sequence) {
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id).resized_source);
    }

    ov::Tensor input_ids = get_encoded_input_ids(prompt, metrics);

    // Compute per-layer embeddings if the model is available
    if (m_per_layer_embeddings_requests) {
        m_lm_extra_inputs["per_layer_inputs"] = compute_per_layer_embeddings(input_ids);
    }

    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor encoded_image_token = m_tokenizer.encode(IMAGE_TOKEN, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] +=
        ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];

    return utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderGemma4::get_position_ids(const size_t inputs_embeds_size,
                                                                                     const size_t history_size) {
    // position_ids in Gemma4 are 1-indexed (same as Gemma3)
    return IInputsEmbedder::get_position_ids(inputs_embeds_size, history_size + 1);
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderGemma4::get_generation_phase_position_ids(
    const size_t inputs_embeds_size,
    const size_t history_size,
    int64_t rope_delta) {
    // position_ids in Gemma4 are 1-indexed (same as Gemma3)
    return IInputsEmbedder::get_position_ids(inputs_embeds_size, history_size + 1);
}

const std::unordered_map<std::string, ov::Tensor>& InputsEmbedderGemma4::get_lm_extra_inputs() const {
    return m_lm_extra_inputs;
}

}  // namespace ov::genai
