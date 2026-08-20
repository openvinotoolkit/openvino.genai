// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/gemma4/audio_encoder.hpp"

#include <algorithm>
#include <cstring>

#include "openvino/openvino.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

constexpr size_t MAX_AUDIO_SAMPLES = 480'000;

size_t get_unified_feature_size(const std::shared_ptr<ov::Model>& model) {
    // Expected input_features shape: [1, num_frames, feature_size].
    const ov::PartialShape& input_shape = model->input("input_features").get_partial_shape();
    return input_shape[2].get_length();
}

std::unique_ptr<CircularBufferQueue<ov::InferRequest>> compile_audio_model(const std::shared_ptr<ov::Model>& model,
                                                                           const std::string& device,
                                                                           const ov::AnyMap& properties) {
    auto compiled =
        utils::singleton_core().compile_model(model,
                                              device,
                                              utils::get_model_properties(properties, "audio_embeddings", device));
    return std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled.get_property(ov::optimal_number_of_infer_requests),
        [&compiled]() -> ov::InferRequest {
            return compiled.create_infer_request();
        });
}

}  // namespace

AudioEncoderGemma4::AudioEncoderGemma4(const std::filesystem::path& model_dir,
                                       VLMModelType model_type,
                                       const std::string& device,
                                       const ov::AnyMap& properties)
    : m_model_type(model_type) {
    const std::filesystem::path model_path = model_dir / "openvino_audio_embeddings_model.xml";
    if (!std::filesystem::exists(model_path)) {
        return;
    }
    if (m_model_type == VLMModelType::GEMMA4) {
        m_feature_extractor.emplace(model_dir);
    }
    const std::shared_ptr<ov::Model> model = utils::singleton_core().read_model(model_path);
    if (m_model_type == VLMModelType::GEMMA4_UNIFIED) {
        m_unified_feature_size = get_unified_feature_size(model);
    }
    m_ireq_queue = compile_audio_model(model, device, properties);
}

AudioEncoderGemma4::AudioEncoderGemma4(const ModelsMap& models_map,
                                       VLMModelType model_type,
                                       const std::filesystem::path& config_dir_path,
                                       const std::string& device,
                                       const ov::AnyMap& properties)
    : m_model_type(model_type) {
    if (!models_map.count("audio_embeddings")) {
        return;
    }
    if (m_model_type == VLMModelType::GEMMA4) {
        m_feature_extractor.emplace(config_dir_path);
    }
    const auto& [model, weights] = utils::get_model_weights_pair(models_map, "audio_embeddings");
    const std::shared_ptr<ov::Model> audio_model = utils::singleton_core().read_model(model, weights);
    if (m_model_type == VLMModelType::GEMMA4_UNIFIED) {
        m_unified_feature_size = get_unified_feature_size(audio_model);
    }
    m_ireq_queue = compile_audio_model(audio_model, device, properties);
}

ov::Tensor AudioEncoderGemma4::encode(const ov::Tensor& audio) {
    OPENVINO_ASSERT(m_ireq_queue, "Gemma4 audio embeddings model is not available");

    switch (m_model_type) {
    case VLMModelType::GEMMA4_UNIFIED:
        return encode_unified(audio);
    case VLMModelType::GEMMA4:
        return encode_e_models(audio);
    default:
        OPENVINO_THROW("Unsupported model type for Gemma4 audio encoder");
    }
}

ov::Tensor AudioEncoderGemma4::prepare_unified_input(const ov::Tensor& audio) const {
    OPENVINO_ASSERT(m_unified_feature_size > 0, "Gemma4 Unified audio feature size is not initialized");
    OPENVINO_ASSERT(audio.get_element_type() == ov::element::f32,
                    "Gemma4 Unified audio input must be float32 PCM, got ",
                    audio.get_element_type());
    OPENVINO_ASSERT(audio.get_shape().size() == 1,
                    "Gemma4 Unified audio input must be a 1-D tensor of 16 kHz mono PCM samples, got rank ",
                    audio.get_shape().size());
    OPENVINO_ASSERT(audio.get_size() > 0, "Gemma4 Unified audio input must not be empty");
    OPENVINO_ASSERT(audio.get_size() <= MAX_AUDIO_SAMPLES,
                    "Gemma4 Unified audio input exceeds the 30 second limit of ",
                    MAX_AUDIO_SAMPLES,
                    " samples at 16 kHz, got ",
                    audio.get_size());

    const size_t num_frames = (audio.get_size() + m_unified_feature_size - 1) / m_unified_feature_size;
    ov::Tensor input_features(ov::element::f32, {1, num_frames, m_unified_feature_size});
    float* feature_data = input_features.data<float>();
    std::fill(feature_data, feature_data + input_features.get_size(), 0.0f);
    std::copy(audio.data<const float>(), audio.data<const float>() + audio.get_size(), feature_data);
    return input_features;
}

ov::Tensor AudioEncoderGemma4::encode_unified(const ov::Tensor& audio) {
    const ov::Tensor input_features = prepare_unified_input(audio);
    CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue.get());
    ov::InferRequest& request = guard.get();
    request.set_tensor("input_features", input_features);
    request.infer();

    // Expected last_hidden_state shape: [1, tokens, hidden_size].
    const ov::Tensor& output = request.get_tensor("last_hidden_state");
    const ov::Shape& output_shape = output.get_shape();

    ov::Tensor result(ov::element::f32, {output_shape[1], output_shape[2]});
    std::memcpy(result.data<float>(), output.data<const float>(), output.get_byte_size());
    return result;
}

ov::Tensor AudioEncoderGemma4::encode_e_models(const ov::Tensor& audio) {
    const Gemma4AudioFeatures features = m_feature_extractor->extract(audio);
    CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue.get());
    ov::InferRequest& request = guard.get();
    request.set_tensor("input_features", features.input_features);
    request.set_tensor("input_features_mask", features.input_features_mask);
    request.infer();

    // Expected last_hidden_state shape: [1, tokens, hidden_size].
    const ov::Tensor& output = request.get_tensor("last_hidden_state");
    const ov::Shape& output_shape = output.get_shape();

    // Expected boolean attention_mask shape: [1, tokens].
    const ov::Tensor& output_mask = request.get_tensor("attention_mask");

    const bool* output_mask_data = output_mask.data<const bool>();
    const size_t valid_tokens = std::count(output_mask_data, output_mask_data + output_shape[1], true);
    const size_t hidden_size = output_shape[2];
    ov::Tensor result(ov::element::f32, {valid_tokens, hidden_size});
    const float* source = output.data<const float>();
    float* destination = result.data<float>();
    for (size_t token = 0; token < output_shape[1]; ++token) {
        if (output_mask_data[token]) {
            std::memcpy(destination, source + token * hidden_size, hidden_size * sizeof(float));
            destination += hidden_size;
        }
    }
    return result;
}

}  // namespace ov::genai
