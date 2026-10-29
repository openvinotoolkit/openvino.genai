// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/lfm3_omni/classes.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "json_utils.hpp"
#include "utils.hpp"

namespace ov::genai {

AudioEncoderLFM3::FeatureConfig AudioEncoderLFM3::read_feature_config(const std::filesystem::path& config_dir_path) {
    const auto path = config_dir_path / "processor_config.json";
    std::ifstream stream(path);
    OPENVINO_ASSERT(stream.is_open(), "LFM3-Omni requires '", path.string(), "' with a feature_extractor section");
    const auto parsed = nlohmann::json::parse(stream);
    OPENVINO_ASSERT(parsed.contains("feature_extractor"), "'", path.string(), "' has no feature_extractor section");
    const auto& section = parsed.at("feature_extractor");

    using utils::read_json_param;
    FeatureConfig config;
    size_t win_length = config.n_fft;
    float normalization_offset = 4.0f;
    float normalization_divisor = 4.0f;
    read_json_param(section, "feature_size", config.feature_size);
    read_json_param(section, "sampling_rate", config.sampling_rate);
    read_json_param(section, "n_fft", config.n_fft);
    read_json_param(section, "hop_length", config.hop_length);
    read_json_param(section, "log_floor", config.log_floor);
    read_json_param(section, "win_length", win_length);
    read_json_param(section, "normalization_offset", normalization_offset);
    read_json_param(section, "normalization_divisor", normalization_divisor);

    // Audio input is 16 kHz by API contract, and WhisperFeatureExtractor fixes the window and (x + 4) / 4.
    OPENVINO_ASSERT(config.sampling_rate == 16000, "LFM3-Omni audio features expect sampling_rate 16000, got ",
                    config.sampling_rate);
    OPENVINO_ASSERT(win_length == config.n_fft, "LFM3-Omni audio features expect win_length == n_fft");
    OPENVINO_ASSERT(normalization_offset == 4.0f && normalization_divisor == 4.0f,
                    "LFM3-Omni audio features expect normalization_offset and normalization_divisor of 4");
    return config;
}

AudioEncoderLFM3::AudioEncoderLFM3(const std::shared_ptr<ov::Model>& model,
                                   const std::filesystem::path& config_dir_path,
                                   const std::string& device,
                                   const ov::AnyMap& properties)
    : AudioEncoderLFM3(model, read_feature_config(config_dir_path), device, properties) {}

AudioEncoderLFM3::AudioEncoderLFM3(const std::shared_ptr<ov::Model>& model,
                                   const FeatureConfig& config,
                                   const std::string& device,
                                   const ov::AnyMap& properties)
    : m_feature_extractor(config.feature_size, config.sampling_rate, config.n_fft, config.hop_length),
      m_log_floor(config.log_floor) {
    auto compiled = utils::singleton_core().compile_model(model, device,
                                                          utils::get_model_properties(properties, "audio_encoder", device));
    ov::genai::utils::print_compiled_model_properties(compiled, "VLM audio encoder model");
    m_ireq_queue = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled.get_property(ov::optimal_number_of_infer_requests),
        [&compiled]() -> ov::InferRequest {
            return compiled.create_infer_request();
        });
}

ov::Tensor AudioEncoderLFM3::encode(const ov::Tensor& audio) {
    OPENVINO_ASSERT(audio.get_element_type() == ov::element::f32 && audio.get_shape().size() == 1 &&
                        audio.get_size() > 0,
                    "LFM3-Omni audio must be a non-empty 1-D f32 tensor of 16 kHz PCM samples");
    const float* samples = audio.data<const float>();
    const WhisperFeatures features =
        m_feature_extractor.extract_uncentered(std::vector<float>(samples, samples + audio.get_size()), m_log_floor);
    // The flattened [feature_size, n_frames] layout is the encoder's [1, num_mel_bins, num_frames] input.
    ov::Tensor input_features(ov::element::f32, {1, features.feature_size, features.n_frames});
    std::copy(features.data.begin(), features.data.end(), input_features.data<float>());

    CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue.get());
    ov::InferRequest& request = guard.get();
    request.set_tensor("input_features", input_features);
    request.infer();
    const ov::Tensor& output = request.get_tensor("last_hidden_state");
    const ov::Shape& shape = output.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "LFM3-Omni audio encoder output must be [1, tokens, hidden]");
    // The output buffer belongs to the pooled request; copy it before the guard returns the request.
    ov::Tensor result(output.get_element_type(), {shape[1], shape[2]});
    std::copy_n(output.data<const float>(), output.get_size(), result.data<float>());
    return result;
}

InputsEmbedderLFM3Omni::InputsEmbedderLFM3Omni(const VLMConfig& vlm_config,
                                               const std::filesystem::path& model_dir,
                                               const Tokenizer& tokenizer,
                                               const std::string& device,
                                               const ov::AnyMap device_config)
    : InputsEmbedderLFM3VL(vlm_config, model_dir, tokenizer, device, device_config),
      m_audio_encoder(std::make_unique<AudioEncoderLFM3>(
          utils::singleton_core().read_model(model_dir / "openvino_audio_encoder_model.xml"),
          model_dir,
          device,
          device_config)) {}

InputsEmbedderLFM3Omni::InputsEmbedderLFM3Omni(const VLMConfig& vlm_config,
                                               const ModelsMap& models_map,
                                               const Tokenizer& tokenizer,
                                               const std::filesystem::path& config_dir_path,
                                               const std::string& device,
                                               const ov::AnyMap device_config)
    : InputsEmbedderLFM3VL(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    const auto& [model, weights] = utils::get_model_weights_pair(models_map, "audio_encoder");
    m_audio_encoder = std::make_unique<AudioEncoderLFM3>(utils::singleton_core().read_model(model, weights),
                                                         config_dir_path,
                                                         device,
                                                         device_config);
}

std::vector<EncodedAudio> InputsEmbedderLFM3Omni::encode_audios(const std::vector<ov::Tensor>& audios) {
    std::vector<EncodedAudio> encoded_audios;
    encoded_audios.reserve(audios.size());
    for (const ov::Tensor& audio : audios) {
        ov::Tensor features = m_audio_encoder->encode(audio);
        const size_t num_audio_tokens = features.get_shape()[0];
        encoded_audios.push_back({std::move(features), num_audio_tokens});
    }
    return encoded_audios;
}

NormalizedPrompt InputsEmbedderLFM3Omni::normalize_prompt(const std::string& prompt,
                                                          size_t base_image_id,
                                                          size_t base_video_id,
                                                          size_t base_audio_id,
                                                          const std::vector<EncodedImage>& images,
                                                          const std::vector<EncodedVideo>& videos,
                                                          const std::vector<EncodedAudio>& audios) const {
    const std::string native_tag =
        m_vlm_config.vision_start_token + m_vlm_config.audio_token + m_vlm_config.vision_end_token;
    // Audio goes first, so untagged audio ends up after the prepended images and videos, right before the text.
    auto [unified_prompt, audios_sequence] =
        normalize(prompt, native_tag, native_tag, base_audio_id, audios.size(), ModalityType::AUDIO);
    size_t offset = 0;
    for (size_t audio_id : audios_sequence) {
        std::string expanded_tag = m_vlm_config.vision_start_token;
        for (size_t idx = 0; idx < audios.at(audio_id - base_audio_id).num_audio_tokens; ++idx) {
            expanded_tag += m_vlm_config.audio_token;
        }
        expanded_tag += m_vlm_config.vision_end_token;
        const size_t pos = unified_prompt.find(native_tag, offset);
        OPENVINO_ASSERT(pos != std::string::npos, "LFM3-Omni audio tag is missing during prompt normalization.");
        unified_prompt.replace(pos, native_tag.size(), expanded_tag);
        offset = pos + expanded_tag.size();
    }
    auto result = InputsEmbedderLFM3VL::normalize_prompt(unified_prompt, base_image_id, base_video_id, images, videos);
    result.audios_sequence = std::move(audios_sequence);
    return result;
}

}  // namespace ov::genai
