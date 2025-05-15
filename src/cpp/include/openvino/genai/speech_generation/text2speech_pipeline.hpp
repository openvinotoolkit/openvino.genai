// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "openvino/genai/speech_generation/speech_generation_perf_metrics.hpp"

namespace ov {
namespace genai {
class Text2SpeechPipelineImpl;

/**
 * Structure that stores the result from the generate method, including a list of waveform tensors
 * sampled at 16 kHz, along with performance metrics
 */
struct Text2SpeechDecodedResults {
    std::vector<ov::Tensor> speeches;
    SpeechGenerationPerfMetrics perf_metrics;
};

/**
 * Text to speech pipelines which provides unified API to all supported models types.
 */
class OPENVINO_GENAI_EXPORTS Text2SpeechPipeline {
public:
    /**
     * Initializes text to speech pipelines from a folder with models and performs compilation after it
     * @param models_path A models path to read models and config files from
     * @param device A single device used for all models
     * @param properties Properties to pass to 'compile_model' or other pipeline properties like LoRA adapters
     */
    Text2SpeechPipeline(const std::filesystem::path& models_path,
                        const std::string& device,
                        const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Text2SpeechPipeline(const std::filesystem::path& models_path, const std::string& device, Properties&&... properties)
        : Text2SpeechPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    /**
     * Generates speeches based on input texts
     * @param text input text for which to generate speech
     * @param speaker_embedding Optional speaker embedding tensor representing the unique characteristics of a speaker's
     * voice. If not provided for SpeechT5 TSS model, the 7306th vector from the validation set of the
     * `Matthijs/cmu-arctic-xvectors` dataset is used by default.
     * @param properties Speech generation parameters specified as properties
     * @returns raw audios of the input texts spoken in the specified speaker's voice, with a sample rate of 16 kHz
     */
    Text2SpeechDecodedResults generate(const std::string& text,
                                       const ov::Tensor& speaker_embedding = ov::Tensor(),
                                       const ov::AnyMap& properties = {}) {
        return generate(std::vector<std::string>{text}, speaker_embedding, properties);
    }

    /**
     * Generates speeches based on input texts
     * @param texts input texts for which to generate speeches
     * @param speaker_embedding Optional speaker embedding tensor representing the unique characteristics of a speaker's
     * voice. If not provided for SpeechT5 TSS model, the 7306th vector from the validation set of the
     * `Matthijs/cmu-arctic-xvectors` dataset is used by default.
     * @param properties Speech generation parameters specified as properties
     * @returns raw audios of the input texts spoken in the specified speaker's voice, with a sample rate of 16 kHz
     */
    Text2SpeechDecodedResults generate(const std::vector<std::string>& texts,
                                       const ov::Tensor& speaker_embedding = ov::Tensor(),
                                       const ov::AnyMap& properties = {});

    template <typename... Properties>
    Text2SpeechDecodedResults generate(const std::vector<std::string>& texts,
                                       const ov::Tensor& speaker_embedding = ov::Tensor(),
                                       Properties&&... properties) {
        return generate(texts, speaker_embedding, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /// @brief Extract GenerationConfig used to get default values.
    /// @return Default values used.
    SpeechGenerationConfig get_generation_config() const;

    /// @brief Override default values for GenerationConfig
    /// @param new_config A config to override default values with.
    void set_generation_config(const SpeechGenerationConfig& new_config);

private:
    std::shared_ptr<Text2SpeechPipelineImpl> m_impl;
    SpeechGenerationConfig m_speech_gen_config;
};

}  // namespace genai
}  // namespace ov
