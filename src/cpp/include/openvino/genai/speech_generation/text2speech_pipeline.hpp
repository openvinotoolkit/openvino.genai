// Copyright (C) 2023-2026 Intel Corporation
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
 * and performance metrics.
 */
struct Text2SpeechDecodedResults {
    std::vector<ov::Tensor> speeches;
    uint32_t output_sample_rate = 0;
    SpeechGenerationPerfMetrics perf_metrics;
};

/**
 * Lightweight token representation for direct speech synthesis.
 *
 * Applications can map external G2P/tokenizer outputs (for example Python Misaki
 * MTokens) into this stable OV GenAI type without introducing hard dependency on
 * any external token class.
 */
struct OPENVINO_GENAI_EXPORTS SpeechToken {
    std::string phonemes;
    bool whitespace = false;
    std::string text;
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
    * @returns raw audios of the input texts spoken in the specified speaker's voice; sample rate is provided in
    * `Text2SpeechDecodedResults::output_sample_rate`
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
    * @returns raw audios of the input texts spoken in the specified speaker's voice; sample rate is provided in
    * `Text2SpeechDecodedResults::output_sample_rate`
     */
    Text2SpeechDecodedResults generate(const std::vector<std::string>& texts,
                                       const ov::Tensor& speaker_embedding = ov::Tensor(),
                                       const ov::AnyMap& properties = {});

    /**
     * Generates speech from precomputed Kokoro phoneme chunks for a single input.
      *
      * NOTE: This API is supported only for Kokoro backend. SpeechT5 backend throws an exception.
     *
     * @param phoneme_chunks phoneme chunks to synthesize and concatenate into one output speech
      * @param speaker_embedding Optional speaker embedding tensor. It is ignored for Kokoro backend.
     * @param properties Speech generation parameters specified as properties
     * @returns raw audio for a single synthesized speech; sample rate is provided in
     * `Text2SpeechDecodedResults::output_sample_rate`
     */
    Text2SpeechDecodedResults generate_from_phonemes(const std::vector<std::string>& phoneme_chunks,
                                                     const ov::Tensor& speaker_embedding = ov::Tensor(),
                                                     const ov::AnyMap& properties = {});

    /**
     * Generates speech from precomputed Kokoro phoneme chunks for multiple inputs.
      *
      * NOTE: This API is supported only for Kokoro backend. SpeechT5 backend throws an exception.
     *
     * @param phoneme_chunks nested list where each item is a list of phoneme chunks for one output speech
      * @param speaker_embedding Optional speaker embedding tensor. It is ignored for Kokoro backend.
     * @param properties Speech generation parameters specified as properties
     * @returns raw audios for synthesized speeches; sample rate is provided in
     * `Text2SpeechDecodedResults::output_sample_rate`
     */
    Text2SpeechDecodedResults generate_from_phonemes(const std::vector<std::vector<std::string>>& phoneme_chunks,
                                                     const ov::Tensor& speaker_embedding = ov::Tensor(),
                                                     const ov::AnyMap& properties = {});

    /**
     * Generates speech from precomputed token stream for a single input.
      *
      * NOTE: This API is supported only for Kokoro backend. SpeechT5 backend throws an exception.
     *
     * @param tokens token sequence used to build Kokoro phoneme chunks
      * @param speaker_embedding Optional speaker embedding tensor. It is ignored for Kokoro backend.
     * @param properties Speech generation parameters specified as properties
     */
    Text2SpeechDecodedResults generate_from_tokens(const std::vector<SpeechToken>& tokens,
                                                   const ov::Tensor& speaker_embedding = ov::Tensor(),
                                                   const ov::AnyMap& properties = {});

    /**
     * Generates speech from precomputed token streams for multiple inputs.
      *
      * NOTE: This API is supported only for Kokoro backend. SpeechT5 backend throws an exception.
     *
     * @param token_batches nested list where each item is a token sequence for one output speech
      * @param speaker_embedding Optional speaker embedding tensor. It is ignored for Kokoro backend.
     * @param properties Speech generation parameters specified as properties
     */
    Text2SpeechDecodedResults generate_from_tokens(const std::vector<std::vector<SpeechToken>>& token_batches,
                                                   const ov::Tensor& speaker_embedding = ov::Tensor(),
                                                   const ov::AnyMap& properties = {});

    /**
     * Runs text preprocessing/phonemization and returns Kokoro phoneme chunks for one input text.
     *
     * NOTE: This API is temporary and currently exposed only for debugging/parity analysis
     * between misaki-cpp and Python Misaki preprocessing behavior.
     *
     * @param text input text to phonemize
     * @param properties Speech generation parameters specified as properties
     * @returns phoneme chunks used by Kokoro backend prior to acoustic inference
     */
    std::vector<std::string> phonemize(const std::string& text,
                                       const ov::AnyMap& properties = {});

    /**
     * Runs text preprocessing/phonemization and returns Kokoro phoneme chunks for each input text.
     *
     * NOTE: This API is temporary and currently exposed only for debugging/parity analysis
     * between misaki-cpp and Python Misaki preprocessing behavior.
     *
     * @param texts input texts to phonemize
     * @param properties Speech generation parameters specified as properties
     * @returns a list of per-input phoneme chunk lists
     */
    std::vector<std::vector<std::string>> phonemize(const std::vector<std::string>& texts,
                                                    const ov::AnyMap& properties = {});

    template <typename... Properties>
    Text2SpeechDecodedResults generate(const std::vector<std::string>& texts,
                                       const ov::Tensor& speaker_embedding = ov::Tensor(),
                                       Properties&&... properties) {
        return generate(texts, speaker_embedding, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    template <typename... Properties>
    Text2SpeechDecodedResults generate_from_phonemes(const std::vector<std::string>& phoneme_chunks,
                                                     const ov::Tensor& speaker_embedding = ov::Tensor(),
                                                     Properties&&... properties) {
        return generate_from_phonemes(phoneme_chunks, speaker_embedding, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    template <typename... Properties>
    Text2SpeechDecodedResults generate_from_phonemes(const std::vector<std::vector<std::string>>& phoneme_chunks,
                                                     const ov::Tensor& speaker_embedding = ov::Tensor(),
                                                     Properties&&... properties) {
        return generate_from_phonemes(phoneme_chunks, speaker_embedding, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    template <typename... Properties>
    Text2SpeechDecodedResults generate_from_tokens(const std::vector<SpeechToken>& tokens,
                                                   const ov::Tensor& speaker_embedding = ov::Tensor(),
                                                   Properties&&... properties) {
        return generate_from_tokens(tokens, speaker_embedding, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    template <typename... Properties>
    Text2SpeechDecodedResults generate_from_tokens(const std::vector<std::vector<SpeechToken>>& token_batches,
                                                   const ov::Tensor& speaker_embedding = ov::Tensor(),
                                                   Properties&&... properties) {
        return generate_from_tokens(token_batches, speaker_embedding, ov::AnyMap{std::forward<Properties>(properties)...});
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
