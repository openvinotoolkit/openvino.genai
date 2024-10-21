// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>
#include <optional>
#include <variant>

#include "openvino/core/any.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/whisper_generation_config.hpp"

namespace ov::genai {

using OptionalWhisperGenerationConfig = std::optional<WhisperGenerationConfig>;

using RawSpeechInput = std::vector<float>;

struct WhisperDecodedResultChunk {
    // start of chunk in seconds
    float start_ts;

    // end of chunk in seconds
    // -1.0f if chunk started but model did not predict an ending timestamp
    // can happen if audio is cut off in the middle of a word
    float end_ts = -1.0f;
    std::string text;
};

struct WhisperDecodedResults : public DecodedResults {
    std::optional<std::vector<WhisperDecodedResultChunk>> chunks = std::nullopt;
};

class OPENVINO_GENAI_EXPORTS WhisperPipeline {
    class Impl;
    std::unique_ptr<Impl> m_impl;

public:
    /**
     * @brief Constructs an WhisperPipeline from xml/bin files, tokenizers and configuration in the
     * same dir.
     *
     * @param model_path Path to the dir model xml/bin files, tokenizers and generation_configs.json
     * @param device optional device
     * @param plugin_config optional plugin_config
     */
    WhisperPipeline(const std::string& model_path,
                    const std::string& device,
                    const ov::AnyMap& plugin_config = {});

    /**
     * @brief Constructs an WhisperPipeline from xml/bin files, tokenizers and configuration in the
     * same dir. Accepts arbitrary list of optional properties.
     *
     * @param model_path Path to the dir model xml/bin files, tokenizers and generation_configs.json
     * @param device optional device
     * @param device_config optional device_config
     */
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    WhisperPipeline(const std::string& root_dir,
                    const std::string& device,
                    Properties&&... device_config)
        : WhisperPipeline(root_dir, device, ov::AnyMap{std::forward<Properties>(device_config)...}) { }

    ~WhisperPipeline();

    /**
     * @brief High level generate that receives raw speech as a vector of floats and returns decoded output.
     *
     * @param raw_speech_input raw speech input. Required to be normalized to near [-1, 1] range and have 16k Hz
     * sampling rate.
     * @param generation_config optional GenerationConfig
     * @param streamer optional streamer
     * @return WhisperDecodedResults decoded resulting text transcription
     */
    WhisperDecodedResults generate(const RawSpeechInput& raw_speech_input,
                                   OptionalWhisperGenerationConfig generation_config = std::nullopt,
                                   StreamerVariant streamer = std::monostate());

    /**
     * @brief High level generate that receives raw speech as a vector of floats and returns decoded output.
     * properties can be in any order pipe.generate(..., ov::genai::max_new_tokens(100),
     * ov::genai::streamer(lambda_func)).
     *
     * @param raw_speech_input raw speech input
     * @param properties properties
     * @return WhisperDecodedResults decoded resulting text transcription
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<WhisperDecodedResults, Properties...> generate(const RawSpeechInput& raw_speech_input,
                                                                              Properties&&... properties) {
        return generate(raw_speech_input, AnyMap{std::forward<Properties>(properties)...});
    }
    WhisperDecodedResults generate(const RawSpeechInput& raw_speech_input, const ov::AnyMap& config_map);

    ov::genai::Tokenizer get_tokenizer();
    WhisperGenerationConfig get_generation_config() const;
    void set_generation_config(const WhisperGenerationConfig& config);
};
}  // namespace ov::genai
