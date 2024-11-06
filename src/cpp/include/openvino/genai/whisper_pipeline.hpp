// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <variant>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/whisper_generation_config.hpp"

namespace ov::genai {

using OptionalWhisperGenerationConfig = std::optional<WhisperGenerationConfig>;

using RawSpeechInput = std::vector<float>;

/**
 * @brief base class for chunk streamers. In order to use inherit from from this class and implement put, and methods
 *
 * @param m_tokenizer tokenizer
 */
class OPENVINO_GENAI_EXPORTS ChunkStreamerBase : public StreamerBase {
public:
    /// @brief put is called every time new token chunk is generated,
    /// @return bool flag to indicate whether generation should be stopped, if return true generation stops
    virtual bool put_chunk(std::vector<int64_t> tokens) = 0;
};

// Return flag corresponds whether generation should be stopped: false means continue generation, true means stop.
using ChunkStreamerVariant =
    std::variant<std::function<bool(std::string)>, std::shared_ptr<ChunkStreamerBase>, std::monostate>;

struct WhisperDecodedResultChunk {
    // start of chunk in seconds
    float start_ts;

    // end of chunk in seconds
    // -1.0f if chunk started but model did not predict an ending timestamp
    // can happen if audio is cut off in the middle of a words
    float end_ts = -1.0f;
    std::string text;
};

struct WhisperDecodedResults : public DecodedResults {
    std::optional<std::vector<WhisperDecodedResultChunk>> chunks = std::nullopt;
};

/**
 * @brief Automatic speech recognition pipeline
 */
class OPENVINO_GENAI_EXPORTS WhisperPipeline {
    class WhisperPipelineImplBase;
    std::unique_ptr<WhisperPipelineImplBase> m_impl;

    class StaticWhisperPipeline;
    class WhisperPipelineStatefulImpl;

public:
    /**
     * @brief Constructs a WhisperPipeline from xml/bin files, tokenizers and configuration in the
     * same dir.
     *
     * @param models_path Path to the dir model xml/bin files, tokenizers and generation_configs.json
     * @param device optional device
     * @param properties optional properties
     */
    WhisperPipeline(const std::filesystem::path& models_path,
                    const std::string& device,
                    const ov::AnyMap& properties = {});

    /**
     * @brief Constructs a WhisperPipeline from xml/bin files, tokenizers and configuration in the
     * same dir. Accepts arbitrary list of optional properties.
     *
     * @param models_path Path to the dir model xml/bin files, tokenizers and generation_configs.json
     * @param device optional device
     * @param properties optional properties
     */
    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    WhisperPipeline(const std::filesystem::path& models_path, const std::string& device, Properties&&... properties)
        : WhisperPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    ~WhisperPipeline();

    /**
     * @brief High level generate that receives raw speech as a vector of floats and returns decoded output.
     *
     * @param raw_speech_input raw speech input. Required to be normalized to near [-1, 1] range and have 16k Hz
     * sampling rate.
     * @param generation_config optional GenerationConfig
     * @param streamer optional streamer. Streamer supported for short-form audio (< 30 seconds) with
     * `return_timestamps=False` only
     * @return WhisperDecodedResults decoded resulting text transcription
     */
    WhisperDecodedResults generate(const RawSpeechInput& raw_speech_input,
                                   OptionalWhisperGenerationConfig generation_config = std::nullopt,
                                   ChunkStreamerVariant streamer = std::monostate());

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

OPENVINO_GENAI_EXPORTS std::pair<std::string, Any> streamer(ChunkStreamerVariant func);
OPENVINO_GENAI_EXPORTS std::pair<std::string, Any> generation_config(const WhisperGenerationConfig& config);
}  // namespace ov::genai
