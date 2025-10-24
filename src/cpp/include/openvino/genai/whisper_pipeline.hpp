// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <variant>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/whisper_generation_config.hpp"

namespace ov {
namespace genai {

using OptionalWhisperGenerationConfig = std::optional<WhisperGenerationConfig>;

using RawSpeechInput = std::vector<float>;

/**
 * Base class for chunk streamers. In order to use inherit from from this class and implement put, and methods
 */
class OPENVINO_DEPRECATED(
    "ChunkStreamerBase is deprecated and will be removed in 2026.0.0 release. Use StreamerBase instead.")
    OPENVINO_GENAI_EXPORTS ChunkStreamerBase : public StreamerBase {
public:
    /// @brief put_chunk is called every time new token chunk is generated,
    /// @return bool flag to indicate whether generation should be stopped, if return true generation stops
    OPENVINO_DEPRECATED("ChunkStreamerBase is deprecated and will be removed in 2026.0.0 "
                        "release. Use StreamerBase instead.")
    virtual bool put_chunk(std::vector<int64_t> tokens) {
        OPENVINO_THROW(
            "ChunkStreamerBase is deprecated and will be removed in 2026.0.0 release. Use StreamerBase instead.");
        return true;
    };

    /// @brief put is called every time new token is decoded. Deprecated. Please, use write instead.
    /// @return bool flag to indicate whether generation should be stopped, if return true generation stops
    OPENVINO_DEPRECATED(
        "ChunkStreamerBase is deprecated and will be removed in 2026.0.0 release. Use StreamerBase instead.")
    bool put(int64_t token) override {
        OPENVINO_THROW(
            "ChunkStreamerBase is deprecated and will be removed in 2026.0.0 release. Use StreamerBase instead.");
        return true;
    };

    /// @brief write is called every time new vector of tokens is decoded, in case of assisting or prompt lookup
    /// decoding
    /// @return StreamingStatus flag to indicate whether generation should be countinue to run or stopped or cancelled
    OPENVINO_DEPRECATED(
        "ChunkStreamerBase is deprecated and will be removed in 2026.0.0 release. Use StreamerBase instead.")
    StreamingStatus write(const std::vector<int64_t>& tokens) override {
        OPENVINO_SUPPRESS_DEPRECATED_START
        return put_chunk(tokens) ? StreamingStatus::STOP : StreamingStatus::RUNNING;
        OPENVINO_SUPPRESS_DEPRECATED_END
    };

    /// @brief write is called every time new token is decoded
    /// @return StreamingStatus flag to indicate whether generation should be countinue to run or stopped or cancelled
    OPENVINO_DEPRECATED(
        "ChunkStreamerBase is deprecated and will be removed in 2026.0.0 release. Use StreamerBase instead.")
    StreamingStatus write(int64_t token) override {
        OPENVINO_SUPPRESS_DEPRECATED_START
        return put(token) ? StreamingStatus::STOP : StreamingStatus::RUNNING;
        OPENVINO_SUPPRESS_DEPRECATED_END
    };

    /// @brief end is called at the end of generation. It can be used to flush cache if your own streamer has one
    OPENVINO_DEPRECATED(
        "ChunkStreamerBase is deprecated and will be removed in 2026.0.0 release. Use StreamerBase instead.")
    virtual void end() override = 0;

    ~ChunkStreamerBase() override;
};

struct WhisperRawPerfMetrics {
    /** @brief Duration for each features extraction call */
    std::vector<MicroSeconds> features_extraction_durations;
};

struct OPENVINO_GENAI_EXPORTS WhisperPerfMetrics : public PerfMetrics {
    /** @brief Mean and standard deviation of Features Extraction Duration in milliseconds */
    MeanStdPair features_extraction_duration;

    MeanStdPair get_features_extraction_duration();

    WhisperPerfMetrics() = default;

    WhisperPerfMetrics(PerfMetrics& perf_metrics) : PerfMetrics(perf_metrics), features_extraction_duration(){};

    void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt) override;

    WhisperPerfMetrics operator+(const WhisperPerfMetrics& metrics) const;
    WhisperPerfMetrics& operator+=(const WhisperPerfMetrics& right);

    WhisperRawPerfMetrics whisper_raw_metrics;
};

struct WhisperDecodedResultChunk {
    // start of chunk in seconds
    float start_ts;

    // end of chunk in seconds
    // -1.0f if chunk started but model did not predict an ending timestamp
    // can happen if audio is cut off in the middle of a word
    float end_ts = -1.0f;
    std::string text;
};

struct WhisperDecodedResults {
    std::vector<std::string> texts;
    std::vector<float> scores;
    std::optional<std::vector<WhisperDecodedResultChunk>> chunks = std::nullopt;
    WhisperPerfMetrics perf_metrics;

    operator std::string() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    operator std::vector<std::string>() const {
        return texts;
    }

    friend std::ostream& operator<<(std::ostream& os, const WhisperDecodedResults& dr) {
        OPENVINO_ASSERT(dr.scores.size() == dr.texts.size(),
                        "The number of scores and texts doesn't match in WhisperDecodedResults.");
        if (dr.texts.empty()) {
            return os;
        }
        if (dr.texts.size() == 1) {
            os << dr.texts[0];
            return os;
        }
        for (size_t i = 0; i < dr.texts.size() - 1; ++i) {
            os << std::to_string(dr.scores[i]) << ": " << dr.texts[i] << '\n';
        }
        return os << std::to_string(dr.scores.back()) << ": " << dr.texts.back();
    }
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

OPENVINO_GENAI_EXPORTS std::pair<std::string, Any> generation_config(const WhisperGenerationConfig& config);
}  // namespace genai
}  // namespace ov
