// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "openvino/genai/automatic_speech_recognition/generation_config.hpp"
#include "openvino/genai/automatic_speech_recognition/perf_metrics.hpp"
#include "openvino/genai/llm_pipeline.hpp"

namespace ov::genai {

/// One waveform (`std::vector<float>`) or a batch of independent waveforms
/// (`std::vector<std::vector<float>>`). Support for B > 1 depends on the selected model.
using AudioInputs = std::variant<std::vector<float>, std::vector<std::vector<float>>>;

/// Time-aligned text chunk — used for both segment-level and word-level timestamps.
/// For segments: text is the decoded segment, token_ids are the tokens in that segment.
/// For words: text is the individual word, token_ids are the tokens composing that word.
struct ASRDecodedResultChunk {
    float start_ts;  // start of chunk in seconds
    float end_ts;    // end of chunk in seconds (-1.0f if not predicted by model)
    std::string text;
    std::vector<int64_t> token_ids;
};

struct OPENVINO_GENAI_EXPORTS ASRDecodedResults {
    std::vector<std::string> texts;
    std::vector<float> scores;
    /**
     * @brief Language associated with each transcription.
     *
     * Each entry contains the detected language when the model supports language identification, or the requested
     * language when generation forces one. An entry is empty when no language was requested and the model does not
     * support language identification, as with Fun-ASR.
     */
    std::vector<std::string> languages;
    ASRPerfMetrics perf_metrics;

    std::optional<std::vector<std::vector<ASRDecodedResultChunk>>> chunks = std::nullopt;
    std::optional<std::vector<std::vector<ASRDecodedResultChunk>>> words = std::nullopt;

    operator std::string() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    };

    operator std::vector<std::string>() const {
        return texts;
    };

    friend std::ostream& operator<<(std::ostream& os, const ASRDecodedResults& dr) {
        OPENVINO_ASSERT(dr.scores.size() == dr.texts.size(), "The number of scores and texts doesn't match.");
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

class ASRPipelineImplBase;

class OPENVINO_GENAI_EXPORTS ASRPipeline {
    std::unique_ptr<ASRPipelineImplBase> m_impl;

public:
    ASRPipeline(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    ASRPipeline(const std::filesystem::path& models_path, const std::string& device, Properties&&... properties)
        : ASRPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    ~ASRPipeline();

    ASRDecodedResults generate(const AudioInputs& audio_inputs,
                               const std::optional<ASRGenerationConfig>& generation_config = std::nullopt,
                               StreamerVariant streamer = std::monostate());

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ASRDecodedResults, Properties...> generate(const AudioInputs& audio_inputs,
                                                                              Properties&&... properties) {
        return generate(audio_inputs, AnyMap{std::forward<Properties>(properties)...});
    }

    ASRDecodedResults generate(const AudioInputs& audio_inputs, const ov::AnyMap& config_map);

    Tokenizer get_tokenizer();
    ASRGenerationConfig get_generation_config() const;
    void set_generation_config(const ASRGenerationConfig& config);
};

}  // namespace ov::genai
