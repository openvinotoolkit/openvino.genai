// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>

#include "openvino/genai/automatic_speech_recognition/generation_config.hpp"
#include "openvino/genai/automatic_speech_recognition/perf_metrics.hpp"
#include "openvino/genai/llm_pipeline.hpp"

namespace ov::genai {

// duplicates whisper definition
// todo: address input type std::vector<float> vs ov::Tensor open
// todo: deprecate WhisperPipeline or move to a separate header
using RawSpeechInput = std::vector<float>;

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
    std::string language;
    ASRPerfMetrics perf_metrics;

    std::optional<std::vector<ASRDecodedResultChunk>> chunks = std::nullopt;
    std::optional<std::vector<ASRDecodedResultChunk>> words = std::nullopt;

    // is it really needed?
    operator std::string() const;
    // is it really needed?
    operator std::vector<std::string>() const;

    friend std::ostream& operator<<(std::ostream& os, const ASRDecodedResults& dr);
};

class OPENVINO_GENAI_EXPORTS ASRPipeline {
    class ASRPipelineImplBase;
    class WhisperASRPipelineAdapter;
    std::unique_ptr<ASRPipelineImplBase> m_impl;

public:
    ASRPipeline(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    ASRPipeline(const std::filesystem::path& models_path, const std::string& device, Properties&&... properties)
        : ASRPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    ~ASRPipeline();

    ASRDecodedResults generate(const RawSpeechInput& raw_speech_input,
                               std::optional<ASRGenerationConfig> generation_config = std::nullopt,
                               StreamerVariant streamer = std::monostate());

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ASRDecodedResults, Properties...> generate(const RawSpeechInput& raw_speech_input,
                                                                              Properties&&... properties) {
        return generate(raw_speech_input, AnyMap{std::forward<Properties>(properties)...});
    }

    ASRDecodedResults generate(const RawSpeechInput& raw_speech_input, const ov::AnyMap& config_map);

    Tokenizer get_tokenizer();
    ASRGenerationConfig get_generation_config() const;
    void set_generation_config(const ASRGenerationConfig& config);
};

}  // namespace ov::genai
