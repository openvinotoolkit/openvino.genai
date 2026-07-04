// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/generation_config.hpp"

namespace ov::genai {

class OPENVINO_GENAI_EXPORTS ASRGenerationConfig : public GenerationConfig {
public:
    ASRGenerationConfig();
    explicit ASRGenerationConfig(const std::filesystem::path& json_path);

    /**
     * @brief Language token to use for generation
     * In the form of <|en|> for Whisper models. Can be set for multilingual models only.
     * In the form of English for Qwen3-ASR models.
     */
    std::optional<std::string> language = std::nullopt;

    // Whether to return segment-level timestamps.
    bool return_timestamps = false;

    // Whisper parameters

    int64_t decoder_start_token_id = 50258;
    int64_t pad_token_id = 50257;
    int64_t translate_token_id = 50358;
    int64_t transcribe_token_id = 50359;
    int64_t prev_sot_token_id = 50361;
    int64_t no_timestamps_token_id = 50363;

    // Token suppression
    std::vector<int64_t> begin_suppress_tokens;
    std::vector<int64_t> suppress_tokens;

    size_t max_initial_timestamp_index = 50;

    bool is_multilingual = true;

    // Task to use for generation, either “translate” or “transcribe”.
    // Can be set for multilingual models only.
    std::optional<std::string> task = std::nullopt;

    // Language token to token_id map. Initialized from the generation_config.json lang_to_id dictionary.
    std::map<std::string, int64_t> lang_to_id;

    // If `true` the pipeline will return word-level timestamps.
    // When enabled ov::genai::word_timestamps(true) property should be passed to ASRPipeline constructor:
    // ASRPipeline("model_path", "CPU", ov::genai::word_timestamps(true));
    bool word_timestamps = false;

    // Encoder attention alignment heads used for word-level timestamps prediction.
    // Each pair represents (layer_index, head_index).
    std::vector<std::pair<size_t, size_t>> alignment_heads;

    /*
     * Initial prompt tokens passed as a previous transcription (after `<|startofprev|>` token) to the first processing
     * window. Can be used to steer the model to use particular spellings or styles.
     *
     * Example:
     *  auto result = pipeline.generate(raw_speech);
     *  //  He has gone and gone for good answered Paul Icrom who...
     *
     *  auto result = pipeline.generate(raw_speech, ov::genai::initial_prompt("Polychrome"));
     *  //  He has gone and gone for good answered Polychrome who...
     */
    std::optional<std::string> initial_prompt = std::nullopt;

    /*
     * Hotwords tokens passed as a previous transcription (after `<|startofprev|>` token) to the all processing windows.
     * Can be used to steer the model to use particular spellings or styles.
     *
     * Example:
     *  auto result = pipeline.generate(raw_speech);
     *  //  He has gone and gone for good answered Paul Icrom who...
     *
     *  auto result = pipeline.generate(raw_speech, ov::genai::hotwords("Polychrome"));
     *  //  He has gone and gone for good answered Polychrome who...
     */
    std::optional<std::string> hotwords = std::nullopt;

    // Qwen3-ASR parameters

    std::optional<std::string> context = std::nullopt;

    using GenerationConfig::update_generation_config;
    void update_generation_config(const ov::AnyMap& config_map = {}) override;

    void validate() const override;
};

OPENVINO_GENAI_EXPORTS std::pair<std::string, ov::Any> generation_config(const ASRGenerationConfig& config);

/*
 * utils that allow to use generate and operator() in the following way:
 * pipe.generate(raw_speech, ov::genai::max_new_tokens(200), ov::genai::return_timestamps(true), ...)
 */

static constexpr ov::Property<std::string> language{"language"};
static constexpr ov::Property<bool> return_timestamps{"return_timestamps"};

static constexpr ov::Property<int64_t> decoder_start_token_id{"decoder_start_token_id"};
static constexpr ov::Property<int64_t> pad_token_id{"pad_token_id"};
static constexpr ov::Property<int64_t> translate_token_id{"translate_token_id"};
static constexpr ov::Property<int64_t> transcribe_token_id{"transcribe_token_id"};
static constexpr ov::Property<int64_t> prev_sot_token_id{"prev_sot_token_id"};
static constexpr ov::Property<int64_t> no_timestamps_token_id{"no_timestamps_token_id"};
static constexpr ov::Property<std::vector<int64_t>> begin_suppress_tokens{"begin_suppress_tokens"};
static constexpr ov::Property<std::vector<int64_t>> suppress_tokens{"suppress_tokens"};
static constexpr ov::Property<std::string> task{"task"};
static constexpr ov::Property<std::map<std::string, int64_t>> lang_to_id{"lang_to_id"};
static constexpr ov::Property<bool> is_multilingual{"is_multilingual"};
static constexpr ov::Property<size_t> max_initial_timestamp_index{"max_initial_timestamp_index"};
static constexpr ov::Property<bool> word_timestamps{"word_timestamps"};
static constexpr ov::Property<std::vector<std::pair<size_t, size_t>>> alignment_heads{"alignment_heads"};
static constexpr ov::Property<std::string> initial_prompt{"initial_prompt"};
static constexpr ov::Property<std::string> hotwords{"hotwords"};
static constexpr ov::Property<std::string> context{"context"};

}  // namespace ov::genai
