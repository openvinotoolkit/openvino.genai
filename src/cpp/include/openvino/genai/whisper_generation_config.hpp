// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/tokenizer.hpp"
#include "openvino/runtime/compiled_model.hpp"

namespace ov {
namespace genai {

/**
 * @brief Structure to keep generation config parameters.
 *
 * Generic parameters:
 * @param max_length the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
 *        `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
 * @param max_new_tokens the maximum numbers of tokens to generate, excluding the number of tokens in the prompt.
 * max_new_tokens has priority over max_length.
 * @param ignore_eos if set to true, then generation will not stop even if <eos> token is met.
 * @param eos_token_id token_id of <eos> (end of sentence)
 */
class OPENVINO_GENAI_EXPORTS WhisperGenerationConfig {
public:
    // todo: default constructor to be removed due to supress tokens initialization
    // or set default supress tokens
    WhisperGenerationConfig() = default;
    explicit WhisperGenerationConfig(const std::string& json_path);

    // Generic
    size_t max_new_tokens = SIZE_MAX;
    size_t max_length = SIZE_MAX;

    // Whisper specific
    std::vector<int64_t> begin_suppress_tokens;
    std::vector<int64_t> suppress_tokens;

    int64_t decoder_start_token_id = 50258;     // "<|startoftranscript|>"
    int64_t language_token_id = 50259;          // "<|en|>"
    int64_t eos_token_id = 50257;               // "<|endoftext|>"
    int64_t pad_token_id = 50257;               // "<|endoftext|>"
    int64_t translate_token_id = 50358;         // "<|translate|>"
    int64_t transcribe_token_id = 50359;        // "<|transcribe|>"
    int64_t no_timestamps_token_id = 50363;     // "<|notimestamps|>"
    int64_t begin_timestamps_token_id = 50364;  // "<|0.00|>"

    /** @brief sets eos_token_id to tokenizer_eos_token_id if eos_token_id is less than 0.
     * Otherwise verifies eos_token_id == tokenizer_eos_token_id.
     */
    void set_eos_token_id(size_t tokenizer_eos_token_id);
    size_t get_max_new_tokens(size_t prompt_length = 0) const;

    void update_generation_config(const ov::AnyMap& config_map = {});

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> update_generation_config(Properties&&... properties) {
        return update_generation_config(AnyMap{std::forward<Properties>(properties)...});
    }

    /// @brief checks that are no conflicting parameters, e.g. do_sample=true and num_beams > 1.
    /// @throws Exception if config is invalid.
    void validate() const;
};

/*
 * utils that allow to use generate and operator() in the following way:
 * pipe.generate(input_ids, ov::genai::max_new_tokens(200), ov::genai::temperature(1.0f),...)
 * pipe(text, ov::genai::max_new_tokens(200), ov::genai::temperature(1.0f),...)
 */

static constexpr ov::Property<std::vector<size_t>> begin_suppress_tokens{"begin_suppress_tokens"};
static constexpr ov::Property<std::vector<size_t>> suppress_tokens{"suppress_tokens"};
static constexpr ov::Property<size_t> decoder_start_token_id{"decoder_start_token_id"};

static constexpr ov::Property<size_t> pad_token_id{"pad_token_id"};
static constexpr ov::Property<size_t> transcribe_token_id{"transcribe_token_id"};
static constexpr ov::Property<size_t> translate_token_id{"translate_token_id"};
static constexpr ov::Property<size_t> no_timestamps_token_id{"no_timestamps_token_id"};
static constexpr ov::Property<size_t> begin_timestamps_token_id{"begin_timestamps_token_id"};

}  // namespace genai
}  // namespace ov
