// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "context_tokens.hpp"

namespace {
std::pair<std::vector<int64_t>, float> tokenize(std::string&& text,
                                                const ov::genai::WhisperGenerationConfig& config,
                                                ov::genai::Tokenizer& tokenizer) {
    if (text.empty()) {
        return {{}, 0.0f};
    }

    auto start_time = std::chrono::steady_clock::now();
    auto encoded = tokenizer.encode(text, ov::genai::add_special_tokens(false));
    auto duration = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start_time);

    auto input_ids = encoded.input_ids;
    auto input_ids_data = input_ids.data<int64_t>();

    std::vector<int64_t> prompt_tokens;
    prompt_tokens.reserve(input_ids.get_size());

    // even with ov::genai::add_special_tokens(false) tokenizer adds next special tokens. Ticket: 159569
    std::set<int64_t> special_tokens{config.decoder_start_token_id, config.eos_token_id, config.no_timestamps_token_id};

    for (size_t i = 0; i < input_ids.get_size(); i++) {
        if (special_tokens.count(input_ids_data[i])) {
            continue;
        }

        prompt_tokens.emplace_back(input_ids_data[i]);
    }

    return {prompt_tokens, duration};
}
}  // namespace

namespace ov {
namespace genai {

std::pair<WhisperContextTokens, float> prepare_context_tokens(const WhisperGenerationConfig& config,
                                                              Tokenizer& tokenizer) {
    WhisperContextTokens context_tokens;
    float duration = 0.0f;

    if (config.initial_prompt.has_value()) {
        auto [initial_prompt_tokens, initial_prompt_duration] =
            tokenize(" " + *config.initial_prompt, config, tokenizer);
        context_tokens.initial_prompt = std::move(initial_prompt_tokens);
        duration += initial_prompt_duration;
    }

    if (config.hotwords.has_value()) {
        auto [hotwords_tokens, hotwords_duration] = tokenize(" " + *config.hotwords, config, tokenizer);
        context_tokens.hotwords = std::move(hotwords_tokens);
        duration += hotwords_duration;
    }

    return {context_tokens, duration};
}

std::vector<int64_t> get_prompt_tokens(const WhisperContextTokens& context_tokens,
                                       const WhisperGenerationConfig& config,
                                       size_t chunk_offset) {
    bool should_add_initial_prompt = !context_tokens.initial_prompt.empty() && chunk_offset == 0;
    bool should_add_hotwords = !context_tokens.hotwords.empty();

    if (!should_add_initial_prompt && !should_add_hotwords) {
        return {};
    }

    std::vector<int64_t> prompt_tokens{config.prev_sot_token_id};

    if (should_add_initial_prompt) {
        prompt_tokens.insert(prompt_tokens.end(),
                             context_tokens.initial_prompt.begin(),
                             context_tokens.initial_prompt.end());
    }

    if (should_add_hotwords) {
        prompt_tokens.insert(prompt_tokens.end(), context_tokens.hotwords.begin(), context_tokens.hotwords.end());
    }

    return prompt_tokens;
}

}  // namespace genai
}  // namespace ov
