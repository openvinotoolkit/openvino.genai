// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include "nlohmann/json.hpp"

#include "generation_config.hpp"

#include "openvino/core/except.hpp"

void GenerationConfig::set_eos_token_id(size_t tokenizer_eos_token_id) {
    if (eos_token_id < 0) {
        eos_token_id = tokenizer_eos_token_id;
    } else {
        OPENVINO_ASSERT(eos_token_id == tokenizer_eos_token_id,
            "EOS token ID is different in generation config (", eos_token_id, ") and tokenizer (",
            tokenizer_eos_token_id, ")");
    }
}

void GenerationConfig::validate() const {
    OPENVINO_ASSERT(min_new_tokens <= max_new_tokens, "min_new_tokens must be less or equal max_new_tokens");
    OPENVINO_ASSERT(min_new_tokens >= 0, "min_new_tokens must be greater 0");
    OPENVINO_ASSERT(max_new_tokens >= 0, "max_new_tokens must be greater 0");
    if (is_beam_search()) {
        OPENVINO_ASSERT(no_repeat_ngram_size > 0, "no_repeat_ngram_size must be positive");
    } else {
        OPENVINO_ASSERT(repetition_penalty >= 0.0f, "repetition penalty must be a positive value");
        OPENVINO_ASSERT(frequence_penalty >= -2.0f && frequence_penalty <= 2.0f, "frequence_penalty penalty must be a [-2; +2]");
        OPENVINO_ASSERT(presence_penalty >= -2.0f && presence_penalty <= 2.0f, "presence_penalty penalty must be a [-2; +2]");
        if (is_multinomial()) {
            OPENVINO_ASSERT(top_p > 0.0f && top_p <= 1.0f, "top_p must be in the interval (0, 1]");
            OPENVINO_ASSERT(temperature >= 0.0f, "temperature must be a positive value");
        }
    }
}

GenerationConfig GenerationConfig::from_file(const std::string& generation_config_json) {
    std::ifstream f(generation_config_json);
    nlohmann::json json_data = nlohmann::json::parse(f);

    GenerationConfig config;

    config.bos_token_id = json_data.value("bos_token_id", -1);
    config.eos_token_id = json_data.value("eos_token_id", -1);
    config.pad_token_id = json_data.value("pad_token_id", -1);

    config.num_return_sequences = json_data.value("num_return_sequences", 1);

    config.max_new_tokens = json_data.value("max_new_tokens", std::numeric_limits<size_t>::max());
    config.min_new_tokens = json_data.value("min_new_tokens", 0);
    config.max_length = json_data.value("max_length", std::numeric_limits<size_t>::max());

    config.temperature = json_data.value("temperature", 0.0f);
    config.do_sample = json_data.value("do_sample", false);
    config.top_p = json_data.value("top_p", 0.0f);

    // beam_search_params
    config.num_groups = json_data.value("num_beam_groups", 1);
    config.diversity_penalty = json_data.value("diversity_penalty", 1.0f);
    config.repetition_penalty = json_data.value("repetition_penalty", 1.0f);
    config.frequence_penalty = json_data.value("frequence_penalty", 0.0f);
    config.presence_penalty = json_data.value("presence_penalty", 0.0f);
    const int num_beams = json_data.value("num_beams", 1);
    config.group_size = num_beams / config.num_groups;

    return config;
}

GenerationConfig GenerationConfig::greedy() {
    GenerationConfig greedy_params;
    greedy_params.temperature = 0.0f;
    greedy_params.ignore_eos = true;
    greedy_params.num_return_sequences = 1;
    greedy_params.repetition_penalty = 3.0f;
    greedy_params.presence_penalty = 0.1f;
    greedy_params.frequence_penalty = 0.01f;
    greedy_params.max_new_tokens = 30;
    return greedy_params;
}

GenerationConfig GenerationConfig::beam_search() {
    GenerationConfig beam_search;
    beam_search.num_groups = 2;
    beam_search.num_return_sequences = 3;
    beam_search.group_size = 2;
    beam_search.max_new_tokens = 100;
    beam_search.diversity_penalty = 2.0f;
    return beam_search;
}

GenerationConfig GenerationConfig::multinomial() {
    GenerationConfig multinomial;
    multinomial.do_sample = true;
    multinomial.temperature = 0.9f;
    multinomial.top_p = 0.9f;
    multinomial.top_k = 20;
    multinomial.num_return_sequences = 3;
    multinomial.presence_penalty = 0.01f;
    multinomial.frequence_penalty = 0.1f;
    multinomial.min_new_tokens = 15;
    multinomial.max_new_tokens = 30;
    return multinomial;
}
