// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include "nlohmann/json.hpp"

#include "generation_config.hpp"

GenerationConfig GenerationConfig::from_file(const std::string& generation_config_json) {
    std::ifstream f(generation_config_json);
    nlohmann::json json_data = nlohmann::json::parse(f);

    GenerationConfig config;

    config.bos_token_id = json_data.value("bos_token_id", -1);
    config.eos_token_id = json_data.value("eos_token_id", -1);
    config.pad_token_id = json_data.value("pad_token_id", -1);

    config.num_return_sequences = json_data.value("num_return_sequences", 1);

    config.max_new_tokens = json_data.value("max_new_tokens", std::numeric_limits<size_t>::max());
    config.max_length = json_data.value("max_length", std::numeric_limits<size_t>::max());

    config.temperature = json_data.value("temperature", 0.0f);
    config.do_sample = json_data.value("do_sample", false);
    config.top_p = json_data.value("top_p", 0.0f);

    // beam_search_params
    config.num_groups = json_data.value("num_beam_groups", 1);
    config.diversity_penalty = json_data.value("diversity_penalty", 1.0f);
    const int num_beams = json_data.value("num_beams", 1);
    config.group_size = num_beams / config.num_groups;

    return config;
}

GenerationConfig GenerationConfig::greedy() {
    GenerationConfig greedy_params;
    greedy_params.temperature = 0.0f;
    greedy_params.ignore_eos = true;
    greedy_params.num_return_sequences = 1;
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
    return multinomial;
}
