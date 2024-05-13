// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <limits>

#include <nlohmann/json.hpp>
#include <openvino/runtime/core.hpp>

#include "openvino/genai/generation_config.hpp"

#include "generation_config_helper.hpp"

namespace ov {

GenerationConfig::GenerationConfig(std::string json_path) {
    std::ifstream f(json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open '" + json_path + "' with generation config");

    nlohmann::json data = nlohmann::json::parse(f);

    if (data.contains("max_new_tokens")) max_new_tokens = data["max_new_tokens"];
    if (data.contains("max_length")) max_length = data["max_length"];
    // note that ignore_eos is not present in HF GenerationConfig
    if (data.contains("num_beam_groups")) num_beam_groups = data["num_beam_groups"];
    if (data.contains("num_beams")) num_beams = data["num_beams"];
    if (data.contains("diversity_penalty")) diversity_penalty = data["diversity_penalty"];
    if (data.contains("length_penalty")) length_penalty = data["length_penalty"];
    if (data.contains("num_return_sequences")) num_return_sequences = data["num_return_sequences"];
    if (data.contains("no_repeat_ngram_size")) no_repeat_ngram_size = data["no_repeat_ngram_size"];
    // stop_criteria will be processed below
    if (data.contains("temperature")) temperature = data["temperature"];
    if (data.contains("top_p")) top_p = data["top_p"];
    if (data.contains("top_k")) top_k = data["top_k"];
    if (data.contains("do_sample")) do_sample = data["do_sample"];
    if (data.contains("repetition_penalty")) repetition_penalty = data["repetition_penalty"];
    if (data.contains("pad_token_id")) pad_token_id = data["pad_token_id"];
    if (data.contains("bos_token_id")) bos_token_id = data["bos_token_id"];
    if (data.contains("eos_token_id")) eos_token_id = data["eos_token_id"];
    if (data.contains("bos_token")) bos_token = data["bos_token"];
    if (data.contains("eos_token")) eos_token = data["eos_token"];

    if (data.contains("early_stopping")) {
        auto field_type = data["early_stopping"].type();
        if (field_type == nlohmann::json::value_t::string && data["early_stopping"] == "never") {
            stop_criteria = StopCriteria::never;
        } else if (field_type == nlohmann::json::value_t::boolean && data["early_stopping"] == true) {
            stop_criteria = StopCriteria::early;
        } else if (field_type == nlohmann::json::value_t::boolean && data["early_stopping"] == false) {
            stop_criteria = StopCriteria::heuristic;
        }
    }
}

GenerationConfig GenerationConfigHelper::anymap_to_generation_config(const ov::AnyMap& config_map) {
    GenerationConfig config = m_config;

    if (config_map.count("max_new_tokens")) config.max_new_tokens = config_map.at("max_new_tokens").as<size_t>();
    if (config_map.count("max_length")) config.max_length = config_map.at("max_length").as<size_t>();
    if (config_map.count("ignore_eos")) config.ignore_eos = config_map.at("ignore_eos").as<bool>();
    if (config_map.count("num_beam_groups")) config.num_beam_groups = config_map.at("num_beam_groups").as<size_t>();
    if (config_map.count("num_beams")) config.num_beams = config_map.at("num_beams").as<size_t>();
    if (config_map.count("diversity_penalty")) config.diversity_penalty = config_map.at("diversity_penalty").as<float>();
    if (config_map.count("length_penalty")) config.length_penalty = config_map.at("length_penalty").as<float>();
    if (config_map.count("num_return_sequences")) config.num_return_sequences = config_map.at("num_return_sequences").as<size_t>();
    if (config_map.count("no_repeat_ngram_size")) config.no_repeat_ngram_size = config_map.at("no_repeat_ngram_size").as<size_t>();
    if (config_map.count("stop_criteria")) config.stop_criteria = config_map.at("stop_criteria").as<StopCriteria>();
    if (config_map.count("temperature")) config.temperature = config_map.at("temperature").as<float>();
    if (config_map.count("top_p")) config.top_p = config_map.at("top_p").as<float>();
    if (config_map.count("top_k")) config.top_k = config_map.at("top_k").as<int>();
    if (config_map.count("do_sample")) config.do_sample = config_map.at("do_sample").as<bool>();
    if (config_map.count("repetition_penalty")) config.repetition_penalty = config_map.at("repetition_penalty").as<float>();
    if (config_map.count("pad_token_id")) config.pad_token_id = config_map.at("pad_token_id").as<int64_t>();
    if (config_map.count("bos_token_id")) config.bos_token_id = config_map.at("bos_token_id").as<int64_t>();
    if (config_map.count("eos_token_id")) config.eos_token_id = config_map.at("eos_token_id").as<int64_t>();
    if (config_map.count("bos_token")) config.bos_token = config_map.at("bos_token").as<std::string>();
    if (config_map.count("eos_token")) config.eos_token = config_map.at("eos_token").as<std::string>();
   
    return config;
}

size_t GenerationConfigHelper::get_max_new_tokens(size_t prompt_length) {
    // max_new_tokens has priority over max_length, only if max_new_tokens was not specified use max_length
    if (m_config.max_new_tokens != SIZE_MAX) {
        return m_config.max_new_tokens;
    } else {
        return m_config.max_length - prompt_length;
    }
}

bool GenerationConfigHelper::is_greedy_decoding() const {
    return !m_config.do_sample && !is_beam_search();
}

bool GenerationConfigHelper::is_beam_search() const {
    return m_config.num_beams > 1;
}

bool GenerationConfigHelper::is_multimomial() const {
    return m_config.do_sample;
}

}  // namespace ov
