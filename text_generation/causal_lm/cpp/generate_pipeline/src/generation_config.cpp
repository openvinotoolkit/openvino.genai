// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <functional>
#include <nlohmann/json.hpp>
#include <fstream>
// #include <group_beam_searcher.hpp>  // used only for StopCriteria
#include <limits>
// #include "llm_tokenizer.hpp"
#include "generation_config.hpp"
#include "generation_config_helper.hpp"

namespace ov {

GenerationConfig::GenerationConfig(std::string json_path) {
    std::ifstream f(json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open '" + json_path + "' with generation config");

    nlohmann::json data = nlohmann::json::parse(f);

    bos_token_id = data.value("bos_token_id", 0);
    eos_token_id = data.value("eos_token_id", 0);
    eos_token = data.value("eos_token", "</s>");

    pad_token_id = data.value("pad_token_id", 0);
    m_num_return_sequences = data.value("num_return_sequences", 1);
    
    max_new_tokens = data.value("max_new_tokens", SIZE_MAX);
    max_length = data.value("max_length", SIZE_MAX);
    
    temperature = data.value("temperature", 0.0f);
    do_sample = data.value("do_sample", false);
    top_p = data.value("top_p", 0.0f);
    
    // beam_search_params
    num_groups = data.value("num_beam_groups", 1);
    diversity_penalty = data.value("diversity_penalty", 1.0f);
    int num_beams = data.value("num_beams", 1);
    group_size = num_beams / num_groups;
    OPENVINO_ASSERT(num_beams % num_groups == 0, "number of beams should be divisible by number of groups");
}


size_t GenerationConfigHelper::get_max_new_tokens(size_t prompt_length) {
    // max_new_tokens has priority over max_length,
    // only if max_new_tokens was not specified use max_length
    if (config.max_new_tokens != SIZE_MAX) {
        return config.max_new_tokens;
    } else {
        return config.max_length - prompt_length;
    }
}

bool GenerationConfigHelper::is_greedy_sampling() const {
    return !config.do_sample && !is_beam_search() && !is_speculative();
}

bool GenerationConfigHelper::is_beam_search() const {
    return config.num_groups * config.group_size > 1;
}

bool GenerationConfigHelper::is_multimomial() const {
    return config.do_sample;
}

bool GenerationConfigHelper::is_speculative() const {
    return is_assistant_ov_defined || is_assistant_request_defined;
}

ov::InferRequest GenerationConfigHelper::get_assistant_model(std::string device, const ov::AnyMap& config) {
    if (is_assistant_request_defined) {
        return assistant_model;
    } else if (is_assistant_ov_defined) {
        assistant_model = ov::Core().compile_model(m_assistant_ov_model, device, config).create_infer_request();
        is_assistant_request_defined = true;
        return assistant_model;
    } else {
        OPENVINO_THROW("assistant model is not specified");
    }
}

} // namespace ov



// // forward declaration
// class Sequence;

// // forward declaration
// namespace ov {
// class LLMPipeline;
// }

// namespace {

// // TODO: LEAVE ONLY ONE PLACE FOR DEFAULT VALUES
// static const ov::AnyMap default_generation_config_map = {
//     // Generic
//     {"max_new_tokens", SIZE_MAX},
//     {"max_length", SIZE_MAX},
//     {"m_ignore_eos", false},
//     {"m_bos_token", "</s>"},
//     {"m_eos_token", "</s>"},
    
//     // Beam search specific
//     {"m_num_groups", 1},
//     {"m_group_size", 1},
//     {"m_diversity_penalty", 1.0f},  // 0.0 means no diversity
//     {"m_num_return_sequences", 1},  // is used by beam search, in other case is equal to batch size
//     // {"stop_criteria", StopCriteria::heuristic},  // todo: align with the latest beam searcher

//     {"m_repetition_penalty", 1.0f},
//     {"m_length_penalty", 1.0f},
//     {"m_no_repeat_ngram_size", std::numeric_limits<size_t>::max()},
//     {"early_finish", [](const Sequence&) {return false; }},
    
//     // Multinomial
//     {"m_temperature", 0.0f},
//     {"m_top_k", -1},
//     {"m_top_p", 1.0f},
//     {"m_do_sample", false},
    
//     // special tokens
//     {"m_bos_token_id", 0},
//     {"m_eos_token_id", 2}, // todo: check form where it's better to extract from rt_info or from tokenizer_config.json
//     {"m_pad_token_id", 0},
    
//     // assistive decoding
//     {"m_assistant_model", ov::InferRequest()},
//     {"m_num_assistant_tokens", 5},
//     {"m_seq_len_axis", 2},
// };

// }

// namespace ov {
// size_t get_max_new_tokens(size_t prompt_length = 0) {
//         // max_new_tokens has priority over max_length,
//         // only if m_max_new_tokens was not specified use max_length
//         if (m_max_new_tokens != SIZE_MAX) {
//             return m_max_new_tokens;
//         } else {
//             return m_max_length - prompt_length;
//         }
//     }

//     void max_new_tokens(size_t max_new_tokens) {
//         const auto& r = ::default_generation_config_map.find("sdf") != ::default_generation_config_map.end();

//         m_max_new_tokens = max_new_tokens;
//     }

//     void max_length(size_t max_length) {
//         m_max_length = max_length;
//     }

//     void ignore_eos(bool ignore_eos) {
//         m_ignore_eos = ignore_eos;
//     }

//     void eos_token(std::string eos_token) {
//         m_eos_token = eos_token;
//     }

//     void num_return_sequences(size_t num_return_sequences) {
//         m_num_return_sequences = num_return_sequences;
//     }

//     void num_groups(size_t num_groups) {
//         m_num_groups = num_groups;
//     }

//     void group_size(size_t group_size) {
//         m_group_size = group_size;
//     }

//     void diversity_penalty(float diversity_penalty) {
//         m_diversity_penalty = diversity_penalty;
//     }

//     void length_penalty(float length_penalty) {
//         m_length_penalty = length_penalty;
//     }

//     void no_repeat_ngram_size(size_t no_repeat_ngram_size) {
//         m_no_repeat_ngram_size = no_repeat_ngram_size;
//     }

//     void temperature(float temperature) {
//         m_temperature = temperature;
//     }

//     void top_k(size_t top_k) {
//         m_top_k = top_k;
//     }

//     void top_p(size_t top_p) {
//         m_top_p = top_p;
//     }

//     void do_sample(bool do_sample) {
//         m_do_sample = do_sample;
//     }

//     void repetition_penalty(float repetition_penalty) {
//         m_repetition_penalty = repetition_penalty;
//     }

//     void bos_token_id(int64_t bos_token_id) {
//         m_bos_token_id = bos_token_id;
//     }

//     void eos_token_id(int64_t eos_token_id) {
//         m_eos_token_id = eos_token_id;
//     }

//     void pad_token_id(int64_t pad_token_id) {
//         m_pad_token_id = pad_token_id;
//     }

//     GenerationConfig() = default;

//     GenerationConfig(std::string json_path) {
//         std::ifstream f(json_path);
//         OPENVINO_ASSERT(f.is_open(), "Failed to open '" + json_path + "' with generation config");

//         nlohmann::json data = nlohmann::json::parse(f);

//         m_bos_token_id = data.value("bos_token_id", 0);
//         m_eos_token_id = data.value("eos_token_id", 0);
//         m_eos_token = data.value("eos_token", "</s>");

//         m_pad_token_id = data.value("pad_token_id", 0);
//         m_num_return_sequences = data.value("num_return_sequences", 1);
        
//         m_max_new_tokens = data.value("max_new_tokens", SIZE_MAX);
//         m_max_length = data.value("max_length", SIZE_MAX);
        
//         m_temperature = data.value("temperature", 0.0f);
//         m_do_sample = data.value("do_sample", false);
//         m_top_p = data.value("top_p", 0.0f);
        
//         // beam_search_params
//         m_num_groups = data.value("num_beam_groups", 1);
//         m_diversity_penalty = data.value("diversity_penalty", 1.0f);
//         int num_beams = data.value("num_beams", 1);
//         m_group_size = num_beams / m_num_groups;
//         OPENVINO_ASSERT(num_beams % m_num_groups == 0, "number of beams should be divisible by number of groups");
//     }


//     static GenerationConfig greedy() {
//         GenerationConfig greedy_params;
//         greedy_params.m_temperature = 0.0f;
//         greedy_params.m_ignore_eos = true;
//         return greedy_params;
//     }

//     static GenerationConfig beam_search() {
//         GenerationConfig beam_search;
//         beam_search.m_num_groups = 3;
//         beam_search.m_group_size = 5;
//         beam_search.m_max_new_tokens = 10;
//         beam_search.m_diversity_penalty = 2.0f;
//         return beam_search;
//     }

//     static GenerationConfig multimomial() {
//         GenerationConfig multimomial;
//         multimomial.m_temperature = 0.8f;
//         multimomial.m_top_p = 0.8;
//         multimomial.m_top_k = 20;
//         multimomial.m_do_sample = 20;
//         return multimomial;
//     }
    
//     template <typename T>
//     static GenerationConfig assistive_decoding(T& assistant_model) {
//         GenerationConfig assistive;
//         assistive.assistant_model(assistant_model);
//         return assistive;
//     }

//     bool is_greedy_sampling() const {
//         return !m_do_sample && !is_beam_search() && !is_speculative();
//     }

//     bool is_beam_search() const {
//         return m_num_groups * m_group_size > 1;
//     }

//     bool is_multimomial() const {
//         return m_do_sample;
//     }

//     // for speculative decoding
//     void assistant_model(const ov::InferRequest& assistant_model) {
//         m_assistant_model = assistant_model;
//         is_assistant_request_defined = true;
//     }

//     void assistant_model(ov::CompiledModel& assistant_model) {
//         m_assistant_model = assistant_model.create_infer_request();
//         is_assistant_request_defined = true;
//     }

//     void assistant_model(const std::shared_ptr<const ov::Model>& assistant_model) {
//         m_assistant_ov_model = assistant_model;
//         is_assistant_ov_defined = true;
//     }

//     void assistant_model(std::string assistant_model) {
//         auto is_xml = [](std::string path) -> bool { return path.compare(path.length() - 4, 4, ".xml") == 0;};
//         if (!is_xml(assistant_model))
//             assistant_model += "/openvino_model.xml";

//         m_assistant_ov_model = ov::Core().read_model(assistant_model);
//         is_assistant_ov_defined = true;
//     }

//     void set_streamer(std::function<void (std::vector<int64_t>&&, ov::LLMPipeline&)> callback) {
//         m_callback = callback;
//     }

//     ov::InferRequest get_assistant_model(std::string device="CPU", const ov::AnyMap& config={}) {
//         if (is_assistant_request_defined) {
//             return m_assistant_model;
//         } else if (is_assistant_ov_defined) {
//             m_assistant_model = ov::Core().compile_model(m_assistant_ov_model, device, config).create_infer_request();
//             is_assistant_request_defined = true;
//             return m_assistant_model;
//         } else {
//             OPENVINO_THROW("assistant model is not specified");
//         }
//     }
    
//     void num_assistant_tokens(int64_t num_assistant_tokens) {
//         m_num_assistant_tokens = num_assistant_tokens;
//     }

//     bool is_speculative() const {
//         return is_assistant_ov_defined || is_assistant_request_defined;
//     }

//     // for Assistive/Speculative decoding
//     ov::InferRequest m_assistant_model;
//     size_t m_num_assistant_tokens = 5;
//     size_t m_seq_len_axis = 2;
    
//     static GenerationConfig anymap_to_generation_config(const ov::AnyMap& genereation_config_map = {}) {
//         // need to load default values and update only those keys that are specified in genereation_config_map
//         auto tmp_map = default_generation_config_map;
        
//         for (auto it = genereation_config_map.begin(); it != genereation_config_map.end(); ++it) {
//             tmp_map[it->first] = it->second;
//         }

//         GenerationConfig config;
        
//         // general arguments
//         config.m_max_new_tokens = tmp_map.at("m_max_new_tokens").as<size_t>();
//         config.m_max_length = tmp_map.at("m_max_length").as<size_t>();
//         config.m_ignore_eos = tmp_map.at("m_ignore_eos").as<bool>();
//         config.m_eos_token = tmp_map.at("m_eos_token").as<int64_t>();

//         // Beam search specific
//         config.m_num_groups = tmp_map.at("m_num_groups").as<size_t>();
//         config.m_group_size = tmp_map.at("m_group_size").as<size_t>();
//         config.m_diversity_penalty = tmp_map.at("m_diversity_penalty").as<size_t>();
//         config.m_num_return_sequences = tmp_map.at("m_num_return_sequences").as<size_t>();
        
//         config.m_repetition_penalty = tmp_map.at("m_repetition_penalty").as<size_t>();
//         config.m_length_penalty = tmp_map.at("m_length_penalty").as<size_t>();
//         config.m_no_repeat_ngram_size = tmp_map.at("m_no_repeat_ngram_size").as<size_t>();
//         config.early_finish = tmp_map.at("early_finish").as<std::function<bool(const Sequence&)>>();

//         // Multinomial
//         config.m_temperature = tmp_map.at("m_temperature").as<size_t>();
//         config.m_top_k = tmp_map.at("m_top_k").as<size_t>();
//         config.m_top_p = tmp_map.at("m_top_p").as<size_t>();
//         config.m_do_sample = tmp_map.at("m_do_sample").as<bool>();

//         // special tokens
//         config.m_bos_token_id = tmp_map.at("m_bos_token_id").as<int64_t>();
//         config.m_eos_token_id = tmp_map.at("m_eos_token_id").as<int64_t>();
//         config.m_pad_token_id = tmp_map.at("m_pad_token_id").as<int64_t>();
//         return config;
//     }    
// }
    
// private:
//     std::shared_ptr<const ov::Model> m_assistant_ov_model;
//     bool is_assistant_request_defined = false;
//     bool is_assistant_ov_defined = false;
// };
