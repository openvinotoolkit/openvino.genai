// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <functional>
#include <nlohmann/json.hpp>
#include <fstream>
#include <group_beam_searcher.hpp>  // used only for StopCriteria
#include <limits>

// forward declaration
class Sequence;

// forward declaration
class LLMPipeline;

// Similar to HuggingFace GenerationConfig
struct GenerationConfig {
    // todo: add copy constructor
    
    // Generic
    size_t m_max_new_tokens = SIZE_MAX;
    size_t m_max_length = SIZE_MAX; // m_max_new_tokens should have priority over m_max_length
    bool m_ignore_eos = false;
    int64_t m_eos_token = 2; // There's no way to extract special token values from the tokenizer for now

    // Beam search specific
    size_t m_num_groups = 1;
    size_t m_group_size = 1; // beam_width
    float m_diversity_penalty = 1.0f; // 0.0 means no diversity
    size_t m_num_return_sequences = 3;  // is used by beam search, in other case is equal to batch size
    StopCriteria stop_criteria = StopCriteria::heuristic;
    
    
    float m_repetition_penalty = 1.0f;
    float m_length_penalty = 1.0f;
    size_t m_no_repeat_ngram_size = std::numeric_limits<size_t>::max();
    std::function<bool(const Sequence&)> early_finish = [](const Sequence&) {return false; };

    // Multinomial
    float m_temperature = 0.0f; // by default we use greedy sampling
    int m_top_k = -1; // maybe to assign vocab_size ?
    float m_top_p = 1.0f; // by default convsider all tokens
    bool m_do_sample;

    // special tokens
    int64_t m_bos_token_id = 0;
    int64_t m_eos_token_id = 0;  // todo: do we need both m_eos_token and m_eos_token_id?
    int64_t m_pad_token_id = 0;

    std::function<void (std::vector<int64_t>&&, LLMPipeline&)> m_callback = [](std::vector<int64_t>&& tokens, LLMPipeline& pipe){ ;};


    size_t get_max_new_tokens(size_t prompt_length = 0) {
        // max_new_tokens has priority over max_length,
        // only if m_max_new_tokens was not specified use max_length
        if (m_max_new_tokens != SIZE_MAX) {
            return m_max_new_tokens;
        } else {
            return m_max_length - prompt_length;
        }
    }

    GenerationConfig& max_new_tokens(size_t max_new_tokens) {
        this->m_max_new_tokens = max_new_tokens;
         return *this;
     }

    GenerationConfig& max_length(size_t max_length) {
        this->m_max_length = max_length;
         return *this;
     }

    GenerationConfig& ignore_eos(bool ignore_eos) {
        this->m_ignore_eos = ignore_eos;
         return *this;
     }

    GenerationConfig& eos_token(int64_t eos_token) {
        this->m_eos_token = eos_token;
         return *this;
     }

    GenerationConfig& num_return_sequences(size_t num_return_sequences) {
        this->m_num_return_sequences = num_return_sequences;
         return *this;
     }

    GenerationConfig& num_groups(size_t num_groups) {
        this->m_num_groups = num_groups;
         return *this;
     }

    GenerationConfig& group_size(size_t group_size) {
        this->m_group_size = group_size;
         return *this;
     }
    GenerationConfig& diversity_penalty(float diversity_penalty) {
        this->m_diversity_penalty = diversity_penalty;
         return *this;
     }

    GenerationConfig& length_penalty(float length_penalty) {
        this->m_length_penalty = length_penalty;
         return *this;
     }
    GenerationConfig& no_repeat_ngram_size(size_t no_repeat_ngram_size) {
        this->m_no_repeat_ngram_size = no_repeat_ngram_size;
         return *this;
     }

    GenerationConfig& temperature(float temperature) {
        this->m_temperature = temperature;
         return *this;
     }
    GenerationConfig& top_k(size_t top_k) {
        this->m_top_k = top_k;
         return *this;
     }

    GenerationConfig& top_p(size_t top_p) {
        this->m_top_p = top_p;
         return *this;
     }
    GenerationConfig& do_sample(bool do_sample) {
        this->m_do_sample = do_sample;
         return *this;
     }

    GenerationConfig& repetition_penalty(float repetition_penalty) {
        this->m_repetition_penalty = repetition_penalty;
         return *this;
     }

    GenerationConfig& bos_token_id(int64_t bos_token_id) {
        this->m_bos_token_id = bos_token_id;
         return *this;
     }

    GenerationConfig& eos_token_id(int64_t eos_token_id) {
        this->m_eos_token_id = eos_token_id;
         return *this;
     }

    GenerationConfig& pad_token_id(int64_t pad_token_id) {
        this->m_pad_token_id = pad_token_id;
         return *this;
     }

    GenerationConfig& set_callback(std::function<void (std::vector<int64_t>&&, LLMPipeline&)> callback) {
        this->m_callback = callback;
         return *this;
     }

    GenerationConfig() = default;

    GenerationConfig(std::string json_path) {
        std::ifstream f(json_path);
        nlohmann::json data = nlohmann::json::parse(f);

        m_bos_token_id = data.value("bos_token_id", 0);
        m_eos_token_id = data.value("eos_token_id", 0);

        m_pad_token_id = data.value("pad_token_id", 0);
        m_num_return_sequences = data.value("num_return_sequences", 1);
        
        m_max_new_tokens = data.value("max_new_tokens", SIZE_MAX);
        m_max_length = data.value("max_length", SIZE_MAX);
        
        m_temperature = data.value("temperature", 0.0f);
        m_do_sample = data.value("do_sample", false);
        m_top_p = data.value("top_p", 0.0f);
        
        // beam_search_params
        m_num_groups = data.value("num_beam_groups", 1);
        m_diversity_penalty = data.value("diversity_penalty", 1.0f);
        int num_beams = data.value("num_beams", 1);
        m_group_size = num_beams / m_num_groups;
    }

    static GenerationConfig greedy() {
        GenerationConfig greedy_params;
        greedy_params.m_temperature = 0.0f;
        greedy_params.m_ignore_eos = true;
        return greedy_params;
    }

    static GenerationConfig beam_search() {
        GenerationConfig beam_search;
        beam_search.m_num_groups = 3;
        beam_search.m_group_size = 5;
        beam_search.m_max_new_tokens = 10;
        beam_search.m_diversity_penalty = 2.0f;
        return beam_search;
    }

    static GenerationConfig multimomial() {
        GenerationConfig multimomial;
        multimomial.m_temperature = 0.8f;
        multimomial.m_top_p = 0.8;
        multimomial.m_top_k = 20;
        multimomial.m_do_sample = 20;
        return multimomial;
    }

    bool is_gready_sampling() const {
        return !m_do_sample && !is_beam_search();
    }

    bool is_beam_search() const {
        return m_num_groups * m_group_size > 1;
    }

    bool is_multimomial() const {
        return m_do_sample;
    }
    
};
