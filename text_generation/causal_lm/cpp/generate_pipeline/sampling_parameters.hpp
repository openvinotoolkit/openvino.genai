
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <functional>
#include <nlohmann/json.hpp>
#include <fstream>

enum class StopCriteria {early, heuristic, never};

// forward declaration
class Sequence;

// SamplingParameters is similar to HuggingFace GenerationConfig 
// and has parameters that are not present in the original SamplingParameters for continous batching
struct SamplingParameters {
    // Generic
    size_t max_new_tokens = 100;
    size_t max_length = 100; // max_new tokens should have priority over max_new_tokens
    bool ignore_eos = false;
    int64_t eos_token = 2; // There's no way to extract special token values from the tokenizer for now

    // Beam search specific
    size_t n_groups = 1;
    size_t group_size = 1; // beam_width
    float diversity_penalty = 1.0f; // 0.0 means no diversity
    StopCriteria stop_criteria = StopCriteria::heuristic;
    float length_penalty = 1.0f;
    size_t no_repeat_ngram_size = std::numeric_limits<size_t>::max();
    std::function<bool(const Sequence&)> early_finish = [](const Sequence&) {return false; };

    // Multinomial
    float temperature = 0.0f; // by default we use greedy sampling
    int top_k = -1; // maybe to assign vocab_size ?
    float top_p = 1.0f; // by default convsider all tokens
    bool do_sample;

    // special tokens
    int64_t bos_token_id = 0;
    int64_t eos_token_id = 0;
    int64_t pad_token_id = 0;

    SamplingParameters() = default;

    SamplingParameters(std::string json_path) {
        std::ifstream f(json_path);
        nlohmann::json data = nlohmann::json::parse(f);

        bos_token_id = data.value("bos_token_id", 0);
        eos_token_id = data.value("eos_token_id", 0);
        max_length = data.value("max_length", 0);
        pad_token_id = data.value("pad_token_id", 0);
        
        temperature = data.value("temperature", 0.0f);
        do_sample = data.value("do_sample", false);
        top_p = data.value("top_p", 0.0f);
    }

    static SamplingParameters greedy() {
        SamplingParameters greedy_params;
        greedy_params.temperature = 0.0f;
        greedy_params.ignore_eos = true;
        return greedy_params;
    }

    static SamplingParameters beam_search() {
        SamplingParameters beam_search;
        beam_search.n_groups = 2;
        beam_search.group_size = 2;
        beam_search.max_new_tokens = 100;
        beam_search.diversity_penalty = 2.0f;
        return beam_search;
    }

    static SamplingParameters multimomial() {
        SamplingParameters multimomial;
        multimomial.temperature = 0.8f;
        multimomial.top_p = 0.8;
        multimomial.top_k = 20;
        multimomial.do_sample = 20;
        return multimomial;
    }

    bool is_gready_sampling() const {
        return !do_sample && !is_beam_search();
    }

    bool is_beam_search() const {
        return n_groups * group_size > 1;
    }

    bool is_multimomial() const {
        return do_sample;
    }
    
};

enum class SamplingAlgorithm{greedy, multinomial, baeam_search};
