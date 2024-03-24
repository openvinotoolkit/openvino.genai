
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <functional>

enum class StopCriteria {early, heuristic, never};

// forward declaration
class Sequence;

struct SamplingParameters {
    // Generic
    size_t max_new_tokens = 100;
    bool ignore_eos = false;
    int64_t eos_token = 2; // There's no way to extract special token values from the tokenizer for now

    // Beam search specific
    size_t n_groups = 1;
    size_t group_size = 1; // beam_width
    float diversity_penalty = 1.0f; // 0.0 means no diversity
    StopCriteria stop_criteria = StopCriteria::heuristic;
    float length_penalty = 1.0f;
    size_t no_repeat_ngram_size = 40;
    std::function<bool(const Sequence&)> early_finish = [](const Sequence&) {return false; };

    // Multinomial
    float temperature = 0.0f; // by default we use greedy sampling
    int top_k = -1; // maybe to assign vocab_size ?
    float top_p = 1.0f; // by default convsider all tokens

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
        return multimomial;
    }

    bool is_gready_sampling() const {
        return temperature == 0.0f && !is_beam_search();
    }

    bool is_beam_search() const {
        return n_groups * group_size > 1;
    }
};
