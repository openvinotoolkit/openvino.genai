// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <openvino/runtime/tensor.hpp>
#include "openvino/genai/generation_config.hpp"

// Modifyed Knuth–Morris–Pratt algorithm which returns tokens following after every needle occurance in haystack
std::vector<int64_t> kmp_search(const std::vector<int64_t>& haystack, const std::vector<int64_t>& needle);

struct Token {
    float log_prob;
    int64_t idx;
};

// std::vector<Token> log_softmax(const ov::Tensor& logits, const size_t batch_idx);

struct Beam {
    float score = -std::numeric_limits<float>::infinity();  // The bigger, the better
    std::vector<int64_t> tokens;
    size_t global_beam_idx = 0;
};

bool greater(const Beam& left, const Beam& right);

// enum class StopCriteria { early, heuristic, never };

struct Parameters {
    std::vector<std::vector<int64_t>> prompts;
    int64_t eos_token;
    size_t n_groups = 3;
    size_t group_size = 5;
    float diversity_penalty = 1.0;
    size_t max_new_tokens = 20;
    ov::StopCriteria stop_criteria = ov::StopCriteria::heuristic;
    float length_penalty = 1.0;
    size_t no_repeat_ngram_size = std::numeric_limits<size_t>::max();

    std::function<bool(const Beam&)> early_finish = [](const Beam&) {
        return false;
    };
};

struct Group {
    std::vector<Beam> ongoing;   // Best beams in front
    std::vector<Beam> min_heap;  // The worst of the best completed beams is the first
    bool done = false;

    void finish(Beam&& beam, const Parameters& parameters);

    void is_done(const Parameters& parameters);
};


struct GroupBeamSearcher {
    Parameters parameters;
    std::vector<std::vector<Group>> prompts_groups;

    GroupBeamSearcher(Parameters parameters);

    std::pair<std::vector<int64_t>, std::vector<int32_t>> select_next_tokens(const ov::Tensor& logits);

    std::pair<std::vector<int64_t>, std::vector<int32_t>> select_prompt_next_tokens(const ov::Tensor& logits,
                                                                                    const std::vector<int64_t>& prompt,
                                                                                    std::vector<Group>& groups);
};

// Consume group_beam_searcher because beams are consumed
std::vector<std::vector<std::vector<Beam>>> finalize(GroupBeamSearcher&& group_beam_searcher);
