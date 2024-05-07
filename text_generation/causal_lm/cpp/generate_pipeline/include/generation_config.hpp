// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <functional>
#include <nlohmann/json.hpp>
#include <fstream>
// #include <group_beam_searcher.hpp>  // used only for StopCriteria
#include <limits>
#include "llm_tokenizer.hpp"
#include <variant>

// forward declaration
class Sequence;

namespace ov {

// Similar to HuggingFace GenerationConfig
class GenerationConfig {
public:
    // Generic
    size_t max_new_tokens;
    size_t max_length;
    bool ignore_eos;
    std::string eos_token;

    // Beam search specific
    size_t num_groups;
    size_t group_size;
    float diversity_penalty;
    size_t m_num_return_sequences;
    // StopCriteria stop_criteria = StopCriteria::heuristic;
    
    float repetition_penalty;
    float length_penalty;
    size_t no_repeat_ngram_size;
    std::function<bool(const Sequence&)> early_finish = [](const Sequence&) {return false; };

    // Multinomial
    float temperature;
    int top_k;
    float top_p;
    bool do_sample;
    std::variant<std::string, ov::CompiledModel, ov::InferRequest> draft_model;  // todo: remove or try to add ov::Model const ov::Model&,

    // special tokens
    int64_t bos_token_id;
    int64_t eos_token_id;
    int64_t pad_token_id;

    GenerationConfig() = default;

    GenerationConfig(std::string json_path);
};

} // namespace ov
