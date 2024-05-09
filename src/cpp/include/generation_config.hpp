// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include "llm_tokenizer.hpp"
#include <variant>


namespace ov {

enum class StopCriteria { early, heuristic, never };

class GenerationConfig {
public:
    GenerationConfig() = default;
    GenerationConfig(std::string json_path);

    // Generic
    size_t max_new_tokens;
    size_t max_length;
    bool ignore_eos;

    // Beam search specific
    size_t num_groups;
    size_t group_size;
    float diversity_penalty;
    float length_penalty;
    size_t m_num_return_sequences;
    size_t no_repeat_ngram_size;
    StopCriteria stop_criteria;
    
    // Multinomial
    float temperature;
    float top_p;
    size_t top_k;
    bool do_sample;
    float repetition_penalty;

    // special tokens
    int64_t bos_token_id;
    int64_t eos_token_id;
    int64_t pad_token_id;
    
    // used for chat scenario
    std::string eos_token;  
    std::string bos_token; 
    
    // speculative sampling
    std::variant<std::string, ov::CompiledModel, ov::InferRequest> draft_model;  // todo: remove or try to add ov::Model const ov::Model&,
};

} // namespace ov
