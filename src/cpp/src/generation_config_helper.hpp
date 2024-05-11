// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/generation_config.hpp"

namespace ov {


class GenerationConfigHelper {
public:
    GenerationConfig m_config;

    GenerationConfigHelper() = default;
    
    GenerationConfigHelper(const GenerationConfig& config): m_config(config) {};

    size_t get_max_new_tokens(size_t prompt_length = 0);
    
    bool is_greedy_decoding() const;

    bool is_beam_search() const;

    bool is_multimomial() const;

    GenerationConfig anymap_to_generation_config(const ov::AnyMap& config_map = {});

};

} // namespace ov
