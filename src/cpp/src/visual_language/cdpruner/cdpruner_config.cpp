// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cdpruner_config.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace ov::genai::cdpruner {

void Config::update_from_env() {
    // CDPRUNER_SPLIT_THRESHOLD
    if (const char* env = std::getenv("CDPRUNER_SPLIT_THRESHOLD")) {
        try {
            split_threshold = std::stoul(env);
        } catch (...) {
            split_threshold = 0;
        }
    }

    // CDPRUNER_USE_CL_KERNEL
    if (const char* env = std::getenv("CDPRUNER_USE_CL_KERNEL")) {
        std::string val(env);
        use_cl_kernel = (val == "1" || val == "true" || val == "TRUE");
    }

    // CDPRUNER_ENABLE_FRAME_CHUNKING
    if (const char* env = std::getenv("CDPRUNER_ENABLE_FRAME_CHUNKING")) {
        std::string val(env);
        enable_frame_chunking = (val == "1" || val == "true" || val == "TRUE");
    }
}

bool Config::operator==(const Config& other) const {
    return pruning_ratio == other.pruning_ratio && std::abs(relevance_weight - other.relevance_weight) < 1e-6f &&
           device == other.device && std::abs(numerical_threshold - other.numerical_threshold) < 1e-9f &&
           use_negative_relevance == other.use_negative_relevance && split_threshold == other.split_threshold &&
           enable_frame_chunking == other.enable_frame_chunking;
}

bool Config::operator!=(const Config& other) const {
    return !(*this == other);
}

}  // namespace ov::genai::cdpruner