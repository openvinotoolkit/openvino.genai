// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <cstddef>
#include <cmath>

namespace ov::genai::cdpruner {

/// @brief Configuration structure for CDPruner algorithm
struct Config {
    /// @brief Percentage of visual tokens to retain after pruning (0-100)
    size_t viusal_tokens_retain_percentage = 50;
    
    /// @brief Weight for balancing relevance vs diversity (0.0 to 1.0)
    float relevance_weight = 0.5f;
    
    /// @brief Whether to enable pruning functionality
    bool enable_pruning = true;
    
    /// @brief Device to run CDPruner computations on
    std::string device = "CPU";

    /// @brief Whether to enable debug output
    bool pruning_debug_mode = false;

    /// @brief Threshold for numerical stability
    float numerical_threshold = 1e-6f;

    /// @brief Whether to apply negative mean for relevance calculation
    /// This is needed for CLIP-based models (like LLaVA) due to counterintuitive similarity values
    bool use_negative_relevance = false;

    /// @brief Compare two Config structures for equality
    /// @param other The other Config to compare with
    /// @return true if all configuration parameters are equal, false otherwise
    bool operator==(const Config& other) const {
        return viusal_tokens_retain_percentage == other.viusal_tokens_retain_percentage &&
               std::abs(relevance_weight - other.relevance_weight) < 1e-6f && enable_pruning == other.enable_pruning &&
               device == other.device && pruning_debug_mode == other.pruning_debug_mode &&
               std::abs(numerical_threshold - other.numerical_threshold) < 1e-9f &&
               use_negative_relevance == other.use_negative_relevance;
    }

    /// @brief Compare two Config structures for inequality
    /// @param other The other Config to compare with
    /// @return true if any configuration parameters differ, false otherwise
    bool operator!=(const Config& other) const {
        return !(*this == other);
    }
};

} // namespace ov::genai::cdpruner 