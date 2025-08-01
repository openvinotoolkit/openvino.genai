// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <cstddef>

namespace ov::genai::cdpruner {

/// @brief Configuration structure for CDPruner algorithm
struct Config {
    /// @brief Number of visual tokens to retain after pruning
    size_t num_visual_tokens = 64;
    
    /// @brief Weight for balancing relevance vs diversity (0.0 to 1.0)
    float relevance_weight = 0.5f;
    
    /// @brief Whether to enable pruning functionality
    bool enable_pruning = true;
    
    /// @brief Device to run CDPruner computations on
    std::string device = "CPU";
    
    /// @brief Whether to enable debug output
    bool debug_mode = false;
    
    /// @brief Threshold for numerical stability
    float numerical_threshold = 1e-6f;
};

} // namespace ov::genai::cdpruner 