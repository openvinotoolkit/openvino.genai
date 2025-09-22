// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstddef>
#include <string>

namespace ov::genai::cdpruner {

/// @brief Configuration structure for CDPruner algorithm
struct Config {
    /// @brief Percentage of visual tokens to retain after pruning (0-100)
    size_t pruning_ratio = 0;

    /// @brief Weight for balancing relevance vs diversity (0.0 to 1.0)
    float relevance_weight = 0.5f;

    /// @brief Device to run CDPruner computations on
    std::string device = "CPU";

    /// @brief Whether to enable debug output
    bool pruning_debug_mode = false;

    /// @brief Threshold for numerical stability
    float numerical_threshold = 1e-6f;

    /// @brief Whether to apply negative mean for relevance calculation
    /// This is needed for CLIP-based models (like LLaVA) due to counterintuitive similarity values
    bool use_negative_relevance = false;

    /// @brief Whether to use OpenCL kernel for DPP computation
    /// When true, uses OpenCL GPU acceleration for DPP selection
    /// When false, uses traditional CPU-based DPP algorithm
    bool use_cl_kernel = true;

    /// @brief Threshold for splitting large kernel matrices (internal use only)
    /// When visual tokens exceed this threshold, the kernel matrix will be split
    /// for parallel processing. This parameter is not exposed in public API.
    size_t split_threshold = 2000;

    /// @brief Whether to enable frame-level chunking for multi-frame video processing
    /// When true, each frame in multi-frame input will be processed separately for DPP pruning
    /// When false, all frames will be concatenated and processed together
    /// Default is false to maintain existing behavior
    bool enable_frame_chunking = false;

    /// @brief Compare two Config structures for equality
    /// @param other The other Config to compare with
    /// @return true if all configuration parameters are equal, false otherwise
    bool operator==(const Config& other) const;

    /// @brief Compare two Config structures for inequality
    /// @param other The other Config to compare with
    /// @return true if any configuration parameters differ, false otherwise
    bool operator!=(const Config& other) const;
};

}  // namespace ov::genai::cdpruner