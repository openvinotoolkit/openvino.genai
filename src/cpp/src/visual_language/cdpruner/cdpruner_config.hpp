// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstddef>
#include <string>

namespace ov::genai::cdpruner {

/// @brief Configuration structure for CDPruner algorithm
struct Config {
    /// @brief Percentage of visual tokens to prune [0-100)
    size_t pruning_ratio = 0;

    /// @brief Weight for balancing relevance vs diversity (0.0 to 1.0)
    float relevance_weight = 0.5f;

    /**
     * @brief Update configuration parameters from environment variables.
     *
     * The following environment variables are read:
     *   - CDPRUNER_USE_CL_KERNEL: Use OpenCL kernel for DPP computation (boolean, "0" or "1").
     *   - CDPRUNER_SPLIT_THRESHOLD: Threshold for splitting large kernel matrices (integer).
     *   - CDPRUNER_ENABLE_FRAME_CHUNKING: Enable frame-level chunking for multi-frame video processing (boolean, "0" or
     * "1").
     *
     * If an environment variable is not set, the default value specified in the Config struct is used.
     */
    void update_from_env();
    /// @brief Device to run CDPruner computations on
    std::string device = "CPU";

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
    /// If the number of visual tokens exceeds this value, the kernel matrix will be split
    /// and processed in parallel to improve efficiency. This parameter is for internal optimization
    /// and is not exposed in the public API.
    size_t split_threshold = 1;

    /// @brief Enable frame-level chunking for multi-frame video input
    /// If true, each frame in a multi-frame input is processed independently for DPP pruning.
    /// If false, all frames are concatenated and processed as a single batch.
    /// Default: true
    bool enable_frame_chunking = true;

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