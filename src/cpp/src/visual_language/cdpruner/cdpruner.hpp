// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <iostream>
#include <vector>

#include "cdpruner_config.hpp"
#include "conditional_kernel.hpp"
#include "fast_dpp.hpp"
#include "openvino/runtime/tensor.hpp"
#include "relevance_calculator.hpp"

namespace ov::genai::cdpruner {

/**
 * @brief Statistics about the pruning operation
 */
struct PruningStatistics {
    size_t total_tokens = 0;     ///< Total number of visual tokens before pruning
    size_t selected_tokens = 0;  ///< Number of tokens selected after pruning
    float pruning_ratio = 0.0f;  ///< Ratio of tokens pruned (0-1)
    size_t batch_size = 0;       ///< Batch size processed
};

class CDPruner {
public:
    /**
     * @brief Constructor
     * @param config Configuration for CDPruner
     */
    explicit CDPruner(const Config& config);

    /**
     * @brief Select diverse and relevant visual tokens
     * @param visual_features Input visual features [B, N, D]
     * @param text_features Input text features [M, D]
     * @param silent If true, suppress detailed logging output
     * @return Selected token indices for each batch [B, T]
     */
    std::vector<std::vector<size_t>> select_tokens(const ov::Tensor& visual_features,
                                                   const ov::Tensor& text_features,
                                                   bool silent = false);

    /**
     * @brief Apply pruning and return only selected features
     * @param visual_features Input visual features [B, N, D]
     * @param text_features Input text features [M, D]
     * @param silent If true, suppress detailed logging output
     * @return Pruned visual features [B, T, D] where T is calculated from pruning_ratio
     */
    ov::Tensor apply_pruning(const ov::Tensor& visual_features, const ov::Tensor& text_features, bool silent = false);

    /**
     * @brief Apply pruning to multiple visual features and return concatenated result
     * @param visual_features_list Vector of input visual features, each [B, N, D]
     * @param text_features Input text features [M, D]
     * @return Concatenated pruned visual features [B, T*num_frames, D] where T is calculated from pruning_ratio
     */
    ov::Tensor apply_pruning(const std::vector<ov::Tensor>& visual_features_list, const ov::Tensor& text_features);

    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const Config& get_config() const {
        return m_config;
    }

    /**
     * @brief Update configuration dynamically
     * @param new_config New configuration to apply
     * @return true if configuration was updated successfully
     */
    bool update_config(const Config& new_config);

    /**
     * @brief Compute current pruning ratio
     * @return Ratio of selected tokens to default token count
     */
    float compute_pruning_ratio() const;

    /**
     * @brief Get statistics from the last pruning operation
     * @return Pruning statistics
     */
    PruningStatistics get_last_pruning_statistics() const;

    const std::vector<std::vector<size_t>>& get_last_selected_tokens() const {
        return m_last_selected_tokens;
    }

private:
    /**
     * @brief Validate configuration parameters
     * @param config Configuration to validate
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate_config(const Config& config);

    /**
     * @brief Validate input tensor shapes and types
     * @param visual_features Visual features tensor
     * @param text_features Text features tensor
     * @throws std::invalid_argument if tensors are invalid
     */
    void validate_input_tensors(const ov::Tensor& visual_features, const ov::Tensor& text_features);

    Config m_config;                            ///< Configuration
    RelevanceCalculator m_relevance_calc;       ///< Relevance computation module
    ConditionalKernelBuilder m_kernel_builder;  ///< Kernel matrix construction module (with OpenVINO ops)
    FastGreedyDPP m_dpp_selector;               ///< DPP selection module

    mutable PruningStatistics m_last_statistics;                      ///< Statistics from last operation
    mutable std::vector<std::vector<size_t>> m_last_selected_tokens;  ///< Token indices selected during last pruning
};

}  // namespace ov::genai::cdpruner