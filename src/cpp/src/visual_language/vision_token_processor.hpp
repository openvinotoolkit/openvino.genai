// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>
#include <optional>
#include "openvino/runtime/tensor.hpp"
#include "visual_language/cdpruner/cdpruner.hpp"
#include "visual_language/cdpruner/cdpruner_config.hpp"

namespace ov::genai {

/**
 * @brief Vision token processor for optimizing visual features.
 * 
 * This class provides a separate abstraction for post-processing visual tokens,
 * including pruning, compression, and other optimization techniques.
 * Currently implements visual token pruning using CDPruner algorithm.
 */
class VisionTokenProcessor {
public:
    /**
     * @brief Construct a new Vision Token Processor
     * @param device Device to use for processing operations
     * @param config Optional CDPruner configuration
     */
    explicit VisionTokenProcessor(const std::string& device, 
                                  const cdpruner::Config& config = cdpruner::Config{});

    /**
     * @brief Process (prune) visual features based on text features
     * @param visual_features Vector of visual feature tensors to process
     * @param text_features Text features for relevance calculation
     * @return Processed (pruned) visual features tensor
     */
    ov::Tensor process(const std::vector<ov::Tensor>& visual_features,
                      const ov::Tensor& text_features);

    /**
     * @brief Check if processor is available and ready
     * @return true if processor is initialized and available
     */
    bool is_available() const { return m_pruner != nullptr; }

    /**
     * @brief Set processor configuration
     * @param config New configuration to apply
     */
    void set_config(const cdpruner::Config& config);

    /**
     * @brief Get current processor configuration
     * @return Current configuration
     */
    cdpruner::Config get_config() const;

    /**
     * @brief Get statistics from last processing operation
     * @return Pruning statistics if available
     */
    std::optional<cdpruner::PruningStatistics> get_last_statistics() const;

    /**
     * @brief Get indices of selected tokens from last processing
     * @return Vector of selected token indices for each frame/image
     */
    std::vector<std::vector<size_t>> get_last_selected_tokens() const;

private:
    /// @brief CDPruner instance for token pruning
    std::unique_ptr<cdpruner::CDPruner> m_pruner;
};

} // namespace ov::genai
