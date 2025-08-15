// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/runtime/tensor.hpp"
#include "cdpruner_config.hpp"

namespace ov::genai::cdpruner {

/// @brief Class for computing relevance scores between visual and text features
class RelevanceCalculator {
public:
    /// @brief Constructor
    /// @param config Configuration for the calculator
    explicit RelevanceCalculator(const Config& config);
    
    /// @brief Compute relevance scores between visual embeddings and text embeddings
    /// @param visual_embeds Visual feature embeddings [B, N, C]
    /// @param text_embeds Text feature embeddings [M, C]
    /// @return Relevance scores tensor [B, N]
    ov::Tensor compute(const ov::Tensor& visual_embeds, const ov::Tensor& text_embeds);

private:
    /// @brief L2 normalize tensor along the last dimension
    /// @param input Input tensor to normalize
    /// @return Normalized tensor
    ov::Tensor l2_normalize(const ov::Tensor& input);
    
    /// @brief Min-max normalize tensor
    /// @param input Input tensor to normalize
    /// @return Normalized tensor
    ov::Tensor min_max_normalize(const ov::Tensor& input);
    
    /// @brief Compute matrix multiplication between visual and text embeddings
    /// @param visual_embeds Visual embeddings [B, N, C]
    /// @param text_embeds Text embeddings [M, C]
    /// @return Similarity matrix [B, N, M]
    ov::Tensor matrix_multiply(const ov::Tensor& visual_embeds, const ov::Tensor& text_embeds);

    /// @brief Compute mean across the last dimension with optional negation
    /// @param relevance_matrix Input relevance matrix [B, N, M]
    /// @param use_negative Whether to apply negation before computing mean
    /// @return Mean relevance scores [B, N]
    ov::Tensor compute_mean(const ov::Tensor& relevance_matrix, bool use_negative);

    Config m_config;
};

} // namespace ov::genai::cdpruner 