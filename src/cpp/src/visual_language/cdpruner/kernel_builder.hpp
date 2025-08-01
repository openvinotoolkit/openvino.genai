// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cdpruner_config.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai::cdpruner {

/**
 * @brief Builder for conditional kernel matrices used in DPP-based token selection
 * 
 * This class implements the conditional kernel matrix construction that combines
 * visual feature similarity with relevance-based weighting as described in the
 * CDPruner paper. The kernel matrix is computed as:
 * 
 * L̃ = diag(r) · L · diag(r)
 * 
 * where L is the similarity matrix and r is the relevance vector.
 */
class ConditionalKernelBuilder {
public:
    /// @brief Constructor
    /// @param config Configuration for the kernel builder
    explicit ConditionalKernelBuilder(const Config& config);
    
    /// @brief Build conditional kernel matrix L̃ = diag(r) · L · diag(r)
    /// @param visual_features Visual feature embeddings [B, N, D]
    /// @param relevance_scores Relevance scores [B, N]
    /// @return Conditional kernel matrix [B, N, N]
    ov::Tensor build(const ov::Tensor& visual_features, const ov::Tensor& relevance_scores);

private:
    /// @brief Compute similarity matrix between visual features
    /// @param features Visual feature embeddings [B, N, D]
    /// @return Similarity matrix [B, N, N]
    ov::Tensor compute_similarity_matrix(const ov::Tensor& features);
    
    /// @brief GPU-accelerated similarity matrix computation using OpenVINO MatMul
    /// @param features Normalized visual features [B, N, D]
    /// @return Similarity matrix [B, N, N]
    ov::Tensor compute_similarity_matrix_gpu(const ov::Tensor& features);
    
    /// @brief Create diagonal matrix from vector
    /// @param scores Vector of scores [B, N]
    /// @return Diagonal matrix [B, N, N]
    ov::Tensor create_diagonal_matrix(const ov::Tensor& scores);
    
    /// @brief Perform batch matrix multiplication
    /// @param a First tensor [B, N, K]
    /// @param b Second tensor [B, K, M]
    /// @return Result tensor [B, N, M]
    ov::Tensor batch_matrix_multiply(const ov::Tensor& a, const ov::Tensor& b);
    
    /// @brief L2 normalize features along the last dimension
    /// @param features Input features [B, N, D]
    /// @return Normalized features [B, N, D]
    ov::Tensor l2_normalize_features(const ov::Tensor& features);
    
    /// @brief Build conditional kernel matrix using relevance weighting
    /// @param similarity_matrix Base similarity matrix [B, N, N]
    /// @param relevance_scores Token relevance scores [B, N]
    /// @return Conditional kernel matrix [B, N, N]
    ov::Tensor build_conditional_kernel(const ov::Tensor& similarity_matrix, 
                                      const ov::Tensor& relevance_scores);
    
    Config m_config;
};

} // namespace ov::genai::cdpruner 