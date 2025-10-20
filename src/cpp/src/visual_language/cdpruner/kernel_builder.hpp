// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <openvino/runtime/infer_request.hpp>

#include "cdpruner_config.hpp"
#include "openvino/runtime/infer_request.hpp"
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
    /// @param input_param Input parameter for relevance scores or text features
    /// @return Conditional kernel matrix [B, N, N]
    ov::Tensor build(const ov::Tensor& visual_features, const ov::Tensor& input_param);

    /// @brief Compute relevance scores and kernel matrix using OpenVINO ops model
    /// @param visual_features Visual feature embeddings [B, N, D]
    /// @param text_features Text feature embeddings [B, M, D]
    /// @return Conditional kernel matrix [B, N, N]
    ov::Tensor compute_conditional_kernel_with_model(const ov::Tensor& visual_features, const ov::Tensor& text_features);

private:
    /// @brief Build conditional kernel using OpenVINO ops model
    /// @param visual_features Visual feature embeddings [B, N, D]
    /// @param text_features Text feature embeddings [B, M, D]
    /// @return Conditional kernel matrix [B, N, N]
    ov::Tensor build_with_ov_model(const ov::Tensor& visual_features, const ov::Tensor& text_features);

    /// @brief Build conditional kernel using traditional pipeline
    /// @param visual_features Visual feature embeddings [B, N, D]
    /// @param relevance_scores Relevance scores [B, N]
    /// @return Conditional kernel matrix [B, N, N]
    ov::Tensor build_with_normal_pipeline(const ov::Tensor& visual_features, const ov::Tensor& relevance_scores);

    /// @brief Log detailed performance metrics for kernel construction steps
    void print_kernel_build_performance(size_t batch_size,
                                        size_t num_tokens,
                                        size_t feature_dim,
                                        size_t total_operations,
                                        const std::chrono::microseconds& normalize_duration,
                                        const std::chrono::microseconds& similarity_duration,
                                        const std::chrono::microseconds& conditional_duration,
                                        const std::chrono::microseconds& total_kernel_duration);
    /// @brief Create OpenVINO ops model for relevance and kernel computation
    /// @return Shared pointer to the OpenVINO model
    std::shared_ptr<ov::Model> create_conditional_kernel_model();

    /// @brief Create OpenVINO ops model for similarity matrix computation
    /// @return Shared pointer to the OpenVINO model for similarity computation
    std::shared_ptr<ov::Model> create_similarity_matrix_model();

    /// @brief Compute similarity matrix between visual features
    /// @param features Visual feature embeddings [B, N, D]
    /// @return Similarity matrix [B, N, N]
    ov::Tensor compute_similarity_matrix(const ov::Tensor& features);

    /// @brief GPU-accelerated similarity matrix computation using OpenVINO MatMul
    /// @param features Normalized visual features [B, N, D]
    /// @return Similarity matrix [B, N, N]
    ov::Tensor compute_similarity_matrix_with_model(const ov::Tensor& features);

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
    ov::Tensor compute_conditional_kernel_normal(const ov::Tensor& similarity_matrix, const ov::Tensor& relevance_scores);

    /// @brief Create min-max normalization subgraph using OpenVINO ops
    /// @param input Input node to normalize
    /// @return Normalized node
    std::shared_ptr<ov::Node> create_min_max_normalize_ops(std::shared_ptr<ov::Node> input);

    Config m_config;

    // Precompiled infer requests for performance optimization
    ov::InferRequest m_similarity_infer_request;
    ov::InferRequest m_conditional_kernel_infer_request;
    bool m_requests_initialized;
};

}  // namespace ov::genai::cdpruner