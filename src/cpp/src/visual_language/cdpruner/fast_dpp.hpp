// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cdpruner_config.hpp"
#include "openvino/runtime/tensor.hpp"
#include <vector>

namespace ov::genai::cdpruner {

/**
 * @brief Fast greedy DPP (Determinantal Point Process) algorithm for token selection
 * 
 * This class implements the fast greedy approximation algorithm for maximizing
 * the determinant of a subset selection from a kernel matrix. The algorithm
 * is based on the CDPruner paper and provides O(T²N) complexity where T is
 * the number of tokens to select and N is the total number of tokens.
 * 
 * The core algorithm follows these steps:
 * 1. Initialize diagonal scores (marginal gains)
 * 2. Greedily select tokens with maximum marginal gain
 * 3. Update orthogonalized vectors using Gram-Schmidt process
 * 4. Update marginal gains by subtracting orthogonal projections
 */
class FastGreedyDPP {
public:
    /// @brief Constructor
    /// @param config Configuration for the DPP selector
    explicit FastGreedyDPP(const Config& config);
    
    /**
     * @brief Select diverse tokens using fast greedy DPP algorithm
     * @param kernel Conditional kernel matrix [B, N, N]
     * @param num_tokens Number of tokens to select
     * @return Selected token indices for each batch [B, T]
     */
    std::vector<std::vector<size_t>> select(const ov::Tensor& kernel, size_t num_tokens);

    /**
     * @brief Create boolean mask from selected indices
     * @param selected_indices Selected indices for each batch [B, T]
     * @param total_tokens Total number of tokens
     * @return Boolean mask [B*N] where true indicates selected tokens
     */
    static std::vector<bool> create_mask(const std::vector<std::vector<size_t>>& selected_indices, 
                                       size_t total_tokens);

    /**
     * @brief Compute approximate determinant for validation
     * @param kernel Kernel matrix [1, N, N] (single batch only)
     * @param selected_indices Selected token indices
     * @return Approximated determinant value
     */
    static float compute_determinant_approximation(const ov::Tensor& kernel, 
                                                 const std::vector<size_t>& selected_indices);

private:
    /**
     * @brief Select tokens for a single batch
     * @param kernel Kernel matrix [B, N, N]
     * @param batch_idx Batch index to process
     * @param num_tokens Number of tokens to select
     * @return Selected token indices for this batch
     */
    std::vector<size_t> select_single_batch(const ov::Tensor& kernel, size_t batch_idx, size_t num_tokens);

    /**
     * @brief Find index with maximum value
     * @param scores Score tensor [N]
     * @return Index of maximum value
     */
    size_t argmax(const ov::Tensor& scores);

    /**
     * @brief Update orthogonal vector using Gram-Schmidt process
     * @param kernel Kernel matrix [B, N, N]
     * @param batch_idx Current batch index
     * @param selected_idx Newly selected token index
     * @param iteration Current iteration (number of previously selected tokens)
     * @param cis Orthogonalized vectors [T, N]
     * @param di2s Current diagonal scores [N]
     */
    void update_orthogonal_vector(const ov::Tensor& kernel, size_t batch_idx, size_t selected_idx, 
                                size_t iteration, ov::Tensor& cis, const ov::Tensor& di2s);

    /**
     * @brief Update marginal gains after selecting a token
     * @param iteration Current iteration
     * @param selected_idx Newly selected token index
     * @param cis Orthogonalized vectors [T, N]
     * @param di2s Diagonal scores to update [N]
     */
    void update_marginal_gains(size_t iteration, size_t selected_idx, 
                             const ov::Tensor& cis, ov::Tensor& di2s);

    Config m_config;
};

} // namespace ov::genai::cdpruner 