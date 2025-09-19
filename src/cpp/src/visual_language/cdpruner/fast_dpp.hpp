// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "cdpruner_config.hpp"
#include "openvino/openvino.hpp"

#ifdef ENABLE_OPENCL_DPP
#    include <CL/opencl.hpp>
#    include <memory>
#endif

namespace ov::genai::cdpruner {

#ifdef ENABLE_OPENCL_DPP
/**
 * @brief OpenCL-accelerated DPP implementation
 *
 * This class provides GPU acceleration for DPP token selection using OpenCL.
 * It implements the same algorithm as FastGreedyDPP but runs on GPU for better performance.
 */
class OpenCLDPP {
public:
    explicit OpenCLDPP(const Config& config);
    ~OpenCLDPP();

    /**
     * @brief Select diverse tokens using OpenCL GPU acceleration
     * @param kernel Conditional kernel matrix [1, N, N] or splited kernel matrix [2, N/2, N/2]
     * @param num_tokens Number of tokens to select
     * @return Selected token indices for single batch [1, T]
     */
    std::vector<size_t> select(const ov::Tensor& kernel, size_t num_tokens);

    /**
     * @brief Check if OpenCL is available and initialized
     * @return true if OpenCL is ready for computation
     */
    bool is_available() const {
        return m_initialized;
    }

private:
    struct OpenCLState;
    std::unique_ptr<OpenCLState> m_state;
    Config m_config;
    bool m_initialized = false;

    bool initialize_opencl();
    bool load_and_compile_kernels();
    void cleanup_opencl();
    std::vector<size_t> run_dpp_split_kernel_impl(const ov::Tensor& kernel, size_t num_tokens);
};
#endif  // ENABLE_OPENCL_DPP

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
     * @brief Perform parallel DPP selection on two kernel matrices
     * @param kernel_matrix_first First kernel matrix
     * @param kernel_matrix_second Second kernel matrix
     * @param num_tokens_to_keep Total number of tokens to keep
     * @param split_point Split point for adjusting second half indices
     * @return Selected token indices for each batch [B, T]
     */
    std::vector<std::vector<size_t>> select(const ov::Tensor& kernel_matrix_first,
                                            const ov::Tensor& kernel_matrix_second,
                                            size_t num_tokens_to_keep,
                                            size_t split_point);

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
     * @brief Internal CPU-only DPP selection (no OpenCL checks)
     * @param kernel Kernel matrix [B, N, N]
     * @param num_tokens Number of tokens to select
     * @return Selected token indices for each batch [B, T]
     */
    std::vector<std::vector<size_t>> select_cpu_internal(const ov::Tensor& kernel, size_t num_tokens);

#ifdef ENABLE_OPENCL_DPP
    /**
     * @brief Internal OpenCL-only DPP selection (no fallback)
     * @param kernel Kernel matrix [B, N, N]
     * @param num_tokens Number of tokens to select
     * @return Selected token indices for each batch [B, T]
     */
    std::vector<std::vector<size_t>> select_opencl_internal(const ov::Tensor& kernel, size_t num_tokens);

    /**
     * @brief Single batch OpenCL DPP selection
     * @param kernel Kernel matrix [B, N, N]
     * @param batch_idx Batch index to process
     * @param num_tokens Number of tokens to select
     * @return Selected token indices for the specified batch [T]
     */
    std::vector<size_t> select_single_batch_opencl(const ov::Tensor& kernel, size_t batch_idx, size_t num_tokens);
#endif

    /**
     * @brief Perform parallel CPU DPP selection on split matrices
     * @param kernel_matrix_first First kernel matrix [B, N1, N1]
     * @param kernel_matrix_second Second kernel matrix [B, N2, N2]
     * @param tokens_first_half Number of tokens to select from first half
     * @param tokens_second_half Number of tokens to select from second half
     * @param split_point Split point for adjusting second half indices
     * @return Selected token indices for each batch [B, T]
     */
    std::vector<std::vector<size_t>> select_parallel_cpu(const ov::Tensor& kernel_matrix_first,
                                                         const ov::Tensor& kernel_matrix_second,
                                                         size_t tokens_first_half,
                                                         size_t tokens_second_half,
                                                         size_t split_point);

#ifdef ENABLE_OPENCL_DPP
    /**
     * @brief Perform parallel OpenCL DPP selection on split matrices
     * @param kernel_matrix_first First kernel matrix [B, N1, N1]
     * @param kernel_matrix_second Second kernel matrix [B, N2, N2]
     * @param tokens_first_half Number of tokens to select from first half
     * @param tokens_second_half Number of tokens to select from second half
     * @param split_point Split point for adjusting second half indices
     * @return Selected token indices for each batch [B, T]
     */
    std::vector<std::vector<size_t>> select_parallel_opencl(const ov::Tensor& kernel_matrix_first,
                                                            const ov::Tensor& kernel_matrix_second,
                                                            size_t tokens_first_half,
                                                            size_t tokens_second_half,
                                                            size_t split_point);
#endif

    /**
     * @brief Find index with maximum value
     * @param scores Score tensor [N]
     * @return Index of maximum value
     */
    size_t argmax(const ov::Tensor& scores);

    /**
     * @brief Update orthogonal vector using Gram-Schmidt process
     * @param batch_kernel_data Pre-computed kernel data pointer for specific batch
     * @param total_tokens Total number of tokens
     * @param selected_idx Newly selected token index
     * @param iteration Current iteration (number of previously selected tokens)
     * @param cis_data Orthogonalized vectors data pointer [T, N]
     * @param di2s_data Current diagonal scores data pointer [N]
     */
    void update_orthogonal_vector(const float* batch_kernel_data,
                                  size_t total_tokens,
                                  size_t selected_idx,
                                  size_t iteration,
                                  float* cis_data,
                                  const float* di2s_data);

    /**
     * @brief Update marginal gains after selecting a token
     * @param iteration Current iteration
     * @param total_tokens Total number of tokens
     * @param cis_data Orthogonalized vectors data pointer [T, N]
     * @param di2s_data Diagonal scores data pointer to update [N]
     */
    void update_marginal_gains(size_t iteration, size_t total_tokens, const float* cis_data, float* di2s_data);

    Config m_config;

#ifdef ENABLE_OPENCL_DPP
    /// @brief OpenCL DPP implementation for GPU acceleration
    mutable std::unique_ptr<OpenCLDPP> m_opencl_dpp;
#endif
};

}  // namespace ov::genai::cdpruner