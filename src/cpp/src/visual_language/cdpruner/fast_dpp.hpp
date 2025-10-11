// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "cdpruner_config.hpp"
#include "openvino/openvino.hpp"

#ifdef ENABLE_OPENCL_DPP
#    ifdef OV_GPU_USE_OPENCL_HPP
#        include <CL/opencl.hpp>
#    else
#        include <CL/cl2.hpp>
#    endif
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
    // OpenCL kernel source for DPP
    const char* dpp_kernel_split_cl = R"CLC(
        // ================================ DPP Kernel Split Implementation ================================
        #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
        #pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
        #pragma OPENCL EXTENSION cl_intel_printf : enable

        __kernel void dpp_impl(__global const float *input, const int token_num, 
                                      int loop_idx, __global float *cis, __global float *di2s,
                                      const float numerical_threshold, const int selected_token_num,
                                      __global int *output_ids,
                                      __local float *local_max_array,
                                      __local int *local_max_ids)
        {
            uint batch_idx = get_global_id(0);
            uint gid_1 = get_global_id(1);

            uint lid_0 = get_local_id(1);
            uint lsz_1 = get_local_size(1);

            // Step 1: Get max idx of di2s in all workgroups
            float local_max = -INFINITY;
            int local_max_id = -1;

            __global float *di2s_data = di2s + batch_idx * token_num;

            for (int i = lid_0; i < token_num; i += lsz_1) {
                if (di2s_data[i] > local_max) {
                    local_max = di2s_data[i];
                    local_max_id = i;
                }
            }
            local_max_array[lid_0] = local_max;
            local_max_ids[lid_0] = local_max_id;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (size_t s = lsz_1 / 2; s > 0; s >>= 1) {
                if (lid_0 < s && local_max_array[lid_0 + s] > local_max_array[lid_0] ) {
                    local_max_array[lid_0] = local_max_array[lid_0 + s];
                    local_max_ids[lid_0] = local_max_ids[lid_0 + s];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            const int selected_idx = local_max_ids[0];

            // Step 2: Update orthogonal vector.
            if (gid_1 >= token_num)
                return;

            __global const float *input_data = input + batch_idx * token_num * token_num;
            __global float *cis_data = cis + batch_idx * token_num * selected_token_num;
            __global int* output_ids_data = output_ids + batch_idx * selected_token_num;

            float norm_factor = sqrt(di2s_data[selected_idx] + numerical_threshold);
            size_t j = gid_1;

            size_t kernel_idx = selected_idx * token_num + j;
            float kernel_val = input_data[kernel_idx];

            float projection = 0.0f;
            __global float *cis_selected_t = cis_data + selected_token_num * selected_idx;
            __global float *cis_t = cis_data + selected_token_num * j;

            __attribute__((opencl_unroll_hint(4))) for (size_t prev_t = 0; prev_t < loop_idx; ++prev_t)
            {
                projection += cis_selected_t[prev_t] * cis_t[prev_t];
            }

            size_t cis_current_idx = loop_idx + j * selected_token_num;
            cis_data[cis_current_idx] = (kernel_val - projection) / norm_factor;

            // step 3: Update_marginal_gains
            size_t cis_idx = loop_idx + j * selected_token_num;
            float eis_j = cis_data[cis_idx];

            if (selected_idx == j) {
                di2s_data[selected_idx] = -INFINITY;
                output_ids_data[loop_idx] = selected_idx;
            }
            else {
                di2s_data[j] -= eis_j * eis_j;
                di2s_data[j] = di2s_data[j] != di2s_data[j] ? -FLT_MAX : di2s_data[j];
            }
        }
        )CLC";
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
     * @brief Perform parallel DPP selection on split matrices
     * @param kernel_matrix_first First kernel matrix [B, N1, N1]
     * @param kernel_matrix_second Second kernel matrix [B, N2, N2]
     * @param tokens_first_half Number of tokens to select from first half
     * @param tokens_second_half Number of tokens to select from second half
     * @param split_point Split point for adjusting second half indices
     * @return Selected token indices for each batch [B, T]
     */
    std::vector<std::vector<size_t>> select_parallel(const ov::Tensor& kernel_matrix_first,
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