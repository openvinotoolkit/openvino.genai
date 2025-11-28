// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "fast_dpp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "logger.hpp"
#include "openvino/openvino.hpp"
#include "utils.hpp"

#ifdef ENABLE_OPENCL_DPP
#    include "fast_dpp_cl.hpp"
#endif

// SIMD headers
#if defined(OPENVINO_ARCH_X86_64)
#    ifdef _MSC_VER
#        include <intrin.h>
#    else
#        include <x86intrin.h>
#    endif
#endif

namespace ov::genai::cdpruner {

/**
 * Performs element-wise subtraction of a scaled input vector from an output vector:
 *     out[i] -= scalar * in[i]   for i in [0, size)
 *
 * Parameters:
 *   - out: Pointer to the output array of floats. Must have at least 'size' elements.
 *   - in: Pointer to the input array of floats. Must have at least 'size' elements.
 *   - scalar: Scalar multiplier applied to each element of 'in'.
 *   - size: Number of elements to process.
 */
inline void simd_vector_sub_scalar_mul(float* out, const float* in, float scalar, size_t size) {
    size_t i = 0;

#ifdef __AVX__
    // AVX: Process 8 floats at a time
    const __m256 scalar_vec = _mm256_set1_ps(scalar);
    for (; i + 8 <= size; i += 8) {
        __m256 out_vec = _mm256_loadu_ps(&out[i]);
        __m256 in_vec = _mm256_loadu_ps(&in[i]);
        __m256 mul_result = _mm256_mul_ps(scalar_vec, in_vec);
        __m256 result = _mm256_sub_ps(out_vec, mul_result);
        _mm256_storeu_ps(&out[i], result);
    }
#elif defined(__SSE2__)
    // SSE2: Process 4 floats at a time
    const __m128 scalar_vec = _mm_set1_ps(scalar);
    for (; i + 4 <= size; i += 4) {
        __m128 out_vec = _mm_loadu_ps(&out[i]);
        __m128 in_vec = _mm_loadu_ps(&in[i]);
        __m128 mul_result = _mm_mul_ps(scalar_vec, in_vec);
        __m128 result = _mm_sub_ps(out_vec, mul_result);
        _mm_storeu_ps(&out[i], result);
    }
#endif

    // Process remaining elements with scalar code
    for (; i < size; ++i) {
        out[i] -= scalar * in[i];
    }
}

// SIMD optimized vector multiplication by scalar: out[i] *= scalar
inline void simd_vector_mul_scalar(float* out, float scalar, size_t size) {
    size_t i = 0;

#ifdef __AVX__
    // AVX: Process 8 floats at a time
    const __m256 scalar_vec = _mm256_set1_ps(scalar);
    for (; i + 8 <= size; i += 8) {
        __m256 out_vec = _mm256_loadu_ps(&out[i]);
        __m256 result = _mm256_mul_ps(out_vec, scalar_vec);
        _mm256_storeu_ps(&out[i], result);
    }
#elif defined(__SSE2__)
    // SSE2: Process 4 floats at a time
    const __m128 scalar_vec = _mm_set1_ps(scalar);
    for (; i + 4 <= size; i += 4) {
        __m128 out_vec = _mm_loadu_ps(&out[i]);
        __m128 result = _mm_mul_ps(out_vec, scalar_vec);
        _mm_storeu_ps(&out[i], result);
    }
#endif

    // Process remaining elements with scalar code
    for (; i < size; ++i) {
        out[i] *= scalar;
    }
}

FastGreedyDPP::FastGreedyDPP(const Config& config) : m_config(config) {
    // Load config from env
    m_config.update_from_env();
}

std::vector<std::vector<size_t>> FastGreedyDPP::select(const ov::Tensor& kernel, size_t num_tokens) {
    // Input validation
    OPENVINO_ASSERT(kernel.get_shape().size() == 3, "Kernel must be 3D tensor [B, N, N]");

    auto shape = kernel.get_shape();
    size_t batch_size = shape[0];
    size_t total_tokens = shape[1];

    OPENVINO_ASSERT(shape[1] == shape[2], "Kernel matrix must be square [B, N, N]");

    OPENVINO_ASSERT(num_tokens <= total_tokens,
                    "Cannot select more tokens [",
                    num_tokens,
                    "] than available [",
                    total_tokens,
                    "]");

#ifdef ENABLE_OPENCL_DPP
    // For small token counts, CPU implementation is faster due to GPU kernel launch overhead
    // Empirically determined threshold where GPU acceleration becomes beneficial
    if (total_tokens < 16) {
        return select_cpu_internal(kernel, num_tokens);
    }

    // Try OpenCL GPU acceleration if enabled and available
    if (m_config.use_cl_kernel) {
        // Initialize OpenCL DPP if not already done
        if (!m_opencl_dpp) {
            m_opencl_dpp = std::make_unique<OpenCLDPP>(m_config);
        }

        // Use OpenCL if available
        if (m_opencl_dpp && m_opencl_dpp->is_available()) {
            GENAI_DEBUG("[CDPruner] Using OpenCL DPP kernel for selection");
            return select_opencl_internal(kernel, num_tokens);
        }
    }
#endif

    // Use CPU implementation
    GENAI_DEBUG("[CDPruner] Using DPP native CPU implementation for selection");
    return select_cpu_internal(kernel, num_tokens);
}

// Parallel DPP selection on two kernel matrices
std::vector<std::vector<size_t>> FastGreedyDPP::select(const ov::Tensor& kernel_matrix_first,
                                                       const ov::Tensor& kernel_matrix_second,
                                                       size_t num_tokens_to_keep,
                                                       size_t split_point) {
    // Distribute tokens to keep between both halves
    size_t tokens_first_half = num_tokens_to_keep / 2;
    size_t tokens_second_half = num_tokens_to_keep - tokens_first_half;
#ifdef ENABLE_OPENCL_DPP
    // Check if OpenCL DPP is enabled and available
    if (m_config.use_cl_kernel) {
        GENAI_DEBUG("[CDPruner] Using OpenCL DPP parallel selection");
        return select_parallel_opencl(kernel_matrix_first,
                                      kernel_matrix_second,
                                      tokens_first_half,
                                      tokens_second_half,
                                      split_point);
    }
#endif
    // Fallback to parallel CPU processing
    GENAI_DEBUG("[CDPruner] Using DPP native CPU parallel selection");
    return this->select_parallel(kernel_matrix_first,
                                 kernel_matrix_second,
                                 tokens_first_half,
                                 tokens_second_half,
                                 split_point);
}

// Parallel CPU DPP selection on split matrices
std::vector<std::vector<size_t>> FastGreedyDPP::select_parallel(const ov::Tensor& kernel_matrix_first,
                                                                const ov::Tensor& kernel_matrix_second,
                                                                size_t tokens_first_half,
                                                                size_t tokens_second_half,
                                                                size_t split_point) {
    // Launch parallel tasks for DPP selection
    std::future<std::vector<std::vector<size_t>>> dpp_first_future = std::async(std::launch::async, [&]() {
        return this->select_cpu_internal(kernel_matrix_first, tokens_first_half);
    });

    std::future<std::vector<std::vector<size_t>>> dpp_second_future = std::async(std::launch::async, [&]() {
        return this->select_cpu_internal(kernel_matrix_second, tokens_second_half);
    });

    // Wait for both DPP selections to complete
    auto selected_first_batches = dpp_first_future.get();
    auto selected_second_batches = dpp_second_future.get();

    // Process all batches, not just the first one
    std::vector<std::vector<size_t>> batch_results;

    // Ensure both have same number of batches
    size_t num_batches = std::min(selected_first_batches.size(), selected_second_batches.size());
    batch_results.reserve(num_batches);

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        const auto& selected_first = selected_first_batches[batch_idx];
        const auto& selected_second = selected_second_batches[batch_idx];

        std::vector<size_t> merged_selection;
        merged_selection.reserve(selected_first.size() + selected_second.size());

        // Add first half selections (indices unchanged)
        for (size_t idx : selected_first) {
            merged_selection.push_back(idx);
        }

        // Add second half selections (adjust indices by split_point)
        for (size_t idx : selected_second) {
            merged_selection.push_back(idx + split_point);
        }

        // Sort final result to maintain order
        std::sort(merged_selection.begin(), merged_selection.end());

        batch_results.push_back(std::move(merged_selection));
    }

    return batch_results;
}

#ifdef ENABLE_OPENCL_DPP
// Parallel OpenCL DPP selection on split matrices
std::vector<std::vector<size_t>> FastGreedyDPP::select_parallel_opencl(const ov::Tensor& kernel_matrix_first,
                                                                       const ov::Tensor& kernel_matrix_second,
                                                                       size_t tokens_first_half,
                                                                       size_t tokens_second_half,
                                                                       size_t split_point) {
    // Initialize OpenCL DPP if not already done
    if (!m_opencl_dpp) {
        m_opencl_dpp = std::make_unique<OpenCLDPP>(m_config);
    }

    // Verify OpenCL is available
    if (!m_opencl_dpp || !m_opencl_dpp->is_available()) {
        return select_parallel(kernel_matrix_first,
                               kernel_matrix_second,
                               tokens_first_half,
                               tokens_second_half,
                               split_point);
    }

    // Get tensor shapes: [B, tokens/2, tokens/2]
    auto first_shape = kernel_matrix_first.get_shape();
    auto second_shape = kernel_matrix_second.get_shape();

    // Verify shapes are compatible
    OPENVINO_ASSERT(first_shape.size() == 3 && second_shape.size() == 3,
                    "Input kernel matrices must be 3D tensors with shape [B, tokens, tokens]");
    OPENVINO_ASSERT(first_shape[1] == first_shape[2] && second_shape[1] == second_shape[2],
                    "Kernel matrices must be square");
    OPENVINO_ASSERT(first_shape[0] == second_shape[0], "Kernel matrices must have the same batch size");

    size_t original_batch_size = first_shape[0];
    size_t first_tokens = first_shape[1];
    size_t second_tokens = second_shape[1];
    size_t num_tokens_to_keep = tokens_first_half + tokens_second_half;
    ov::Tensor merged_kernel;

    // Check if both matrices have the same token size
    if (first_tokens == second_tokens) {
        // Create merged tensor with shape [original_batch_size, 2, tokens, tokens]
        merged_kernel = ov::Tensor(ov::element::f32, {original_batch_size, 2, first_tokens, first_tokens});
        float* merged_data = merged_kernel.data<float>();

        const float* first_data = kernel_matrix_first.data<const float>();
        const float* second_data = kernel_matrix_second.data<const float>();

        size_t matrix_size = first_tokens * first_tokens;
        size_t matrix_size_bytes = matrix_size * sizeof(float);

        // Copy data for each original batch
        for (size_t b = 0; b < original_batch_size; ++b) {
            // Copy first half: merged[b][0] = first[b]
            size_t first_src_offset = b * matrix_size;
            size_t first_dst_offset = b * 2 * matrix_size + 0 * matrix_size;
            std::memcpy(merged_data + first_dst_offset, first_data + first_src_offset, matrix_size_bytes);

            // Copy second half: merged[b][1] = second[b]
            size_t second_src_offset = b * matrix_size;
            size_t second_dst_offset = b * 2 * matrix_size + 1 * matrix_size;
            std::memcpy(merged_data + second_dst_offset, second_data + second_src_offset, matrix_size_bytes);
        }

    } else {
        // Create merged tensor with padding for different sizes
        size_t max_tokens = std::max(first_tokens, second_tokens);
        merged_kernel = ov::Tensor(ov::element::f32, {original_batch_size, 2, max_tokens, max_tokens});
        float* merged_data = merged_kernel.data<float>();

        // Initialize with zeros
        std::memset(merged_data, 0, merged_kernel.get_byte_size());

        const float* first_data = kernel_matrix_first.data<const float>();
        const float* second_data = kernel_matrix_second.data<const float>();

        // Copy data for each original batch with padding
        for (size_t b = 0; b < original_batch_size; ++b) {
            // Copy first matrix with padding
            for (size_t i = 0; i < first_tokens; ++i) {
                size_t src_offset = b * first_tokens * first_tokens + i * first_tokens;
                size_t dst_offset = b * 2 * max_tokens * max_tokens + 0 * max_tokens * max_tokens + i * max_tokens;
                std::memcpy(merged_data + dst_offset, first_data + src_offset, first_tokens * sizeof(float));
            }

            // Copy second matrix with padding
            for (size_t i = 0; i < second_tokens; ++i) {
                size_t src_offset = b * second_tokens * second_tokens + i * second_tokens;
                size_t dst_offset = b * 2 * max_tokens * max_tokens + 1 * max_tokens * max_tokens + i * max_tokens;
                std::memcpy(merged_data + dst_offset, second_data + src_offset, second_tokens * sizeof(float));
            }
        }
    }

    // Use DPP selector with merged tensor
    // Process each batch individually
    std::vector<std::vector<size_t>> batch_results;
    batch_results.reserve(original_batch_size);

    // The merged_kernel has shape [original_batch_size, 2, max_tokens, max_tokens] representing two halves
    // Extract each batch and process separately
    size_t max_tokens = (first_tokens == second_tokens) ? first_tokens : std::max(first_tokens, second_tokens);
    const float* merged_data = merged_kernel.data<const float>();
    for (size_t batch_idx = 0; batch_idx < original_batch_size; ++batch_idx) {
        // Extract single batch matrix with shape [2, max_tokens, max_tokens]
        ov::Tensor single_batch_kernel(ov::element::f32, {2, max_tokens, max_tokens});
        float* single_batch_data = single_batch_kernel.data<float>();

        // Copy data for this batch: [2, max_tokens, max_tokens]
        size_t batch_matrix_size = 2 * max_tokens * max_tokens;
        size_t batch_offset = batch_idx * batch_matrix_size;
        std::memcpy(single_batch_data, merged_data + batch_offset, batch_matrix_size * sizeof(float));

        // Call select_opencl_internal with single batch matrix
        auto selected_tokens = m_opencl_dpp->select(single_batch_kernel, num_tokens_to_keep);

        std::vector<size_t> merged_selection;

        // convert splited selected_batches to merged_selection
        for (size_t idx = 0; idx < selected_tokens.size(); ++idx) {
            if (idx < num_tokens_to_keep / 2) {
                merged_selection.push_back(selected_tokens[idx]);
            } else {
                merged_selection.push_back(selected_tokens[idx] + split_point);
            }
        }

        // Sort final result to maintain order
        std::sort(merged_selection.begin(), merged_selection.end());
        merged_selection.erase(std::unique(merged_selection.begin(), merged_selection.end()), merged_selection.end());
        batch_results.push_back(std::move(merged_selection));
    }

    return batch_results;
}
#endif

std::vector<size_t> FastGreedyDPP::select_single_batch(const ov::Tensor& kernel, size_t batch_idx, size_t num_tokens) {
    auto shape = kernel.get_shape();
    size_t total_tokens = shape[1];

    // Get kernel data pointer once for reuse
    const float* kernel_data = kernel.data<const float>();

    // Initialize working tensors for this batch
    // cis: Orthogonalized vectors [T, N] where T is the number of selected tokens
    ov::Tensor cis(ov::element::f32, {num_tokens, total_tokens});

    // di2s: Diagonal elements (marginal gains) [N]
    ov::Tensor di2s(ov::element::f32, {total_tokens});

    // Copy diagonal elements from kernel for this batch
    float* di2s_data = di2s.data<float>();

    for (size_t i = 0; i < total_tokens; ++i) {
        size_t diag_idx = batch_idx * total_tokens * total_tokens + i * total_tokens + i;
        di2s_data[i] = kernel_data[diag_idx];
    }

    std::vector<size_t> selected_indices;
    selected_indices.reserve(num_tokens);

    float* cis_data = cis.data<float>();
    std::memset(cis_data, 0, cis.get_byte_size());

    // Greedy selection loop - this is the core DPP algorithm
    for (size_t t = 0; t < num_tokens; ++t) {
        // Find the token with maximum marginal gain
        size_t best_idx = argmax(di2s);
        selected_indices.push_back(best_idx);

        // Get batch-specific kernel data pointer and tensor data pointers
        const float* batch_kernel_data = kernel_data + batch_idx * total_tokens * total_tokens;
        float* cis_data = cis.data<float>();
        const float* di2s_data_const = di2s.data<const float>();

        // Compute the new orthogonalized vector e_i
        // eis = (kernel[batch, best_idx] - sum(cis[:t] * cis[:t, best_idx])) / sqrt(di2s[best_idx])
        update_orthogonal_vector(batch_kernel_data, total_tokens, best_idx, t, cis_data, di2s_data_const);

        // Update marginal gains by subtracting the squared new orthogonal vector
        // di2s -= square(eis)
        update_marginal_gains(t, total_tokens, cis_data, di2s_data);

        // Set the selected token's gain to negative infinity to prevent re-selection
        di2s_data[best_idx] = -std::numeric_limits<float>::infinity();
    }

    return selected_indices;
}

// Internal CPU-only DPP selection (no OpenCL checks)
std::vector<std::vector<size_t>> FastGreedyDPP::select_cpu_internal(const ov::Tensor& kernel, size_t num_tokens) {
    auto shape = kernel.get_shape();
    size_t batch_size = shape[0];

    // Debug output: report which SIMD instruction set is being used
    {
        static bool simd_logged = false;
        if (!simd_logged) {
#ifdef __AVX__
            GENAI_WARN("[CDPruner] Using AVX SIMD instructions for vector operations (8 floats/operation)");
#elif defined(__SSE2__)
            GENAI_WARN("[CDPruner] Using SSE2 SIMD instructions for vector operations (4 floats/operation)");
#else
            GENAI_WARN("[CDPruner] Using scalar operations (no SIMD acceleration)");
#endif
            simd_logged = true;
        }
    }

    std::vector<std::vector<size_t>> batch_results(batch_size);

    // Process each batch independently
    for (size_t b = 0; b < batch_size; ++b) {
        batch_results[b] = select_single_batch(kernel, b, num_tokens);
    }

    return batch_results;
}

#ifdef ENABLE_OPENCL_DPP
// Internal OpenCL-only DPP selection (no fallback)
std::vector<std::vector<size_t>> FastGreedyDPP::select_opencl_internal(const ov::Tensor& kernel, size_t num_tokens) {
    auto shape = kernel.get_shape();
    size_t batch_size = shape[0];

    std::vector<std::vector<size_t>> batch_results(batch_size);

    // Process each batch independently, similar to CPU version
    for (size_t b = 0; b < batch_size; ++b) {
        batch_results[b] = select_single_batch_opencl(kernel, b, num_tokens);
    }

    return batch_results;
}

// Single batch OpenCL selection (similar to select_single_batch for CPU)
std::vector<size_t> FastGreedyDPP::select_single_batch_opencl(const ov::Tensor& kernel,
                                                              size_t batch_idx,
                                                              size_t num_tokens) {
    // Extract single batch kernel matrix for OpenCL processing
    auto shape = kernel.get_shape();
    size_t total_tokens = shape[1];

    // Create a single batch tensor [1, total_tokens, total_tokens]
    ov::Tensor single_batch_kernel(ov::element::f32, {1, total_tokens, total_tokens});
    float* single_batch_data = single_batch_kernel.data<float>();
    const float* kernel_data = kernel.data<const float>();

    // Copy the specific batch data
    size_t batch_matrix_size = total_tokens * total_tokens;
    size_t batch_offset = batch_idx * batch_matrix_size;
    std::memcpy(single_batch_data, kernel_data + batch_offset, batch_matrix_size * sizeof(float));

    // Call OpenCL DPP with single batch
    return m_opencl_dpp->select(single_batch_kernel, num_tokens);
}
#endif

size_t FastGreedyDPP::argmax(const ov::Tensor& scores) {
    const float* data = scores.data<const float>();
    size_t size = scores.get_size();

    OPENVINO_ASSERT(size > 0, "Cannot find argmax of empty tensor");

    size_t best_idx = 0;
    float best_value = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < size; ++i) {
        if (data[i] > best_value) {
            best_value = data[i];
            best_idx = i;
        }
    }

    return best_idx;
}

void FastGreedyDPP::update_orthogonal_vector(const float* batch_kernel_data,
                                             size_t total_tokens,
                                             size_t selected_idx,
                                             size_t iteration,
                                             float* cis_data,
                                             const float* di2s_data) {
    // This implements the key DPP orthogonalization step:
    // eis = (kernel[batch, selected_idx] - sum(cis[:iteration] * cis[:iteration, selected_idx])) /
    // sqrt(di2s[selected_idx])

    // Get the normalization factor
    float norm_factor = std::sqrt(di2s_data[selected_idx] + m_config.numerical_threshold);
    float inv_norm = 1.0f / norm_factor;

    // Get kernel row for selected token (already offset to correct batch)
    const float* kernel_row = batch_kernel_data + selected_idx * total_tokens;

    // Get output position in cis matrix
    float* cis_out = cis_data + iteration * total_tokens;

    // Copy kernel row to cis output
    std::memcpy(cis_out, kernel_row, total_tokens * sizeof(float));

    // Subtract projections from all previous orthogonal vectors
    for (size_t prev_t = 0; prev_t < iteration; ++prev_t) {
        const float* cis_prev_row = cis_data + prev_t * total_tokens;
        float cis_sel = cis_prev_row[selected_idx];

        if (std::abs(cis_sel) < 1e-10f)
            continue;

        // SIMD optimized vector subtraction: cis_out[j] -= cis_sel * cis_prev_row[j]
        simd_vector_sub_scalar_mul(cis_out, cis_prev_row, cis_sel, total_tokens);
    }

    // SIMD optimized vector multiplication: cis_out[j] *= inv_norm
    simd_vector_mul_scalar(cis_out, inv_norm, total_tokens);
}

void FastGreedyDPP::update_marginal_gains(size_t iteration,
                                          size_t total_tokens,
                                          const float* cis_data,
                                          float* di2s_data) {
    // This implements: di2s -= square(eis)
    // where eis is the newly computed orthogonal vector cis[iteration, :]

    // Update marginal gains for all tokens
    for (size_t j = 0; j < total_tokens; ++j) {
        // Skip updating if this token is already selected (marked as negative infinity)
        if (di2s_data[j] == -std::numeric_limits<float>::infinity()) {
            continue;
        }

        size_t cis_idx = iteration * total_tokens + j;
        float eis_j = cis_data[cis_idx];
        // Subtract the squared orthogonal component
        if (std::isnan(eis_j)) {
            di2s_data[j] = -std::numeric_limits<float>::max();
            continue;
        }
        di2s_data[j] -= eis_j * eis_j;
    }
}

}  // namespace ov::genai::cdpruner
