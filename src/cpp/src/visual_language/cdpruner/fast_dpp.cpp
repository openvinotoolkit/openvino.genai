// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "fast_dpp.hpp"
#include "openvino/openvino.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ov::genai::cdpruner {

FastGreedyDPP::FastGreedyDPP(const Config& config) : m_config(config) {
    // Constructor implementation
}

std::vector<std::vector<size_t>> FastGreedyDPP::select(const ov::Tensor& kernel, size_t num_tokens) {
    // Input validation
    if (kernel.get_shape().size() != 3) {
        throw std::invalid_argument("Kernel must be 3D tensor [B, N, N]");
    }
    
    auto shape = kernel.get_shape();
    size_t batch_size = shape[0];
    size_t total_tokens = shape[1];
    
    if (shape[1] != shape[2]) {
        throw std::invalid_argument("Kernel matrix must be square [B, N, N]");
    }
    
    if (num_tokens > total_tokens) {
        throw std::invalid_argument("Cannot select more tokens than available");
    }
    
    std::vector<std::vector<size_t>> batch_results(batch_size);
    
    // Process each batch independently
    for (size_t b = 0; b < batch_size; ++b) {
        batch_results[b] = select_single_batch(kernel, b, num_tokens);
    }
    
    return batch_results;
}

std::vector<size_t> FastGreedyDPP::select_single_batch(const ov::Tensor& kernel, size_t batch_idx, size_t num_tokens) {
    auto shape = kernel.get_shape();
    size_t total_tokens = shape[1];
    
    // Initialize working tensors for this batch
    // cis: Orthogonalized vectors [T, N] where T is the number of selected tokens
    ov::Tensor cis(ov::element::f32, {num_tokens, total_tokens});
    
    // di2s: Diagonal elements (marginal gains) [N]
    ov::Tensor di2s(ov::element::f32, {total_tokens});
    
    // Copy diagonal elements from kernel for this batch
    const float* kernel_data = kernel.data<const float>();
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
        
        // Compute the new orthogonalized vector e_i
        // eis = (kernel[batch, best_idx] - sum(cis[:t] * cis[:t, best_idx])) / sqrt(di2s[best_idx])
        update_orthogonal_vector(kernel, batch_idx, best_idx, t, cis, di2s);
        
        // Update marginal gains by subtracting the squared new orthogonal vector
        // di2s -= square(eis)
        update_marginal_gains(t, best_idx, cis, di2s);
        
        // Set the selected token's gain to negative infinity to prevent re-selection
        di2s_data[best_idx] = -std::numeric_limits<float>::infinity();
    }
    
    // Sort the selected indices for deterministic output
    std::sort(selected_indices.begin(), selected_indices.end());
    
    return selected_indices;
}

size_t FastGreedyDPP::argmax(const ov::Tensor& scores) {
    const float* data = scores.data<const float>();
    size_t size = scores.get_size();
    
    if (size == 0) {
        throw std::invalid_argument("Cannot find argmax of empty tensor");
    }
    
    size_t best_idx = 0;
    float best_value = data[0];
    
    for (size_t i = 1; i < size; ++i) {
        if (data[i] > best_value) {
            best_value = data[i];
            best_idx = i;
        }
    }
    
    return best_idx;
}

void FastGreedyDPP::update_orthogonal_vector(const ov::Tensor& kernel, size_t batch_idx, size_t selected_idx, 
                                           size_t iteration, ov::Tensor& cis, const ov::Tensor& di2s) {
    // This implements the key DPP orthogonalization step:
    // eis = (kernel[batch, selected_idx] - sum(cis[:iteration] * cis[:iteration, selected_idx])) / sqrt(di2s[selected_idx])
    
    auto kernel_shape = kernel.get_shape();
    size_t total_tokens = kernel_shape[1];
    
    const float* kernel_data = kernel.data<const float>();
    const float* di2s_data = di2s.data<const float>();
    float* cis_data = cis.data<float>();
    
    // Get the normalization factor
    float norm_factor = std::sqrt(di2s_data[selected_idx] + m_config.numerical_threshold);
    
    // Compute the new orthogonal vector for each token
    for (size_t j = 0; j < total_tokens; ++j) {
        // Get kernel[batch_idx, selected_idx, j]
        size_t kernel_idx = batch_idx * total_tokens * total_tokens + selected_idx * total_tokens + j;
        float kernel_val = kernel_data[kernel_idx];
        
        // Subtract the projection onto previously selected vectors
        // sum(cis[:iteration, selected_idx] * cis[:iteration, j])
        float projection = 0.0f;
        for (size_t prev_t = 0; prev_t < iteration; ++prev_t) {
            size_t cis_selected_idx = prev_t * total_tokens + selected_idx;
            size_t cis_j_idx = prev_t * total_tokens + j;
            projection += cis_data[cis_selected_idx] * cis_data[cis_j_idx];
        }
        
        // Store the orthogonalized vector element
        size_t cis_current_idx = iteration * total_tokens + j;
        cis_data[cis_current_idx] = (kernel_val - projection) / norm_factor;
    }
}

void FastGreedyDPP::update_marginal_gains(size_t iteration, size_t selected_idx, 
                                        const ov::Tensor& cis, ov::Tensor& di2s) {
    // This implements: di2s -= square(eis)
    // where eis is the newly computed orthogonal vector cis[iteration, :]
    
    auto cis_shape = cis.get_shape();
    size_t total_tokens = cis_shape[1];
    
    const float* cis_data = cis.data<const float>();
    float* di2s_data = di2s.data<float>();
    
    // Update marginal gains for all tokens
    for (size_t j = 0; j < total_tokens; ++j) {
        size_t cis_idx = iteration * total_tokens + j;
        float eis_j = cis_data[cis_idx];
        
        // Subtract the squared orthogonal component
        di2s_data[j] -= eis_j * eis_j;
    }
}

std::vector<bool> FastGreedyDPP::create_mask(const std::vector<std::vector<size_t>>& selected_indices, 
                                           size_t total_tokens) {
    if (selected_indices.empty()) {
        return std::vector<bool>(total_tokens, false);
    }
    
    size_t batch_size = selected_indices.size();
    std::vector<bool> mask(batch_size * total_tokens, false);
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t idx : selected_indices[b]) {
            if (idx < total_tokens) {
                mask[b * total_tokens + idx] = true;
            }
        }
    }
    
    return mask;
}

float FastGreedyDPP::compute_determinant_approximation(const ov::Tensor& kernel, 
                                                     const std::vector<size_t>& selected_indices) {
    // This is a simplified approximation for validation purposes
    // In practice, the greedy algorithm approximates the determinant maximization
    
    if (selected_indices.empty()) {
        return 0.0f;
    }
    
    auto shape = kernel.get_shape();
    size_t batch_size = shape[0];
    
    if (batch_size != 1) {
        throw std::invalid_argument("Determinant approximation only supports single batch");
    }
    
    const float* kernel_data = kernel.data<const float>();
    size_t total_tokens = shape[1];
    
    // Compute the product of diagonal elements of selected tokens as approximation
    float det_approx = 1.0f;
    for (size_t idx : selected_indices) {
        size_t diag_idx = idx * total_tokens + idx;
        det_approx *= kernel_data[diag_idx];
    }
    
    return det_approx;
}

} // namespace ov::genai::cdpruner 