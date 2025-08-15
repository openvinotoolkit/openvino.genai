// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "kernel_builder.hpp"
#include "openvino/openvino.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <iostream>

namespace ov::genai::cdpruner {

ConditionalKernelBuilder::ConditionalKernelBuilder(const Config& config) : m_config(config) {
    // Constructor implementation
    try {
        // Create a simple MatMul model dynamically
        auto input_features = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1});

        // Transpose features for batch matrix multiplication: [B, N, D] -> [B, D, N]
        auto axes_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
        auto features_transposed = std::make_shared<ov::op::v1::Transpose>(input_features, axes_order);

        // Batch matrix multiplication: [B, N, D] @ [B, D, N] = [B, N, N]
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input_features, features_transposed);

        auto result = std::make_shared<ov::op::v0::Result>(matmul);
        auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_features});

        // Compile model for GPU
        ov::Core core;
        ov::CompiledModel compiled_model;

        if (m_config.device == "GPU") {
            compiled_model = core.compile_model(model, "GPU");
        } else {
            // Fallback to CPU if GPU not available
            compiled_model = core.compile_model(model, "CPU");
        }

        // Create inference request
        infer_request = compiled_model.create_infer_request();
    } catch (const std::exception& e) {
        if (m_config.pruning_debug_mode) {
            std::cout << "Error occurred while building kernel: " << e.what() << std::endl;
        }
    }
}

ov::Tensor ConditionalKernelBuilder::build(const ov::Tensor& visual_features, const ov::Tensor& relevance_scores) {
    // Input validation
    if (visual_features.get_shape().size() != 3) {
        throw std::invalid_argument("Visual features must be 3D tensor [B, N, D]");
    }
    if (relevance_scores.get_shape().size() != 2) {
        throw std::invalid_argument("Relevance scores must be 2D tensor [B, N]");
    }
    
    auto visual_shape = visual_features.get_shape();
    auto relevance_shape = relevance_scores.get_shape();
    
    size_t batch_size = visual_shape[0];
    size_t num_tokens = visual_shape[1];
    size_t feature_dim = visual_shape[2];
    
    // Check shape consistency
    if (relevance_shape[0] != batch_size || relevance_shape[1] != num_tokens) {
        throw std::invalid_argument("Visual features and relevance scores must have consistent batch size and token count");
    }
    
    // Performance timing for kernel building steps
    auto kernel_build_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n==== Kernel Build Performance Analysis ====" << std::endl;
    std::cout << "Input tensors: visual_features[" << batch_size << ", " << num_tokens << ", " << feature_dim 
              << "], relevance_scores[" << batch_size << ", " << num_tokens << "]" << std::endl;
    
    // Step 1: L2 normalize visual features along the last dimension
    // This is equivalent to: image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)
    auto normalize_start = std::chrono::high_resolution_clock::now();
    ov::Tensor normalized_features = l2_normalize_features(visual_features);
    auto normalize_end = std::chrono::high_resolution_clock::now();
    auto normalize_duration = std::chrono::duration_cast<std::chrono::microseconds>(normalize_end - normalize_start);
    
    // Step 2: Compute similarity matrix L = normalized_features @ normalized_features.T
    // This gives us the base similarity matrix [B, N, N]
    auto similarity_start = std::chrono::high_resolution_clock::now();
    // ov::Tensor similarity_matrix = compute_similarity_matrix(normalized_features);
    ov::Tensor similarity_matrix = compute_similarity_matrix_gpu(normalized_features);
    auto similarity_end = std::chrono::high_resolution_clock::now();
    auto similarity_duration = std::chrono::duration_cast<std::chrono::microseconds>(similarity_end - similarity_start);
    
    // Step 3: Build conditional kernel matrix L̃ = diag(r) · L · diag(r)
    // Following CDPruner: kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)
    auto conditional_start = std::chrono::high_resolution_clock::now();
    ov::Tensor conditional_kernel = build_conditional_kernel(similarity_matrix, relevance_scores);
    auto conditional_end = std::chrono::high_resolution_clock::now();
    auto conditional_duration = std::chrono::duration_cast<std::chrono::microseconds>(conditional_end - conditional_start);
    
    auto kernel_build_end = std::chrono::high_resolution_clock::now();
    auto total_kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_build_end - kernel_build_start);
    
    // Print performance breakdown
    std::cout << "Kernel Build Breakdown:" << std::endl;
    std::cout << "  L2 normalization [" << batch_size << ", " << num_tokens << ", " << feature_dim << "]: " 
              << normalize_duration.count() << " us (" 
              << (static_cast<double>(normalize_duration.count()) / total_kernel_duration.count() * 100) << "%)" << std::endl;
    std::cout << "  Similarity matrix [" << batch_size << ", " << num_tokens << ", " << num_tokens << "]: " 
              << similarity_duration.count() << " us (" 
              << (static_cast<double>(similarity_duration.count()) / total_kernel_duration.count() * 100) << "%)" << std::endl;
    std::cout << "  Conditional kernel [" << batch_size << ", " << num_tokens << ", " << num_tokens << "]: " 
              << conditional_duration.count() << " us (" 
              << (static_cast<double>(conditional_duration.count()) / total_kernel_duration.count() * 100) << "%)" << std::endl;
    
    std::cout << "Total kernel build time: " << total_kernel_duration.count() << " us (" 
              << (total_kernel_duration.count() / 1000.0) << " ms)" << std::endl;
    
    // Performance metrics
    size_t total_operations = batch_size * num_tokens * num_tokens; // Dominant operation is N^2
    std::cout << "Kernel build throughput: " << (static_cast<double>(total_operations) / total_kernel_duration.count() * 1000000) 
              << " ops/sec" << std::endl;
    std::cout << "==========================================\n" << std::endl;
    
    return conditional_kernel;
}

// GPU-accelerated similarity matrix computation using OpenVINO
ov::Tensor ConditionalKernelBuilder::compute_similarity_matrix_gpu(const ov::Tensor& features) {
    // features: [B, N, D] - normalized visual features
    // Result: [B, N, N] - similarity matrix
    
    auto shape = features.get_shape();
    size_t batch_size = shape[0];
    size_t num_tokens = shape[1];
    size_t feature_dim = shape[2];

    if (infer_request) {
        // Set input tensor
        infer_request.set_input_tensor(features);

        // Run inference
        infer_request.infer();

        // Get output tensor
        auto output_tensor = infer_request.get_output_tensor();

        return output_tensor;

    } else {
        // Fallback to CPU implementation if GPU fails
        if (m_config.pruning_debug_mode) {
            std::cout << "GPU MatMul failed, falling back to CPU." << std::endl;
        }
        return compute_similarity_matrix(features);
    }
}

ov::Tensor ConditionalKernelBuilder::compute_similarity_matrix(const ov::Tensor& features) {
    // features: [B, N, D] - normalized visual features
    // Result: [B, N, N] - similarity matrix
    
    auto shape = features.get_shape();
    size_t batch_size = shape[0];
    size_t num_tokens = shape[1];
    size_t feature_dim = shape[2];
    
    ov::Tensor similarity_matrix(ov::element::f32, {batch_size, num_tokens, num_tokens});
    
    const float* features_data = features.data<const float>();
    float* similarity_data = similarity_matrix.data<float>();
    
    // Compute similarity matrix for each batch
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < num_tokens; ++i) {
            for (size_t j = 0; j < num_tokens; ++j) {
                float dot_product = 0.0f;
                
                // Compute dot product between token i and token j in batch b
                for (size_t k = 0; k < feature_dim; ++k) {
                    size_t idx_i = b * num_tokens * feature_dim + i * feature_dim + k;
                    size_t idx_j = b * num_tokens * feature_dim + j * feature_dim + k;
                    dot_product += features_data[idx_i] * features_data[idx_j];
                }
                
                size_t sim_idx = b * num_tokens * num_tokens + i * num_tokens + j;
                similarity_data[sim_idx] = dot_product;
            }
        }
    }
    
    return similarity_matrix;
}

ov::Tensor ConditionalKernelBuilder::create_diagonal_matrix(const ov::Tensor& scores) {
    // scores: [B, N]
    // Result: [B, N, N] - diagonal matrices
    
    auto shape = scores.get_shape();
    size_t batch_size = shape[0];
    size_t num_tokens = shape[1];
    
    ov::Tensor diagonal_matrix(ov::element::f32, {batch_size, num_tokens, num_tokens});
    
    // Initialize with zeros
    float* diag_data = diagonal_matrix.data<float>();
    std::memset(diag_data, 0, diagonal_matrix.get_byte_size());
    
    // Set diagonal elements
    const float* scores_data = scores.data<const float>();
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < num_tokens; ++i) {
            size_t diag_idx = b * num_tokens * num_tokens + i * num_tokens + i;
            size_t score_idx = b * num_tokens + i;
            diag_data[diag_idx] = scores_data[score_idx];
        }
    }
    
    return diagonal_matrix;
}

ov::Tensor ConditionalKernelBuilder::batch_matrix_multiply(const ov::Tensor& a, const ov::Tensor& b) {
    // a: [B, N, K], b: [B, K, M]
    // Result: [B, N, M]
    
    auto a_shape = a.get_shape();
    auto b_shape = b.get_shape();
    
    size_t batch_size = a_shape[0];
    size_t result_rows = a_shape[1];
    size_t inner_dim = a_shape[2];
    size_t result_cols = b_shape[2];
    
    if (b_shape[0] != batch_size || b_shape[1] != inner_dim) {
        throw std::invalid_argument("Incompatible matrix dimensions for batch multiplication");
    }
    
    ov::Tensor result(ov::element::f32, {batch_size, result_rows, result_cols});
    
    const float* a_data = a.data<const float>();
    const float* b_data = b.data<const float>();
    float* result_data = result.data<float>();
    
    // Perform batch matrix multiplication
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t i = 0; i < result_rows; ++i) {
            for (size_t j = 0; j < result_cols; ++j) {
                float sum = 0.0f;
                
                for (size_t k = 0; k < inner_dim; ++k) {
                    size_t a_idx = batch * result_rows * inner_dim + i * inner_dim + k;
                    size_t b_idx = batch * inner_dim * result_cols + k * result_cols + j;
                    sum += a_data[a_idx] * b_data[b_idx];
                }
                
                size_t result_idx = batch * result_rows * result_cols + i * result_cols + j;
                result_data[result_idx] = sum;
            }
        }
    }
    
    return result;
}

ov::Tensor ConditionalKernelBuilder::l2_normalize_features(const ov::Tensor& features) {
    // features: [B, N, D]
    // Result: [B, N, D] - L2 normalized features
    
    auto shape = features.get_shape();
    size_t batch_size = shape[0];
    size_t num_tokens = shape[1];
    size_t feature_dim = shape[2];
    
    ov::Tensor normalized_features(features.get_element_type(), shape);
    
    const float* input_data = features.data<const float>();
    float* output_data = normalized_features.data<float>();
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < num_tokens; ++i) {
            // Compute L2 norm for token (b, i)
            float norm = 0.0f;
            for (size_t j = 0; j < feature_dim; ++j) {
                size_t idx = b * num_tokens * feature_dim + i * feature_dim + j;
                norm += input_data[idx] * input_data[idx];
            }
            norm = std::sqrt(norm + m_config.numerical_threshold); // Add epsilon for stability
            
            // Normalize token (b, i)
            for (size_t j = 0; j < feature_dim; ++j) {
                size_t idx = b * num_tokens * feature_dim + i * feature_dim + j;
                output_data[idx] = input_data[idx] / norm;
            }
        }
    }
    
    return normalized_features;
}

ov::Tensor ConditionalKernelBuilder::build_conditional_kernel(const ov::Tensor& similarity_matrix, const ov::Tensor& relevance_scores) {
    // similarity_matrix: [B, N, N]
    // relevance_scores: [B, N]
    // Result: [B, N, N] - conditional kernel matrix
    
    // Implementation of: kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)
    // This is equivalent to: kernel[b, i, j] = relevance[b, i] * similarity[b, i, j] * relevance[b, j]
    
    auto sim_shape = similarity_matrix.get_shape();
    size_t batch_size = sim_shape[0];
    size_t num_tokens = sim_shape[1];
    
    ov::Tensor conditional_kernel(ov::element::f32, sim_shape);
    
    const float* sim_data = similarity_matrix.data<const float>();
    const float* rel_data = relevance_scores.data<const float>();
    float* kernel_data = conditional_kernel.data<float>();
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < num_tokens; ++i) {
            for (size_t j = 0; j < num_tokens; ++j) {
                size_t sim_idx = b * num_tokens * num_tokens + i * num_tokens + j;
                size_t rel_i_idx = b * num_tokens + i;
                size_t rel_j_idx = b * num_tokens + j;
                
                // Apply conditional weighting: r[i] * similarity[i,j] * r[j]
                kernel_data[sim_idx] = rel_data[rel_i_idx] * sim_data[sim_idx] * rel_data[rel_j_idx];
            }
        }
    }
    
    return conditional_kernel;
}

} // namespace ov::genai::cdpruner 