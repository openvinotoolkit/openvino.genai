// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "kernel_builder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "logger.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/openvino.hpp"
#include "utils.hpp"

namespace ov::genai::cdpruner {

ConditionalKernelBuilder::ConditionalKernelBuilder(const Config& config)
    : m_config(config),
      m_requests_initialized(false) {
    // Precompile models and create infer requests for performance optimization
    try {
        ov::Core core;

        // Compile and create infer request for conditional kernel model
        auto kernel_model = create_conditional_kernel_model();
        ov::CompiledModel compiled_kernel_model;
        compiled_kernel_model = core.compile_model(kernel_model, m_config.device);
        m_conditional_kernel_infer_request = compiled_kernel_model.create_infer_request();

        // Always compile similarity matrix model for potential GPU acceleration
        auto similarity_model = create_similarity_matrix_model();
        ov::CompiledModel compiled_similarity_model;
        compiled_similarity_model = core.compile_model(similarity_model, m_config.device);
        m_similarity_infer_request = compiled_similarity_model.create_infer_request();

        m_requests_initialized = true;

    } catch (const std::exception& e) {
        Logger::warn("[CDPruner] ConditionalKernelBuilder: InferRequest initialization failed, will use fallback: " +
                     std::string(e.what()));
        m_requests_initialized = false;
    }
}

ov::Tensor ConditionalKernelBuilder::build(const ov::Tensor& visual_features, const ov::Tensor& input_param) {
    // Input validation
    if (visual_features.get_shape().size() != 3) {
        throw std::invalid_argument("Visual features must be 3D tensor [B, N, D]");
    }

    auto visual_shape = visual_features.get_shape();
    size_t batch_size = visual_shape[0];
    size_t num_tokens = visual_shape[1];
    size_t feature_dim = visual_shape[2];

    ov::Tensor conditional_kernel;
    try {
        conditional_kernel = build_with_ov_model(visual_features, input_param);
    } catch (const std::exception& e) {
        Logger::warn("[CDPruner] ConditionalKernelBuilder: OV model failed, falling back to normal pipeline: " +
                     std::string(e.what()));
        conditional_kernel = build_with_normal_pipeline(visual_features, input_param);
    }

    return conditional_kernel;
}

ov::Tensor ConditionalKernelBuilder::build_with_ov_model(const ov::Tensor& visual_features,
                                                         const ov::Tensor& text_features) {
    auto visual_shape = visual_features.get_shape();
    size_t batch_size = visual_shape[0];
    size_t num_tokens = visual_shape[1];
    size_t feature_dim = visual_shape[2];
    size_t total_operations = batch_size * num_tokens * num_tokens;  // Dominant operation is N^2

    // Use OV model for building the kernel matrix
    if (text_features.get_shape().size() != 2) {
        throw std::invalid_argument("Text features must be 2D tensor [N, D]");
    }

    // Check shape consistency
    if (text_features.get_shape()[1] != feature_dim) {
        throw std::invalid_argument("Visual features and text features must have the same feature dimension");
    }
    ov::Tensor conditional_kernel = compute_conditional_kernel_with_model(visual_features, text_features);
    return conditional_kernel;
}

ov::Tensor ConditionalKernelBuilder::build_with_normal_pipeline(const ov::Tensor& visual_features,
                                                                const ov::Tensor& relevance_scores) {
    auto visual_shape = visual_features.get_shape();
    size_t batch_size = visual_shape[0];
    size_t num_tokens = visual_shape[1];
    size_t feature_dim = visual_shape[2];
    size_t total_operations = batch_size * num_tokens * num_tokens;  // Dominant operation is N^2

    if (relevance_scores.get_shape().size() != 2) {
        throw std::invalid_argument("Relevance scores must be 2D tensor [B, N]");
    }
    auto relevance_shape = relevance_scores.get_shape();
    // Check shape consistency
    if (relevance_shape[0] != batch_size || relevance_shape[1] != num_tokens) {
        throw std::invalid_argument(
            "Visual features and relevance scores must have consistent batch size and token count");
    }

    // Performance timing for kernel building steps
    auto kernel_build_start = std::chrono::high_resolution_clock::now();

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
    ov::Tensor similarity_matrix = compute_similarity_matrix_with_model(normalized_features);
    auto similarity_end = std::chrono::high_resolution_clock::now();
    auto similarity_duration = std::chrono::duration_cast<std::chrono::microseconds>(similarity_end - similarity_start);

    // Step 3: Build conditional kernel matrix L̃ = diag(r) · L · diag(r)
    // Following CDPruner: kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)
    auto conditional_start = std::chrono::high_resolution_clock::now();
    ov::Tensor conditional_kernel = compute_conditional_kernel_normal(similarity_matrix, relevance_scores);
    auto conditional_end = std::chrono::high_resolution_clock::now();
    auto conditional_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(conditional_end - conditional_start);

    auto kernel_build_end = std::chrono::high_resolution_clock::now();
    auto total_kernel_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(kernel_build_end - kernel_build_start);

    print_kernel_build_performance(batch_size,
                                   num_tokens,
                                   feature_dim,
                                   total_operations,
                                   normalize_duration,
                                   similarity_duration,
                                   conditional_duration,
                                   total_kernel_duration);

    return conditional_kernel;
}

void ConditionalKernelBuilder::print_kernel_build_performance(size_t batch_size,
                                                              size_t num_tokens,
                                                              size_t feature_dim,
                                                              size_t total_operations,
                                                              const std::chrono::microseconds& normalize_duration,
                                                              const std::chrono::microseconds& similarity_duration,
                                                              const std::chrono::microseconds& conditional_duration,
                                                              const std::chrono::microseconds& total_kernel_duration) {
    if (!utils::env_setup_for_print_debug_info())
        return;
    std::ostringstream ss;
    ss << "[CDPruner]   L2 normalization [" << batch_size << ", " << num_tokens << ", " << feature_dim
       << "]: " << normalize_duration.count() << " us ("
       << (static_cast<double>(normalize_duration.count()) / total_kernel_duration.count() * 100) << "%)" << std::endl;
    ss << "[CDPruner]   Similarity matrix [" << batch_size << ", " << num_tokens << ", " << num_tokens
       << "]: " << similarity_duration.count() << " us ("
       << (static_cast<double>(similarity_duration.count()) / total_kernel_duration.count() * 100) << "%)" << std::endl;
    ss << "[CDPruner]   Conditional kernel [" << batch_size << ", " << num_tokens << ", " << num_tokens
       << "]: " << conditional_duration.count() << " us ("
       << (static_cast<double>(conditional_duration.count()) / total_kernel_duration.count() * 100) << "%)"
       << std::endl;
    ss << "[CDPruner] Total kernel build time: " << total_kernel_duration.count() << " us ("
       << (total_kernel_duration.count() / 1000.0) << " ms)" << std::endl;
    ss << "[CDPruner] Kernel build throughput: "
       << (static_cast<double>(total_operations) / total_kernel_duration.count() * 1000000) << " ops/sec" << std::endl;
    std::cout << ss.str();
}

// GPU-accelerated similarity matrix computation using OpenVINO
ov::Tensor ConditionalKernelBuilder::compute_similarity_matrix_with_model(const ov::Tensor& features) {
    // features: [B, N, D] - normalized visual features
    // Result: [B, N, N] - similarity matrix

    if (!m_requests_initialized) {
        // Fallback to CPU implementation if infer requests not initialized
        Logger::warn("[CDPruner] Using CPU fallback for similarity matrix computation");
        return compute_similarity_matrix(features);
    }

    try {
        // Use preinitialized infer request
        m_similarity_infer_request.set_input_tensor(features);
        m_similarity_infer_request.infer();
        auto output_tensor = m_similarity_infer_request.get_output_tensor();

        return output_tensor;

    } catch (const std::exception& e) {
        // Fallback to CPU implementation if GPU fails
        Logger::warn("[CDPruner] GPU MatMul failed, falling back to CPU.");
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
            norm = std::sqrt(norm + m_config.numerical_threshold);  // Add epsilon for stability

            // Normalize token (b, i)
            for (size_t j = 0; j < feature_dim; ++j) {
                size_t idx = b * num_tokens * feature_dim + i * feature_dim + j;
                output_data[idx] = input_data[idx] / norm;
            }
        }
    }

    return normalized_features;
}

ov::Tensor ConditionalKernelBuilder::compute_conditional_kernel_normal(const ov::Tensor& similarity_matrix,
                                                                       const ov::Tensor& relevance_scores) {
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

ov::Tensor ConditionalKernelBuilder::compute_conditional_kernel_with_model(const ov::Tensor& visual_features,
                                                                           const ov::Tensor& text_features) {
    // Input validation
    if (visual_features.get_shape().size() != 3) {
        throw std::invalid_argument("Visual features must be 3D tensor [B, N, D]");
    }
    if (text_features.get_shape().size() != 2) {
        throw std::invalid_argument("Text features must be 2D tensor [M, D]");
    }

    auto visual_shape = visual_features.get_shape();
    auto text_shape = text_features.get_shape();

    if (visual_shape[2] != text_shape[1]) {
        throw std::invalid_argument("Visual and text features must have same feature dimension");
    }

    if (!m_requests_initialized) {
        throw std::runtime_error("Conditional kernel infer request not initialized. Cannot use GPU acceleration.");
    }

    // Use preinitialized infer request
    m_conditional_kernel_infer_request.set_input_tensor(0, visual_features);  // visual features
    m_conditional_kernel_infer_request.set_input_tensor(1, text_features);    // text features
    m_conditional_kernel_infer_request.infer();

    // Get output tensor
    auto conditional_kernel = m_conditional_kernel_infer_request.get_output_tensor(0);

    return conditional_kernel;
}

std::shared_ptr<ov::Model> ConditionalKernelBuilder::create_similarity_matrix_model() {
    // Create a simple MatMul model for similarity matrix computation
    // Input: [B, N, D] normalized features
    // Output: [B, N, N] similarity matrix

    auto input_features = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1});

    // Transpose features for batch matrix multiplication: [B, N, D] -> [B, D, N]
    auto axes_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
    auto features_transposed = std::make_shared<ov::op::v1::Transpose>(input_features, axes_order);

    // Batch matrix multiplication: [B, N, D] @ [B, D, N] = [B, N, N]
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input_features, features_transposed);

    auto result = std::make_shared<ov::op::v0::Result>(matmul);

    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       ov::ParameterVector{input_features},
                                       "SimilarityMatrix_Model");
}

std::shared_ptr<ov::Model> ConditionalKernelBuilder::create_conditional_kernel_model() {
    // Input parameters
    auto visual_input =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1}  // [B, N, D]
        );
    auto text_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1}  // [M, D]
    );

    // ========== RELEVANCE COMPUTATION ==========
    // Step 1.1: L2 normalize visual features (will be reused for kernel computation)
    auto axes = ov::op::v0::Constant::create(ov::element::i32, {1}, {2});
    auto visual_l2_norm = std::make_shared<ov::op::v0::NormalizeL2>(visual_input,
                                                                    axes,
                                                                    m_config.numerical_threshold,
                                                                    ov::op::EpsMode::ADD);

    // Step 1.2: L2 normalize text features
    auto axes_text = ov::op::v0::Constant::create(element::i32, {1}, {1});
    auto text_l2_norm = std::make_shared<ov::op::v0::NormalizeL2>(text_input,
                                                                  axes_text,
                                                                  m_config.numerical_threshold,
                                                                  ov::op::EpsMode::ADD);

    // Step 1.3: Compute similarity matrix [B, N, M]
    auto text_transposed =
        std::make_shared<ov::op::v1::Transpose>(text_l2_norm,
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0}));

    auto text_visual_similarity = std::make_shared<ov::op::v0::MatMul>(visual_l2_norm, text_transposed);

    // Step 1.4: Compute mean similarity over text dimension
    auto mean_similarity = std::make_shared<ov::op::v1::ReduceMean>(
        text_visual_similarity,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),  // axis=2
        false                                                               // keep_dims=false
    );

    // Step 1.5: Apply negative transformation based on configuration
    std::shared_ptr<ov::Node> processed_mean;
    if (m_config.use_negative_relevance) {
        processed_mean = std::make_shared<ov::op::v1::Multiply>(
            mean_similarity,
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f}));
    } else {
        processed_mean = mean_similarity;
    }

    // Step 1.6: Min-max normalization to get relevance scores
    auto relevance_scores = create_min_max_normalize_ops(processed_mean);

    // ========== KERNEL COMPUTATION ==========
    // Step 2.1: Compute visual self-similarity matrix [B, N, N]
    auto visual_transposed = std::make_shared<ov::op::v1::Transpose>(
        visual_l2_norm,  // Reuse normalized visual features
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1}));

    auto visual_self_similarity = std::make_shared<ov::op::v0::MatMul>(visual_l2_norm, visual_transposed);

    // Step 2.2: Build conditional kernel matrix
    auto relevance_i = std::make_shared<ov::op::v0::Unsqueeze>(
        relevance_scores,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2})  // axis=2
    );
    auto relevance_j = std::make_shared<ov::op::v0::Unsqueeze>(
        relevance_scores,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1})  // axis=1
    );

    // Conditional kernel: relevance[i] * similarity[i,j] * relevance[j]
    auto temp_kernel = std::make_shared<ov::op::v1::Multiply>(relevance_i, visual_self_similarity);
    auto conditional_kernel = std::make_shared<ov::op::v1::Multiply>(temp_kernel, relevance_j);

    // Create outputs
    auto kernel_result = std::make_shared<ov::op::v0::Result>(conditional_kernel);

    return std::make_shared<ov::Model>(ov::ResultVector{kernel_result},
                                       ov::ParameterVector{visual_input, text_input},
                                       "CDPruner_Kernel_Model");
}

std::shared_ptr<ov::Node> ConditionalKernelBuilder::create_min_max_normalize_ops(std::shared_ptr<ov::Node> input) {
    // Find min and max values along the last dimension
    auto min_vals = std::make_shared<ov::op::v1::ReduceMin>(
        input,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),  // axis=1
        true                                                                // keep_dims=true
    );
    auto max_vals = std::make_shared<ov::op::v1::ReduceMax>(
        input,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),  // axis=1
        true                                                                // keep_dims=true
    );

    // Compute range (max - min)
    auto range = std::make_shared<ov::op::v1::Subtract>(max_vals, min_vals);

    // Add small epsilon to avoid division by zero
    auto epsilon = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {m_config.numerical_threshold});
    auto range_with_eps = std::make_shared<ov::op::v1::Add>(range, epsilon);

    // Normalize: (input - min) / (max - min)
    auto shifted = std::make_shared<ov::op::v1::Subtract>(input, min_vals);
    return std::make_shared<ov::op::v1::Divide>(shifted, range_with_eps);
}

}  // namespace ov::genai::cdpruner