// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "conditional_kernel.hpp"

#include <algorithm>
#include <cmath>
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
        GENAI_WARN("[CDPruner] ConditionalKernelBuilder: InferRequest initialization failed, will use fallback: " +
                   std::string(e.what()));
        m_requests_initialized = false;
    }
}

ov::Tensor ConditionalKernelBuilder::build(const ov::Tensor& visual_features, const ov::Tensor& input_param) {
    // Input validation
    OPENVINO_ASSERT(visual_features.get_shape().size() == 3, "Visual features must be 3D tensor [B, N, D]");

    auto visual_shape = visual_features.get_shape();
    size_t batch_size = visual_shape[0];
    size_t num_tokens = visual_shape[1];
    size_t feature_dim = visual_shape[2];

    ov::Tensor conditional_kernel;
    try {
        conditional_kernel = build_with_ov_model(visual_features, input_param);
    } catch (const std::exception& e) {
        GENAI_WARN("[CDPruner] ConditionalKernelBuilder: OV model failed, falling back to normal pipeline: " +
                   std::string(e.what()));
        conditional_kernel = build_with_normal_pipeline(visual_features, input_param);
    }

    return conditional_kernel;
}

ov::Tensor ConditionalKernelBuilder::build_with_ov_model(const ov::Tensor& visual_features,
                                                         const ov::Tensor& text_features) {
    // Use OV model for building the kernel matrix
    OPENVINO_ASSERT(text_features.get_shape().size() == 2, "Text features must be 2D tensor [N, D]");

    auto visual_shape = visual_features.get_shape();
    size_t feature_dim = visual_shape[2];

    // Check shape consistency
    OPENVINO_ASSERT(text_features.get_shape()[1] == feature_dim,
                    "Visual features and text features must have the same feature dimension");
    ov::Tensor conditional_kernel = compute_conditional_kernel_with_model(visual_features, text_features);
    return conditional_kernel;
}

ov::Tensor ConditionalKernelBuilder::build_with_normal_pipeline(const ov::Tensor& visual_features,
                                                                const ov::Tensor& relevance_scores) {
    auto visual_shape = visual_features.get_shape();
    size_t batch_size = visual_shape[0];
    size_t num_tokens = visual_shape[1];
    size_t feature_dim = visual_shape[2];

    OPENVINO_ASSERT(relevance_scores.get_shape().size() == 2, "Relevance scores must be 2D tensor [B, N]");
    auto relevance_shape = relevance_scores.get_shape();
    // Check shape consistency
    OPENVINO_ASSERT(relevance_shape[0] == batch_size && relevance_shape[1] == num_tokens,
                    "Visual features and relevance scores must have consistent batch size and token count");

    // Step 1: L2 normalize visual features along the last dimension
    // This is equivalent to: image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)
    ov::Tensor normalized_features = l2_normalize_features(visual_features);

    // Step 2: Compute similarity matrix L = normalized_features @ normalized_features.T
    // This gives us the base similarity matrix [B, N, N]
    ov::Tensor similarity_matrix = compute_similarity_matrix_with_model(normalized_features);

    // Step 3: Build conditional kernel matrix L̃ = diag(r) · L · diag(r)
    // Following CDPruner: kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)
    ov::Tensor conditional_kernel = compute_conditional_kernel_normal(similarity_matrix, relevance_scores);

    return conditional_kernel;
}

// GPU-accelerated similarity matrix computation using OpenVINO
ov::Tensor ConditionalKernelBuilder::compute_similarity_matrix_with_model(const ov::Tensor& features) {
    // features: [B, N, D] - normalized visual features
    // Result: [B, N, N] - similarity matrix

    if (!m_requests_initialized) {
        // Fallback to CPU implementation if infer requests not initialized
        GENAI_WARN("[CDPruner] Using CPU fallback for similarity matrix computation");
        return compute_similarity_matrix(features);
    }

    try {
        // Use preinitialized infer request
        m_similarity_infer_request.set_input_tensor(features);
        m_similarity_infer_request.infer();
        auto output_tensor_ref = m_similarity_infer_request.get_output_tensor();

        // Create deep copy to avoid data corruption when InferRequest is reused
        ov::Tensor output_tensor(output_tensor_ref.get_element_type(), output_tensor_ref.get_shape());
        std::memcpy(output_tensor.data(), output_tensor_ref.data(), output_tensor_ref.get_byte_size());

        return output_tensor;

    } catch (const std::exception& e) {
        // Fallback to CPU implementation if GPU fails
        GENAI_WARN("[CDPruner] GPU MatMul failed, falling back to CPU.");
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

    OPENVINO_ASSERT(b_shape[0] == batch_size && b_shape[1] == inner_dim,
                    "Incompatible matrix dimensions for batch multiplication");

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

    // CDPruner's relevance weighting formula:
    //
    // 1. Exponential relevance transformation:
    //    α = θ / (2 × (1 - θ)), where θ is relevance_weight
    //    weighted_relevance[i] = exp(α × normalized_relevance[i])
    //
    // 2. Conditional kernel construction:
    //    kernel[i,j] = weighted_relevance[i] × similarity[i,j] × weighted_relevance[j]
    const float w = m_config.relevance_weight;

    float alpha = 0.0f;
    if (w < 1.0f && w > 0.0f) {
        alpha = w / (2.0f * (1.0f - w));
    }

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < num_tokens; ++i) {
            for (size_t j = 0; j < num_tokens; ++j) {
                size_t sim_idx = b * num_tokens * num_tokens + i * num_tokens + j;
                size_t rel_i_idx = b * num_tokens + i;
                size_t rel_j_idx = b * num_tokens + j;

                float base_similarity = sim_data[sim_idx];

                if (w == 0.0f) {
                    // Pure similarity matrix (no relevance weighting)
                    kernel_data[sim_idx] = base_similarity;
                } else if (w == 1.0f) {
                    // Pure CDPruner conditional weighting (no exponential transform)
                    // Direct multiplication: relevance[i] × similarity[i,j] × relevance[j]
                    float conditional_weight = rel_data[rel_i_idx] * base_similarity * rel_data[rel_j_idx];
                    kernel_data[sim_idx] = conditional_weight;
                } else {
                    // CDPruner with exponential relevance transformation (0 < w < 1)
                    float weighted_rel_i = std::exp(alpha * rel_data[rel_i_idx]);
                    float weighted_rel_j = std::exp(alpha * rel_data[rel_j_idx]);
                    kernel_data[sim_idx] = weighted_rel_i * base_similarity * weighted_rel_j;
                }
            }
        }
    }

    return conditional_kernel;
}

ov::Tensor ConditionalKernelBuilder::compute_conditional_kernel_with_model(const ov::Tensor& visual_features,
                                                                           const ov::Tensor& text_features) {
    // Input validation
    OPENVINO_ASSERT(visual_features.get_shape().size() == 3, "Visual features must be 3D tensor [B, N, D]");
    OPENVINO_ASSERT(text_features.get_shape().size() == 2, "Text features must be 2D tensor [M, D]");

    auto visual_shape = visual_features.get_shape();
    auto text_shape = text_features.get_shape();

    OPENVINO_ASSERT(visual_shape[2] == text_shape[1], "Visual and text features must have same feature dimension");

    OPENVINO_ASSERT(m_requests_initialized,
                    "Conditional kernel infer request not initialized. Cannot use GPU acceleration.");

    // Use preinitialized infer request
    m_conditional_kernel_infer_request.set_input_tensor(0, visual_features);
    m_conditional_kernel_infer_request.set_input_tensor(1, text_features);
    m_conditional_kernel_infer_request.infer();

    // Get output tensor reference
    auto conditional_kernel_ref = m_conditional_kernel_infer_request.get_output_tensor(0);

    // CRITICAL: Create a deep copy to avoid data corruption when InferRequest is reused
    // The InferRequest may reuse or modify its output tensor on the next inference call,
    // which would invalidate any references to this tensor
    ov::Tensor conditional_kernel(conditional_kernel_ref.get_element_type(), conditional_kernel_ref.get_shape());
    std::memcpy(conditional_kernel.data(), conditional_kernel_ref.data(), conditional_kernel_ref.get_byte_size());

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

    return std::make_shared<ov::Model>(ov::ResultVector{std::move(result)},
                                       ov::ParameterVector{std::move(input_features)},
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
    auto axes_text = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
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
    auto relevance_scores = create_min_max_normalize_ops(std::move(processed_mean));

    // ========== KERNEL COMPUTATION ==========
    // Step 2.1: Compute visual self-similarity matrix [B, N, N]
    auto visual_transposed = std::make_shared<ov::op::v1::Transpose>(
        visual_l2_norm,  // Reuse normalized visual features
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1}));

    auto visual_self_similarity = std::make_shared<ov::op::v0::MatMul>(visual_l2_norm, visual_transposed);

    // Step 2.2: Build conditional kernel matrix with relevance weighting
    auto relevance_i = std::make_shared<ov::op::v0::Unsqueeze>(
        relevance_scores,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2})  // axis=2
    );
    auto relevance_j = std::make_shared<ov::op::v0::Unsqueeze>(
        relevance_scores,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1})  // axis=1
    );

    // Conditional weighting: relevance[i] * similarity[i,j] * relevance[j]
    auto temp_kernel = std::make_shared<ov::op::v1::Multiply>(relevance_i, visual_self_similarity);
    auto conditional_weighted = std::make_shared<ov::op::v1::Multiply>(temp_kernel, relevance_j);

    // Blend between pure similarity and conditional weighting using relevance_weight
    // kernel = (1-w) * similarity + w * conditional_weighted
    auto weight = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {m_config.relevance_weight});
    auto one_minus_weight =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f - m_config.relevance_weight});

    auto similarity_part = std::make_shared<ov::op::v1::Multiply>(visual_self_similarity, one_minus_weight);
    auto conditional_part = std::make_shared<ov::op::v1::Multiply>(conditional_weighted, weight);
    auto conditional_kernel = std::make_shared<ov::op::v1::Add>(similarity_part, conditional_part);

    // Create outputs
    auto kernel_result = std::make_shared<ov::op::v0::Result>(conditional_kernel);

    return std::make_shared<ov::Model>(ov::ResultVector{std::move(kernel_result)},
                                       ov::ParameterVector{std::move(visual_input), std::move(text_input)},
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
