// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "relevance_calculator.hpp"
#include "openvino/openvino.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace ov::genai::cdpruner {

RelevanceCalculator::RelevanceCalculator(const Config& config) : m_config(config) {
    // Constructor implementation
}

ov::Tensor RelevanceCalculator::compute(const ov::Tensor& visual_embeds, const ov::Tensor& text_embeds) {
    // Input validation
    if (visual_embeds.get_shape().size() != 3) {
        throw std::invalid_argument("Visual embeddings must be 3D tensor [B, N, C]");
    }
    if (text_embeds.get_shape().size() != 2) {
        throw std::invalid_argument("Text embeddings must be 2D tensor [M, C]");
    }
    
    auto visual_shape = visual_embeds.get_shape();
    auto text_shape = text_embeds.get_shape();
    
    size_t batch_size = visual_shape[0];
    size_t num_visual_tokens = visual_shape[1];
    size_t visual_dim = visual_shape[2];
    size_t num_text_tokens = text_shape[0];
    size_t text_dim = text_shape[1];
    
    // For simplicity, we assume visual and text embeddings have the same dimension
    // In practice, they might need to be projected to the same space
    if (visual_dim != text_dim) {
        throw std::invalid_argument("Visual and text embeddings must have the same feature dimension");
    }
    
    // Step 1: L2 normalize visual embeddings along the last dimension
    ov::Tensor visual_normalized = l2_normalize(visual_embeds);
    
    // Step 2: L2 normalize text embeddings along the last dimension  
    ov::Tensor text_normalized = l2_normalize(text_embeds);
    
    // Step 3: Compute cosine similarity between visual and text embeddings
    // relevance = visual_embeds @ text_embeds.T  => [B, N, M]
    ov::Tensor relevance_matrix = matrix_multiply(visual_normalized, text_normalized);
    
    // Step 4: Take negative mean across text tokens dimension to get relevance scores
    // This follows the CDPruner implementation: relevance = (-relevance).mean(dim=-1)
    ov::Tensor relevance_scores = compute_negative_mean(relevance_matrix);
    
    // Step 5: Min-max normalize the relevance scores
    ov::Tensor normalized_relevance = min_max_normalize(relevance_scores);
    
    return normalized_relevance;
}

ov::Tensor RelevanceCalculator::l2_normalize(const ov::Tensor& input) {
    auto shape = input.get_shape();
    ov::Tensor result(input.get_element_type(), shape);
    
    const float* input_data = input.data<const float>();
    float* result_data = result.data<float>();
    
    if (shape.size() == 2) {
        // For 2D tensor [M, C], normalize along last dimension
        size_t num_tokens = shape[0];
        size_t feature_dim = shape[1];
        
        for (size_t i = 0; i < num_tokens; ++i) {
            // Compute L2 norm for token i
            float norm = 0.0f;
            for (size_t j = 0; j < feature_dim; ++j) {
                size_t idx = i * feature_dim + j;
                norm += input_data[idx] * input_data[idx];
            }
            norm = std::sqrt(norm + m_config.numerical_threshold); // Add small epsilon for stability
            
            // Normalize token i
            for (size_t j = 0; j < feature_dim; ++j) {
                size_t idx = i * feature_dim + j;
                result_data[idx] = input_data[idx] / norm;
            }
        }
    } else if (shape.size() == 3) {
        // For 3D tensor [B, N, C], normalize along last dimension
        size_t batch_size = shape[0];
        size_t num_tokens = shape[1];
        size_t feature_dim = shape[2];
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < num_tokens; ++i) {
                // Compute L2 norm for token (b, i)
                float norm = 0.0f;
                for (size_t j = 0; j < feature_dim; ++j) {
                    size_t idx = b * num_tokens * feature_dim + i * feature_dim + j;
                    norm += input_data[idx] * input_data[idx];
                }
                norm = std::sqrt(norm + m_config.numerical_threshold);
                
                // Normalize token (b, i)
                for (size_t j = 0; j < feature_dim; ++j) {
                    size_t idx = b * num_tokens * feature_dim + i * feature_dim + j;
                    result_data[idx] = input_data[idx] / norm;
                }
            }
        }
    } else {
        throw std::invalid_argument("L2 normalization only supports 2D and 3D tensors");
    }
    
    return result;
}

ov::Tensor RelevanceCalculator::min_max_normalize(const ov::Tensor& input) {
    auto shape = input.get_shape();
    ov::Tensor result(input.get_element_type(), shape);
    
    const float* input_data = input.data<const float>();
    float* result_data = result.data<float>();
    
    if (shape.size() == 2) {
        // For 2D tensor [B, N], normalize each batch independently
        size_t batch_size = shape[0];
        size_t num_tokens = shape[1];
        
        for (size_t b = 0; b < batch_size; ++b) {
            // Find min and max for batch b
            float min_val = std::numeric_limits<float>::infinity();
            float max_val = -std::numeric_limits<float>::infinity();
            
            for (size_t i = 0; i < num_tokens; ++i) {
                size_t idx = b * num_tokens + i;
                min_val = std::min(min_val, input_data[idx]);
                max_val = std::max(max_val, input_data[idx]);
            }
            
            // Avoid division by zero
            float range = max_val - min_val;
            if (range < m_config.numerical_threshold) {
                range = 1.0f; // If all values are the same, set to 1
            }
            
            // Normalize batch b
            for (size_t i = 0; i < num_tokens; ++i) {
                size_t idx = b * num_tokens + i;
                result_data[idx] = (input_data[idx] - min_val + m_config.numerical_threshold) / range;
            }
        }
    } else {
        throw std::invalid_argument("Min-max normalization only supports 2D tensors");
    }
    
    return result;
}

ov::Tensor RelevanceCalculator::matrix_multiply(const ov::Tensor& visual_embeds, const ov::Tensor& text_embeds) {
    // visual_embeds: [B, N, C]
    // text_embeds: [M, C] 
    // Result: [B, N, M]
    
    auto visual_shape = visual_embeds.get_shape();
    auto text_shape = text_embeds.get_shape();
    
    size_t batch_size = visual_shape[0];
    size_t num_visual_tokens = visual_shape[1];
    size_t feature_dim = visual_shape[2];
    size_t num_text_tokens = text_shape[0];
    
    ov::Tensor result(ov::element::f32, {batch_size, num_visual_tokens, num_text_tokens});
    
    const float* visual_data = visual_embeds.data<const float>();
    const float* text_data = text_embeds.data<const float>();
    float* result_data = result.data<float>();
    
    // Perform batch matrix multiplication: visual @ text.T
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < num_visual_tokens; ++i) {
            for (size_t j = 0; j < num_text_tokens; ++j) {
                float dot_product = 0.0f;
                
                // Compute dot product between visual token (b,i) and text token j
                for (size_t k = 0; k < feature_dim; ++k) {
                    size_t visual_idx = b * num_visual_tokens * feature_dim + i * feature_dim + k;
                    size_t text_idx = j * feature_dim + k;
                    dot_product += visual_data[visual_idx] * text_data[text_idx];
                }
                
                size_t result_idx = b * num_visual_tokens * num_text_tokens + i * num_text_tokens + j;
                result_data[result_idx] = dot_product;
            }
        }
    }
    
    return result;
}

ov::Tensor RelevanceCalculator::compute_negative_mean(const ov::Tensor& relevance_matrix) {
    // relevance_matrix: [B, N, M]
    // Result: [B, N] - mean across the last dimension with negation
    
    auto shape = relevance_matrix.get_shape();
    size_t batch_size = shape[0];
    size_t num_visual_tokens = shape[1];
    size_t num_text_tokens = shape[2];
    
    ov::Tensor result(ov::element::f32, {batch_size, num_visual_tokens});
    
    const float* input_data = relevance_matrix.data<const float>();
    float* result_data = result.data<float>();
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < num_visual_tokens; ++i) {
            float sum = 0.0f;
            
            // Compute mean across text tokens for visual token (b, i)
            for (size_t j = 0; j < num_text_tokens; ++j) {
                size_t idx = b * num_visual_tokens * num_text_tokens + i * num_text_tokens + j;
                sum += input_data[idx];
            }
            
            float mean_val = sum / static_cast<float>(num_text_tokens);
            size_t result_idx = b * num_visual_tokens + i;
            result_data[result_idx] = -mean_val; // Apply negation as in CDPruner
        }
    }
    
    return result;
}

} // namespace ov::genai::cdpruner 