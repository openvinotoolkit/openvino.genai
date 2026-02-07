// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file cdpruner.cpp
 * @brief Conditional Determinantal Point Process (DPP) based Visual Token Pruner
 *
 * Algorithm Workflow:
 * ===================
 *
 * The CDPruner implements a three-stage pipeline for visual token pruning:
 *
 * 1. Relevance Calculation (RelevanceCalculator)
 *    - Computes relevance scores between visual tokens and text features
 *    - Uses cosine similarity: score = (V · T) / (||V|| * ||T||)
 *    - Input: Visual features [B, N, D] and Text features [M, D]
 *    - Output: Relevance scores [B, N, M]
 *
 * 2. Conditional Kernel Construction (ConditionalKernelBuilder)
 *    - Builds a conditional kernel matrix combining:
 *      a) Token similarity: S[i,j] = visual[i] · visual[j]
 *      b) Text relevance: R[i] = max(relevance_scores[i])
 *    - Kernel formula: K[i,j] = λ·R[i]·R[j]·S[i,j] + (1-λ)·S[i,j]
 *    - λ = relevance_weight (default: 0.5)
 *    - Output: Kernel matrix [N, N]
 *
 * 3. DPP-based Token Selection (FastGreedyDPP)
 *    - Selects diverse and relevant tokens using greedy DPP algorithm
 *    - Maximizes: Quality (relevance) + Diversity (orthogonality)
 *    - Iteratively selects tokens that maximize marginal gain
 *    - Supports both traditional CPU and OpenCL GPU acceleration
 *    - Output: Selected token indices [T] where T = N * (1 - pruning_ratio/100)
 *
 * Key Features:
 * - Adaptive pruning based on token count (split_threshold for large sequences)
 * - Quality-diversity trade-off controlled by relevance_weight
 * - Configurable pruning ratio (0-100%)
 * - Optional OpenCL GPU acceleration for DPP selection
 *
 * Usage Example:
 * ```cpp
 * Config config;
 * config.pruning_ratio = 50;  // Keep 50% of tokens
 * config.relevance_weight = 1.0;
 *
 * CDPruner pruner(config);
 * auto pruned_features = pruner.apply_pruning(visual_features, text_features);
 * ```
 */

#include "cdpruner.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "logger.hpp"
#include "openvino/openvino.hpp"
#include "utils.hpp"

namespace ov::genai::cdpruner {

CDPruner::CDPruner(const Config& config)
    : m_config(config),
      m_relevance_calc(config),
      m_kernel_builder(config),
      m_dpp_selector(config) {
    m_config.update_from_env();
    validate_config(config);
}

std::vector<std::vector<size_t>> CDPruner::select_tokens(const ov::Tensor& visual_features,
                                                         const ov::Tensor& text_features,
                                                         bool silent) {
    validate_input_tensors(visual_features, text_features);

    const auto& visual_shape = visual_features.get_shape();
    size_t batch_size = visual_shape[0];
    size_t total_tokens = visual_shape[1];
    size_t feature_dim = visual_shape[2];

    size_t num_tokens_to_keep = static_cast<size_t>(std::round(total_tokens * (1.0 - m_config.pruning_ratio / 100.0)));

    try {
        std::vector<std::vector<size_t>> selected_tokens;

        // Determine whether to use token splitting strategy
        // Splitting is introduced to handle large token sequences more efficiently:
        // 1. Computational efficiency: Greedy DPP algorithm complexity is O(N×M²) where N is total
        //    tokens and M is tokens to select. Splitting into halves could provide ~4× speedup for large sequences.
        // 2. Parallel processing: Two halves can be processed independently, enabling parallelization.
        bool use_splitting = m_config.split_threshold > 0 && total_tokens > m_config.split_threshold;

        size_t split_point = 0;
        size_t first_half_size = 0;
        size_t second_half_size = 0;

        if (use_splitting) {
            split_point = total_tokens / 2;
            first_half_size = split_point;
            second_half_size = total_tokens - split_point;
        }

        const float* visual_data = visual_features.data<const float>();
        ov::Tensor visual_first_half;
        ov::Tensor visual_second_half;

        if (use_splitting) {
            visual_first_half = ov::Tensor(ov::element::f32,
                                           {batch_size, first_half_size, feature_dim},
                                           const_cast<float*>(visual_data));
            visual_second_half = ov::Tensor(ov::element::f32,
                                            {batch_size, second_half_size, feature_dim},
                                            const_cast<float*>(visual_data + batch_size * first_half_size * feature_dim));
        } else {
            visual_first_half = visual_features;
        }

        ov::Tensor kernel_matrix_first;
        ov::Tensor kernel_matrix_second;

        try {
            // OpenVINO ops model approach - compute relevance scores and build kernel matrix via OV model
            auto kernel_start = std::chrono::high_resolution_clock::now();

            // Building single kernel matrix for all the tokens or first half.
            kernel_matrix_first = m_kernel_builder.build(visual_first_half, text_features);
            if (use_splitting) {
                // Building kernel matrix for second half
                kernel_matrix_second = m_kernel_builder.build(visual_second_half, text_features);
            }

            auto kernel_end = std::chrono::high_resolution_clock::now();
            auto kernel_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(kernel_end - kernel_start).count();
            if (!silent) {
                GENAI_DEBUG("Kernel building (OV model) time: %ld ms", kernel_duration);
            }
        } catch (const std::exception&) {
            // Fallback to CPU-based kernel construction when GPU/OV model approach fails
            // Using traditional step-by-step computation:
            auto fallback_start = std::chrono::high_resolution_clock::now();

            if (use_splitting) {
                // Compute relevance scores for both halves
                ov::Tensor relevance_scores_first = m_relevance_calc.compute(visual_first_half, text_features);
                ov::Tensor relevance_scores_second = m_relevance_calc.compute(visual_second_half, text_features);

                // Build kernel matrices using relevance scores
                kernel_matrix_first = m_kernel_builder.build(visual_first_half, relevance_scores_first);
                kernel_matrix_second = m_kernel_builder.build(visual_second_half, relevance_scores_second);
            } else {
                // Process entire tensor without splitting
                ov::Tensor relevance_scores = m_relevance_calc.compute(visual_first_half, text_features);

                // Build single kernel matrix using relevance scores
                kernel_matrix_first = m_kernel_builder.build(visual_first_half, relevance_scores);
            }

            auto fallback_end = std::chrono::high_resolution_clock::now();
            auto fallback_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(fallback_end - fallback_start).count();
            if (!silent) {
                GENAI_DEBUG("Kernel building (CPU fallback) time: %ld ms", fallback_duration);
            }
        }

        // Apply DPP selection
        auto dpp_start = std::chrono::high_resolution_clock::now();

        if (use_splitting) {
            // Use parallel DPP selection for split processing
            selected_tokens =
                m_dpp_selector.select(kernel_matrix_first, kernel_matrix_second, num_tokens_to_keep, split_point);
        } else {
            // Direct DPP selection for single tensor processing
            selected_tokens = m_dpp_selector.select(kernel_matrix_first, num_tokens_to_keep);
        }

        auto dpp_end = std::chrono::high_resolution_clock::now();
        auto dpp_duration = std::chrono::duration_cast<std::chrono::milliseconds>(dpp_end - dpp_start).count();
        if (!silent) {
            GENAI_DEBUG("DPP selection time: %ld ms", dpp_duration);
        }

        return selected_tokens;
    } catch (const std::exception& e) {
        OPENVINO_THROW("CDPruner::select_tokens failed: ", e.what());
    }
}

ov::Tensor CDPruner::apply_pruning(const ov::Tensor& visual_features, const ov::Tensor& text_features, bool silent) {
    if (m_config.pruning_ratio <= 0 || m_config.pruning_ratio >= 100)
        return visual_features;
    const auto& visual_shape = visual_features.get_shape();
    const auto& text_shape = text_features.get_shape();
    size_t batch_size = visual_shape[0];
    size_t total_tokens = visual_shape[1];
    size_t feature_dim = visual_shape[2];

    size_t num_tokens_to_keep = static_cast<size_t>(std::round(total_tokens * (1 - m_config.pruning_ratio / 100.0)));

    auto selected_tokens = select_tokens(visual_features, text_features, silent);
    m_last_selected_tokens = selected_tokens;

    // Determine actual number of selected tokens (may differ from num_tokens_to_keep due to odd->even adjustment)
    size_t actual_selected_tokens = selected_tokens.empty() ? 0 : selected_tokens[0].size();

    ov::Tensor pruned_features(visual_features.get_element_type(), {batch_size, actual_selected_tokens, feature_dim});

    const float* input_data = visual_features.data<const float>();
    float* output_data = pruned_features.data<float>();

    const size_t feature_size_bytes = feature_dim * sizeof(float);

    for (size_t b = 0; b < batch_size; ++b) {
        const auto& batch_selected = selected_tokens[b];

        for (size_t t = 0; t < batch_selected.size(); ++t) {
            size_t src_token_idx = batch_selected[t];
            const float* src_ptr = input_data + b * total_tokens * feature_dim + src_token_idx * feature_dim;
            float* dst_ptr = output_data + b * actual_selected_tokens * feature_dim + t * feature_dim;
            std::memcpy(dst_ptr, src_ptr, feature_size_bytes);
        }
    }

    m_last_statistics.total_tokens = total_tokens;
    m_last_statistics.selected_tokens = actual_selected_tokens;
    m_last_statistics.pruning_ratio = 1.0f - static_cast<float>(actual_selected_tokens) / total_tokens;
    m_last_statistics.batch_size = batch_size;

    return pruned_features;
}

ov::Tensor CDPruner::apply_pruning(const std::vector<ov::Tensor>& visual_features_list,
                                   const ov::Tensor& text_features) {
    if (visual_features_list.empty()) {
        return ov::Tensor();
    }

    // Handle single feature case by calling existing method
    if (visual_features_list.size() == 1u) {
        return apply_pruning(visual_features_list[0], text_features);
    }

    const auto& first_feature = visual_features_list[0];
    const auto& visual_shape = first_feature.get_shape();
    const auto& text_shape = text_features.get_shape();
    size_t batch_size = visual_shape[0];
    size_t tokens_per_frame = visual_shape[1];
    size_t feature_dim = visual_shape[2];
    size_t total_input_tokens = tokens_per_frame * visual_features_list.size();

    size_t num_tokens_to_keep =
        static_cast<size_t>(std::round(tokens_per_frame * (1 - m_config.pruning_ratio / 100.0)));

    // Apply pruning to each visual feature and collect results (using silent mode)
    std::vector<ov::Tensor> pruned_features_list;
    pruned_features_list.reserve(visual_features_list.size());
    std::vector<std::vector<size_t>> aggregated_selected;

    size_t global_offset = 0;
    GENAI_DEBUG("[CDPruner] Multi-frame pruning: %zu frames", visual_features_list.size());

    for (size_t frame_idx = 0; frame_idx < visual_features_list.size(); ++frame_idx) {
        const auto& visual_feature = visual_features_list[frame_idx];
        size_t frame_tokens = visual_feature.get_shape()[1];
        ov::Tensor pruned_feature = apply_pruning(visual_feature, text_features, true);
        size_t pruned_tokens = pruned_feature.get_shape()[1];
        const auto& frame_selected = m_last_selected_tokens;
        if (aggregated_selected.empty()) {
            aggregated_selected.resize(frame_selected.size());
        }
        for (size_t batch_idx = 0; batch_idx < frame_selected.size(); ++batch_idx) {
            auto& aggregated = aggregated_selected[batch_idx];
            const auto& frame_indices = frame_selected[batch_idx];
            aggregated.reserve(aggregated.size() + frame_indices.size());
            for (size_t index : frame_indices) {
                aggregated.push_back(index + global_offset);
            }
        }
        pruned_features_list.push_back(std::move(pruned_feature));
        global_offset += visual_feature.get_shape()[1];
    }

    m_last_selected_tokens = std::move(aggregated_selected);
    // Calculate actual total tokens by summing each frame's pruned tokens
    // (frames may have different sizes after pruning)
    const auto& first_pruned_feature = pruned_features_list[0];
    const size_t actual_batch_size = first_pruned_feature.get_shape()[0];
    const size_t actual_hidden_dim = first_pruned_feature.get_shape()[2];

    size_t actual_total_tokens = 0;
    for (const auto& feature : pruned_features_list) {
        actual_total_tokens += feature.get_shape()[1];
    }

    GENAI_DEBUG("[CDPruner] Concatenating %zu frames with total %zu tokens",
                pruned_features_list.size(),
                actual_total_tokens);

    ov::Tensor concatenated_features(first_pruned_feature.get_element_type(),
                                     {actual_batch_size, actual_total_tokens, actual_hidden_dim});
    float* concat_data = concatenated_features.data<float>();

    size_t offset_elements = 0;

    for (const auto& feature : pruned_features_list) {
        size_t frame_tokens = feature.get_shape()[1];
        size_t frame_size_bytes = frame_tokens * actual_hidden_dim * sizeof(float);

        std::memcpy(concat_data + offset_elements, feature.data(), frame_size_bytes);
        offset_elements += frame_tokens * actual_hidden_dim;
    }

    m_last_statistics.total_tokens = total_input_tokens;
    m_last_statistics.selected_tokens = actual_total_tokens;
    m_last_statistics.pruning_ratio = 1.0f - static_cast<float>(actual_total_tokens) / total_input_tokens;
    m_last_statistics.batch_size = actual_batch_size;

    return concatenated_features;
}

float CDPruner::compute_pruning_ratio() const {
    return m_config.pruning_ratio / 100.0f;
}

PruningStatistics CDPruner::get_last_pruning_statistics() const {
    return m_last_statistics;
}

void CDPruner::validate_config(const Config& config) {
    if (config.pruning_ratio == 0) {
        GENAI_INFO("pruning_ratio is 0, pruning disabled!");
        return;  // Pruning disabled, no validation needed
    }

    OPENVINO_ASSERT(config.pruning_ratio >= 1 && config.pruning_ratio <= 100,
                    "pruning ratio must be between 0 and 100");

    OPENVINO_ASSERT(config.relevance_weight >= 0.0f && config.relevance_weight <= 1.0f,
                    "relevance weight must be between 0.0 and 1.0");

    OPENVINO_ASSERT(config.numerical_threshold >= 0.0f, "numerical_threshold must be positive");

    OPENVINO_ASSERT(!config.device.empty(), "device cannot be empty");
}

bool CDPruner::update_config(const Config& new_config) {
    if (m_config == new_config)
        return true;
    try {
        // Validate the new configuration first
        validate_config(new_config);

        // Determine what needs to be reinitialized based on config changes
        bool need_reinit_relevance = new_config.use_negative_relevance != m_config.use_negative_relevance;
        bool need_reinit_kernel = (new_config.use_negative_relevance != m_config.use_negative_relevance ||
                                   new_config.relevance_weight != m_config.relevance_weight);
        bool need_reinit_dpp = (new_config.use_cl_kernel != m_config.use_cl_kernel);

        // Update the configuration
        m_config = new_config;

        // Reinitialize components with new config
        if (need_reinit_relevance) {
            m_relevance_calc = RelevanceCalculator(m_config);
        }
        if (need_reinit_kernel) {
            m_kernel_builder = ConditionalKernelBuilder(m_config);
        }
        if (need_reinit_dpp) {
            m_dpp_selector = FastGreedyDPP(m_config);
        }

        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void CDPruner::validate_input_tensors(const ov::Tensor& visual_features, const ov::Tensor& text_features) {
    // Validate visual features
    OPENVINO_ASSERT(visual_features.get_shape().size() == 3, "Visual features must be 3D tensor [B, N, D]");

    // Validate text features
    OPENVINO_ASSERT(text_features.get_shape().size() == 2, "Text features must be 2D tensor [M, D]");

    const auto& visual_shape = visual_features.get_shape();
    const auto& text_shape = text_features.get_shape();

    // Check feature dimension consistency
    OPENVINO_ASSERT(visual_shape[2] == text_shape[1], "Visual and text features must have same feature dimension");

    // Calculate actual token count based on percentage
    size_t num_tokens_to_keep = static_cast<size_t>(std::round(visual_shape[1] * (1 - m_config.pruning_ratio / 100.0)));

    // Check if percentage would result in zero tokens
    OPENVINO_ASSERT(num_tokens_to_keep > 0, "Percentage is too small, would result in zero tokens");

    // Check tensor data types
    OPENVINO_ASSERT(
        visual_features.get_element_type() == ov::element::f32 && text_features.get_element_type() == ov::element::f32,
        "Input tensors must be float32 type");
}

}  // namespace ov::genai::cdpruner
