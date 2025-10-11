// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cdpruner.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "openvino/openvino.hpp"
#include "utils.hpp"

namespace ov::genai::cdpruner {

namespace {

bool should_log_cdpruner(bool silent, bool debug_mode) {
    if (ov::genai::utils::env_setup_for_print_debug_info()) {
        return true;
    }
    if (debug_mode) {
        return true;
    }
    return !silent;
}

}  // namespace

CDPruner::CDPruner(const Config& config)
    : m_config(config),
      m_relevance_calc(config),
      m_kernel_builder(config),
      m_dpp_selector(config) {
    // Load config from env
    m_config.update_from_env();
    // Validate configuration
    validate_config(config);
}

std::vector<std::vector<size_t>> CDPruner::select_tokens(const ov::Tensor& visual_features,
                                                         const ov::Tensor& text_features,
                                                         bool silent) {
    // Input validation
    if (m_config.pruning_ratio == 0) {
        return create_all_tokens_selection(visual_features);
    }

    const auto& visual_shape = visual_features.get_shape();
    const auto& text_shape = text_features.get_shape();

    size_t total_tokens = visual_shape[1];
    size_t raw_tokens_to_keep = static_cast<size_t>(std::round(total_tokens * (1.0 - m_config.pruning_ratio / 100.0)));
    // Round up to the next even number of tokens to keep
    size_t num_tokens_to_keep = (raw_tokens_to_keep % 2 == 0) ? raw_tokens_to_keep : raw_tokens_to_keep + 1;

    if (m_config.pruning_ratio == 0 || num_tokens_to_keep >= total_tokens) {
        return create_all_tokens_selection(visual_features);
    }

    validate_input_tensors(visual_features, text_features);

    auto overall_start = std::chrono::high_resolution_clock::now();

    try {
        std::vector<std::vector<size_t>> selected_tokens;
        std::chrono::microseconds dpp_duration{0};
        std::chrono::microseconds relevance_duration{0};
        std::chrono::microseconds kernel_duration{0};
        size_t batch_size = visual_shape[0];
        size_t total_visual_tokens = visual_shape[1];
        size_t feature_dim = visual_shape[2];

        bool use_splitting = m_config.split_threshold > 0 && total_visual_tokens > m_config.split_threshold;

        size_t split_point = 0;
        size_t first_half_size = 0;
        size_t second_half_size = 0;

        if (use_splitting) {
            split_point = total_visual_tokens / 2;
            first_half_size = split_point;
            second_half_size = total_visual_tokens - split_point;
        }

        const float* visual_data = visual_features.data<const float>();
        ov::Tensor visual_first_half;
        ov::Tensor visual_second_half;

        if (use_splitting) {
            visual_first_half = ov::Tensor(ov::element::f32,
                                           {batch_size, first_half_size, feature_dim},
                                           const_cast<float*>(visual_data));
            visual_second_half =
                ov::Tensor(ov::element::f32,
                           {batch_size, second_half_size, feature_dim},
                           const_cast<float*>(visual_data + batch_size * first_half_size * feature_dim));
        } else {
            visual_first_half = visual_features;
        }

        ov::Tensor kernel_matrix_first;
        ov::Tensor kernel_matrix_second;

        std::string computation_mode;
        try {
            // OpenVINO ops model approach
            computation_mode = std::string("OV Model by ") + m_config.device;
            // CDPruner Step 1-2: Compute relevance scores and build kernel matrix via OV model
            auto computation_start = std::chrono::high_resolution_clock::now();

            if (use_splitting) {
                // Building kernel matrix for first half
                kernel_matrix_first = m_kernel_builder.build(visual_first_half, text_features);
                // Building kernel matrix for second half
                kernel_matrix_second = m_kernel_builder.build(visual_second_half, text_features);
            } else {
                // Building single kernel matrix for all the tokens
                kernel_matrix_first = m_kernel_builder.build(visual_first_half, text_features);
            }

            auto computation_end = std::chrono::high_resolution_clock::now();
            auto computation_duration =
                std::chrono::duration_cast<std::chrono::microseconds>(computation_end - computation_start);

            kernel_duration = computation_duration;
        } catch (const std::exception& e) {
            std::cerr << "[CDPruner] Error occurred during kernel building: " << e.what() << std::endl;
            std::cout << "[CDPruner] Falling back to traditional approach..." << std::endl;
            computation_mode = "Traditional Step-by-Step by CPU";
            auto relevance_start = std::chrono::high_resolution_clock::now();
            // CDPruner Step 1: Compute relevance scores
            if (use_splitting) {
                // Compute relevance scores for both halves
                ov::Tensor relevance_scores_first = m_relevance_calc.compute(visual_first_half, text_features);
                ov::Tensor relevance_scores_second = m_relevance_calc.compute(visual_second_half, text_features);

                auto relevance_end = std::chrono::high_resolution_clock::now();
                relevance_duration =
                    std::chrono::duration_cast<std::chrono::microseconds>(relevance_end - relevance_start);

                auto kernel_start = std::chrono::high_resolution_clock::now();

                // Build kernel matrices using relevance scores
                kernel_matrix_first = m_kernel_builder.build(visual_first_half, relevance_scores_first);
                kernel_matrix_second = m_kernel_builder.build(visual_second_half, relevance_scores_second);

                auto kernel_end = std::chrono::high_resolution_clock::now();
                kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start);
            } else {
                // Process entire tensor without splitting
                ov::Tensor relevance_scores = m_relevance_calc.compute(visual_first_half, text_features);

                auto relevance_end = std::chrono::high_resolution_clock::now();
                relevance_duration =
                    std::chrono::duration_cast<std::chrono::microseconds>(relevance_end - relevance_start);

                auto kernel_start = std::chrono::high_resolution_clock::now();

                // Build single kernel matrix using relevance scores
                kernel_matrix_first = m_kernel_builder.build(visual_first_half, relevance_scores);

                auto kernel_end = std::chrono::high_resolution_clock::now();
                kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start);

                // Accumulate kernel duration
                kernel_duration += relevance_duration;
            }
        }

        if (use_splitting) {
            // Use parallel DPP selection for split processing
            auto dpp_start = std::chrono::high_resolution_clock::now();

            selected_tokens =
                m_dpp_selector.select(kernel_matrix_first, kernel_matrix_second, num_tokens_to_keep, split_point);

            auto dpp_end = std::chrono::high_resolution_clock::now();
            dpp_duration = std::chrono::duration_cast<std::chrono::microseconds>(dpp_end - dpp_start);
        } else {
            // Direct DPP selection for single tensor processing
            auto dpp_start = std::chrono::high_resolution_clock::now();

            selected_tokens = m_dpp_selector.select(kernel_matrix_first, num_tokens_to_keep);

            auto dpp_end = std::chrono::high_resolution_clock::now();
            dpp_duration = std::chrono::duration_cast<std::chrono::microseconds>(dpp_end - dpp_start);
        }

        auto overall_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(overall_end - overall_start);

        if (!silent) {
            size_t total_input_tokens = visual_shape[0] * visual_shape[1];
            size_t total_output_tokens = visual_shape[0] * num_tokens_to_keep;

#ifdef ENABLE_OPENCL_DPP
            if (m_config.use_cl_kernel) {
                computation_mode += " + OpenCL GPU DPP";
            }
#endif
            utils::print_cdpruner_performance_summary(computation_mode,
                                                      total_duration,
                                                      kernel_duration,
                                                      dpp_duration,
                                                      total_input_tokens,
                                                      total_output_tokens);
        }

        return selected_tokens;
    } catch (const std::exception& e) {
        throw std::runtime_error("CDPruner::select_tokens failed: " + std::string(e.what()));
    }
}

ov::Tensor CDPruner::apply_pruning(const ov::Tensor& visual_features, const ov::Tensor& text_features, bool silent) {
    const auto& visual_shape = visual_features.get_shape();
    const auto& text_shape = text_features.get_shape();
    size_t batch_size = visual_shape[0];
    size_t total_tokens = visual_shape[1];
    size_t feature_dim = visual_shape[2];

    size_t num_tokens_to_keep = static_cast<size_t>(std::round(total_tokens * (1 - m_config.pruning_ratio / 100.0)));

    size_t text_tokens = text_shape[0];
    size_t text_feature_dim = text_shape[1];

    float pruning_ratio = 1.0f - static_cast<float>(num_tokens_to_keep) / static_cast<float>(total_tokens);
    float reduction_percentage = pruning_ratio * 100.0f;
    size_t tokens_removed = total_tokens - num_tokens_to_keep;

    utils::print_cdpruner_processing_overview(total_tokens,
                                              feature_dim,
                                              text_tokens,
                                              text_feature_dim,
                                              num_tokens_to_keep,
                                              tokens_removed,
                                              reduction_percentage,
                                              m_config.pruning_ratio,
                                              m_config.relevance_weight);

    auto selected_tokens = select_tokens(visual_features, text_features, silent);

    // Determine actual number of selected tokens (may differ from num_tokens_to_keep due to odd->even adjustment)
    size_t actual_selected_tokens = selected_tokens.empty() ? 0 : selected_tokens[0].size();

    ov::Tensor pruned_features(visual_features.get_element_type(), {batch_size, actual_selected_tokens, feature_dim});

    const float* input_data = visual_features.data<const float>();
    float* output_data = pruned_features.data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        const auto& batch_selected = selected_tokens[b];

        for (size_t t = 0; t < batch_selected.size(); ++t) {
            size_t src_token_idx = batch_selected[t];

            for (size_t f = 0; f < feature_dim; ++f) {
                size_t src_idx = b * total_tokens * feature_dim + src_token_idx * feature_dim + f;
                size_t dst_idx = b * actual_selected_tokens * feature_dim + t * feature_dim + f;
                output_data[dst_idx] = input_data[src_idx];
            }
        }
    }

    m_last_statistics.total_tokens = total_tokens;
    m_last_statistics.selected_tokens = actual_selected_tokens;
    m_last_statistics.pruning_ratio = 1.0f - static_cast<float>(actual_selected_tokens) / total_tokens;
    m_last_statistics.batch_size = batch_size;

    if (!selected_tokens.empty() && m_config.pruning_debug_mode) {
        std::cout << "\n[CDPruner] Selected token indices (batch 0): [";
        const auto& first_batch_tokens = selected_tokens[0];
        for (size_t i = 0; i < std::min(static_cast<size_t>(10), first_batch_tokens.size()); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << first_batch_tokens[i];
        }
        if (first_batch_tokens.size() > 10) {
            std::cout << ", ... (+" << (first_batch_tokens.size() - 10) << " more)";
        }
        std::cout << "]" << std::endl;
    }

    return pruned_features;
}

ov::Tensor CDPruner::apply_pruning(const std::vector<ov::Tensor>& visual_features_list,
                                   const ov::Tensor& text_features) {
    if (visual_features_list.empty()) {
        return ov::Tensor();
    }

    // Handle single feature case by calling existing method
    if (visual_features_list.size() == 1) {
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
    size_t total_output_tokens = num_tokens_to_keep * visual_features_list.size();

    size_t text_tokens = text_shape[0];
    size_t text_feature_dim = text_shape[1];

    utils::print_cdpruner_processing_overview(visual_features_list.size(),
                                              tokens_per_frame,
                                              feature_dim,
                                              text_tokens,
                                              text_feature_dim,
                                              num_tokens_to_keep,
                                              total_input_tokens,
                                              total_output_tokens,
                                              m_config.pruning_ratio,
                                              m_config.relevance_weight);

    // Apply pruning to each visual feature and collect results (using silent mode)
    std::vector<ov::Tensor> pruned_features_list;
    pruned_features_list.reserve(visual_features_list.size());

    auto overall_start = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds total_kernel_duration{0};
    std::chrono::microseconds total_dpp_duration{0};

    for (size_t frame_idx = 0; frame_idx < visual_features_list.size(); ++frame_idx) {
        const auto& visual_feature = visual_features_list[frame_idx];
        ov::Tensor pruned_feature = apply_pruning(visual_feature, text_features, true);
        if (m_config.pruning_debug_mode) {
            auto shape = visual_feature.get_shape();
            auto pruned_shape = pruned_feature.get_shape();
            std::cout << "[CDPruner] Frame " << frame_idx << ": [" << shape[1] << " → " << pruned_shape[1] << " tokens]"
                      << std::endl;
        }
        pruned_features_list.push_back(std::move(pruned_feature));
    }

    const auto& first_pruned_feature = pruned_features_list[0];
    const size_t actual_batch_size = first_pruned_feature.get_shape()[0];
    const size_t actual_tokens_per_frame = first_pruned_feature.get_shape()[1];
    const size_t actual_hidden_dim = first_pruned_feature.get_shape()[2];
    const size_t actual_total_tokens = actual_tokens_per_frame * visual_features_list.size();

    ov::Tensor concatenated_features(first_pruned_feature.get_element_type(),
                                     {actual_batch_size, actual_total_tokens, actual_hidden_dim});
    float* concat_data = concatenated_features.data<float>();

    const size_t feature_size_bytes = actual_tokens_per_frame * actual_hidden_dim * sizeof(float);
    size_t offset_elements = 0;

    for (const auto& feature : pruned_features_list) {
        std::memcpy(concat_data + offset_elements, feature.data(), feature_size_bytes);
        offset_elements += actual_tokens_per_frame * actual_hidden_dim;
    }

    auto overall_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(overall_end - overall_start);

    m_last_statistics.total_tokens = total_input_tokens;
    m_last_statistics.selected_tokens = actual_total_tokens;
    m_last_statistics.pruning_ratio = 1.0f - static_cast<float>(actual_total_tokens) / total_input_tokens;
    m_last_statistics.batch_size = actual_batch_size;

#ifdef ENABLE_OPENCL_DPP
    std::string computation_mode = std::string("OV Model by ") + m_config.device;
    if (m_config.use_cl_kernel) {
        computation_mode += " + OpenCL GPU DPP";
    }
#else
    std::string computation_mode = std::string("OV Model by ") + m_config.device + std::string(" + Traditional DPP");
#endif

    utils::print_cdpruner_performance_summary(computation_mode,
                                              total_duration,
                                              visual_features_list.size(),
                                              total_input_tokens,
                                              actual_total_tokens,
                                              actual_batch_size,
                                              actual_hidden_dim);

    return concatenated_features;
}

float CDPruner::compute_pruning_ratio() const {
    return m_config.pruning_ratio / 100.0f;
}

size_t CDPruner::get_default_token_count() const {
    // LLaVA typical token count (can be made configurable)
    return 576;  // 24x24 patches for most LLaVA configurations
}

PruningStatistics CDPruner::get_last_pruning_statistics() const {
    return m_last_statistics;
}

void CDPruner::validate_config(const Config& config) {
    if (config.pruning_ratio == 0)
        return;  // Pruning disabled, no validation needed

    if (config.pruning_ratio < 1 || config.pruning_ratio > 100) {
        throw std::invalid_argument("pruning_ratio must be between 1 and 100 (or 0 to disable)");
    }

    if (config.relevance_weight < 0.0f || config.relevance_weight > 1.0f) {
        throw std::invalid_argument("relevance_weight must be in range [0.0, 1.0]");
    }

    if (config.numerical_threshold < 0.0f) {
        throw std::invalid_argument("numerical_threshold must be positive");
    }

    if (config.device.empty()) {
        throw std::invalid_argument("device cannot be empty");
    }
}

bool CDPruner::update_config(const Config& new_config) {
    try {
        // Validate the new configuration first
        validate_config(new_config);

        // Update the configuration
        m_config = new_config;

        // Reinitialize components with new config
        m_relevance_calc = RelevanceCalculator(new_config);
        m_kernel_builder = ConditionalKernelBuilder(new_config);
        m_dpp_selector = FastGreedyDPP(new_config);
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

void CDPruner::validate_input_tensors(const ov::Tensor& visual_features, const ov::Tensor& text_features) {
    // Validate visual features
    if (visual_features.get_shape().size() != 3) {
        throw std::invalid_argument("Visual features must be 3D tensor [B, N, D]");
    }

    // Validate text features
    if (text_features.get_shape().size() != 2) {
        throw std::invalid_argument("Text features must be 2D tensor [M, D]");
    }

    const auto& visual_shape = visual_features.get_shape();
    const auto& text_shape = text_features.get_shape();

    // Check feature dimension consistency
    if (visual_shape[2] != text_shape[1]) {
        throw std::invalid_argument("Visual and text features must have same feature dimension");
    }

    // Calculate actual token count based on percentage
    size_t num_tokens_to_keep = static_cast<size_t>(std::round(visual_shape[1] * (1 - m_config.pruning_ratio / 100.0)));

    // Check if percentage would result in zero tokens
    if (num_tokens_to_keep == 0) {
        throw std::invalid_argument("Percentage is too small, would result in zero tokens");
    }

    // Check tensor data types
    if (visual_features.get_element_type() != ov::element::f32 ||
        text_features.get_element_type() != ov::element::f32) {
        throw std::invalid_argument("Input tensors must be float32 type");
    }
}

std::vector<std::vector<size_t>> CDPruner::create_all_tokens_selection(const ov::Tensor& visual_features) {
    auto shape = visual_features.get_shape();
    size_t batch_size = shape[0];
    size_t total_tokens = shape[1];

    std::vector<std::vector<size_t>> all_tokens(batch_size);

    for (size_t b = 0; b < batch_size; ++b) {
        all_tokens[b].reserve(total_tokens);
        for (size_t t = 0; t < total_tokens; ++t) {
            all_tokens[b].push_back(t);
        }
    }

    return all_tokens;
}

void CDPruner::print_selection_statistics(const ov::Tensor& visual_features,
                                          const std::vector<std::vector<size_t>>& selected_tokens) {
    const auto& shape = visual_features.get_shape();
    size_t batch_size = shape[0];
    size_t total_tokens = shape[1];
    size_t selected_token_count = static_cast<size_t>(std::round(total_tokens * (1 - m_config.pruning_ratio / 100.0)));

    std::cout << "\n+--- CDPruner Results -----------------------------------+" << std::endl;

    std::cout << "[CDPruner] Summary: " << total_tokens << " -> " << selected_token_count << " tokens (" << std::fixed
              << std::setprecision(1) << (1.0f - static_cast<float>(selected_token_count) / total_tokens) * 100.0f
              << "% reduction)" << std::endl;

    if (batch_size > 0 && !selected_tokens.empty()) {
        for (size_t b = 0; b < batch_size; b++) {
            std::cout << "[CDPruner] Selected indices (batch " << b << "): [";
            for (size_t i = 0; i < selected_tokens[b].size() && i < 10; ++i) {
                if (i > 0)
                    std::cout << ", ";
                std::cout << selected_tokens[b][i];
            }
            if (selected_tokens[b].size() > 10) {
                std::cout << ", ..." << (selected_tokens[b].size() - 10) << " more";
            }
            std::cout << "]" << std::endl;
            if (b > 10) {
                std::cout << "[CDPruner] ... (more batches not shown)" << std::endl;
            }
        }
    }

    std::cout << "+----------------------------------------------------------+" << std::endl;

    m_last_statistics.total_tokens = total_tokens;
    m_last_statistics.selected_tokens = selected_token_count;
    m_last_statistics.pruning_ratio = 1.0f - static_cast<float>(selected_token_count) / total_tokens;
    m_last_statistics.batch_size = batch_size;
}

}  // namespace ov::genai::cdpruner