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

namespace ov::genai::cdpruner {

CDPruner::CDPruner(const Config& config)
    : m_config(config),
      m_relevance_calc(config),
      m_kernel_builder(config),
      m_dpp_selector(config) {
    // Validate configuration
    validate_config(config);
}

std::vector<std::vector<size_t>> CDPruner::select_tokens(const ov::Tensor& visual_features,
                                                         const ov::Tensor& text_features) {
    // Input validation
    if (m_config.pruning_ratio == 0) {
        // If pruning is disabled (ratio = 0), return all tokens
        if (m_config.pruning_debug_mode) {
            std::cout << "[CDPruner] Pruning is disabled (ratio=0). Returning all tokens." << std::endl;
        }
        return create_all_tokens_selection(visual_features);
    }

    // Calculate actual number of tokens to keep based on percentage
    size_t total_tokens = visual_features.get_shape()[1];
    size_t raw_tokens_to_keep = static_cast<size_t>(std::round(total_tokens * (1.0 - m_config.pruning_ratio / 100.0)));
    // Ensure the number of tokens to keep is the nearest even number
    size_t num_tokens_to_keep = (raw_tokens_to_keep % 2 == 0) ? raw_tokens_to_keep : raw_tokens_to_keep + 1;

    if (m_config.pruning_ratio == 0 || num_tokens_to_keep >= total_tokens) {
        if (m_config.pruning_debug_mode) {
            std::cout << "[CDPruner] Warning: pruning_ratio is 0 or results in keeping all tokens. "
                      << "Returning all tokens without pruning." << std::endl;
        }
        return create_all_tokens_selection(visual_features);
    }

    validate_input_tensors(visual_features, text_features);

    // Performance timing setup
    auto overall_start = std::chrono::high_resolution_clock::now();

    // Get input dimensions for context
    auto visual_shape = visual_features.get_shape();
    auto text_shape = text_features.get_shape();

    try {
        std::vector<std::vector<size_t>> selected_tokens;
        std::chrono::microseconds dpp_duration{0};        // Initialize DPP timing variable
        std::chrono::microseconds relevance_duration{0};  // Initialize relevance timing variable
        std::chrono::microseconds kernel_duration{0};     // Initialize kernel building timing variable

        if (m_config.pruning_debug_mode) {
            std::cout << "\n+--- CDPruner Processing Steps ----------------------+" << std::endl;
        }

        // Get input dimensions for processing decision
        auto visual_shape = visual_features.get_shape();
        size_t batch_size = visual_shape[0];
        size_t total_visual_tokens = visual_shape[1];
        size_t feature_dim = visual_shape[2];

        // Decision: split tokens only if total count exceeds threshold
        bool use_splitting = total_visual_tokens > m_config.split_threshold;

        if (m_config.pruning_debug_mode) {
            std::cout << "[CDPruner] Total visual tokens: " << total_visual_tokens << std::endl;
            std::cout << "[CDPruner] Split threshold: " << m_config.split_threshold << std::endl;
            std::cout << "[CDPruner] Using " << (use_splitting ? "split processing" : "single processing") << std::endl;
        }

        size_t split_point = 0;
        size_t first_half_size = 0;
        size_t second_half_size = 0;

        if (use_splitting) {
            split_point = total_visual_tokens / 2;
            first_half_size = split_point;
            second_half_size = total_visual_tokens - split_point;

            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner] Splitting " << total_visual_tokens
                          << " visual tokens into groups: " << first_half_size << " + " << second_half_size
                          << std::endl;
            }
        }

        // Create tensor views based on splitting decision
        const float* visual_data = visual_features.data<const float>();
        ov::Tensor visual_first_half, visual_second_half;

        if (use_splitting) {
            // Create tensor views for first and second halves without copying data
            visual_first_half = ov::Tensor(ov::element::f32,
                                           {batch_size, first_half_size, feature_dim},
                                           const_cast<float*>(visual_data));
            visual_second_half =
                ov::Tensor(ov::element::f32,
                           {batch_size, second_half_size, feature_dim},
                           const_cast<float*>(visual_data + batch_size * first_half_size * feature_dim));
        } else {
            // Use the entire visual features tensor without splitting
            visual_first_half = visual_features;
        }

        // Build kernel matrices using different approaches
        ov::Tensor kernel_matrix_first, kernel_matrix_second;

        std::string computation_mode;
        try {
            // OpenVINO ops model approach
            computation_mode = std::string("OV Model by ") + m_config.device;
            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner] Step 1-2: Computing kernel matrices using ov model on device: "
                          << m_config.device << "..." << std::endl;
            }

            auto computation_start = std::chrono::high_resolution_clock::now();

            if (use_splitting) {
                if (m_config.pruning_debug_mode) {
                    std::cout << "[CDPruner]   Building kernel matrix for first half..." << std::endl;
                }
                kernel_matrix_first = m_kernel_builder.build(visual_first_half, text_features);

                if (m_config.pruning_debug_mode) {
                    std::cout << "[CDPruner]   Building kernel matrix for second half..." << std::endl;
                }
                kernel_matrix_second = m_kernel_builder.build(visual_second_half, text_features);
            } else {
                if (m_config.pruning_debug_mode) {
                    std::cout << "[CDPruner]   Building single kernel matrix for all tokens..." << std::endl;
                }
                kernel_matrix_first = m_kernel_builder.build(visual_first_half, text_features);
            }

            auto computation_end = std::chrono::high_resolution_clock::now();
            auto computation_duration =
                std::chrono::duration_cast<std::chrono::microseconds>(computation_end - computation_start);

            // For OV model approach, this includes both relevance computation and kernel building
            kernel_duration = computation_duration;

            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner]   Kernel building via ov model took: " << computation_duration.count() << " us"
                          << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[CDPruner] Error occurred during kernel building: " << e.what() << std::endl;
            std::cout << "[CDPruner] Falling back to traditional approach..." << std::endl;
            // Traditional step-by-step approach
            computation_mode = "Traditional Step-by-Step by CPU";
            auto relevance_start = std::chrono::high_resolution_clock::now();

            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner] Step 1: Computing relevance scores..." << std::endl;
            }

            if (use_splitting) {
                // Compute relevance scores for both halves
                ov::Tensor relevance_scores_first = m_relevance_calc.compute(visual_first_half, text_features);
                ov::Tensor relevance_scores_second = m_relevance_calc.compute(visual_second_half, text_features);

                auto relevance_end = std::chrono::high_resolution_clock::now();
                relevance_duration =
                    std::chrono::duration_cast<std::chrono::microseconds>(relevance_end - relevance_start);

                if (m_config.pruning_debug_mode) {
                    std::cout << "[CDPruner]   Relevance computation took: " << relevance_duration.count() << " us"
                              << std::endl;
                }

                if (m_config.pruning_debug_mode) {
                    std::cout << "[CDPruner] Step 2: Building kernel matrices..." << std::endl;
                }
                auto kernel_start = std::chrono::high_resolution_clock::now();

                // Build kernel matrices using relevance scores
                kernel_matrix_first = m_kernel_builder.build(visual_first_half, relevance_scores_first);
                kernel_matrix_second = m_kernel_builder.build(visual_second_half, relevance_scores_second);

                auto kernel_end = std::chrono::high_resolution_clock::now();
                kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start);

                if (m_config.pruning_debug_mode) {
                    std::cout << "[CDPruner]   Kernel building took: " << kernel_duration.count() << " us" << std::endl;
                }
            } else {
                // Process entire tensor without splitting
                ov::Tensor relevance_scores = m_relevance_calc.compute(visual_first_half, text_features);

                auto relevance_end = std::chrono::high_resolution_clock::now();
                relevance_duration =
                    std::chrono::duration_cast<std::chrono::microseconds>(relevance_end - relevance_start);

                if (m_config.pruning_debug_mode) {
                    std::cout << "[CDPruner]   Relevance computation took: " << relevance_duration.count() << " us"
                              << std::endl;
                }

                if (m_config.pruning_debug_mode) {
                    std::cout << "[CDPruner] Step 2: Building single kernel matrix..." << std::endl;
                }
                auto kernel_start = std::chrono::high_resolution_clock::now();

                // Build single kernel matrix using relevance scores
                kernel_matrix_first = m_kernel_builder.build(visual_first_half, relevance_scores);

                auto kernel_end = std::chrono::high_resolution_clock::now();
                kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start);

                if (m_config.pruning_debug_mode) {
                    std::cout << "[CDPruner]   Kernel building took: " << kernel_duration.count() << " us" << std::endl;
                }

                // Accumulate kernel duration
                kernel_duration += relevance_duration;
            }
        }

        if (use_splitting) {
            // Use parallel DPP selection for split processing
            auto dpp_start = std::chrono::high_resolution_clock::now();

            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner] Step 3: Selecting " << num_tokens_to_keep << " tokens using parallel DPP..."
                          << std::endl;
            }

            selected_tokens =
                m_dpp_selector.select(kernel_matrix_first, kernel_matrix_second, num_tokens_to_keep, split_point);

            auto dpp_end = std::chrono::high_resolution_clock::now();
            dpp_duration = std::chrono::duration_cast<std::chrono::microseconds>(dpp_end - dpp_start);

            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner]   Parallel DPP selection took: " << dpp_duration.count() << " us" << std::endl;
            }
        } else {
            // Direct DPP selection for single tensor processing
            auto dpp_start = std::chrono::high_resolution_clock::now();

            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner] Step 3: Selecting " << num_tokens_to_keep << " tokens using single DPP..."
                          << std::endl;
            }

            selected_tokens = m_dpp_selector.select(kernel_matrix_first, num_tokens_to_keep);

            auto dpp_end = std::chrono::high_resolution_clock::now();
            dpp_duration = std::chrono::duration_cast<std::chrono::microseconds>(dpp_end - dpp_start);

            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner]   Single DPP selection took: " << dpp_duration.count() << " us" << std::endl;
            }
        }

        // Overall timing summary
        auto overall_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(overall_end - overall_start);

        std::cout << "\n+--- CDPruner Performance Summary -------------------------+" << std::endl;

#ifdef ENABLE_OPENCL_DPP
        if (m_config.use_cl_kernel) {
            computation_mode += " + OpenCL GPU DPP";
        }
#endif

        std::cout << "[CDPruner] Computation mode: " << computation_mode << std::endl;
        std::cout << "[CDPruner] Total processing time: " << total_duration.count() << " us ("
                  << (total_duration.count() / 1000.0) << " ms)" << std::endl;

        // Performance metrics
        size_t total_input_tokens = visual_shape[0] * visual_shape[1];
        size_t total_output_tokens = visual_shape[0] * num_tokens_to_keep;
        std::cout << "[CDPruner] Performance Metrics:" << std::endl;

        // Show timing breakdown based on computation mode
        std::cout << "[CDPruner]   Kernel computation time: " << kernel_duration.count() << " us ("
                  << (kernel_duration.count() / 1000.0) << " ms)" << std::endl;

        std::cout << "[CDPruner]   DPP selection time: " << dpp_duration.count() << " us ("
                  << (dpp_duration.count() / 1000.0) << " ms)" << std::endl;
        std::cout << "[CDPruner]   Overall throughput: "
                  << (static_cast<double>(total_input_tokens) / total_duration.count() * 1000000) << " input tokens/sec"
                  << std::endl;
        std::cout << "[CDPruner]   Pruning efficiency: "
                  << (static_cast<double>(total_output_tokens) / total_duration.count() * 1000000)
                  << " output tokens/sec" << std::endl;
        std::cout << "[CDPruner]   Pruning ratio: "
                  << (1.0 - static_cast<double>(num_tokens_to_keep) / visual_shape[1]) * 100 << "%" << std::endl;
        std::cout << "+----------------------------------------------------------+" << std::endl;
        if (m_config.pruning_debug_mode) {
            print_selection_statistics(visual_features, selected_tokens);
        }

        return selected_tokens;

    } catch (const std::exception& e) {
        throw std::runtime_error("CDPruner::select_tokens failed: " + std::string(e.what()));
    }
}

ov::Tensor CDPruner::apply_pruning(const ov::Tensor& visual_features, const ov::Tensor& text_features) {
    auto visual_shape = visual_features.get_shape();
    size_t batch_size = visual_shape[0];
    size_t total_tokens = visual_shape[1];
    size_t feature_dim = visual_shape[2];

    // Calculate actual number of tokens to keep based on percentage
    size_t num_tokens_to_keep = static_cast<size_t>(std::round(total_tokens * (1 - m_config.pruning_ratio / 100.0)));

    auto text_shape = text_features.get_shape();
    size_t text_tokens = text_shape[0];
    size_t text_feature_dim = text_shape[1];

    // Calculate pruning statistics
    float pruning_ratio = 1.0f - static_cast<float>(num_tokens_to_keep) / static_cast<float>(total_tokens);
    float reduction_percentage = pruning_ratio * 100.0f;
    size_t tokens_removed = total_tokens - num_tokens_to_keep;

    // Print consolidated CDPruner overview
    std::cout << "\n+--- CDPruner Processing Overview -------------------------+" << std::endl;
    std::cout << "[CDPruner] Input:  Vision[" << total_tokens << " tokens x " << feature_dim << "D] + Text["
              << text_tokens << " tokens x " << text_feature_dim << "D]" << std::endl;
    std::cout << "[CDPruner] Config: Keep " << m_config.pruning_ratio << "% (" << num_tokens_to_keep << "/"
              << total_tokens << " tokens) | Weight=" << m_config.relevance_weight << std::endl;
    std::cout << "[CDPruner] Result: " << tokens_removed << " tokens removed (" << std::fixed << std::setprecision(1)
              << reduction_percentage << "% reduction)" << std::endl;
    std::cout << "+----------------------------------------------------------+" << std::endl;

    auto selected_tokens = select_tokens(visual_features, text_features);

    // Determine actual number of selected tokens (may differ from num_tokens_to_keep due to odd->even adjustment)
    size_t actual_selected_tokens = selected_tokens.empty() ? 0 : selected_tokens[0].size();

    // Create output tensor with actual selected tokens
    ov::Tensor pruned_features(visual_features.get_element_type(), {batch_size, actual_selected_tokens, feature_dim});

    const float* input_data = visual_features.data<const float>();
    float* output_data = pruned_features.data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        const auto& batch_selected = selected_tokens[b];

        for (size_t t = 0; t < batch_selected.size(); ++t) {
            size_t src_token_idx = batch_selected[t];

            // Copy features for this token
            for (size_t f = 0; f < feature_dim; ++f) {
                size_t src_idx = b * total_tokens * feature_dim + src_token_idx * feature_dim + f;
                size_t dst_idx = b * actual_selected_tokens * feature_dim + t * feature_dim + f;
                output_data[dst_idx] = input_data[src_idx];
            }
        }
    }

    // Update statistics
    m_last_statistics.total_tokens = total_tokens;
    m_last_statistics.selected_tokens = actual_selected_tokens;
    m_last_statistics.pruning_ratio = 1.0f - static_cast<float>(actual_selected_tokens) / total_tokens;
    m_last_statistics.batch_size = batch_size;

    // Show selected token indices for debugging
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

float CDPruner::compute_pruning_ratio() const {
    // Return the percentage as a ratio (30% -> 0.30)
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

        if (m_config.pruning_debug_mode) {
            std::cout << "[CDPruner] Configuration updated successfully:" << std::endl;
            std::cout << "[CDPruner]   pruning_ratio: " << m_config.pruning_ratio << "%" << std::endl;
            std::cout << "[CDPruner]   relevance_weight: " << m_config.relevance_weight << std::endl;
            std::cout << "[CDPruner]   pruning enabled: " << (m_config.pruning_ratio > 0 ? "true" : "false")
                      << std::endl;
        }

        return true;
    } catch (const std::exception& e) {
        if (m_config.pruning_debug_mode) {
            std::cerr << "[CDPruner] Failed to update configuration: " << e.what() << std::endl;
        }
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

    auto visual_shape = visual_features.get_shape();
    auto text_shape = text_features.get_shape();

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
    auto shape = visual_features.get_shape();
    size_t batch_size = shape[0];
    size_t total_tokens = shape[1];
    size_t selected_token_count = static_cast<size_t>(std::round(total_tokens * (1 - m_config.pruning_ratio / 100.0)));

    // Start CDPruner output block
    std::cout << "\n+--- CDPruner Results -----------------------------------+" << std::endl;

    // Show compact performance summary
    std::cout << "[CDPruner] Summary: " << total_tokens << " -> " << selected_token_count << " tokens (" << std::fixed
              << std::setprecision(1) << (1.0f - static_cast<float>(selected_token_count) / total_tokens) * 100.0f
              << "% reduction)" << std::endl;

    // Show token indices for debugging
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

    // End CDPruner output block
    std::cout << "+----------------------------------------------------------+" << std::endl;

    // Update statistics
    m_last_statistics.total_tokens = total_tokens;
    m_last_statistics.selected_tokens = selected_token_count;
    m_last_statistics.pruning_ratio = 1.0f - static_cast<float>(selected_token_count) / total_tokens;
    m_last_statistics.batch_size = batch_size;
}

}  // namespace ov::genai::cdpruner