// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cdpruner.hpp"
#include "openvino/openvino.hpp"
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <thread>
#include <future>

namespace ov::genai::cdpruner {

CDPruner::CDPruner(const Config& config) 
    : m_config(config)
    , m_relevance_calc(config)
    , m_kernel_builder(config)
    , m_dpp_selector(config) {
    
    // Validate configuration
    validate_config(config);
}

std::vector<std::vector<size_t>> CDPruner::select_tokens(const ov::Tensor& visual_features, 
                                                        const ov::Tensor& text_features) {
    // Input validation
    if (!m_config.enable_pruning) {
        // If pruning is disabled, return all tokens
        if (m_config.pruning_debug_mode) {
            std::cout << "[CDPruner] Pruning is disabled. Returning all tokens." << std::endl;
        }
        return create_all_tokens_selection(visual_features);
    }

    // Calculate actual number of tokens to keep based on percentage
    size_t total_tokens = visual_features.get_shape()[1];
    size_t num_tokens_to_keep = static_cast<size_t>(std::round(total_tokens * m_config.visual_tokens_retain_percentage / 100.0));
    
    if (m_config.visual_tokens_retain_percentage == 0 || num_tokens_to_keep >= total_tokens) {
        if (m_config.pruning_debug_mode) {
            std::cout << "[CDPruner] Warning: visual_tokens_retain_percentage is 0 or results in keeping all tokens. "
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
        std::chrono::microseconds dpp_duration{0}; // Initialize DPP timing variable

        if (m_config.pruning_debug_mode) {
            std::cout << "\n+--- CDPruner Processing Steps ----------------------+" << std::endl;
        }

        if (m_config.use_ops_model) {
            // New OpenVINO ops model approach with visual token splitting
            
            // Split visual tokens into two groups: first 50% and last 50%
            auto visual_shape = visual_features.get_shape();
            size_t batch_size = visual_shape[0];
            size_t total_visual_tokens = visual_shape[1];
            size_t feature_dim = visual_shape[2];
            
            size_t split_point = total_visual_tokens / 2;
            size_t first_half_size = split_point;
            size_t second_half_size = total_visual_tokens - split_point;
            
            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner] Step 1-2: Computing kernel matrices using ov model on device: "
                          << m_config.device << "..." << std::endl;
                std::cout << "[CDPruner] Splitting " << total_visual_tokens << " visual tokens into groups: "
                          << first_half_size << " + " << second_half_size << std::endl;
            }
            
            // Create tensor views for first and second halves without copying data
            const float* visual_data = visual_features.data<const float>();
            
            // Create tensor views pointing to specific regions of original data
            ov::Tensor visual_first_half(ov::element::f32, {batch_size, first_half_size, feature_dim}, 
                                         const_cast<float*>(visual_data));
            ov::Tensor visual_second_half(ov::element::f32, {batch_size, second_half_size, feature_dim}, 
                                         const_cast<float*>(visual_data + batch_size * first_half_size * feature_dim));
            
            auto computation_start = std::chrono::high_resolution_clock::now();
            
            // Step 1-2: Build kernel matrices sequentially (due to single InferRequest object)
            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner]   Building kernel matrix for first half..." << std::endl;
            }
            auto kernel_matrix_first = m_kernel_builder.build(visual_first_half, text_features);
            
            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner]   Building kernel matrix for second half..." << std::endl;
            }
            auto kernel_matrix_second = m_kernel_builder.build(visual_second_half, text_features);
            
            auto computation_end = std::chrono::high_resolution_clock::now();
            auto computation_duration =
                std::chrono::duration_cast<std::chrono::microseconds>(computation_end - computation_start);

            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner]   Kernel building via ov model took: " << computation_duration.count() << " us"
                          << std::endl;
            }

            // Step 3: Select tokens using parallel DPP
            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner] Step 3: Selecting tokens using parallel DPP..." << std::endl;
            }
            auto dpp_start = std::chrono::high_resolution_clock::now();
            
            // Distribute tokens to keep between both halves
            size_t tokens_first_half = num_tokens_to_keep / 2;
            size_t tokens_second_half = num_tokens_to_keep - tokens_first_half;
            
            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner]   Selecting " << tokens_first_half << " tokens from first half, "
                          << tokens_second_half << " tokens from second half in parallel" << std::endl;
            }
            
            // Launch parallel tasks for DPP selection
            std::future<std::vector<std::vector<size_t>>> dpp_first_future = std::async(std::launch::async, [&]() {
                if (m_config.pruning_debug_mode) {
                    std::cout << "[CDPruner] Thread 1: DPP selection for first half..." << std::endl;
                }
                return m_dpp_selector.select(kernel_matrix_first, tokens_first_half);
            });
            
            std::future<std::vector<std::vector<size_t>>> dpp_second_future = std::async(std::launch::async, [&]() {
                if (m_config.pruning_debug_mode) {
                    std::cout << "[CDPruner] Thread 2: DPP selection for second half..." << std::endl;
                }
                return m_dpp_selector.select(kernel_matrix_second, tokens_second_half);
            });
            
            // Wait for both DPP selections to complete
            auto selected_first_batches = dpp_first_future.get();
            auto selected_second_batches = dpp_second_future.get();
            
            std::vector<size_t> selected_first = selected_first_batches[0]; // Take first batch
            std::vector<size_t> selected_second = selected_second_batches[0]; // Take first batch
            
            // Merge results: adjust indices for second half
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
            
            // Store result for this batch
            selected_tokens.push_back(merged_selection);
            
            auto dpp_end = std::chrono::high_resolution_clock::now();
            dpp_duration = std::chrono::duration_cast<std::chrono::microseconds>(dpp_end - dpp_start);

            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner]   DPP selection took: " << dpp_duration.count() << " us" << std::endl;
            }
        } else {
            // Original step-by-step approach
            
            // Split visual tokens into two groups: first 50% and last 50%
            auto visual_shape = visual_features.get_shape();
            size_t batch_size = visual_shape[0];
            size_t total_visual_tokens = visual_shape[1];
            size_t feature_dim = visual_shape[2];
            
            size_t split_point = total_visual_tokens / 2;
            size_t first_half_size = split_point;
            size_t second_half_size = total_visual_tokens - split_point;
            
            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner] Splitting " << total_visual_tokens << " visual tokens into groups: "
                          << first_half_size << " + " << second_half_size << std::endl;
            }
            
            // Create tensor views for first and second halves without copying data
            const float* visual_data = visual_features.data<const float>();
            
            // Create tensor views pointing to specific regions of original data
            ov::Tensor visual_first_half(ov::element::f32, {batch_size, first_half_size, feature_dim}, 
                                         const_cast<float*>(visual_data));
            ov::Tensor visual_second_half(ov::element::f32, {batch_size, second_half_size, feature_dim}, 
                                         const_cast<float*>(visual_data + batch_size * first_half_size * feature_dim));
            
            auto relevance_start = std::chrono::high_resolution_clock::now();
            
            // Step 1: Compute relevance scores sequentially
            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner] Step 1: Computing relevance scores..." << std::endl;
            }
            
            // Compute relevance scores for first half
            ov::Tensor relevance_scores_first = m_relevance_calc.compute(visual_first_half, text_features);
            
            // Compute relevance scores for second half
            ov::Tensor relevance_scores_second = m_relevance_calc.compute(visual_second_half, text_features);
            
            auto relevance_end = std::chrono::high_resolution_clock::now();
            auto relevance_duration = std::chrono::duration_cast<std::chrono::microseconds>(relevance_end - relevance_start);
            
            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner]   Relevance computation took: " << relevance_duration.count() << " us" << std::endl;
            }
            
            // Step 2: Build conditional kernel matrices sequentially
            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner] Step 2: Building kernel matrices..." << std::endl;
            }
            auto kernel_start = std::chrono::high_resolution_clock::now();
            
            // Build kernel matrix for first half
            ov::Tensor kernel_matrix_first = m_kernel_builder.build(visual_first_half, relevance_scores_first);
            
            // Build kernel matrix for second half
            ov::Tensor kernel_matrix_second = m_kernel_builder.build(visual_second_half, relevance_scores_second);
            
            auto kernel_end = std::chrono::high_resolution_clock::now();
            auto kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start);

            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner]   Kernel building took: " << kernel_duration.count() << " us" << std::endl;
            }

            // Step 3: Select tokens using parallel DPP only
            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner] Step 3: Selecting tokens using parallel DPP..." << std::endl;
            }
            auto dpp_start = std::chrono::high_resolution_clock::now();
            
            // Distribute tokens to keep between both halves
            size_t tokens_first_half = num_tokens_to_keep / 2;
            size_t tokens_second_half = num_tokens_to_keep - tokens_first_half;
            
            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner]   Selecting " << tokens_first_half << " tokens from first half, "
                          << tokens_second_half << " tokens from second half in parallel" << std::endl;
            }
            
            // Launch parallel tasks for DPP selection
            std::future<std::vector<std::vector<size_t>>> dpp_first_future = std::async(std::launch::async, [&]() {
                if (m_config.pruning_debug_mode) {
                    std::cout << "[CDPruner] Thread 1: DPP selection for first half..." << std::endl;
                }
                return m_dpp_selector.select(kernel_matrix_first, tokens_first_half);
            });
            
            std::future<std::vector<std::vector<size_t>>> dpp_second_future = std::async(std::launch::async, [&]() {
                if (m_config.pruning_debug_mode) {
                    std::cout << "[CDPruner] Thread 2: DPP selection for second half..." << std::endl;
                }
                return m_dpp_selector.select(kernel_matrix_second, tokens_second_half);
            });
            
            // Wait for both DPP selections to complete
            auto selected_first_batches = dpp_first_future.get();
            auto selected_second_batches = dpp_second_future.get();
            
            std::vector<size_t> selected_first = selected_first_batches[0]; // Take first batch
            std::vector<size_t> selected_second = selected_second_batches[0]; // Take first batch
            
            // Merge results: adjust indices for second half
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
            
            // Store result for this batch
            selected_tokens.push_back(merged_selection);
            auto dpp_end = std::chrono::high_resolution_clock::now();

            dpp_duration = std::chrono::duration_cast<std::chrono::microseconds>(dpp_end - dpp_start);
            
            if (m_config.pruning_debug_mode) {
                std::cout << "[CDPruner]   DPP selection took: " << dpp_duration.count() << " us" << std::endl;
            }
        }
        // Overall timing summary
        auto overall_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(overall_end - overall_start);

        std::cout << "\n+--- CDPruner Performance Summary ----------------------+" << std::endl;
        std::cout << "[CDPruner] Computation mode: "
                  << (m_config.use_ops_model ? (std::string("OV Model by ") + m_config.device)
                                             : "Traditional Step-by-Step by CPU")
                  << std::endl;
        std::cout << "[CDPruner] Total processing time: " << total_duration.count() << " us (" << (total_duration.count() / 1000.0)
                  << " ms)" << std::endl;

        // Performance metrics
        size_t total_input_tokens = visual_shape[0] * visual_shape[1];
        size_t total_output_tokens = visual_shape[0] * num_tokens_to_keep;
        std::cout << "[CDPruner] Performance Metrics:" << std::endl;
        std::cout << "[CDPruner]   DPP selection time: " << dpp_duration.count() << " us (" << (dpp_duration.count() / 1000.0) << " ms)" << std::endl;
        std::cout << "[CDPruner]   Overall throughput: " << (static_cast<double>(total_input_tokens) / total_duration.count() * 1000000) << " input tokens/sec" << std::endl;
        std::cout << "[CDPruner]   Pruning efficiency: " << (static_cast<double>(total_output_tokens) / total_duration.count() * 1000000) << " output tokens/sec" << std::endl;
        std::cout << "[CDPruner]   Pruning ratio: " << (1.0 - static_cast<double>(num_tokens_to_keep) / visual_shape[1]) * 100 << "%" << std::endl;
        std::cout << "+----------------------------------------------------------+" << std::endl;
        if (m_config.pruning_debug_mode) {
            print_selection_statistics(visual_features, selected_tokens);
        }
        
        return selected_tokens;

    } catch (const std::exception& e) {
        throw std::runtime_error("CDPruner::select_tokens failed: " + std::string(e.what()));
    }
}

std::vector<bool> CDPruner::create_pruning_mask(const ov::Tensor& visual_features, 
                                               const ov::Tensor& text_features) {
    auto selected_tokens = select_tokens(visual_features, text_features);
    
    auto visual_shape = visual_features.get_shape();
    size_t batch_size = visual_shape[0];
    size_t total_tokens = visual_shape[1];
    
    return FastGreedyDPP::create_mask(selected_tokens, total_tokens);
}

ov::Tensor CDPruner::apply_pruning(const ov::Tensor& visual_features, 
                                 const ov::Tensor& text_features) {
    auto visual_shape = visual_features.get_shape();
    size_t batch_size = visual_shape[0];
    size_t total_tokens = visual_shape[1];
    size_t feature_dim = visual_shape[2];
    
    // Calculate actual number of tokens to keep based on percentage
    size_t num_tokens_to_keep = static_cast<size_t>(std::round(total_tokens * m_config.visual_tokens_retain_percentage / 100.0));
    
    auto text_shape = text_features.get_shape();
    size_t text_tokens = text_shape[0];
    size_t text_feature_dim = text_shape[1];
    
    // Calculate pruning statistics
    float pruning_ratio = 1.0f - static_cast<float>(num_tokens_to_keep) / static_cast<float>(total_tokens);
    float reduction_percentage = pruning_ratio * 100.0f;
    size_t tokens_removed = total_tokens - num_tokens_to_keep;
    
    // Print consolidated CDPruner overview
    std::cout << "\n+--- CDPruner Processing Overview -----------------------+" << std::endl;
    std::cout << "[CDPruner] Input:  Vision[" << total_tokens << " tokens x " << feature_dim << "D] + Text[" << text_tokens << " tokens x " << text_feature_dim << "D]" << std::endl;
    std::cout << "[CDPruner] Config: Keep " << m_config.visual_tokens_retain_percentage << "% (" << num_tokens_to_keep << "/" << total_tokens << " tokens) | Weight=" << m_config.relevance_weight;
    std::cout << " | " << (m_config.use_ops_model ? "OpenVINO-OPs" : "Traditional") << std::endl;
    std::cout << "[CDPruner] Result: " << tokens_removed << " tokens removed (" << std::fixed << std::setprecision(1) << reduction_percentage << "% reduction)" << std::endl;
    std::cout << "+----------------------------------------------------------+" << std::endl;
    
    auto selected_tokens = select_tokens(visual_features, text_features);
    
    // Create output tensor with selected tokens only
    ov::Tensor pruned_features(visual_features.get_element_type(), 
                              {batch_size, num_tokens_to_keep, feature_dim});
    
    const float* input_data = visual_features.data<const float>();
    float* output_data = pruned_features.data<float>();
    
    for (size_t b = 0; b < batch_size; ++b) {
        const auto& batch_selected = selected_tokens[b];
        
        for (size_t t = 0; t < batch_selected.size(); ++t) {
            size_t src_token_idx = batch_selected[t];
            
            // Copy features for this token
            for (size_t f = 0; f < feature_dim; ++f) {
                size_t src_idx = b * total_tokens * feature_dim + src_token_idx * feature_dim + f;
                size_t dst_idx = b * num_tokens_to_keep * feature_dim + t * feature_dim + f;
                output_data[dst_idx] = input_data[src_idx];
            }
        }
    }
    
    // Update statistics
    m_last_statistics.total_tokens = total_tokens;
    m_last_statistics.selected_tokens = num_tokens_to_keep;
    m_last_statistics.pruning_ratio = 1.0f - static_cast<float>(num_tokens_to_keep) / total_tokens;
    m_last_statistics.batch_size = batch_size;

    // Show selected token indices for debugging
    if (!selected_tokens.empty() && m_config.pruning_debug_mode) {
        std::cout << "\n[CDPruner] Selected token indices (batch 0): [";
        const auto& first_batch_tokens = selected_tokens[0];
        for (size_t i = 0; i < std::min(static_cast<size_t>(10), first_batch_tokens.size()); ++i) {
            if (i > 0) std::cout << ", ";
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
    return m_config.visual_tokens_retain_percentage / 100.0f;
}

size_t CDPruner::get_default_token_count() const {
    // LLaVA typical token count (can be made configurable)
    return 576; // 24x24 patches for most LLaVA configurations
}

PruningStatistics CDPruner::get_last_pruning_statistics() const {
    return m_last_statistics;
}

void CDPruner::validate_config(const Config& config) {
    if (!config.enable_pruning)
        return;

    if (config.visual_tokens_retain_percentage < 0 || config.visual_tokens_retain_percentage > 100) {
        throw std::invalid_argument("visual_tokens_retain_percentage must be between 1 and 100");
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
            std::cout << "[CDPruner]   visual_tokens_retain_percentage: " << m_config.visual_tokens_retain_percentage << "%" << std::endl;
            std::cout << "[CDPruner]   relevance_weight: " << m_config.relevance_weight << std::endl;
            std::cout << "[CDPruner]   enable_pruning: " << (m_config.enable_pruning ? "true" : "false") << std::endl;
            std::cout << "[CDPruner]   use_ops_model: " << (m_config.use_ops_model ? "true" : "false") << std::endl;
        }

        return true;
    } catch (const std::exception& e) {
        if (m_config.pruning_debug_mode) {
            std::cerr << "[CDPruner] Failed to update configuration: " << e.what() << std::endl;
        }
        return false;
    }
}

void CDPruner::validate_input_tensors(const ov::Tensor& visual_features, 
                                    const ov::Tensor& text_features) {
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
    size_t num_tokens_to_keep = static_cast<size_t>(std::round(visual_shape[1] * m_config.visual_tokens_retain_percentage / 100.0));
    
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
    size_t selected_token_count = static_cast<size_t>(std::round(total_tokens * m_config.visual_tokens_retain_percentage / 100.0));
    
    // Start CDPruner output block
    std::cout << "\n+--- CDPruner Results -----------------------------------+" << std::endl;
    
    // Show compact performance summary
    std::cout << "[CDPruner] Summary: " << total_tokens << " -> " << selected_token_count << " tokens ("
              << std::fixed << std::setprecision(1) << (1.0f - static_cast<float>(selected_token_count) / total_tokens) * 100.0f
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

} // namespace ov::genai::cdpruner 