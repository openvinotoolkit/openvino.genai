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
                computation_mode += "<Split>";
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
            computation_mode = "Traditional Step-by-Step";
            auto relevance_start = std::chrono::high_resolution_clock::now();
            // CDPruner Step 1: Compute relevance scores
            if (use_splitting) {
                computation_mode += "<Split>";
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
            print_cdpruner_performance_summary(computation_mode,
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
    size_t tokens_removed = total_tokens - num_tokens_to_keep;

    print_cdpruner_processing_overview(total_tokens,
                                       feature_dim,
                                       text_tokens,
                                       text_feature_dim,
                                       num_tokens_to_keep,
                                       tokens_removed,
                                       m_config.pruning_ratio,
                                       m_config.relevance_weight);

    auto selected_tokens = select_tokens(visual_features, text_features, silent);
    m_last_selected_tokens = selected_tokens;

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

    print_cdpruner_processing_overview(visual_features_list.size(),
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
    std::vector<std::vector<size_t>> aggregated_selected;

    size_t global_offset = 0;

    auto overall_start = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds total_kernel_duration{0};
    std::chrono::microseconds total_dpp_duration{0};

    for (size_t frame_idx = 0; frame_idx < visual_features_list.size(); ++frame_idx) {
        const auto& visual_feature = visual_features_list[frame_idx];
        ov::Tensor pruned_feature = apply_pruning(visual_feature, text_features, true);
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

    m_last_selected_tokens = aggregated_selected;

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
    print_cdpruner_performance_summary(computation_mode,
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

PruningStatistics CDPruner::get_last_pruning_statistics() const {
    return m_last_statistics;
}

void CDPruner::print_cdpruner_processing_overview(size_t total_tokens,
                                                  size_t feature_dim,
                                                  size_t text_tokens,
                                                  size_t text_feature_dim,
                                                  size_t num_tokens_to_keep,
                                                  size_t tokens_removed,
                                                  size_t pruning_ratio,
                                                  float relevance_weight) {
    if (!utils::env_setup_for_print_debug_info())
        return;
    float reduction_percentage = 0.0f;
    if (total_tokens > 0) {
        reduction_percentage = (static_cast<float>(tokens_removed) / static_cast<float>(total_tokens)) * 100.0f;
    }
    std::cout << ("+--- CDPruner Processing Overview -------------------------+") << std::endl;
    std::cout << "[CDPruner] Input:  Vision[" << total_tokens << " tokens x " << feature_dim << "D] + Text["
              << text_tokens << " tokens x " << text_feature_dim << "D]" << std::endl;
    std::cout << "[CDPruner] Config: Keep " << (100 - pruning_ratio) << "% (" << num_tokens_to_keep << "/"
              << total_tokens << " tokens) | Weight=" << relevance_weight
              << " | Split Threshold=" << m_config.split_threshold << std::endl;
    std::cout << "[CDPruner] Result: " << tokens_removed << " tokens removed (" << reduction_percentage
              << "% reduction)" << std::endl;
    std::cout << "+----------------------------------------------------------+" << std::endl;
}

void CDPruner::print_cdpruner_processing_overview(size_t frame_count,
                                                  size_t tokens_per_frame,
                                                  size_t feature_dim,
                                                  size_t text_tokens,
                                                  size_t text_feature_dim,
                                                  size_t num_tokens_to_keep_per_frame,
                                                  size_t total_input_tokens,
                                                  size_t total_output_tokens,
                                                  size_t pruning_ratio,
                                                  float relevance_weight) {
    if (!utils::env_setup_for_print_debug_info())
        return;
    float reduction_percentage = 0.0f;
    if (total_input_tokens > 0) {
        reduction_percentage =
            (1.0f - static_cast<float>(total_output_tokens) / static_cast<float>(total_input_tokens)) * 100.0f;
    }

    std::cout << "+--- CDPruner Multi-Frame Processing Overview -------------+" << std::endl;
    std::cout << "[CDPruner] Input:  " << frame_count << " frames x Vision[" << tokens_per_frame << " tokens x "
              << feature_dim << "D] + Text[" << text_tokens << " tokens x " << text_feature_dim << "D]" << std::endl;
    std::cout << "[CDPruner] Config: Keep " << pruning_ratio << "% (" << num_tokens_to_keep_per_frame << "/"
              << tokens_per_frame << " tokens per frame) | Weight=" << relevance_weight
              << " | Split Threshold=" << m_config.split_threshold << std::endl;
    std::cout << "[CDPruner] Total:  " << total_input_tokens << " → " << total_output_tokens << " tokens ("
              << reduction_percentage << "% reduction)" << std::endl;
    std::cout << "+----------------------------------------------------------+" << std::endl;
}

void CDPruner::print_cdpruner_performance_summary(const std::string& computation_mode,
                                                  std::chrono::microseconds total_duration,
                                                  std::chrono::microseconds kernel_duration,
                                                  std::chrono::microseconds dpp_duration,
                                                  size_t total_input_tokens,
                                                  size_t total_output_tokens) {
    if (!utils::env_setup_for_print_debug_info())
        return;
    std::cout << "+-------------- CDPruner Performance Summary --------------+" << std::endl;
    std::cout << "[CDPruner] Computation mode: " << computation_mode << std::endl;
    std::cout << "[CDPruner] Total processing time: " << total_duration.count() << " us ("
              << total_duration.count() / 1000.0 << " ms)" << std::endl;
    std::cout << "[CDPruner] Performance Metrics:" << std::endl;
    std::cout << "[CDPruner]   Kernel computation time: " << kernel_duration.count() << " us ("
              << kernel_duration.count() / 1000.0 << " ms)" << std::endl;
    std::cout << "[CDPruner]   DPP selection time: " << dpp_duration.count() << " us (" << dpp_duration.count() / 1000.0
              << " ms)" << std::endl;
    if (total_duration.count() > 0) {
        const double total_time_sec = static_cast<double>(total_duration.count()) / 1'000'000.0;
        std::cout << "[CDPruner]   Overall throughput: " << static_cast<double>(total_input_tokens) / total_time_sec
                  << " input tokens/sec" << std::endl;
        std::cout << "[CDPruner]   Pruning efficiency: " << static_cast<double>(total_output_tokens) / total_time_sec
                  << " output tokens/sec" << std::endl;
    } else {
        std::cout << "[CDPruner]   Overall throughput: N/A" << std::endl;
        std::cout << "[CDPruner]   Pruning efficiency: N/A" << std::endl;
    }
    if (total_input_tokens > 0) {
        std::cout << "[CDPruner]   Pruning ratio: "
                  << (1.0 - static_cast<double>(total_output_tokens) / total_input_tokens) * 100 << "%" << std::endl;
    } else {
        std::cout << "[CDPruner]   Pruning ratio: N/A" << std::endl;
    }
    std::cout << "+----------------------------------------------------------+" << std::endl;
}

void CDPruner::print_cdpruner_performance_summary(const std::string& computation_mode,
                                                  std::chrono::microseconds total_duration,
                                                  size_t frame_count,
                                                  size_t total_input_tokens,
                                                  size_t actual_total_tokens,
                                                  size_t actual_batch_size,
                                                  size_t actual_hidden_dim) {
    if (!utils::env_setup_for_print_debug_info())
        return;
    std::cout << "+---------- CDPruner Multi-Frame Processing Performance Summary ---------+" << std::endl;
    std::cout << "[CDPruner] Computation mode: " << computation_mode << std::endl;
    std::cout << "[CDPruner] Total processing time: " << total_duration.count() << " us ("
              << total_duration.count() / 1000.0 << " ms)" << std::endl;
    std::cout << "[CDPruner] Performance Metrics:" << std::endl;

    if (frame_count > 0) {
        std::cout << "[CDPruner]   Frames processed: " << frame_count << std::endl;
        std::cout << "[CDPruner]   Average time per frame: " << total_duration.count() / frame_count << " us ("
                  << total_duration.count() / frame_count / 1000.0 << " ms)" << std::endl;
    } else {
        std::cout << "[CDPruner]   Frames processed: 0" << std::endl;
        std::cout << "[CDPruner]   Average time per frame: N/A" << std::endl;
    }

    if (total_duration.count() > 0) {
        const double duration_us = static_cast<double>(total_duration.count());
        std::cout << "[CDPruner]   Overall throughput: "
                  << (static_cast<double>(total_input_tokens) / duration_us * 1'000'000) << " input tokens/sec"
                  << std::endl;
        std::cout << "[CDPruner]   Pruning efficiency: "
                  << (static_cast<double>(actual_total_tokens) / duration_us * 1'000'000) << " output tokens/sec"
                  << std::endl;
    } else {
        std::cout << "[CDPruner]   Overall throughput: N/A" << std::endl;
        std::cout << "[CDPruner]   Pruning efficiency: N/A" << std::endl;
    }

    if (total_input_tokens > 0) {
        std::cout << "[CDPruner]   Combined pruning ratio: "
                  << (1.0 - static_cast<double>(actual_total_tokens) / total_input_tokens) * 100 << "%" << std::endl;
    } else {
        std::cout << "[CDPruner]   Combined pruning ratio: N/A" << std::endl;
    }

    std::cout << "[CDPruner] Final result: [" << actual_batch_size << ", " << actual_total_tokens << ", "
              << actual_hidden_dim << "] from " << frame_count << " frames" << std::endl;
    std::cout << "+----------------------------------------------------------+" << std::endl;
}

void CDPruner::validate_config(const Config& config) {
    if (config.pruning_ratio == 0)
        return;  // Pruning disabled, no validation needed

    if (config.pruning_ratio < 1 || config.pruning_ratio > 100) {
        throw std::invalid_argument("pruning ratio must be between 0 and 100");
    }

    if (config.relevance_weight < 0.0f || config.relevance_weight > 1.0f) {
        throw std::invalid_argument("relevance weight must be between 0.0 and 1.0");
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
}  // namespace ov::genai::cdpruner