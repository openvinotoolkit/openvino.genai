// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <openvino/runtime/tensor.hpp>

namespace eagle3 {

// Pipeline Step Enumeration

/// @brief Eagle3 pipeline execution stages for structured logging
enum class PipelineStep {
    INIT,       ///< Initialization phase
    ITER,       ///< Inference iteration
    SAMPLE,     ///< Token sampling
    VALID,      ///< Validation phase
    KV_CACHE,   ///< KV cache operations
};

// Utility Functions

/// @brief Get logging prefix for a pipeline step
inline const char* get_step_prefix(PipelineStep step) {
    switch (step) {
    case PipelineStep::INIT:
        return "[EAGLE3-INIT]";
    case PipelineStep::ITER:
        return "[EAGLE3-ITER]";
    case PipelineStep::SAMPLE:
        return "[EAGLE3-SAMPLE]";
    case PipelineStep::VALID:
        return "[EAGLE3-VALID]";
    case PipelineStep::KV_CACHE:
        return "[EAGLE3-KVCACHE]";
    default:
        return "[EAGLE3]";
    }
}

/// @brief Format tensor shape as string
inline std::string format_shape(const ov::Shape& shape) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        ss << shape[i];
        if (i + 1 < shape.size())
            ss << ", ";
    }
    ss << "]";
    return ss.str();
}

/// @brief Format vector of tokens as string, showing last max_show tokens
inline std::string format_tokens(const std::vector<int64_t>& tokens, size_t max_show = 20) {
    std::ostringstream ss;
    ss << "[";
    
    if (tokens.size() > max_show) {
        ss << "(" << (tokens.size() - max_show) << " more) ... ";
    }
    
    const size_t start_idx = (tokens.size() > max_show) ? (tokens.size() - max_show) : 0;
    for (size_t i = start_idx; i < tokens.size(); ++i) {
        if (i > start_idx)
            ss << ", ";
        ss << tokens[i];
    }
    ss << "]";
    return ss.str();
}

// Logging Functions

/// @brief Log debug message for a specific pipeline step
inline void log_debug(PipelineStep step, const std::string& message, bool verbose) {
    if (verbose) {
        std::cout << get_step_prefix(step) << " " << message << std::endl;
    }
}

/// @brief Log general debug message
inline void log_debug(const std::string& message, bool verbose) {
    if (verbose) {
        std::cout << "[EAGLE3-DEBUG] " << message << std::endl;
    }
}

/// @brief Log info message (always shown)
inline void log_info(const std::string& message) {
    std::cout << "[EAGLE3-PIPELINE] " << message << std::endl;
}

// Tensor Logging Functions

/// @brief Log tensor information (shape and type)
inline void log_tensor_info(const std::string& name, const ov::Tensor& tensor, bool verbose) {
    if (!verbose)
        return;
    std::cout << "[EAGLE3-TENSOR] " << name << " shape: " << format_shape(tensor.get_shape())
              << ", type: " << tensor.get_element_type() << std::endl;
}

/// @brief Log tensor content values
inline void log_tensor_content(const std::string& name,
                               const ov::Tensor& tensor,
                               bool verbose,
                               size_t max_elements = 10) {
    if (!verbose || !tensor)
        return;

    const size_t total_elements = tensor.get_size();
    // For key inputs, show all elements
    const bool show_all = (name == "input_ids" || name == "position_ids" || name == "attention_mask");
    const size_t elements_to_show = show_all ? total_elements : std::min(max_elements, total_elements);

    std::ostringstream ss;
    ss << "[EAGLE3-TENSOR] " << name << " values (" << elements_to_show << " elements): ";

    const auto elem_type = tensor.get_element_type();
    if (elem_type == ov::element::i64) {
        const int64_t* data = tensor.data<const int64_t>();
        for (size_t i = 0; i < elements_to_show; ++i) {
            if (i > 0)
                ss << ", ";
            ss << data[i];
        }
    } else if (elem_type == ov::element::f32) {
        const float* data = tensor.data<const float>();
        for (size_t i = 0; i < elements_to_show; ++i) {
            if (i > 0)
                ss << ", ";
            ss << std::fixed << std::setprecision(4) << data[i];
        }
    } else if (elem_type == ov::element::i32) {
        const int32_t* data = tensor.data<const int32_t>();
        for (size_t i = 0; i < elements_to_show; ++i) {
            if (i > 0)
                ss << ", ";
            ss << data[i];
        }
    }
    
    std::cout << ss.str() << std::endl;
}

/// @brief Log model inputs (input_ids, attention_mask, position_ids)
inline void log_model_inputs(const ov::Tensor& input_ids,
                             const ov::Tensor& attention_mask,
                             const ov::Tensor& position_ids,
                             bool verbose) {
    if (!verbose)
        return;

    std::cout << "[EAGLE3-INPUT] ========== MODEL INPUTS ==========" << std::endl;
    log_tensor_info("input_ids", input_ids, verbose);
    log_tensor_content("input_ids", input_ids, verbose, 0);
    log_tensor_info("attention_mask", attention_mask, verbose);
    log_tensor_content("attention_mask", attention_mask, verbose, 0);
    log_tensor_info("position_ids", position_ids, verbose);
    log_tensor_content("position_ids", position_ids, verbose, 0);
    std::cout << "[EAGLE3-INPUT] =================================" << std::endl;
}

/// @brief Log model outputs (logits with top-k, hidden_features)
inline void log_model_outputs(const ov::Tensor& logits, 
                              const ov::Tensor& hidden_features,
                              bool verbose) {
    if (!verbose)
        return;

    std::cout << "[EAGLE3-OUTPUT] ========== MODEL OUTPUTS =========" << std::endl;

    if (logits && logits.get_size() > 0) {
        const auto logits_shape = logits.get_shape();
        log_tensor_info("logits", logits, verbose);

        if (logits_shape.size() == 3) {
            const size_t seq_len = logits_shape[1];
            const size_t vocab_size = logits_shape[2];

            // Show only last few positions for long sequences
            const size_t start_pos = (seq_len > 5) ? (seq_len - 5) : 0;
            if (start_pos > 0) {
                std::cout << "[EAGLE3-OUTPUT] Showing last " << (seq_len - start_pos)
                          << " positions (total: " << seq_len << ")" << std::endl;
            }

            // Show top-k logits for each position
            for (size_t pos = start_pos; pos < seq_len; ++pos) {
                const float* logits_data = logits.data<const float>() + pos * vocab_size;

                // Find top 5 tokens
                std::vector<std::pair<float, int64_t>> top_logits;
                top_logits.reserve(vocab_size);
                for (size_t i = 0; i < vocab_size; ++i) {
                    top_logits.emplace_back(logits_data[i], static_cast<int64_t>(i));
                }
                std::partial_sort(top_logits.begin(),
                                  top_logits.begin() + std::min<size_t>(5, vocab_size),
                                  top_logits.end(),
                                  std::greater<>());

                std::ostringstream ss;
                ss << "[EAGLE3-OUTPUT] Pos " << pos << " top-5: ";
                for (size_t i = 0; i < std::min<size_t>(5, vocab_size); ++i) {
                    if (i > 0)
                        ss << ", ";
                    ss << "token " << top_logits[i].second << ": " << std::fixed 
                       << std::setprecision(2) << top_logits[i].first;
                }
                std::cout << ss.str() << std::endl;
            }
        }
    }

    if (hidden_features && hidden_features.get_size() > 0) {
        log_tensor_info("hidden_features", hidden_features, verbose);
    }

    std::cout << "[EAGLE3-OUTPUT] =================================" << std::endl;
}

// High-level Logging Functions

/// @brief Log generation step with step number
inline void log_generation_step(const std::string& step_name, size_t step_number, bool verbose) {
    if (verbose) {
        std::cout << "\n[EAGLE3] ===== " << step_name << " (step " << step_number 
                  << ") =====" << std::endl;
    }
}

/// @brief Log sequence state for debugging
inline void log_sequence_state(const std::string& context,
                               size_t prompt_len,
                               size_t target_len,
                               size_t draft_len,
                               const std::vector<int64_t>& target_tokens,
                               const std::vector<int64_t>& draft_tokens,
                               bool verbose) {
    if (!verbose)
        return;

    std::cout << "[EAGLE3-STATE] Sequence (" << context << "):" << std::endl;
    std::cout << "  Prompt: " << prompt_len << " tokens" << std::endl;
    std::cout << "  Target model: " << target_len << " tokens" << std::endl;
    std::cout << "  Draft model: " << draft_len << " tokens" << std::endl;

    if (!target_tokens.empty()) {
        std::cout << "  Target generated: " << format_tokens(target_tokens) << std::endl;
    }

    if (!draft_tokens.empty() && draft_tokens != target_tokens) {
        std::cout << "  Draft generated: " << format_tokens(draft_tokens) << std::endl;
    }
}

}  // namespace eagle3
