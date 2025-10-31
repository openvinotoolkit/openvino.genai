// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speculative_decoding_stateful_eagle3.hpp"
#include "continuous_batching/timer.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "speculative_decoding_eagle3_impl.hpp"
#include "utils.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <numeric>

namespace ov::genai {
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
} // ov::genai

namespace {

/**
 * @brief Utility function to stream generated tokens
 */
ov::genai::StreamingStatus stream_generated_tokens(
    std::shared_ptr<ov::genai::StreamerBase> streamer_ptr,
    const std::vector<int64_t>& tokens) {
    if (streamer_ptr) {
        return streamer_ptr->write(tokens);
    }
    return ov::genai::StreamingStatus{};
}

/**
 * @brief Utility to format time duration for logging
 */
std::string format_duration_us(uint64_t microseconds) {
    if (microseconds < 1000) {
        return std::to_string(microseconds) + "μs";
    } else if (microseconds < 1000000) {
        return std::to_string(microseconds / 1000.0) + "ms";
    } else {
        return std::to_string(microseconds / 1000000.0) + "s";
    }
}

/**
 * @brief Extract last token's hidden state from tensor
 */
ov::Tensor extract_last_hidden_state(const ov::Tensor& hidden_features) {
    if (!hidden_features || hidden_features.get_size() == 0) {
        return ov::Tensor{};
    }
    
    auto shape = hidden_features.get_shape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] == 0) {
        return ov::Tensor{};
    }
    
    std::size_t seq_len = shape[1];
    std::size_t hidden_size = shape[2];
    
    ov::Tensor last_hidden(ov::element::f32, {1, 1, hidden_size});
    
    if (hidden_features.get_element_type() == ov::element::f32) {
        const float* src = hidden_features.data<const float>() + (seq_len - 1) * hidden_size;
        std::copy_n(src, hidden_size, last_hidden.data<float>());
    }
    
    return last_hidden;
}

} // anonymous namespace

namespace ov {
namespace genai {

//==================================================================================================
// Eagle3InferWrapper Implementation
//==================================================================================================

Eagle3InferWrapper::Eagle3InferWrapper(const ov::genai::ModelDesc& model_desc)
    : m_device(model_desc.device)
    , m_properties(model_desc.properties)
    , m_generation_config(model_desc.generation_config)
    , m_tokenizer(model_desc.tokenizer) {
    
    log_debug("Initializing Eagle3InferWrapper for device: " + m_device);
    
    // Get KV-cache axes positions early
    m_kv_axes_pos = ov::genai::utils::get_kv_axes_pos(model_desc.model);
    
    // Compile model based on device type (following LLMInferWrapper pattern)
    if (m_device == "NPU") {
        auto [compiled, kv_desc] = ov::genai::utils::compile_decoder_for_npu(
            model_desc.model, m_properties, m_kv_axes_pos);
        m_max_prompt_len = kv_desc.max_prompt_len;
        m_kv_cache_capacity = kv_desc.max_prompt_len + kv_desc.min_response_len;
        m_request = compiled.create_infer_request();
        
        log_debug("NPU: max_prompt_len=" + std::to_string(m_max_prompt_len) + 
                  ", kv_cache_capacity=" + std::to_string(m_kv_cache_capacity));
    } else {
        // TODO: We might need it for manipulations with indices
        // utils::apply_gather_before_matmul_transformation(model_desc.model);
        m_request = ov::genai::utils::singleton_core().compile_model(
            model_desc.model, m_device, m_properties).create_infer_request();
        
        log_debug(m_device + ": model compiled successfully");
    }
    
    // Initialize performance metrics
    m_raw_perf_metrics.m_inference_durations = {ov::genai::MicroSeconds(0.0f)};
    m_raw_perf_metrics.tokenization_durations = {ov::genai::MicroSeconds(0.0f)};
    m_raw_perf_metrics.detokenization_durations = {ov::genai::MicroSeconds(0.0f)};
    
    log_debug("Eagle3InferWrapper initialization completed");
}

void Eagle3InferWrapper::initialize_sequence(const ov::Tensor& input_ids, const ov::Tensor& position_ids) {
    // Initialize sequence
    const int64_t* ids_data = input_ids.data<const int64_t>();
    std::size_t seq_len = input_ids.get_size();
    m_tokens.assign(ids_data, ids_data + seq_len);
    
    if (position_ids) {
        const int64_t* pos_data = position_ids.data<const int64_t>();
        m_positions.assign(pos_data, pos_data + position_ids.get_size());
    }
    
    log_debug("Initialized sequence with " + std::to_string(m_tokens.size()) + " tokens");
}

void Eagle3InferWrapper::append_tokens(const std::vector<int64_t>& tokens) {
    if (tokens.empty()) return;
    
    // Append to sequence
    std::size_t old_size = m_tokens.size();
    m_tokens.insert(m_tokens.end(), tokens.begin(), tokens.end());
    
    // Update positions
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        m_positions.push_back(static_cast<int64_t>(old_size + i));
    }
    
    log_debug("Appended " + std::to_string(tokens.size()) + " tokens - Sequence size: " + 
              std::to_string(m_tokens.size()));
}

void Eagle3InferWrapper::truncate_sequence(std::size_t size) {
    if (size < m_tokens.size()) {
        m_tokens.resize(size);
        m_positions.resize(size);
    }
    
    log_debug("Truncated sequence to size: " + std::to_string(m_tokens.size()));
}

void Eagle3InferWrapper::trim_kv_cache(std::size_t tokens_to_remove) {
    if (tokens_to_remove == 0 || m_processed_tokens == 0) {
        log_debug("No KV cache trimming needed");
        return;
    }
    
    if (tokens_to_remove >= m_processed_tokens) {
        log_debug("Warning: trying to trim more tokens than processed, resetting KV cache");
        // Reset the entire KV cache if removing all or more tokens
        reset_state();
        return;
    }
    
    log_debug("Trimming KV cache: removing " + std::to_string(tokens_to_remove) + " tokens");
    
    // For NPU, KV cache trimming is handled by position IDs on NPUW side
    if (m_device != "NPU") {
        // Trim KV cache values
        ov::genai::utils::KVCacheState to_trim_state;
        to_trim_state.num_tokens_to_trim = tokens_to_remove;
        to_trim_state.seq_length_axis = m_kv_axes_pos.seq_len;
        to_trim_state.reset_mem_state = false;
        
        try {
            ov::genai::utils::trim_kv_cache(m_request, to_trim_state, {});
            log_debug("KV cache trimmed successfully");
        } catch (const std::exception& e) {
            log_debug("Warning: KV cache trimming failed: " + std::string(e.what()));
        }
    }
    
    // Update processed tokens count
    m_processed_tokens -= tokens_to_remove;
    
    log_debug("KV cache trimmed - processed tokens now: " + std::to_string(m_processed_tokens));
}

void Eagle3InferWrapper::reset_state() {
    m_tokens.clear();
    m_positions.clear();
    m_processed_tokens = 0;
    m_last_sampled_token = -1;
    m_metrics.reset();
    
    log_debug("State reset completed");
}

void Eagle3InferWrapper::release_memory() {
    m_request.get_compiled_model().release_memory();
    log_debug("Released compiled model memory");
}

ov::Tensor Eagle3InferWrapper::infer_target_model(const ov::Tensor& input_ids, 
                                                  const ov::Tensor& attention_mask, 
                                                  const ov::Tensor& position_ids) {
    log_debug("Starting target model inference");
    log_model_inputs(input_ids, attention_mask, position_ids);
    
    // Validate NPU constraints
    if (m_device == "NPU") {
        auto prompt_len = input_ids.get_shape()[1];
        if (prompt_len > m_max_prompt_len) {
            throw std::runtime_error("NPU prompt length " + std::to_string(prompt_len) + 
                                   " exceeds maximum " + std::to_string(m_max_prompt_len));
        }
    }
    
    // Set input tensors
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);
    
    if (m_device != "NPU") {
        m_request.get_tensor("beam_idx").set_shape({BATCH_SIZE});
        m_request.get_tensor("beam_idx").data<int32_t>()[0] = 0;
    }
    
    // Execute inference with timing
    uint64_t inference_time_us = execute_inference(input_ids);
    update_performance_metrics(inference_time_us, input_ids.get_shape()[1]);
    
    // Get outputs and log them
    auto logits = get_logits();
    auto hidden_features = get_hidden_features();
    log_model_outputs(logits, hidden_features);
    
    log_debug("Target model inference completed in " + format_duration_us(inference_time_us));
    return logits;
}

ov::Tensor Eagle3InferWrapper::infer_draft_model(const ov::Tensor& input_ids,
                                                 const ov::Tensor& attention_mask,
                                                 const ov::Tensor& position_ids,
                                                 const ov::Tensor& target_hidden_features,
                                                 const ov::Tensor& internal_hidden_features) {
    log_debug("Starting draft model inference");
    log_model_inputs(input_ids, attention_mask, position_ids);
    
    // Set basic input tensors
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);
    
    // Handle hidden state inputs (Eagle3 requires exactly one of target/internal)
    bool has_target = target_hidden_features && target_hidden_features.get_size() > 0;
    bool has_internal = internal_hidden_features && internal_hidden_features.get_size() > 0;
    
    if (!(has_target ^ has_internal)) {
        throw std::runtime_error("Draft model requires exactly one of target_hidden_features or internal_hidden_features");
    }
    
    ov::Tensor target_tensor, internal_tensor;
    
    if (has_target) {
        // Use target hidden features, create placeholder for internal
        auto t_shape = target_hidden_features.get_shape();
        if (t_shape.size() != 3 || t_shape.back() % 3 != 0) {
            throw std::runtime_error("Invalid target hidden features shape for Eagle3");
        }
        
        target_tensor = target_hidden_features;
        auto internal_shape = t_shape;
        internal_shape.back() = t_shape.back() / 3;
        internal_tensor = create_hidden_state_placeholder(internal_shape);
        
        if (m_verbose) {
            log_tensor_info("processed_target_tensor", target_tensor);
            log_tensor_info("created_internal_placeholder", internal_tensor);
        }
    } else {
        // Use internal hidden features, create placeholder for target  
        auto i_shape = internal_hidden_features.get_shape();
        if (i_shape.size() != 3) {
            throw std::runtime_error("Invalid internal hidden features shape for Eagle3");
        }
        
        internal_tensor = internal_hidden_features;
        auto target_shape = i_shape;
        target_shape.back() = i_shape.back() * 3;
        target_tensor = create_hidden_state_placeholder(target_shape);
        
        if (m_verbose) {
            log_tensor_info("processed_internal_tensor", internal_tensor);
            log_tensor_info("created_target_placeholder", target_tensor);
        }
    }
    
    m_request.set_tensor("hidden_states", target_tensor);
    m_request.set_tensor("internal_hidden_states", internal_tensor);
    
    // Print hidden_states and internal_hidden_states values if verbose
    if (m_verbose) {
        std::cout << "[EAGLE3-WRAPPER] Final Hidden State Tensors:" << std::endl;
        log_tensor_info("hidden_states", target_tensor);
        log_tensor_content("hidden_states", target_tensor, 10);  // 0 means show all
        
        log_tensor_info("internal_hidden_states", internal_tensor);
        log_tensor_content("internal_hidden_states", internal_tensor, 10);  // 0 means show all
    }
    
    if (m_device != "NPU") {
        m_request.get_tensor("beam_idx").set_shape({BATCH_SIZE});
        m_request.get_tensor("beam_idx").data<int32_t>()[0] = 0;
    }
    
    // Execute inference with timing
    uint64_t inference_time_us = execute_inference(input_ids);
    update_performance_metrics(inference_time_us, input_ids.get_shape()[1]);
    
    // Get outputs and log them
    auto logits = get_logits();
    auto hidden_features = get_hidden_features();
    log_model_outputs(logits, hidden_features);
    
    log_debug("Draft model inference completed in " + format_duration_us(inference_time_us));
    return logits;
}

void Eagle3InferWrapper::build_model_inputs(int64_t begin_idx, std::size_t size,
                                           ov::Tensor& input_ids, ov::Tensor& attention_mask, 
                                           ov::Tensor& position_ids, bool reset_positions, bool full_attention_mask) {
    const auto& tokens = m_tokens;
    const auto& positions = m_positions;
    
    if (tokens.empty() || size == 0) {
        throw std::runtime_error("Cannot build model inputs: empty sequence or zero size");
    }
    
    // Handle negative indexing
    int64_t sequence_size = static_cast<int64_t>(tokens.size());
    int64_t actual_begin = (begin_idx < 0) ? sequence_size + begin_idx : begin_idx;
    
    if (actual_begin < 0 || actual_begin >= sequence_size || actual_begin + static_cast<int64_t>(size) > sequence_size) {
        throw std::runtime_error("Invalid slice range for model inputs");
    }
    
    // Create tensors for input_ids and position_ids (current input)
    input_ids = ov::Tensor(ov::element::i64, {1, size});
    position_ids = ov::Tensor(ov::element::i64, {1, size});
    
    // Fill input_ids
    std::copy_n(tokens.data() + actual_begin, size, input_ids.data<int64_t>());
    
    // Fill position_ids
    if (reset_positions || positions.empty()) {
        std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + size, actual_begin);
    } else {
        std::copy_n(positions.data() + actual_begin, size, position_ids.data<int64_t>());
    }
    
    // Attention mask: full sequence or just input_ids size
    if (full_attention_mask) {
        attention_mask = ov::Tensor(ov::element::i64, {1, tokens.size()});
        std::fill_n(attention_mask.data<int64_t>(), tokens.size(), 1);
        log_debug("Model inputs built: begin=" + std::to_string(actual_begin) + ", size=" + std::to_string(size) + ", attention_mask.size=" + std::to_string(tokens.size()));
    } else {
        attention_mask = ov::Tensor(ov::element::i64, {1, size});
        std::fill_n(attention_mask.data<int64_t>(), size, 1);
        log_debug("Model inputs built: begin=" + std::to_string(actual_begin) + ", size=" + std::to_string(size) + ", attention_mask.size=" + std::to_string(size));
    }
}

ov::Tensor Eagle3InferWrapper::create_hidden_state_placeholder(const ov::Shape& shape) const {
    ov::Tensor tensor(ov::element::f32, shape);
    std::fill_n(tensor.data<float>(), tensor.get_size(), 0.0f);
    return tensor;
}

std::variant<int64_t, std::vector<int64_t>> Eagle3InferWrapper::sample_tokens(const ov::Tensor& logits, std::size_t count) {
    auto logits_shape = logits.get_shape();
    if (logits_shape.size() != 3 || logits_shape[0] != 1) {
        throw std::runtime_error("Invalid logits shape for sampling");
    }
    
    std::size_t seq_len = logits_shape[1];
    std::size_t vocab_size = logits_shape[2];
    
    if (count > seq_len) {
        throw std::runtime_error("Requested token count exceeds sequence length");
    }
    
    log_debug("Sampling " + std::to_string(count) + " tokens from logits with shape [" + 
              std::to_string(logits_shape[0]) + ", " + std::to_string(seq_len) + ", " + std::to_string(vocab_size) + "]");
    
    auto sample_single = [&](std::size_t pos) -> int64_t {
        const float* logits_data = logits.data<const float>() + pos * vocab_size;
        auto max_it = std::max_element(logits_data, logits_data + vocab_size);
        int64_t token = static_cast<int64_t>(max_it - logits_data);
        float max_prob = *max_it;
        
        if (m_verbose) {
            log_debug("Position " + std::to_string(pos) + ": sampled token " + std::to_string(token) + 
                      " with logit value " + std::to_string(max_prob));
        }
        
        return token;
    };
    
    if (count == 1) {
        int64_t token = sample_single(seq_len - 1);
        m_last_sampled_token = token;
        log_debug("Sampled single token: " + std::to_string(token));
        return token;
    } else {
        std::vector<int64_t> tokens;
        tokens.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            tokens.push_back(sample_single(seq_len - count + i));
        }
        if (!tokens.empty()) {
            m_last_sampled_token = tokens.back();
        }
        
        if (m_verbose) {
            std::cout << "[EAGLE3-WRAPPER] Sampled " << count << " tokens: ";
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                std::cout << tokens[i];
                if (i + 1 < tokens.size()) std::cout << ", ";
            }
            std::cout << std::endl;
        }
        
        return tokens;
    }
}

ov::Tensor Eagle3InferWrapper::get_logits() const {
    return m_request.get_tensor("logits");
}

ov::Tensor Eagle3InferWrapper::get_hidden_features() const {
    try {
        auto hidden_state = m_request.get_tensor("last_hidden_state");
        
        // For NPU, the hidden state may be padded to a fixed length (e.g., 1024)
        // We need to trim the padding from the beginning to match actual token length
        if (m_device == "NPU" && hidden_state && hidden_state.get_size() > 0) {
            auto shape = hidden_state.get_shape();
            
            // Expected shape: [batch=1, padded_seq_len, hidden_size]
            if (shape.size() == 3 && shape[0] == 1) {
                size_t padded_seq_len = shape[1];
                size_t hidden_size = shape[2];
                
                // Get actual sequence length from current input_ids tensor
                auto input_ids = m_request.get_tensor("input_ids");
                size_t actual_seq_len = input_ids.get_shape()[1]; // Real input sequence length
                
                if (actual_seq_len < padded_seq_len) {
                    // Calculate padding amount (at the beginning)
                    size_t padding_len = padded_seq_len - actual_seq_len;
                    
                    log_debug("Trimming NPU hidden state padding: padded_len=" + std::to_string(padded_seq_len) + 
                              ", actual_len=" + std::to_string(actual_seq_len) + 
                              ", removing " + std::to_string(padding_len) + " padding tokens from start");
                    
                    // Create trimmed tensor with actual sequence length
                    ov::Tensor trimmed_hidden(ov::element::f32, {1, actual_seq_len, hidden_size});
                    
                    if (hidden_state.get_element_type() == ov::element::f32) {
                        const float* src = hidden_state.data<const float>();
                        float* dst = trimmed_hidden.data<float>();
                        
                        // Copy from [padding_len:] to remove padding at the beginning
                        size_t offset = padding_len * hidden_size;
                        size_t copy_size = actual_seq_len * hidden_size;
                        std::copy_n(src + offset, copy_size, dst);
                        
                        if (m_verbose) {
                            log_debug("NPU hidden state trimmed: [1, " + std::to_string(padded_seq_len) + 
                                      ", " + std::to_string(hidden_size) + "] -> [1, " + 
                                      std::to_string(actual_seq_len) + ", " + std::to_string(hidden_size) + "]");
                        }
                        
                        return trimmed_hidden;
                    }
                }
            }
        }
        
        return hidden_state;
    } catch (const std::exception&) {
        log_debug("No hidden features tensor found");
        return ov::Tensor{};
    }
}

uint64_t Eagle3InferWrapper::execute_inference(const ov::Tensor& input_ids) {
    auto start = std::chrono::steady_clock::now();
    m_request.infer();
    auto end = std::chrono::steady_clock::now();
    
    // Update processed tokens to current sequence length
    m_processed_tokens = m_tokens.size();
    
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

void Eagle3InferWrapper::update_performance_metrics(uint64_t inference_time_us, std::size_t tokens_count) {
    m_metrics.total_inference_time_us += inference_time_us;
    m_metrics.last_inference_time_us = inference_time_us;
    m_metrics.total_inferences++;
    m_metrics.total_tokens_processed += tokens_count;
    
    // Update raw metrics for compatibility
    m_raw_perf_metrics.m_durations.emplace_back(static_cast<float>(inference_time_us));
    m_raw_perf_metrics.m_inference_durations[0] += ov::genai::MicroSeconds(static_cast<float>(inference_time_us));
    m_raw_perf_metrics.m_batch_sizes.emplace_back(tokens_count);
}

void Eagle3InferWrapper::log_debug(const std::string& message) const {
    if (m_verbose) {
        std::cout << "[EAGLE3-WRAPPER] " << message << std::endl;
    }
}

void Eagle3InferWrapper::log_tensor_info(const std::string& name, const ov::Tensor& tensor) const {
    if (!m_verbose) return;
    
    auto shape = tensor.get_shape();
    std::cout << "[EAGLE3-WRAPPER] " << name << " shape: [";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i + 1 < shape.size()) std::cout << ", ";
    }
    std::cout << "], type: " << tensor.get_element_type() << std::endl;
}

void Eagle3InferWrapper::log_tensor_content(const std::string& name, const ov::Tensor& tensor, std::size_t max_elements) const {
    if (!m_verbose || !tensor) return;
    
    auto shape = tensor.get_shape();
    std::size_t total_elements = tensor.get_size();
    
    // For input_ids, position_ids, attention_mask, hidden_states, and internal_hidden_states, always show all elements, ignore max_elements
    bool show_all = (name == "input_ids" || name == "position_ids" || name == "attention_mask");
    std::size_t elements_to_show = show_all ? total_elements : std::min(max_elements, total_elements);
    
    std::cout << "[EAGLE3-WRAPPER] " << name << " content (" << elements_to_show << " elements): ";
    
    if (tensor.get_element_type() == ov::element::i64) {
        const int64_t* data = tensor.data<const int64_t>();
        for (std::size_t i = 0; i < elements_to_show; ++i) {
            std::cout << data[i];
            if (i + 1 < elements_to_show) std::cout << ", ";
        }
    } else if (tensor.get_element_type() == ov::element::f32) {
        const float* data = tensor.data<const float>();
        for (std::size_t i = 0; i < elements_to_show; ++i) {
            std::cout << std::fixed << std::setprecision(4) << data[i];
            if (i + 1 < elements_to_show) std::cout << ", ";
        }
    } else if (tensor.get_element_type() == ov::element::i32) {
        const int32_t* data = tensor.data<const int32_t>();
        for (std::size_t i = 0; i < elements_to_show; ++i) {
            std::cout << data[i];
            if (i + 1 < elements_to_show) std::cout << ", ";
        }
    }
    
    if (!show_all && elements_to_show < total_elements) {
        std::cout << " ... (+" << (total_elements - elements_to_show) << " more)";
    }
    std::cout << std::endl;
}

void Eagle3InferWrapper::log_model_inputs(const ov::Tensor& input_ids, const ov::Tensor& attention_mask, const ov::Tensor& position_ids) const {
    if (!m_verbose) return;
    
    std::cout << "[EAGLE3-WRAPPER] ========== MODEL INPUTS ==========" << std::endl;
    log_tensor_info("input_ids", input_ids);
    log_tensor_content("input_ids", input_ids, 0);  // 0 means show all for input_ids
    
    log_tensor_info("attention_mask", attention_mask);
    log_tensor_content("attention_mask", attention_mask, 0);
    
    log_tensor_info("position_ids", position_ids);
    log_tensor_content("position_ids", position_ids, 0);  // 0 means show all for position_ids
    std::cout << "[EAGLE3-WRAPPER] =================================" << std::endl;
}

void Eagle3InferWrapper::log_model_outputs(const ov::Tensor& logits, const ov::Tensor& hidden_features) const {
    if (!m_verbose) return;
    
    std::cout << "[EAGLE3-WRAPPER] ========== MODEL OUTPUTS =========" << std::endl;
    log_tensor_info("logits", logits);
    if (logits && logits.get_size() > 0) {
        // For logits, show top 10 values for each position
        auto logits_shape = logits.get_shape();
        if (logits_shape.size() == 3) {
            std::size_t seq_len = logits_shape[1];
            std::size_t vocab_size = logits_shape[2];
            
            // If seq_len > 5, only show last 5 positions
            std::size_t start_pos = (seq_len > 5) ? (seq_len - 5) : 0;
            std::size_t positions_to_show = seq_len - start_pos;
            
            if (start_pos > 0) {
                std::cout << "[EAGLE3-WRAPPER] Showing only last " << positions_to_show 
                          << " positions (total seq_len: " << seq_len << ")" << std::endl;
            }
            
            // Show top 10 logit values for each position (only last 5 if seq_len > 5)
            for (std::size_t pos = start_pos; pos < seq_len; ++pos) {
                const float* logits_data = logits.data<const float>() + pos * vocab_size;
                std::vector<std::pair<float, int64_t>> top_logits;
                
                for (std::size_t i = 0; i < vocab_size; ++i) {
                    top_logits.emplace_back(logits_data[i], static_cast<int64_t>(i));
                }
                
                std::sort(top_logits.begin(), top_logits.end(), std::greater<std::pair<float, int64_t>>());
                
                std::cout << "[EAGLE3-WRAPPER] Position " << pos << " - Top 10 logits: ";
                for (std::size_t i = 0; i < std::min<std::size_t>(10, top_logits.size()); ++i) {
                    std::cout << "token_" << top_logits[i].second << ":" << std::fixed << std::setprecision(3) << top_logits[i].first;
                    if (i + 1 < std::min<std::size_t>(10, top_logits.size())) std::cout << ", ";
                }
                std::cout << std::endl;
                
                // Show first 20 raw logit values for each position
                std::cout << "[EAGLE3-WRAPPER] Position " << pos << " - First 20 raw logits: ";
                for (std::size_t i = 0; i < std::min<std::size_t>(20, vocab_size); ++i) {
                    std::cout << std::fixed << std::setprecision(4) << logits_data[i];
                    if (i + 1 < std::min<std::size_t>(20, vocab_size)) std::cout << ", ";
                }
                std::cout << std::endl;
            }
        }
    }
    
    if (hidden_features && hidden_features.get_size() > 0) {
        log_tensor_info("hidden_features", hidden_features);
        // Show first few elements of the last hidden state
        auto hidden_shape = hidden_features.get_shape();
        if (hidden_shape.size() == 3 && hidden_shape[1] > 0) {
            std::size_t seq_len = hidden_shape[1];
            std::size_t hidden_dim = hidden_shape[2];
            const float* hidden_data = hidden_features.data<const float>() + (seq_len - 1) * hidden_dim;
            
            std::cout << "[EAGLE3-WRAPPER] Last hidden state (first 10 dims): ";
            for (std::size_t i = 0; i < std::min<std::size_t>(10, hidden_dim); ++i) {
                std::cout << std::fixed << std::setprecision(4) << hidden_data[i];
                if (i + 1 < std::min<std::size_t>(10, hidden_dim)) std::cout << ", ";
            }
            if (hidden_dim > 10) std::cout << " ... (+" << (hidden_dim - 10) << " more)";
            std::cout << std::endl;
        }
    }
    std::cout << "[EAGLE3-WRAPPER] =================================" << std::endl;
}

//==================================================================================================
// StatefulEagle3LLMPipeline Implementation  
//==================================================================================================

StatefulEagle3LLMPipeline::StatefulEagle3LLMPipeline(const ov::genai::ModelDesc& main_model_desc,
                                                     const ov::genai::ModelDesc& draft_model_desc,
                                                     const std::vector<int>& hidden_layers_to_abstract)
    : LLMPipelineImplBase(main_model_desc.tokenizer, main_model_desc.generation_config)
    , m_hidden_layers_to_abstract(hidden_layers_to_abstract) {
    
    log_info("Initializing Eagle3 pipeline with main device: " + main_model_desc.device + 
             ", draft device: " + draft_model_desc.device);
    
    // Apply model transformations
    auto main_model = main_model_desc.model;
    auto draft_model = draft_model_desc.model;
    
    // Use main model's tokenizer (draft model shares the same vocabulary)
    m_tokenizer = main_model_desc.tokenizer;
    
    // Apply Eagle3 model transformations
    // For eagle model, we need to obtain hidden layer state as extra output
    // Apply transformations needed to run eagle model:
    // 1. Share embedding weights between main and draft models
    // 2. Target model: hidden state extraction
    // 3. Draft model: hidden state import and extraction
    
    // Validate that hidden_layers were provided
    if (m_hidden_layers_to_abstract.empty()) {
        throw std::runtime_error(
            "Eagle3 requires explicit hidden_layers_list configuration. "
            "Please provide hidden_layers_list in draft model properties, or ensure it can be "
            "auto-deduced from the model's config.json. "
            "Example: config[\"hidden_layers_list\"] = std::vector<int>{2, 16, 29};"
        );
    }
    
    // Step 1: Share embedding weights to reduce memory usage
    ov::genai::share_embedding_weights(main_model, draft_model);
    log_debug("Shared embedding weights between main and draft models");

    // Step 2: Extract draft-to-target mapping from draft model (must be done before other transformations)
    set_draft_target_mapping(draft_model);
    
    // Step 3: Remove the d2t Result node after extraction
    ov::genai::remove_d2t_result_node(draft_model);
    log_debug("Removed d2t Result node from draft model");
    
    // Step 4: Extract hidden states from main model (target model)
    ov::genai::extract_hidden_state_generic(main_model, m_hidden_layers_to_abstract);
    log_debug("Extracted hidden states from main model layers");
    
    // Step 5: Extract hidden states and add inputs for draft model
    ov::genai::extract_hidden_state_generic(draft_model, {-1});
    log_debug("Extracted hidden states from draft model and added hidden state inputs");
    // ov::serialize(main_model, "main_model_sgl.xml");
    // ov::serialize(draft_model, "draft_model_sgl.xml");
    log_debug("Eagle3 model transformations completed for both models");

    auto draft_desc = draft_model_desc;
    if (draft_desc.device == "NPU") {
        draft_desc.properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = MAX_CANDIDATES + 1;
        draft_desc.properties["NPUW_DEVICES"] = "CPU";
        draft_desc.properties["NPUW_ONLINE_PIPELINE"] = "NONE";
    }

    m_draft_model = std::make_unique<Eagle3InferWrapper>(draft_desc);
    
    auto main_desc = main_model_desc;
    if (main_desc.device == "NPU") {
        main_model->set_rt_info("true", "eagle3_mode");
        main_desc.properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = MAX_CANDIDATES + 1;
        main_desc.properties["NPUW_DEVICES"] = "CPU";
    }
    
    m_main_model = std::make_unique<Eagle3InferWrapper>(main_desc);
    
    // Initialize performance metrics
    m_sd_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_sd_perf_metrics.raw_metrics.m_inference_durations = {{ov::genai::MicroSeconds(0.0f)}};
    
    log_info("Eagle3 pipeline initialization completed");
}

StatefulEagle3LLMPipeline::~StatefulEagle3LLMPipeline() {
    m_main_model->release_memory();
    m_draft_model->release_memory();
    log_debug("Eagle3 pipeline destructor called");
}

void StatefulEagle3LLMPipeline::set_draft_target_mapping(const std::shared_ptr<ov::Model>& draft_model) {
    if (!draft_model) {
        throw std::runtime_error("Draft model is null");
    }
    
    // Extract d2t mapping from draft model
    auto d2t_tensor = ov::genai::extract_d2t_mapping_table(draft_model);
    if (!d2t_tensor) {
        throw std::runtime_error("Draft-to-target mapping table not found in draft model. "
                                 "Eagle3 requires d2t mapping for token conversion.");
    }
    
    // Convert ov::op::v0::Constant to ov::Tensor
    auto d2t_shape = d2t_tensor->get_shape();
    ov::Tensor d2t_mapping(d2t_tensor->get_element_type(), d2t_shape);
    
    std::size_t num_elements = d2t_tensor->get_byte_size() / d2t_tensor->get_element_type().size();
    std::memcpy(d2t_mapping.data(), d2t_tensor->get_data_ptr(), d2t_tensor->get_byte_size());
    
    log_info("Extracted and converting d2t mapping from draft model (" + std::to_string(num_elements) + " entries)");
    
    // Validate the tensor
    if (d2t_mapping.get_element_type() != ov::element::i64) {
        throw std::runtime_error("Draft-to-target mapping must be int64 tensor");
    }
    
    m_draft_target_mapping = d2t_mapping;
    log_info("Draft-to-target mapping configured with " + std::to_string(d2t_mapping.get_size()) + " entries");
}

void StatefulEagle3LLMPipeline::set_verbose(bool verbose) {
    m_verbose = verbose;
    if (m_main_model) m_main_model->set_verbose(verbose);
    if (m_draft_model) m_draft_model->set_verbose(verbose);
    log_debug("Verbosity set to " + std::string(verbose ? "enabled" : "disabled"));
}

DecodedResults StatefulEagle3LLMPipeline::generate(
    StringInputs inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {
    ManualTimer generate_timer("StatefulEagle3LLMPipeline::generate()");
    generate_timer.start();
    ManualTimer encode_timer("Encode");
    encode_timer.start();

    std::string prompt = std::visit(overloaded{
        [](const std::string& prompt_str) {
            return prompt_str;
        },
        [](std::vector<std::string>& prompts) {
            OPENVINO_ASSERT(prompts.size() == 1u, "Currently only batch size=1 is supported");
            return prompts.front();
        }
    }, inputs);

    GenerationConfig config = generation_config.has_value() ? *generation_config : m_generation_config;

    ov::genai::TokenizedInputs tokenized_input;
    if (m_is_chat_active) {
        m_chat_history.push_back({{"role", "user"}, {"content", prompt}});
        constexpr bool add_generation_prompt = true;
        prompt = m_tokenizer.apply_chat_template(m_chat_history, add_generation_prompt);
        // for chat ov::genai::add_special_tokens(false) is aligned with stateful pipeline and HF
        tokenized_input = m_tokenizer.encode(prompt, ov::genai::add_special_tokens(false));
    } else {
        if (config.apply_chat_template && !m_tokenizer.get_chat_template().empty()) {
            ChatHistory history({{{"role", "user"}, {"content", prompt}}});
            constexpr bool add_generation_prompt = true;
            auto templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
            tokenized_input = m_tokenizer.encode(templated_prompt, ov::genai::add_special_tokens(false));
        } else {
            // in case when chat_template was not found in tokenizer_config.json or set
            tokenized_input = m_tokenizer.encode(prompt, ov::genai::add_special_tokens(true));
        }
    }

    encode_timer.end();
    auto encoded_results = generate(tokenized_input, config, streamer);

    ManualTimer decode_timer("Decode");
    decode_timer.start();
    DecodedResults decoded_results = {m_tokenizer.decode(encoded_results.tokens), encoded_results.scores};
    decode_timer.end();

    if (m_is_chat_active) {
        auto answer = decoded_results.texts[0];
        if (m_streaming_was_cancelled)
            // If generation process was cancelled by user, let's rollback to previous state of history
            m_chat_history.pop_back();
        else
            m_chat_history.push_back({{"role", "assistant"}, {"content", answer}});
    }

    // Update perf metrics
    decoded_results.perf_metrics = encoded_results.perf_metrics;
    decoded_results.extended_perf_metrics = encoded_results.extended_perf_metrics;
    generate_timer.end();
    auto& raw_counters = decoded_results.perf_metrics.raw_metrics;
    raw_counters.generate_durations.clear();
    raw_counters.generate_durations.emplace_back(generate_timer.get_duration_microsec());
    raw_counters.tokenization_durations.emplace_back(encode_timer.get_duration_microsec());
    raw_counters.detokenization_durations.emplace_back(decode_timer.get_duration_microsec());
    decoded_results.perf_metrics.m_evaluated = false;
    decoded_results.perf_metrics.evaluate_statistics(generate_timer.get_start_time());
    
    return decoded_results;
}

DecodedResults StatefulEagle3LLMPipeline::generate(
    const ChatHistory& history,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {
    ManualTimer generate_timer("StatefulEagle3LLMPipeline::generate()");
    generate_timer.start();
    ManualTimer encode_timer("Encode");
    encode_timer.start();

    GenerationConfig config = generation_config.has_value() ? *generation_config : m_generation_config;

    OPENVINO_ASSERT(config.apply_chat_template, "Chat template must be applied when using ChatHistory in generate method.");
    OPENVINO_ASSERT(!m_tokenizer.get_chat_template().empty(), "Chat template must not be empty when using ChatHistory in generate method.");
    OPENVINO_ASSERT(!history.empty(), "Chat history must not be empty when using ChatHistory in generate method.");

    constexpr bool add_generation_prompt = true;
    auto templated_chat_history = m_tokenizer.apply_chat_template(history, add_generation_prompt);
    // for chat ov::genai::add_special_tokens(false) is aligned with stateful pipeline and HF
    auto tokenized_inputs = m_tokenizer.encode(templated_chat_history, ov::genai::add_special_tokens(false));
    encode_timer.end();
    auto encoded_results = generate(tokenized_inputs, config, streamer);

    ManualTimer decode_timer("Decode");
    decode_timer.start();
    DecodedResults decoded_results = {m_tokenizer.decode(encoded_results.tokens), encoded_results.scores};
    decode_timer.end();
    
    // Update perf metrics
    decoded_results.perf_metrics = encoded_results.perf_metrics;
    decoded_results.extended_perf_metrics = encoded_results.extended_perf_metrics;
    auto& raw_counters = decoded_results.perf_metrics.raw_metrics;
    generate_timer.end();
    raw_counters.generate_durations.clear();
    raw_counters.generate_durations.emplace_back(generate_timer.get_duration_microsec());
    raw_counters.tokenization_durations.emplace_back(encode_timer.get_duration_microsec());
    raw_counters.detokenization_durations.emplace_back(decode_timer.get_duration_microsec());
    decoded_results.perf_metrics.m_evaluated = false;
    decoded_results.perf_metrics.evaluate_statistics(generate_timer.get_start_time());

    return decoded_results;
}

EncodedResults StatefulEagle3LLMPipeline::generate(const EncodedInputs& inputs, 
                                                   OptionalGenerationConfig generation_config, 
                                                   StreamerVariant streamer) {
    ManualTimer generate_timer("StatefulEagle3LLMPipeline::generate");
    generate_timer.start();
    
    auto config = generation_config.has_value() ? *generation_config : m_generation_config;
    m_perf_summary.reset();
    
    log_info("Starting Eagle3 generation with max_new_tokens=" + std::to_string(config.max_new_tokens));
    
    // Extract input tensors
    ov::Tensor input_ids, attention_mask;
    if (auto* tensor_input = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *tensor_input;
        attention_mask = ov::genai::utils::init_attention_mask(input_ids);
    } else if (auto* tokenized_input = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = tokenized_input->input_ids;
        attention_mask = tokenized_input->attention_mask;
    }
    
    // Debug override - overwrite input_ids and attention_mask
    // {
    //     static const int64_t debug_tokens[] = {151644, 872, 198, 40451, 752, 2494, 911, 6864, 30, 151645};
    //     constexpr size_t token_count = sizeof(debug_tokens) / sizeof(debug_tokens[0]);
        
    //     input_ids = ov::Tensor(ov::element::i64, {1, token_count});
    //     std::copy(debug_tokens, debug_tokens + token_count, input_ids.data<int64_t>());
        
    //     attention_mask = ov::Tensor(ov::element::i64, {1, token_count});
    //     std::fill_n(attention_mask.data<int64_t>(), token_count, 1);
    // }
    
    auto prompt_shape = input_ids.get_shape();
    if (prompt_shape[0] != 1) {
        throw std::runtime_error("Only batch size 1 is supported");
    }
    
    std::size_t prompt_len = prompt_shape[1];
    m_perf_summary.prompt_tokens = prompt_len;
    
    // Initialize position IDs
    ov::Tensor position_ids{ov::element::i64, input_ids.get_shape()};
    utils::initialize_position_ids(position_ids, attention_mask);
    
    // Reset model states and initialize sequences
    m_main_model->reset_state();
    m_draft_model->reset_state();
    
    // Initialize main model with full sequence
    m_main_model->initialize_sequence(input_ids, position_ids);
    
    // Initialize draft model with sequence starting from second token (Eagle3 specific)
    std::size_t seq_len = prompt_shape[1];
    if (seq_len > 1) {
        ov::Tensor draft_input_ids(ov::element::i64, {1, seq_len - 1});
        ov::Tensor draft_position_ids(ov::element::i64, {1, seq_len - 1});
        
        const int64_t* ids_data = input_ids.data<const int64_t>();
        const int64_t* pos_data = position_ids.data<const int64_t>();
        
        std::copy_n(ids_data + 1, seq_len - 1, draft_input_ids.data<int64_t>());
        std::copy_n(pos_data, seq_len - 1, draft_position_ids.data<int64_t>());
        
        m_draft_model->initialize_sequence(draft_input_ids, draft_position_ids);
    } else {
        // Single token case - draft gets empty sequence initially
        ov::Tensor empty_ids(ov::element::i64, {1, 0});
        ov::Tensor empty_pos(ov::element::i64, {1, 0});
        m_draft_model->initialize_sequence(empty_ids, empty_pos);
    }
    
    // Initial main model inference
    log_generation_step("Initial Main Model Inference", 0);
    auto initial_start = std::chrono::steady_clock::now();
    auto main_logits = m_main_model->infer_target_model(input_ids, attention_mask, position_ids);
    auto initial_token = std::get<int64_t>(m_main_model->sample_tokens(main_logits, 1));
    auto initial_end = std::chrono::steady_clock::now();
    
    m_perf_summary.main_inference_time_us += 
        std::chrono::duration_cast<std::chrono::microseconds>(initial_end - initial_start).count();
    
    // Get initial hidden features and append first generated token
    auto main_hidden_features = m_main_model->get_hidden_features();
    m_main_model->append_tokens({initial_token});
    m_draft_model->append_tokens({initial_token});
    
    log_debug("Initial token generated: " + std::to_string(initial_token));
    log_sequence_state("after initial token generation");
    
    // Main generation loop using speculative decoding
    std::size_t max_new_tokens = config.max_new_tokens;
    std::size_t generated_tokens = 1; // Count initial token
    // Initial window size should be the draft model sequence length, but subsequent will be 1
    std::size_t window_size = m_draft_model->get_sequence_length();
    auto hidden_window = main_hidden_features;
    bool eos_reached = false;
    
    auto generation_start = std::chrono::steady_clock::now();
    
    while (!eos_reached && generated_tokens < max_new_tokens && 
           m_main_model->get_sequence_length() < prompt_len + max_new_tokens) {
        
        log_generation_step("Speculative Decoding Iteration", generated_tokens);
        log_sequence_state("iteration start");
        
        auto iteration_start = std::chrono::steady_clock::now();
        auto result = run_speculative_iteration(hidden_window, window_size, static_cast<int64_t>(config.eos_token_id));
        auto iteration_end = std::chrono::steady_clock::now();
        
        // Update metrics
        if (result.new_token == static_cast<int64_t>(config.eos_token_id) || result.eos_reached) {
            eos_reached = true;
            log_debug("EOS reached - terminating generation");
        }
        
        if (result.new_token != -1) {
            generated_tokens++;
            m_perf_summary.accepted_tokens += result.accepted_tokens_count;
            log_debug("Generated token " + std::to_string(generated_tokens) + ": " + std::to_string(result.new_token) + 
                      ", accepted " + std::to_string(result.accepted_tokens_count) + " draft tokens");
        }
        
        // Prepare for next iteration
        window_size = result.next_window_size > 0 ? result.next_window_size : 
                      std::min<std::size_t>(1, m_main_model->get_sequence_length());
        hidden_window = result.next_hidden_window ? result.next_hidden_window : m_main_model->get_hidden_features();
        
        log_debug("Next iteration: window_size=" + std::to_string(window_size) + 
                  ", hidden_window_size=" + (hidden_window ? std::to_string(hidden_window.get_size()) : "0"));
        
        // Safety check to prevent infinite loops
        if (result.next_window_size == 0 && result.new_token == -1) {
            log_debug("No progress made, terminating generation");
            break;
        }
        
        m_perf_summary.validation_rounds++;
        log_sequence_state("iteration end");
    }
    
    auto generation_end = std::chrono::steady_clock::now();
    m_perf_summary.total_generation_time_us = 
        std::chrono::duration_cast<std::chrono::microseconds>(generation_end - generation_start).count();
    m_perf_summary.generated_tokens = generated_tokens;
    
    // Log performance summary
    if (m_verbose) {
        log_performance_summary();
    }
    
    // Convert all main model tokens to text and display
    const auto& all_tokens = m_main_model->get_tokens();
    if (!all_tokens.empty()) {
        try {
            std::string decoded_text = m_tokenizer.decode(all_tokens);
            std::cout << "[EAGLE3-FINAL] All generated tokens decoded: \"" << decoded_text << "\"" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "[EAGLE3-FINAL] Failed to decode all tokens: " << e.what() << std::endl;
        }
    }
    
    // Prepare results
    EncodedResults results;
    results.tokens = {m_main_model->get_tokens()};
    results.scores.resize(1);
    results.scores[0] = 0.0f; // Greedy decoding, no scores
    
    // Update performance metrics following the stateful pipeline pattern
    generate_timer.end();
    
    m_sd_perf_metrics.num_input_tokens = prompt_len;
    m_sd_perf_metrics.load_time = this->m_load_time_ms;
    m_sd_perf_metrics.raw_metrics.generate_durations.clear();
    m_sd_perf_metrics.raw_metrics.generate_durations.emplace_back(
        generate_timer.get_duration_microsec());
    
    // Update main and draft model metrics
    m_sd_perf_metrics.main_model_metrics.raw_metrics = m_main_model->get_raw_perf_metrics();
    m_sd_perf_metrics.draft_model_metrics.raw_metrics = m_draft_model->get_raw_perf_metrics();
    
    // Update speculative decoding metrics
    m_sd_metrics.draft_duration = m_perf_summary.draft_inference_time_us / 1e6f;
    m_sd_metrics.main_duration = m_perf_summary.main_inference_time_us / 1e6f;
    
    if (generated_tokens > 0) {
        float acceptance_rate = m_perf_summary.get_acceptance_rate() * 100.0f;
        m_sd_metrics.update_acceptance_rate(0, acceptance_rate);
        m_sd_metrics.update_draft_accepted_tokens(0, m_perf_summary.accepted_tokens);
        m_sd_metrics.update_draft_generated_len(0, m_perf_summary.draft_iterations);
        m_sd_metrics.update_generated_len(generated_tokens);
    }
    
    // Evaluate statistics
    m_sd_perf_metrics.evaluate_statistics(generate_timer.get_start_time());
    
    results.perf_metrics = m_sd_perf_metrics;
    results.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_sd_perf_metrics);
    
    return results;
}

StatefulEagle3LLMPipeline::SpeculativeResult 
StatefulEagle3LLMPipeline::run_speculative_iteration(const ov::Tensor& hidden_window, 
                                                     std::size_t window_size, 
                                                     int64_t eos_token_id) {
    SpeculativeResult result;
    
    log_debug("Starting speculative iteration with window_size=" + std::to_string(window_size));
    
    if (!hidden_window || hidden_window.get_size() == 0) {
        log_debug("Invalid hidden window provided");
        return result;
    }
    
    auto draft_start = std::chrono::steady_clock::now();
    
    // Step 1: Initial draft inference using main hidden features
    ov::Tensor draft_input_ids, draft_attention_mask, draft_position_ids;
    int64_t begin_idx = -static_cast<int64_t>(window_size);
    m_draft_model->build_model_inputs(begin_idx, window_size, 
                                     draft_input_ids, draft_attention_mask, draft_position_ids, true, true);
    
    auto draft_logits = m_draft_model->infer_draft_model(draft_input_ids, draft_attention_mask, draft_position_ids,
                                                        hidden_window, ov::Tensor{});
    
    int64_t first_draft_token = std::get<int64_t>(m_draft_model->sample_tokens(draft_logits, 1));
    first_draft_token = map_draft_token(first_draft_token);
    
    // Record the sequence length before draft generation for rollback
    std::size_t pre_draft_main_len = m_main_model->get_sequence_length();
    std::size_t pre_draft_draft_len = m_draft_model->get_sequence_length();
    
    // Store all draft tokens temporarily (including first one)
    std::vector<int64_t> draft_candidates;
    draft_candidates.push_back(first_draft_token);
    
    // Append first draft token and get its hidden state for subsequent iterations
    m_main_model->append_tokens({first_draft_token});
    m_draft_model->append_tokens({first_draft_token});
    auto draft_hidden = extract_last_hidden_state(m_draft_model->get_hidden_features());
    
    // Step 2: Additional draft iterations  
    for (std::size_t i = 0; i < DEFAULT_DRAFT_ITERATIONS; ++i) {
        m_draft_model->build_model_inputs(-1, 1, 
                                         draft_input_ids, draft_attention_mask, draft_position_ids, false, true);
        
        auto more_logits = m_draft_model->infer_draft_model(draft_input_ids, draft_attention_mask, draft_position_ids,
                                                           ov::Tensor{}, draft_hidden);
        
        int64_t draft_token = std::get<int64_t>(m_draft_model->sample_tokens(more_logits, 1));
        draft_token = map_draft_token(draft_token);
        
        draft_candidates.push_back(draft_token);
        m_main_model->append_tokens({draft_token});
        m_draft_model->append_tokens({draft_token});
        draft_hidden = extract_last_hidden_state(m_draft_model->get_hidden_features());
        
        m_perf_summary.draft_iterations++;
    }
    
    auto draft_end = std::chrono::steady_clock::now();
    m_perf_summary.draft_inference_time_us += 
        std::chrono::duration_cast<std::chrono::microseconds>(draft_end - draft_start).count();
    
    // Step 3: Validation phase
    log_debug("Starting validation phase");
    auto validation_start = std::chrono::steady_clock::now();
    
    std::size_t current_target_len = m_main_model->get_sequence_length();
    std::size_t validation_window = std::min(DEFAULT_VALIDATION_WINDOW, current_target_len);
    
    if (validation_window == 0) {
        log_debug("Validation window too small, skipping validation");
        result.next_window_size = window_size;
        result.next_hidden_window = hidden_window;
        return result;
    }
    
    // Build validation inputs
    ov::Tensor val_input_ids, val_attention_mask, val_position_ids;
    int64_t val_begin_idx = static_cast<int64_t>(current_target_len - validation_window);
    m_main_model->build_model_inputs(val_begin_idx, validation_window,
                                    val_input_ids, val_attention_mask, val_position_ids, false, true);
    
    // Run validation inference
    auto val_logits = m_main_model->infer_target_model(val_input_ids, val_attention_mask, val_position_ids);
    auto sampled_tokens = std::get<std::vector<int64_t>>(m_main_model->sample_tokens(val_logits, validation_window));
    
    // Compare predictions with existing tokens using shift-based validation
    const int64_t* existing_tokens = val_input_ids.data<const int64_t>();
    std::size_t accepted_count = 0;
    int64_t future_token = -1;
    bool mismatch_found = false;
    int validation_position = -1;
    
    // Shift comparison: sampled[i] should match existing[i+1]
    if (validation_window >= 2) {
        for (std::size_t i = 0; i < validation_window - 1; ++i) {
            if (sampled_tokens[i] == existing_tokens[i + 1]) {
                accepted_count++;
            } else {
                // Mismatch found - truncate sequences and use predicted token
                std::size_t truncate_pos = (current_target_len - validation_window) + (i + 1);
                std::size_t tokens_to_trim = current_target_len - truncate_pos;
                
                m_main_model->truncate_sequence(truncate_pos);
                m_main_model->trim_kv_cache(tokens_to_trim);
                
                future_token = sampled_tokens[i];
                mismatch_found = true;
                validation_position = static_cast<int>(i);
                log_debug("Validation mismatch at position " + std::to_string(i) + 
                          ", truncated and trimmed " + std::to_string(tokens_to_trim) + " tokens");
                break;
            }
        }
    }
    
    if (!mismatch_found) {
        // All shifts matched or single token window
        if (validation_window == 1) {
            future_token = sampled_tokens[0];
            validation_position = 0;
        } else {
            future_token = sampled_tokens[validation_window - 1];
            validation_position = static_cast<int>(validation_window - 1);
        }
    }
    
    // Critical fix: Remove rejected draft tokens from sequences
    // Only keep accepted tokens + future token (if any)
    std::size_t draft_tokens_added = draft_candidates.size(); // Total draft tokens added
    std::size_t tokens_to_keep = accepted_count;
    std::size_t tokens_to_remove = draft_tokens_added - tokens_to_keep;
    
    if (tokens_to_remove > 0) {
        log_debug("Removing " + std::to_string(tokens_to_remove) + " rejected draft tokens");
        
        // Truncate both models to remove rejected tokens
        std::size_t new_main_len = pre_draft_main_len + tokens_to_keep;
        std::size_t new_draft_len = pre_draft_draft_len + tokens_to_keep;
        
        m_main_model->truncate_sequence(new_main_len);
        m_draft_model->truncate_sequence(new_draft_len);
        
        // Trim KV cache for both models to match the truncated sequences
        // m_main_model->trim_kv_cache(tokens_to_remove);
        
        // If we have a future token, append it back
        if (future_token != -1) {
            m_main_model->append_tokens({future_token});
            m_draft_model->append_tokens({future_token});
        }
        
        m_perf_summary.rejected_tokens += tokens_to_remove;
        log_debug("Accepted " + std::to_string(accepted_count) + " tokens, rejected " + 
                  std::to_string(tokens_to_remove) + " tokens, KV cache trimmed");
    } else {
        // All draft tokens were accepted
        if (future_token != -1) {
            m_main_model->append_tokens({future_token});
            m_draft_model->append_tokens({future_token});
        }
        log_debug("All " + std::to_string(draft_tokens_added) + " draft tokens accepted");
    }

    m_draft_model->trim_kv_cache(tokens_to_remove-1+tokens_to_keep);
    
    auto validation_end = std::chrono::steady_clock::now();
    m_perf_summary.validation_time_us += 
        std::chrono::duration_cast<std::chrono::microseconds>(validation_end - validation_start).count();
    
    // Build next hidden window - includes accepted draft tokens + future token's hidden states
    ov::Tensor next_hidden;
    auto current_hidden = m_main_model->get_hidden_features();
    if (current_hidden && current_hidden.get_size() > 0) {
        auto h_shape = current_hidden.get_shape();
        if (h_shape.size() == 3 && h_shape[0] == 1 && current_hidden.get_element_type() == ov::element::f32) {
            std::size_t seq_len = h_shape[1];
            std::size_t hidden_dim = h_shape[2];
            
            // Calculate window size: accepted tokens + future token
            std::size_t next_window_len = accepted_count + (future_token != -1 ? 1 : 0);
            
            if (next_window_len > 0 && seq_len >= next_window_len) {
                // Create hidden window containing accepted tokens + future token
                next_hidden = ov::Tensor(ov::element::f32, {1, next_window_len, hidden_dim});
                const float* src_data = current_hidden.data<const float>();
                float* dst_data = next_hidden.data<float>();
                
                // Copy hidden states from the last next_window_len positions
                std::size_t start_pos = 0;
                std::copy_n(src_data + start_pos * hidden_dim, next_window_len * hidden_dim, dst_data);
                
                if (m_verbose) {
                    log_debug("Built next hidden window: " + std::to_string(accepted_count) + 
                              " accepted tokens + " + (future_token != -1 ? "1" : "0") + 
                              " future token (total window size: " + std::to_string(next_window_len) + ")");
                }
            } else {
                // Fallback: use last available hidden state if calculation failed
                if (seq_len > 0) {
                    next_hidden = ov::Tensor(ov::element::f32, {1, 1, hidden_dim});
                    const float* src_data = current_hidden.data<const float>();
                    float* dst_data = next_hidden.data<float>();
                    
                    std::size_t last_pos = seq_len - 1;
                    std::copy_n(src_data + last_pos * hidden_dim, hidden_dim, dst_data);
                    
                    if (m_verbose) {
                        log_debug("Built fallback next hidden window from last position " + std::to_string(last_pos));
                    }
                }
            }
        }
    }
    
    // Check for EOS token
    if (future_token != -1 && future_token == eos_token_id) {
        result.eos_reached = true;
        log_debug("EOS token detected: " + std::to_string(future_token));
    }
    
    // Set result fields - window size = accepted draft tokens + future token
    result.accepted_tokens_count = accepted_count;
    result.next_window_size = accepted_count + (future_token != -1 ? 1 : 0);  // Accepted tokens + future token
    result.new_token = future_token;
    result.next_hidden_window = next_hidden;  // Hidden states for accepted tokens + future token
    
    log_debug("Speculative iteration completed - accepted: " + std::to_string(accepted_count) + 
              ", next_window_size: " + std::to_string(result.next_window_size) + 
              " (accepted tokens + future token)");
    
    return result;
}

int64_t StatefulEagle3LLMPipeline::map_draft_token(int64_t draft_token) const {
    if (!m_draft_target_mapping || m_draft_target_mapping.get_size() == 0) {
        return draft_token; // Identity mapping
    }
    
    std::size_t mapping_size = m_draft_target_mapping.get_size();
    if (draft_token < 0 || static_cast<std::size_t>(draft_token) >= mapping_size) {
        log_debug("Draft token " + std::to_string(draft_token) + " out of mapping range, using identity");
        return draft_token;
    }
    
    const int64_t* mapping_data = m_draft_target_mapping.data<const int64_t>();
    int64_t offset = mapping_data[draft_token];
    int64_t target_token = draft_token + offset;
    
    if (m_verbose) {
        log_debug("Mapped draft token " + std::to_string(draft_token) + " -> " + std::to_string(target_token));
    }
    
    return target_token;
}

std::vector<int64_t> StatefulEagle3LLMPipeline::map_draft_tokens(const std::vector<int64_t>& draft_tokens) const {
    if (!m_draft_target_mapping || m_draft_target_mapping.get_size() == 0) {
        return draft_tokens;
    }
    
    std::vector<int64_t> mapped_tokens;
    mapped_tokens.reserve(draft_tokens.size());
    
    for (int64_t token : draft_tokens) {
        mapped_tokens.push_back(map_draft_token(token));
    }
    
    return mapped_tokens;
}

void StatefulEagle3LLMPipeline::start_chat(const std::string& system_message) {
    m_is_chat_active = true;
    m_chat_history.clear();
    
    if (!system_message.empty()) {
        m_chat_history.push_back({{"role", "system"}, {"content", system_message}});
    }
    
    log_info("Chat session started");
}

void StatefulEagle3LLMPipeline::finish_chat() {
    m_is_chat_active = false;
    m_chat_history.clear();
    log_info("Chat session ended");
}

ov::genai::SpeculativeDecodingMetrics StatefulEagle3LLMPipeline::get_speculative_decoding_metrics() const {
    return m_sd_metrics;
}

void StatefulEagle3LLMPipeline::log_info(const std::string& message) const {
    std::cout << "[EAGLE3-PIPELINE] " << message << std::endl;
}

void StatefulEagle3LLMPipeline::log_debug(const std::string& message) const {
    if (m_verbose) {
        std::cout << "[EAGLE3-PIPELINE-DEBUG] " << message << std::endl;
    }
}

void StatefulEagle3LLMPipeline::log_generation_step(const std::string& step_name, std::size_t step_number) const {
    if (m_verbose) {
        std::cout << "\n[EAGLE3-PIPELINE] ========== STEP " << step_number << ": " << step_name << " =========="<< std::endl;
    }
}

void StatefulEagle3LLMPipeline::log_sequence_state(const std::string& context) const {
    if (!m_verbose) return;
    
    std::cout << "[EAGLE3-PIPELINE] Sequence state (" << context << "):" << std::endl;
    std::cout << "  Main model tokens: " << m_main_model->get_sequence_length() << " tokens" << std::endl;
    std::cout << "  Draft model tokens: " << m_draft_model->get_sequence_length() << " tokens" << std::endl;
    
    // Show all tokens and positions from main model
    const auto& main_tokens = m_main_model->get_tokens();
    const auto& main_positions = m_main_model->get_positions();
    if (!main_tokens.empty()) {
        std::cout << "  Main model tokens: ";
        for (std::size_t i = 0; i < main_tokens.size(); ++i) {
            std::cout << main_tokens[i];
            if (i + 1 < main_tokens.size()) std::cout << ", ";
        }
        std::cout << std::endl;
        
        if (!main_positions.empty()) {
            std::cout << "  Main model positions: ";
            for (std::size_t i = 0; i < main_positions.size(); ++i) {
                std::cout << main_positions[i];
                if (i + 1 < main_positions.size()) std::cout << ", ";
            }
            std::cout << std::endl;
        }
    }
    
    // Show all tokens and positions from draft model
    const auto& draft_tokens = m_draft_model->get_tokens();
    const auto& draft_positions = m_draft_model->get_positions();
    if (!draft_tokens.empty()) {
        std::cout << "  Draft model tokens: ";
        for (std::size_t i = 0; i < draft_tokens.size(); ++i) {
            std::cout << draft_tokens[i];
            if (i + 1 < draft_tokens.size()) std::cout << ", ";
        }
        std::cout << std::endl;
        
        if (!draft_positions.empty()) {
            std::cout << "  Draft model positions: ";
            for (std::size_t i = 0; i < draft_positions.size(); ++i) {
                std::cout << draft_positions[i];
                if (i + 1 < draft_positions.size()) std::cout << ", ";
            }
            std::cout << std::endl;
        }
    }
}

void StatefulEagle3LLMPipeline::log_performance_summary() const {
    const auto& perf = m_perf_summary;
    const auto& main_metrics = m_main_model->get_metrics();
    const auto& draft_metrics = m_draft_model->get_metrics();
    
    std::cout << "\n[EAGLE3-PERFORMANCE-SUMMARY]\n";
    std::cout << "==================================================\n";
    std::cout << "Generation Statistics:\n";
    std::cout << "  Prompt tokens: " << perf.prompt_tokens << "\n";
    std::cout << "  Generated tokens: " << perf.generated_tokens << "\n";
    std::cout << "  Total generation time: " << format_duration_us(perf.total_generation_time_us) << "\n";
    std::cout << "  Tokens per second: " << std::fixed << std::setprecision(2) 
              << (perf.total_generation_time_us > 0 ? 
                  (perf.generated_tokens * 1000000.0) / perf.total_generation_time_us : 0.0) << "\n";
    
    std::cout << "\nSpeculative Decoding Metrics:\n";
    std::cout << "  Draft iterations: " << perf.draft_iterations << "\n";
    std::cout << "  Validation rounds: " << perf.validation_rounds << "\n";
    std::cout << "  Accepted tokens: " << perf.accepted_tokens << "\n"; 
    std::cout << "  Rejected tokens: " << perf.rejected_tokens << "\n";
    std::cout << "  Acceptance rate: " << std::fixed << std::setprecision(1) 
              << (perf.get_acceptance_rate() * 100.0) << "%\n";
    std::cout << "  Estimated speedup: " << std::fixed << std::setprecision(2) 
              << perf.get_speedup() << "x\n";
    
    std::cout << "\nModel Performance:\n";
    std::cout << "  Main model avg inference: " << format_duration_us(static_cast<uint64_t>(main_metrics.get_average_inference_time_us())) << "\n";
    std::cout << "  Draft model avg inference: " << format_duration_us(static_cast<uint64_t>(draft_metrics.get_average_inference_time_us())) << "\n";
    std::cout << "  Main model total time: " << format_duration_us(perf.main_inference_time_us) << "\n";
    std::cout << "  Draft model total time: " << format_duration_us(perf.draft_inference_time_us) << "\n";
    std::cout << "  Validation time: " << format_duration_us(perf.validation_time_us) << "\n";
    std::cout << "==================================================\n" << std::endl;
}

ov::Tensor StatefulEagle3LLMPipeline::slice_hidden_features(const ov::Tensor& hidden_features, 
                                                           std::size_t start_pos, 
                                                           std::size_t length) const {
    if (!hidden_features || hidden_features.get_size() == 0) {
        return ov::Tensor{};
    }
    
    auto shape = hidden_features.get_shape();
    if (shape.size() != 3 || shape[0] != 1) {
        return hidden_features; // Return original if unexpected shape
    }
    
    std::size_t seq_len = shape[1];
    std::size_t hidden_dim = shape[2];
    
    if (start_pos >= seq_len || length == 0) {
        return ov::Tensor{};
    }
    
    std::size_t actual_length = std::min(length, seq_len - start_pos);
    ov::Tensor sliced(ov::element::f32, {1, actual_length, hidden_dim});
    
    if (hidden_features.get_element_type() == ov::element::f32) {
        const float* src_data = hidden_features.data<const float>() + start_pos * hidden_dim;
        std::copy_n(src_data, actual_length * hidden_dim, sliced.data<float>());
    }
    
    return sliced;
}

ov::Tensor StatefulEagle3LLMPipeline::combine_hidden_windows(const ov::Tensor& confirmed_hidden, 
                                                            const ov::Tensor& new_hidden) const {
    if (!confirmed_hidden || confirmed_hidden.get_size() == 0) {
        return new_hidden;
    }
    
    if (!new_hidden || new_hidden.get_size() == 0) {
        return confirmed_hidden;
    }
    
    auto conf_shape = confirmed_hidden.get_shape();
    auto new_shape = new_hidden.get_shape();
    
    if (conf_shape.size() != 3 || new_shape.size() != 3 || 
        conf_shape[0] != 1 || new_shape[0] != 1 ||
        conf_shape[2] != new_shape[2]) {
        return confirmed_hidden; // Return first tensor if shapes incompatible
    }
    
    std::size_t conf_seq_len = conf_shape[1];
    std::size_t new_seq_len = new_shape[1];
    std::size_t hidden_dim = conf_shape[2];
    std::size_t combined_seq_len = conf_seq_len + new_seq_len;
    
    ov::Tensor combined(ov::element::f32, {1, combined_seq_len, hidden_dim});
    
    if (confirmed_hidden.get_element_type() == ov::element::f32 && 
        new_hidden.get_element_type() == ov::element::f32) {
        
        float* dst_data = combined.data<float>();
        
        // Copy confirmed hidden states
        std::copy_n(confirmed_hidden.data<const float>(), conf_seq_len * hidden_dim, dst_data);
        
        // Copy new hidden states
        std::copy_n(new_hidden.data<const float>(), new_seq_len * hidden_dim, 
                   dst_data + conf_seq_len * hidden_dim);
    }
    
    return combined;
}

} // namespace genai
} // namespace ov
