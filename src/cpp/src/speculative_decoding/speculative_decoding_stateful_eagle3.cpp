// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speculative_decoding_stateful_eagle3.hpp"
#include "speculative_decoding_utils.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "continuous_batching/timer.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "speculative_decoding_eagle3_impl.hpp"
#include "utils.hpp"

namespace ov::genai {
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;
}  // namespace ov::genai

namespace {

// Stream generated tokens to output
ov::genai::StreamingStatus stream_generated_tokens(std::shared_ptr<ov::genai::StreamerBase> streamer_ptr,
                                                   const std::vector<int64_t>& tokens) {
    if (streamer_ptr) {
        return streamer_ptr->write(tokens);
    }
    return ov::genai::StreamingStatus{};
}

// Extract last token's hidden state using zero-copy ROI
// Input: [batch=1, seq_len, hidden_size] -> Output: [1, 1, hidden_size]
ov::Tensor extract_last_hidden_state(const ov::Tensor& hidden_features) {
    OPENVINO_ASSERT(hidden_features && hidden_features.get_size() > 0, "Hidden features tensor is empty");

    auto shape = hidden_features.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && shape[1] > 0,
                    "Expected shape [1, seq_len, hidden_size], got [",
                    shape.size() == 3
                        ? std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ", " + std::to_string(shape[2])
                        : "invalid",
                    "]");

    std::size_t seq_len = shape[1];
    std::size_t hidden_size = shape[2];

    return ov::Tensor(hidden_features, {0, seq_len - 1, 0}, {1, seq_len, hidden_size});
}

}  // anonymous namespace

namespace ov {
namespace genai {

//==================================================================================================
// Eagle3InferWrapperBase Implementation
//==================================================================================================

Eagle3InferWrapperBase::Eagle3InferWrapperBase(const ov::genai::ModelDesc& model_desc)
    : m_device(model_desc.device),
      m_properties(model_desc.properties),
      m_generation_config(model_desc.generation_config),
      m_tokenizer(model_desc.tokenizer) {
    log_debug("Initializing for device: " + m_device);

    m_kv_axes_pos = ov::genai::utils::get_kv_axes_pos(model_desc.model);

    if (m_device == "NPU") {
        auto [compiled, kv_desc] =
            ov::genai::utils::compile_decoder_for_npu(model_desc.model, m_properties, m_kv_axes_pos);
        m_max_prompt_len = kv_desc.max_prompt_len;
        m_kv_cache_capacity = kv_desc.max_prompt_len + kv_desc.min_response_len;
        m_request = compiled.create_infer_request();

        log_debug("NPU compiled: max_prompt=" + std::to_string(m_max_prompt_len) +
                  ", kv_capacity=" + std::to_string(m_kv_cache_capacity));
    } else {
        m_request = ov::genai::utils::singleton_core()
                        .compile_model(model_desc.model, m_device, m_properties)
                        .create_infer_request();
        log_debug(m_device + " compiled successfully");
    }

    // Initialize metrics
    m_raw_perf_metrics.m_inference_durations = {ov::genai::MicroSeconds(0.0f)};
    m_raw_perf_metrics.tokenization_durations = {ov::genai::MicroSeconds(0.0f)};
    m_raw_perf_metrics.detokenization_durations = {ov::genai::MicroSeconds(0.0f)};

    log_debug("Initialization completed");
}

void Eagle3InferWrapperBase::append_tokens(const std::vector<int64_t>& tokens) {
    if (tokens.empty())
        return;

    std::size_t old_size = m_tokens.size();
    m_tokens.insert(m_tokens.end(), tokens.begin(), tokens.end());

    for (std::size_t i = 0; i < tokens.size(); ++i) {
        m_positions.push_back(static_cast<int64_t>(old_size + i));
    }

    log_debug("Appended " + std::to_string(tokens.size()) + " tokens, total: " + std::to_string(m_tokens.size()));
}

void Eagle3InferWrapperBase::truncate_sequence(std::size_t size) {
    if (size < m_tokens.size()) {
        m_tokens.resize(size);
        m_positions.resize(size);
        log_debug("Truncated to: " + std::to_string(size));
    }
}

void Eagle3InferWrapperBase::trim_kv_cache(std::size_t tokens_to_remove) {
    if (tokens_to_remove == 0 || m_processed_tokens == 0) {
        return;
    }

    OPENVINO_ASSERT(tokens_to_remove < m_processed_tokens, "Cannot trim more tokens than processed");

    log_debug("Trimming KV cache: " + std::to_string(tokens_to_remove) + " tokens");

    // NPU handles KV trimming via position IDs
    if (m_device != "NPU") {
        ov::genai::utils::KVCacheState state;
        state.num_tokens_to_trim = tokens_to_remove;
        state.seq_length_axis = m_kv_axes_pos.seq_len;
        state.reset_mem_state = false;
        ov::genai::utils::trim_kv_cache(m_request, state, {});
    }

    m_processed_tokens -= tokens_to_remove;
    log_debug("KV trimmed, processed: " + std::to_string(m_processed_tokens));
}

void Eagle3InferWrapperBase::reset_state() {
    m_tokens.clear();
    m_positions.clear();
    m_processed_tokens = 0;
    m_last_sampled_token = -1;

    m_raw_perf_metrics.m_inference_durations = {ov::genai::MicroSeconds(0.0f)};
    m_raw_perf_metrics.m_durations.clear();
    m_raw_perf_metrics.m_batch_sizes.clear();

    log_debug("State reset");
}

void Eagle3InferWrapperBase::release_memory() {
    m_request.get_compiled_model().release_memory();
    log_debug("Memory released");
}

void Eagle3InferWrapperBase::build_model_inputs(std::size_t token_count,
                                                ov::Tensor& input_ids,
                                                ov::Tensor& attention_mask,
                                                ov::Tensor& position_ids) {
    OPENVINO_ASSERT(!m_tokens.empty() && token_count > 0, "Cannot build inputs: empty sequence or zero token count");
    OPENVINO_ASSERT(!m_positions.empty(), "Position IDs not initialized");

    const std::size_t seq_len = m_tokens.size();
    OPENVINO_ASSERT(token_count <= seq_len, "Requested ", token_count, " tokens but only ", seq_len, " available");

    const std::size_t start_pos = seq_len - token_count;

    input_ids = ov::Tensor(ov::element::i64, {1, token_count});
    position_ids = ov::Tensor(ov::element::i64, {1, token_count});

    int64_t* input_ids_ptr = input_ids.data<int64_t>();
    int64_t* position_ids_ptr = position_ids.data<int64_t>();

    std::memcpy(input_ids_ptr, m_tokens.data() + start_pos, token_count * sizeof(int64_t));
    std::memcpy(position_ids_ptr, m_positions.data() + start_pos, token_count * sizeof(int64_t));

    // Attention mask length = last_position_id + 1 (total KV cache size)
    const std::size_t attention_mask_len = static_cast<std::size_t>(position_ids_ptr[token_count - 1] + 1);

    attention_mask = ov::Tensor(ov::element::i64, {1, attention_mask_len});
    std::fill_n(attention_mask.data<int64_t>(), attention_mask_len, 1);
}

ov::Tensor Eagle3InferWrapperBase::create_hidden_state_placeholder(const ov::Shape& shape) const {
    ov::Tensor tensor(ov::element::f32, shape);
    std::fill_n(tensor.data<float>(), tensor.get_size(), 0.0f);
    return tensor;
}

// TODO: Use already provided Sampler API, that will support both greedy and
//       multinomial decoding.
std::variant<int64_t, std::vector<int64_t>> Eagle3InferWrapperBase::sample_tokens(const ov::Tensor& logits,
                                                                                  std::size_t count) {
    auto shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Invalid logits shape for sampling");

    std::size_t seq_len = shape[1];
    std::size_t vocab_size = shape[2];
    OPENVINO_ASSERT(count <= seq_len, "Requested count exceeds sequence length");

    log_debug("Sampling " + std::to_string(count) + " tokens from logits [" + std::to_string(shape[0]) + ", " +
              std::to_string(seq_len) + ", " + std::to_string(vocab_size) + "]");

    auto sample_single = [&](std::size_t pos) -> int64_t {
        const float* data = logits.data<const float>() + pos * vocab_size;
        auto max_it = std::max_element(data, data + vocab_size);
        int64_t token = static_cast<int64_t>(max_it - data);

        if (m_verbose) {
            log_debug("Pos " + std::to_string(pos) + ": token " + std::to_string(token) +
                      " (logit: " + std::to_string(*max_it) + ")");
        }
        return token;
    };

    if (count == 1) {
        int64_t token = sample_single(seq_len - 1);
        m_last_sampled_token = token;
        log_debug("Sampled: " + std::to_string(token));
        return token;
    }

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
            if (i + 1 < tokens.size())
                std::cout << ", ";
        }
        std::cout << std::endl;
    }

    return tokens;
}

ov::Tensor Eagle3InferWrapperBase::get_logits() const {
    return m_request.get_tensor("logits");
}

ov::Tensor Eagle3InferWrapperBase::get_hidden_features() const {
    auto hidden_state = m_request.get_tensor("last_hidden_state");
    auto shape = hidden_state.get_shape();

    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1,
                    "Expected [1, seq_len, hidden_size], got [",
                    shape.size() == 3
                        ? std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ", " + std::to_string(shape[2])
                        : "invalid",
                    "]");

    std::size_t output_seq_len = shape[1];
    std::size_t hidden_size = shape[2];

    auto input_ids = m_request.get_tensor("input_ids");
    std::size_t actual_seq_len = input_ids.get_shape()[1];

    if (output_seq_len == actual_seq_len) {
        return hidden_state;
    }

    OPENVINO_ASSERT(actual_seq_len <= output_seq_len,
                    "Actual length (",
                    actual_seq_len,
                    ") exceeds output (",
                    output_seq_len,
                    ")");

    log_debug("Trimming hidden: " + std::to_string(output_seq_len) + " -> " + std::to_string(actual_seq_len));

    // if NPU device is used, the output may be padded, trim it via ROI
    std::size_t start_offset = output_seq_len - actual_seq_len;
    ov::Tensor trimmed(hidden_state, {0, start_offset, 0}, {1, output_seq_len, hidden_size});

    log_debug("Trimmed via ROI: [1, " + std::to_string(actual_seq_len) + ", " + std::to_string(hidden_size) + "]");

    return trimmed;
}

uint64_t Eagle3InferWrapperBase::execute_inference() {
    auto start = std::chrono::steady_clock::now();
    m_request.infer();
    auto end = std::chrono::steady_clock::now();

    // Update processed tokens to current sequence length
    m_processed_tokens = m_tokens.size();

    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

void Eagle3InferWrapperBase::update_performance_metrics(uint64_t inference_time_us, std::size_t tokens_count) {
    m_raw_perf_metrics.m_durations.emplace_back(static_cast<float>(inference_time_us));
    m_raw_perf_metrics.m_inference_durations[0] += ov::genai::MicroSeconds(static_cast<float>(inference_time_us));
    m_raw_perf_metrics.m_batch_sizes.emplace_back(tokens_count);
}

void Eagle3InferWrapperBase::log_debug(const std::string& message) const {
    if (m_verbose) {
        std::cout << "[EAGLE3-WRAPPER] " << message << std::endl;
    }
}

void Eagle3InferWrapperBase::log_tensor_info(const std::string& name, const ov::Tensor& tensor) const {
    if (!m_verbose)
        return;

    auto shape = tensor.get_shape();
    std::cout << "[EAGLE3-WRAPPER] " << name << " shape: [";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i + 1 < shape.size())
            std::cout << ", ";
    }
    std::cout << "], type: " << tensor.get_element_type() << std::endl;
}

void Eagle3InferWrapperBase::log_tensor_content(const std::string& name,
                                                const ov::Tensor& tensor,
                                                std::size_t max_elements) const {
    if (!m_verbose || !tensor)
        return;

    auto shape = tensor.get_shape();
    std::size_t total_elements = tensor.get_size();

    // For input_ids, position_ids, attention_mask, always show all elements, ignore max_elements
    bool show_all = (name == "input_ids" || name == "position_ids" || name == "attention_mask");
    std::size_t elements_to_show = show_all ? total_elements : std::min(max_elements, total_elements);

    std::cout << "[EAGLE3-WRAPPER] " << name << " content (" << elements_to_show << " elements): ";

    if (tensor.get_element_type() == ov::element::i64) {
        const int64_t* data = tensor.data<const int64_t>();
        for (std::size_t i = 0; i < elements_to_show; ++i) {
            std::cout << data[i];
            if (i + 1 < elements_to_show)
                std::cout << ", ";
        }
    } else if (tensor.get_element_type() == ov::element::f32) {
        const float* data = tensor.data<const float>();
        for (std::size_t i = 0; i < elements_to_show; ++i) {
            std::cout << std::fixed << std::setprecision(4) << data[i];
            if (i + 1 < elements_to_show)
                std::cout << ", ";
        }
    } else if (tensor.get_element_type() == ov::element::i32) {
        const int32_t* data = tensor.data<const int32_t>();
        for (std::size_t i = 0; i < elements_to_show; ++i) {
            std::cout << data[i];
            if (i + 1 < elements_to_show)
                std::cout << ", ";
        }
    }

    if (!show_all && elements_to_show < total_elements) {
        std::cout << " ... (+" << (total_elements - elements_to_show) << " more)";
    }
    std::cout << std::endl;
}

void Eagle3InferWrapperBase::log_model_inputs(const ov::Tensor& input_ids,
                                              const ov::Tensor& attention_mask,
                                              const ov::Tensor& position_ids) const {
    if (!m_verbose)
        return;

    std::cout << "[EAGLE3-WRAPPER] ========== MODEL INPUTS ==========" << std::endl;
    log_tensor_info("input_ids", input_ids);
    log_tensor_content("input_ids", input_ids, 0);  // 0 means show all for input_ids

    log_tensor_info("attention_mask", attention_mask);
    log_tensor_content("attention_mask", attention_mask, 0);

    log_tensor_info("position_ids", position_ids);
    log_tensor_content("position_ids", position_ids, 0);  // 0 means show all for position_ids
    std::cout << "[EAGLE3-WRAPPER] =================================" << std::endl;
}

void Eagle3InferWrapperBase::log_model_outputs(const ov::Tensor& logits, const ov::Tensor& hidden_features) const {
    if (!m_verbose)
        return;

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
                    std::cout << "token_" << top_logits[i].second << ":" << std::fixed << std::setprecision(3)
                              << top_logits[i].first;
                    if (i + 1 < std::min<std::size_t>(10, top_logits.size()))
                        std::cout << ", ";
                }
                std::cout << std::endl;

                // Show first 20 raw logit values for each position
                std::cout << "[EAGLE3-WRAPPER] Position " << pos << " - First 20 raw logits: ";
                for (std::size_t i = 0; i < std::min<std::size_t>(20, vocab_size); ++i) {
                    std::cout << std::fixed << std::setprecision(4) << logits_data[i];
                    if (i + 1 < std::min<std::size_t>(20, vocab_size))
                        std::cout << ", ";
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
                if (i + 1 < std::min<std::size_t>(10, hidden_dim))
                    std::cout << ", ";
            }
            if (hidden_dim > 10)
                std::cout << " ... (+" << (hidden_dim - 10) << " more)";
            std::cout << std::endl;
        }
    }
    std::cout << "[EAGLE3-WRAPPER] =================================" << std::endl;
}

//==================================================================================================
// Eagle3TargetModelWrapper Implementation
//==================================================================================================

Eagle3TargetModelWrapper::Eagle3TargetModelWrapper(const ov::genai::ModelDesc& model_desc)
    : Eagle3InferWrapperBase(model_desc) {
    log_debug("Target model initialized");
}

void Eagle3TargetModelWrapper::initialize_sequence(const ov::Tensor& input_ids, const ov::Tensor& position_ids) {
    const int64_t* ids_data = input_ids.data<const int64_t>();
    std::size_t seq_len = input_ids.get_size();
    m_tokens.assign(ids_data, ids_data + seq_len);

    if (position_ids) {
        const int64_t* pos_data = position_ids.data<const int64_t>();
        m_positions.assign(pos_data, pos_data + position_ids.get_size());
    }

    log_debug("Sequence initialized: " + std::to_string(m_tokens.size()) + " tokens");
}

InferenceOutput Eagle3TargetModelWrapper::infer(const ov::Tensor& input_ids,
                                                const ov::Tensor& attention_mask,
                                                const ov::Tensor& position_ids) {
    log_debug("Target inference start");
    log_model_inputs(input_ids, attention_mask, position_ids);

    if (m_device == "NPU") {
        auto prompt_len = input_ids.get_shape()[1];
        OPENVINO_ASSERT(prompt_len <= m_max_prompt_len,
                        "NPU prompt length ",
                        prompt_len,
                        " exceeds max ",
                        m_max_prompt_len);
    }

    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);

    if (m_device != "NPU") {
        m_request.get_tensor("beam_idx").set_shape({BATCH_SIZE});
        m_request.get_tensor("beam_idx").data<int32_t>()[0] = 0;
    }

    uint64_t time_us = execute_inference();
    update_performance_metrics(time_us, input_ids.get_shape()[1]);

    InferenceOutput output;
    output.logits = get_logits();
    output.hidden_features = get_hidden_features();
    log_model_outputs(output.logits, output.hidden_features);

    log_debug("Target inference done: " + std::to_string(time_us / 1000.0) + "ms");
    return output;
}

//==================================================================================================
// Eagle3DraftModelWrapper Implementation
//==================================================================================================

Eagle3DraftModelWrapper::Eagle3DraftModelWrapper(const ov::genai::ModelDesc& model_desc)
    : Eagle3InferWrapperBase(model_desc) {
    log_debug("Draft model initialized");
}

void Eagle3DraftModelWrapper::initialize_sequence(const ov::Tensor& input_ids, const ov::Tensor& position_ids) {
    // Eagle3: draft uses tokens[1:] with positions [0, 1, ..., n-2]
    const int64_t* ids_data = input_ids.data<const int64_t>();
    std::size_t total_len = input_ids.get_size();

    OPENVINO_ASSERT(total_len >= 2, "Draft model requires at least 2 tokens, got ", total_len);

    std::size_t actual_len = total_len - 1;
    m_tokens.assign(ids_data + 1, ids_data + total_len);

    if (position_ids) {
        m_positions.resize(actual_len);
        std::iota(m_positions.begin(), m_positions.end(), 0);
    }

    log_debug("Sequence initialized: " + std::to_string(m_tokens.size()) + " tokens (skipped first, positions 0 to " +
              std::to_string(actual_len - 1) + ")");
}

InferenceOutput Eagle3DraftModelWrapper::infer(const ov::Tensor& input_ids,
                                               const ov::Tensor& attention_mask,
                                               const ov::Tensor& position_ids,
                                               const ov::Tensor& target_hidden_features,
                                               const ov::Tensor& internal_hidden_features) {
    log_debug("Draft inference start");
    log_model_inputs(input_ids, attention_mask, position_ids);

    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);

    // Eagle3 requires exactly one hidden state input
    bool has_target = target_hidden_features && target_hidden_features.get_size() > 0;
    bool has_internal = internal_hidden_features && internal_hidden_features.get_size() > 0;

    OPENVINO_ASSERT(has_target ^ has_internal, "Draft model requires exactly one of target/internal hidden features");

    ov::Tensor target_tensor, internal_tensor;

    if (has_target) {
        auto t_shape = target_hidden_features.get_shape();
        OPENVINO_ASSERT(t_shape.size() == 3 && t_shape.back() % 3 == 0, "Invalid target hidden features shape");

        target_tensor = target_hidden_features;
        auto internal_shape = t_shape;
        internal_shape.back() = t_shape.back() / 3;
        internal_tensor = create_hidden_state_placeholder(internal_shape);

        log_tensor_info("target_tensor", target_tensor);
        log_tensor_info("internal_placeholder", internal_tensor);
    } else {
        auto i_shape = internal_hidden_features.get_shape();
        OPENVINO_ASSERT(i_shape.size() == 3, "Invalid internal hidden features shape");

        internal_tensor = internal_hidden_features;
        auto target_shape = i_shape;
        target_shape.back() = i_shape.back() * 3;
        target_tensor = create_hidden_state_placeholder(target_shape);

        log_tensor_info("internal_tensor", internal_tensor);
        log_tensor_info("target_placeholder", target_tensor);
    }

    m_request.set_tensor("hidden_states", target_tensor);
    m_request.set_tensor("internal_hidden_states", internal_tensor);

    if (m_verbose) {
        std::cout << "[EAGLE3-WRAPPER] Hidden State Tensors:" << std::endl;
        log_tensor_info("hidden_states", target_tensor);
        log_tensor_content("hidden_states", target_tensor, 10);
        log_tensor_info("internal_hidden_states", internal_tensor);
        log_tensor_content("internal_hidden_states", internal_tensor, 10);
    }

    if (m_device != "NPU") {
        m_request.get_tensor("beam_idx").set_shape({BATCH_SIZE});
        m_request.get_tensor("beam_idx").data<int32_t>()[0] = 0;
    }

    uint64_t time_us = execute_inference();
    update_performance_metrics(time_us, input_ids.get_shape()[1]);

    InferenceOutput output;
    output.logits = get_logits();
    output.hidden_features = get_hidden_features();
    log_model_outputs(output.logits, output.hidden_features);

    log_debug("Draft inference done: " + std::to_string(time_us / 1000.0) + "ms");
    return output;
}

//==================================================================================================
// StatefulEagle3LLMPipeline Implementation
//==================================================================================================

StatefulEagle3LLMPipeline::StatefulEagle3LLMPipeline(const ov::genai::ModelDesc& main_model_desc,
                                                     const ov::genai::ModelDesc& draft_model_desc,
                                                     const std::vector<int>& hidden_layers_to_abstract)
    : LLMPipelineImplBase(main_model_desc.tokenizer, main_model_desc.generation_config),
      m_hidden_layers_to_abstract(hidden_layers_to_abstract) {
    ov::genai::speculative_decoding::ensure_num_assistant_tokens_is_set(m_generation_config);
    m_draft_iterations = m_generation_config.num_assistant_tokens;

    log_info("Initializing Eagle3: main=" + main_model_desc.device + ", draft=" + draft_model_desc.device +
             ", iterations=" + std::to_string(m_draft_iterations));

    auto main_model = main_model_desc.model;
    auto draft_model = draft_model_desc.model;
    m_tokenizer = main_model_desc.tokenizer;

    OPENVINO_ASSERT(!m_hidden_layers_to_abstract.empty(),
                    "Eagle3 requires hidden_layers_list configuration. "
                    "Provide it in properties or config.json. "
                    "Example: config[\"hidden_layers_list\"] = std::vector<int>{2, 16, 29}");

    // Model transformations
    ov::genai::share_embedding_weights(main_model, draft_model);
    log_debug("Shared embedding weights");

    set_draft_target_mapping(draft_model);

    // Currently, the d2t node is stored in the draft model
    // If it is not removed, it will affect the splitting and compilation of NPUW
    // TODO: Root cause and better to remove this logic in model conversion step
    ov::genai::remove_d2t_result_node(draft_model);
    log_debug("Removed d2t node");

    ov::genai::extract_hidden_state_generic(main_model, m_hidden_layers_to_abstract, main_model_desc.device);
    log_debug("Extracted main model hidden states");

    ov::genai::extract_hidden_state_generic(draft_model, {-1}, draft_model_desc.device);
    log_debug("Model transformations completed");

    std::size_t validation_window = m_draft_iterations + 1;

    auto draft_desc = draft_model_desc;
    if (draft_desc.device == "NPU") {
        draft_desc.properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = validation_window;
        draft_desc.properties["NPUW_DEVICES"] = "CPU";
        // TODO: Partition issue for draft model, low priority since it has only one repeat block
        draft_desc.properties["NPUW_ONLINE_PIPELINE"] = "NONE";
    }
    m_draft_model = std::make_unique<Eagle3DraftModelWrapper>(draft_desc);

    auto main_desc = main_model_desc;
    if (main_desc.device == "NPU") {
        main_desc.properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = validation_window;
        main_desc.properties["NPUW_DEVICES"] = "CPU";
        // Set rt_info to identify Eagle3 mode in NPUW
        main_model->set_rt_info("true", "eagle3_mode");
    }
    m_main_model = std::make_unique<Eagle3TargetModelWrapper>(main_desc);

    m_sd_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_sd_perf_metrics.raw_metrics.m_inference_durations = {{ov::genai::MicroSeconds(0.0f)}};

    log_info("Eagle3 initialization completed");
}

StatefulEagle3LLMPipeline::~StatefulEagle3LLMPipeline() {
    m_main_model->release_memory();
    m_draft_model->release_memory();
    log_debug("Pipeline destroyed");
}

void StatefulEagle3LLMPipeline::set_draft_target_mapping(const std::shared_ptr<ov::Model>& draft_model) {
    OPENVINO_ASSERT(draft_model, "Draft model is null");

    auto d2t_tensor = ov::genai::extract_d2t_mapping_table(draft_model);
    OPENVINO_ASSERT(d2t_tensor, "Draft-to-target mapping not found. Eagle3 requires d2t mapping.");

    OPENVINO_ASSERT(d2t_tensor->get_element_type() == ov::element::i64, "Draft-to-target mapping must be int64");

    ov::Tensor d2t_mapping(ov::element::i64, d2t_tensor->get_shape());
    std::memcpy(d2t_mapping.data(), d2t_tensor->get_data_ptr(), d2t_tensor->get_byte_size());

    m_draft_target_mapping = std::move(d2t_mapping);
    log_info("D2T mapping: " + std::to_string(m_draft_target_mapping.get_size()) + " entries");
}

void StatefulEagle3LLMPipeline::set_verbose(bool verbose) {
    if (m_main_model)
        m_main_model->set_verbose(verbose);
    if (m_draft_model)
        m_draft_model->set_verbose(verbose);
    log_debug("Verbose: " + std::string(verbose ? "on" : "off"));
}

GenerationConfig StatefulEagle3LLMPipeline::resolve_generation_config(OptionalGenerationConfig generation_config) {
    GenerationConfig config = generation_config.value_or(m_generation_config);

    std::size_t prev_draft_iterations = m_draft_iterations;
    ov::genai::speculative_decoding::ensure_num_assistant_tokens_is_set(config);
    m_draft_iterations = config.num_assistant_tokens;

    // Log if draft_iterations changed from default
    if (m_draft_iterations != prev_draft_iterations) {
        if (m_draft_iterations == 0) {
            log_info("Speculative decoding DISABLED (num_assistant_tokens=0), using target model only");
        } else if (is_verbose()) {
            log_debug("Draft iterations updated: " + std::to_string(prev_draft_iterations) + " -> " +
                      std::to_string(m_draft_iterations));
        }
    }

    if (config.stop_token_ids.empty())
        config.stop_token_ids = m_generation_config.stop_token_ids;
    if (config.eos_token_id == -1)
        config.set_eos_token_id(m_generation_config.eos_token_id);
    config.validate();
    return config;
}

DecodedResults StatefulEagle3LLMPipeline::generate(StringInputs inputs,
                                                   OptionalGenerationConfig generation_config,
                                                   StreamerVariant streamer) {
    ManualTimer generate_timer("StatefulEagle3LLMPipeline::generate()");
    generate_timer.start();
    ManualTimer encode_timer("Encode");
    encode_timer.start();

    std::string prompt =
        std::visit(overloaded{[](const std::string& prompt_str) {
                                  return prompt_str;
                              },
                              [](std::vector<std::string>& prompts) {
                                  OPENVINO_ASSERT(prompts.size() == 1u, "Currently only batch size=1 is supported");
                                  return prompts.front();
                              }},
                   inputs);

    GenerationConfig config = resolve_generation_config(generation_config);

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

DecodedResults StatefulEagle3LLMPipeline::generate(const ChatHistory& history,
                                                   OptionalGenerationConfig generation_config,
                                                   StreamerVariant streamer) {
    ManualTimer generate_timer("StatefulEagle3LLMPipeline::generate()");
    generate_timer.start();
    ManualTimer encode_timer("Encode");
    encode_timer.start();

    GenerationConfig config = resolve_generation_config(generation_config);

    OPENVINO_ASSERT(config.apply_chat_template,
                    "Chat template must be applied when using ChatHistory in generate method.");
    OPENVINO_ASSERT(!m_tokenizer.get_chat_template().empty(),
                    "Chat template must not be empty when using ChatHistory in generate method.");
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

    auto config = resolve_generation_config(generation_config);

    // Create streamer for streaming output
    std::shared_ptr<StreamerBase> streamer_ptr = ov::genai::utils::create_streamer(streamer, m_tokenizer);

    log_info("Starting Eagle3 generation with max_new_tokens=" + std::to_string(config.max_new_tokens) +
             ", draft_iterations=" + std::to_string(m_draft_iterations));

    // Extract input tensors
    ov::Tensor input_ids, attention_mask;
    if (auto* tensor_input = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *tensor_input;
        attention_mask = ov::genai::utils::init_attention_mask(input_ids);
    } else if (auto* tokenized_input = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = tokenized_input->input_ids;
        attention_mask = tokenized_input->attention_mask;
    }

    auto prompt_shape = input_ids.get_shape();
    if (prompt_shape[0] != 1) {
        throw std::runtime_error("Only batch size 1 is supported");
    }

    std::size_t prompt_len = prompt_shape[1];

    m_prompt_length = prompt_len;

    log_debug("Prompt length: " + std::to_string(prompt_len) + " tokens");

    // Initialize position IDs
    ov::Tensor position_ids{ov::element::i64, input_ids.get_shape()};
    utils::initialize_position_ids(position_ids, attention_mask);

    // Reset model states and initialize sequences
    m_main_model->reset_state();
    m_draft_model->reset_state();

    // Initialize main model with full sequence
    m_main_model->initialize_sequence(input_ids, position_ids);

    // Initialize draft model with sequence starting from second token
    m_draft_model->initialize_sequence(input_ids, position_ids);

    // Initial main model inference
    log_generation_step("Initial Main Model Inference", 0);
    auto main_output = m_main_model->infer(input_ids, attention_mask, position_ids);
    auto initial_token = std::get<int64_t>(m_main_model->sample_tokens(main_output.logits, 1));

    // Get initial hidden features and append first generated token
    auto main_hidden_features = main_output.hidden_features;
    m_main_model->append_tokens({initial_token});
    m_draft_model->append_tokens({initial_token});

    // Stream the initial token
    auto streaming_status = stream_generated_tokens(streamer_ptr, std::vector<int64_t>{initial_token});

    log_debug("Initial token generated: " + std::to_string(initial_token));
    log_sequence_state("after initial token generation");

    // Main generation loop
    std::size_t max_new_tokens = config.max_new_tokens;
    std::size_t generated_tokens = 1;  // Count initial token
    bool eos_reached = false;

    // Track metrics for speculative decoding
    std::size_t total_draft_accepted = 0;   // Number of draft tokens accepted by main model
    std::size_t total_draft_generated = 0;  // Total draft tokens generated (including rejected)
    std::size_t total_iterations = 0;       // Number of speculative iterations

    // Speculative decoding loop
    std::size_t token_count = m_draft_model->get_sequence_length();
    auto target_hidden_states = main_hidden_features;

    while (!eos_reached && generated_tokens < max_new_tokens &&
            m_main_model->get_sequence_length() < prompt_len + max_new_tokens &&
            (streaming_status == ov::genai::StreamingStatus::RUNNING)) {
        log_generation_step("Speculative Decoding Iteration", generated_tokens);
        log_sequence_state("iteration start");

        auto result =
            run_speculative_iteration(target_hidden_states, token_count, static_cast<int64_t>(config.eos_token_id));

        // Stream validated tokens
        streaming_status = stream_generated_tokens(streamer_ptr, result.validated_tokens);

        // Update iteration counter
        total_iterations++;

        // Update draft token statistics
        total_draft_generated += m_draft_iterations;  // Each iteration generates m_draft_iterations draft tokens
        total_draft_accepted +=
            result.accepted_tokens_count;  // Number of draft tokens accepted (not including main model's token)

        if (result.new_token == static_cast<int64_t>(config.eos_token_id) || result.eos_reached) {
            eos_reached = true;
            log_debug("EOS reached - terminating generation");
        }

        // Validate that speculative iteration produced valid results
        OPENVINO_ASSERT(result.new_token != -1, "Speculative iteration must produce a valid token");
        OPENVINO_ASSERT(result.next_window_size > 0, "Speculative iteration must produce valid next_window_size");
        OPENVINO_ASSERT(result.next_hidden_window && result.next_hidden_window.get_size() > 0,
                        "Speculative iteration must produce valid next_hidden_window");

        generated_tokens++;
        log_debug("Generated token " + std::to_string(generated_tokens) + ": " +
                    std::to_string(result.new_token) + ", accepted " +
                    std::to_string(result.accepted_tokens_count) + " draft tokens out of " +
                    std::to_string(m_draft_iterations));

        // Prepare for next iteration
        token_count = result.next_window_size;
        target_hidden_states = result.next_hidden_window;

        log_debug("Next iteration: token_count=" + std::to_string(token_count) +
                    ", hidden_states_size=" + std::to_string(target_hidden_states.get_size()));

        log_sequence_state("iteration end");
    }

    m_streaming_was_cancelled = (streaming_status == ov::genai::StreamingStatus::CANCEL);
    if (streamer_ptr) {  // push streamer's cache
        streamer_ptr->end();
    }

    // Prepare results using main model's tokens as source of truth
    EncodedResults results;
    results.tokens = {m_main_model->get_tokens()};
    results.scores.resize(1);
    results.scores[0] = 0.0f;  // Greedy decoding, no scores

    // Display final tokens if verbose
    if (is_verbose() && !results.tokens[0].empty()) {
        try {
            std::string decoded_text = m_tokenizer.decode(results.tokens[0]);
            std::cout << "[EAGLE3-FINAL] All tokens decoded (" << results.tokens[0].size() << " tokens): \""
                      << decoded_text << "\"" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "[EAGLE3-FINAL] Failed to decode tokens: " << e.what() << std::endl;
        }
    }

    // Update performance metrics following the standard stateful speculative decoding pattern
    generate_timer.end();

    m_sd_perf_metrics.num_input_tokens = prompt_len;
    m_sd_perf_metrics.load_time = this->m_load_time_ms;
    m_sd_perf_metrics.raw_metrics.generate_durations.clear();
    m_sd_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_timer.get_duration_microsec());

    // Update main and draft model metrics from their RawPerfMetrics
    m_sd_perf_metrics.main_model_metrics.raw_metrics = m_main_model->get_raw_perf_metrics();
    m_sd_perf_metrics.draft_model_metrics.raw_metrics = m_draft_model->get_raw_perf_metrics();

    // Set num_accepted_tokens - this represents draft tokens accepted by main model
    m_sd_perf_metrics.num_accepted_tokens = total_draft_accepted;

    // Update speculative decoding metrics based on collected data
    if (generated_tokens > 0) {
        // Calculate acceptance rate: accepted draft tokens / total draft tokens generated
        float acceptance_rate = total_draft_generated > 0
                                    ? (static_cast<float>(total_draft_accepted) / total_draft_generated * 100.0f)
                                    : 0.0f;

        m_sd_metrics.update_acceptance_rate(0, acceptance_rate);
        m_sd_metrics.update_draft_accepted_tokens(0, total_draft_accepted);
        m_sd_metrics.update_draft_generated_len(0, total_draft_generated);
        m_sd_metrics.update_generated_len(generated_tokens);
    }

    // Evaluate statistics using standard interface
    m_sd_perf_metrics.evaluate_statistics(generate_timer.get_start_time());

    results.perf_metrics = m_sd_perf_metrics;
    results.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_sd_perf_metrics);

    return results;
}

StatefulEagle3LLMPipeline::SpeculativeResult StatefulEagle3LLMPipeline::run_speculative_iteration(
    const ov::Tensor& target_hidden_states,
    std::size_t token_count,
    int64_t eos_token_id) {
    SpeculativeResult result;

    log_debug("Starting speculative iteration with token_count=" + std::to_string(token_count));

    if (!target_hidden_states || target_hidden_states.get_size() == 0) {
        log_debug("Invalid target hidden states provided");
        return result;
    }

    // Step 1: Initial draft inference using target model's hidden states
    ov::Tensor draft_input_ids, draft_attention_mask, draft_position_ids;
    m_draft_model->build_model_inputs(token_count, draft_input_ids, draft_attention_mask, draft_position_ids);

    auto draft_output = m_draft_model->infer(draft_input_ids,
                                             draft_attention_mask,
                                             draft_position_ids,
                                             target_hidden_states,
                                             /*internal_hidden_states=*/ov::Tensor{});

    int64_t first_draft_token = std::get<int64_t>(m_draft_model->sample_tokens(draft_output.logits, 1));
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
    auto internal_hidden_states = extract_last_hidden_state(draft_output.hidden_features);

    // Step 2: Additional draft iterations
    for (std::size_t i = 0; i < m_draft_iterations - 1; ++i) {
        m_draft_model->build_model_inputs(1, draft_input_ids, draft_attention_mask, draft_position_ids);

        auto more_output = m_draft_model->infer(draft_input_ids,
                                                draft_attention_mask,
                                                draft_position_ids,
                                                /*target_hidden_states*/ ov::Tensor{},
                                                internal_hidden_states);

        int64_t draft_token = std::get<int64_t>(m_draft_model->sample_tokens(more_output.logits, 1));
        draft_token = map_draft_token(draft_token);

        draft_candidates.push_back(draft_token);
        m_main_model->append_tokens({draft_token});
        m_draft_model->append_tokens({draft_token});
        internal_hidden_states = extract_last_hidden_state(more_output.hidden_features);
    }

    // Step 3: Validation - main model validates draft tokens with shift comparison
    log_debug("Starting validation phase with " + std::to_string(m_draft_iterations) + " draft tokens");

    std::size_t validation_window_size = m_draft_iterations + 1;

    ov::Tensor val_input_ids, val_attention_mask, val_position_ids;
    m_main_model->build_model_inputs(validation_window_size, val_input_ids, val_attention_mask, val_position_ids);

    // Run main model validation inference
    auto val_output = m_main_model->infer(val_input_ids, val_attention_mask, val_position_ids);
    auto sampled_tokens =
        std::get<std::vector<int64_t>>(m_main_model->sample_tokens(val_output.logits, validation_window_size));

    // Compare main model predictions with draft tokens (shift comparison)
    const int64_t* existing_tokens = val_input_ids.data<const int64_t>();
    std::size_t accepted_count = 0;

    for (std::size_t i = 0; i < m_draft_iterations; ++i) {
        if (sampled_tokens[i] == existing_tokens[i + 1]) {
            accepted_count++;
        } else {
            log_debug("Validation mismatch at position " + std::to_string(i) + ": expected " +
                      std::to_string(existing_tokens[i + 1]) + ", got " + std::to_string(sampled_tokens[i]));
            break;
        }
    }

    // The main model's prediction at accepted_count position becomes the new token
    int64_t main_predicted_token = sampled_tokens[accepted_count];

    // Calculate tokens to accept and reject
    std::size_t tokens_to_remove_from_draft = m_draft_iterations - accepted_count;
    std::size_t total_accepted_tokens = accepted_count + 1;  // accepted drafts + main prediction

    log_debug("Validation result: accepted " + std::to_string(accepted_count) + "/" +
              std::to_string(m_draft_iterations) +
              " draft tokens, main_predicted_token=" + std::to_string(main_predicted_token));

    // Rollback both models to pre-draft state, then append accepted tokens
    m_main_model->truncate_sequence(pre_draft_main_len);
    m_draft_model->truncate_sequence(pre_draft_draft_len);

    // Append accepted draft tokens + main predicted token to both models
    std::vector<int64_t> tokens_to_append;
    tokens_to_append.reserve(total_accepted_tokens);
    for (std::size_t i = 0; i < accepted_count; ++i) {
        tokens_to_append.push_back(draft_candidates[i]);
    }
    tokens_to_append.push_back(main_predicted_token);

    m_main_model->append_tokens(tokens_to_append);
    m_draft_model->append_tokens(tokens_to_append);

    // Trim KV cache for rejected draft tokens
    if (tokens_to_remove_from_draft > 0) {
        m_main_model->trim_kv_cache(tokens_to_remove_from_draft);
        m_draft_model->trim_kv_cache(tokens_to_remove_from_draft);
    }

    log_debug("Accepted total " + std::to_string(total_accepted_tokens) + " tokens (" + std::to_string(accepted_count) +
              " draft + 1 main prediction), rejected " + std::to_string(tokens_to_remove_from_draft) +
              " draft tokens.");

    // Build next hidden window for next iteration
    auto current_hidden = val_output.hidden_features;
    OPENVINO_ASSERT(current_hidden && current_hidden.get_size() > 0,
                    "Hidden features from validation output must exist");

    auto h_shape = current_hidden.get_shape();
    OPENVINO_ASSERT(h_shape.size() == 3 && h_shape[0] == 1,
                    "Invalid hidden state shape for next window construction");

    std::size_t seq_len = h_shape[1];
    std::size_t hidden_dim = h_shape[2];
    std::size_t next_window_len = total_accepted_tokens;

    OPENVINO_ASSERT(seq_len >= next_window_len,
                    "Hidden state seq_len (",
                    seq_len,
                    ") < next_window_len (",
                    next_window_len,
                    ")");

    // Extract hidden states for accepted tokens
    // Input: [1, seq_len, hidden_dim] -> Output: [1, next_window_len, hidden_dim]
    ov::Tensor next_hidden = ov::Tensor(current_hidden, {0, 0, 0}, {1, next_window_len, hidden_dim});

    log_debug("Built next hidden window with " + std::to_string(next_window_len) + " positions (ROI, zero-copy)");

    // Check for EOS token
    if (main_predicted_token == eos_token_id) {
        result.eos_reached = true;
        log_debug("EOS token detected: " + std::to_string(main_predicted_token));
    }

    // Set result fields
    // IMPORTANT: accepted_tokens_count is ONLY the number of DRAFT tokens accepted by main model
    // It does NOT include the main model's own prediction token
    result.accepted_tokens_count = accepted_count;    // Only draft tokens accepted
    result.next_window_size = total_accepted_tokens;  // Total tokens for next iteration (draft + main)
    result.new_token = main_predicted_token;          // Main model's prediction
    result.next_hidden_window = next_hidden;
    result.validated_tokens = tokens_to_append;  // Return validated tokens for streaming

    log_debug("Speculative iteration completed - accepted " + std::to_string(accepted_count) +
              " draft tokens + 1 main prediction = " + std::to_string(total_accepted_tokens) +
              " tokens for next iteration");

    return result;
}

int64_t StatefulEagle3LLMPipeline::map_draft_token(int64_t draft_token) const {
    if (!m_draft_target_mapping || m_draft_target_mapping.get_size() == 0) {
        return draft_token;
    }

    std::size_t mapping_size = m_draft_target_mapping.get_size();
    if (draft_token < 0 || static_cast<std::size_t>(draft_token) >= mapping_size) {
        log_debug("Token " + std::to_string(draft_token) + " out of range, identity mapping");
        return draft_token;
    }

    const int64_t* data = m_draft_target_mapping.data<const int64_t>();
    int64_t target_token = draft_token + data[draft_token];

    if (is_verbose()) {
        log_debug("Mapped: " + std::to_string(draft_token) + " -> " + std::to_string(target_token));
    }

    return target_token;
}

std::vector<int64_t> StatefulEagle3LLMPipeline::map_draft_tokens(const std::vector<int64_t>& draft_tokens) const {
    if (!m_draft_target_mapping || m_draft_target_mapping.get_size() == 0) {
        return draft_tokens;
    }

    std::vector<int64_t> mapped;
    mapped.reserve(draft_tokens.size());
    for (int64_t token : draft_tokens) {
        mapped.push_back(map_draft_token(token));
    }
    return mapped;
}

void StatefulEagle3LLMPipeline::start_chat(const std::string& system_message) {
    m_is_chat_active = true;
    m_chat_history.clear();

    if (!system_message.empty()) {
        m_chat_history.push_back({{"role", "system"}, {"content", system_message}});
    }
    log_info("Chat started");
}

void StatefulEagle3LLMPipeline::finish_chat() {
    m_is_chat_active = false;
    m_chat_history.clear();
    log_info("Chat ended");
}

ov::genai::SpeculativeDecodingMetrics StatefulEagle3LLMPipeline::get_speculative_decoding_metrics() const {
    return m_sd_metrics;
}

void StatefulEagle3LLMPipeline::log_info(const std::string& message) const {
    std::cout << "[EAGLE3-PIPELINE] " << message << std::endl;
}

void StatefulEagle3LLMPipeline::log_debug(const std::string& message) const {
    if (is_verbose()) {
        std::cout << "[EAGLE3-DEBUG] " << message << std::endl;
    }
}

void StatefulEagle3LLMPipeline::log_generation_step(const std::string& step_name, std::size_t step_number) const {
    if (is_verbose()) {
        std::cout << "\n[EAGLE3] ===== STEP " << step_number << ": " << step_name << " =====" << std::endl;
    }
}

void StatefulEagle3LLMPipeline::log_sequence_state(const std::string& context) const {
    if (!is_verbose())
        return;

    std::cout << "[EAGLE3-PIPELINE] Sequence state (" << context << "):" << std::endl;
    std::cout << "  Prompt length: " << m_prompt_length << " tokens" << std::endl;
    std::cout << "  Main model tokens: " << m_main_model->get_sequence_length() << " tokens" << std::endl;
    std::cout << "  Draft model tokens: " << m_draft_model->get_sequence_length() << " tokens" << std::endl;

    // Show all tokens and positions from main model
    const auto& main_tokens = m_main_model->get_tokens();
    const auto& main_positions = m_main_model->get_positions();
    if (!main_tokens.empty()) {
        std::cout << "  Main model tokens: ";
        for (std::size_t i = 0; i < main_tokens.size(); ++i) {
            std::cout << main_tokens[i];
            if (i + 1 < main_tokens.size())
                std::cout << ", ";
        }
        std::cout << std::endl;

        if (!main_positions.empty()) {
            std::cout << "  Main model positions: ";
            for (std::size_t i = 0; i < main_positions.size(); ++i) {
                std::cout << main_positions[i];
                if (i + 1 < main_positions.size())
                    std::cout << ", ";
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
            if (i + 1 < draft_tokens.size())
                std::cout << ", ";
        }
        std::cout << std::endl;

        if (!draft_positions.empty()) {
            std::cout << "  Draft model positions: ";
            for (std::size_t i = 0; i < draft_positions.size(); ++i) {
                std::cout << draft_positions[i];
                if (i + 1 < draft_positions.size())
                    std::cout << ", ";
            }
            std::cout << std::endl;
        }
    }
}

ov::Tensor StatefulEagle3LLMPipeline::slice_hidden_features(const ov::Tensor& hidden_features,
                                                            std::size_t start_pos,
                                                            std::size_t length) const {
    if (!hidden_features || hidden_features.get_size() == 0) {
        return ov::Tensor{};
    }

    auto shape = hidden_features.get_shape();
    if (shape.size() != 3 || shape[0] != 1) {
        return hidden_features;
    }

    std::size_t seq_len = shape[1];
    std::size_t hidden_dim = shape[2];

    if (start_pos >= seq_len || length == 0) {
        return ov::Tensor{};
    }

    std::size_t actual_len = std::min(length, seq_len - start_pos);
    ov::Tensor sliced(ov::element::f32, {1, actual_len, hidden_dim});

    if (hidden_features.get_element_type() == ov::element::f32) {
        const float* src = hidden_features.data<const float>() + start_pos * hidden_dim;
        std::copy_n(src, actual_len * hidden_dim, sliced.data<float>());
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

    if (conf_shape.size() != 3 || new_shape.size() != 3 || conf_shape[0] != 1 || new_shape[0] != 1 ||
        conf_shape[2] != new_shape[2]) {
        return confirmed_hidden;
    }

    std::size_t conf_len = conf_shape[1];
    std::size_t new_len = new_shape[1];
    std::size_t hidden_dim = conf_shape[2];
    std::size_t total_len = conf_len + new_len;

    ov::Tensor combined(ov::element::f32, {1, total_len, hidden_dim});

    if (confirmed_hidden.get_element_type() == ov::element::f32 && new_hidden.get_element_type() == ov::element::f32) {
        float* dst = combined.data<float>();
        std::copy_n(confirmed_hidden.data<const float>(), conf_len * hidden_dim, dst);
        std::copy_n(new_hidden.data<const float>(), new_len * hidden_dim, dst + conf_len * hidden_dim);
    }

    return combined;
}

}  // namespace genai
}  // namespace ov
