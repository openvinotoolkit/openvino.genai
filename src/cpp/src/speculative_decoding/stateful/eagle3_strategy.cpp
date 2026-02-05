// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eagle3_strategy.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>

#include "continuous_batching/timer.hpp"
#include "eagle3_utils.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "speculative_decoding/eagle3_debug_utils.hpp"
#include "speculative_decoding/eagle3_model_transforms.hpp"
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

ov::genai::StreamingStatus stream_generated_tokens(std::shared_ptr<ov::genai::StreamerBase> streamer_ptr,
                                                   const std::vector<int64_t>& tokens) {
    if (streamer_ptr) {
        return streamer_ptr->write(tokens);
    }
    return ov::genai::StreamingStatus{};
}

}  // anonymous namespace

namespace ov::genai {

Eagle3InferWrapperBase::Eagle3InferWrapperBase(const ModelDesc& model_desc)
    : m_device(model_desc.device),
      m_properties(model_desc.properties),
      m_tokenizer(model_desc.tokenizer),
      m_sampler(model_desc.tokenizer) {
    m_kv_axes_pos = utils::get_kv_axes_pos(model_desc.model);

    if (m_device == "NPU") {
        auto [compiled, kv_desc] = utils::compile_decoder_for_npu(model_desc.model, m_properties, m_kv_axes_pos);
        m_max_prompt_len = kv_desc.max_prompt_len;
        m_request = compiled.create_infer_request();

        eagle3::log_debug(eagle3::PipelineStep::INIT,
                          "NPU compiled: max_prompt=" + std::to_string(m_max_prompt_len),
                          m_verbose);
    } else {
        m_request =
            utils::singleton_core().compile_model(model_desc.model, m_device, m_properties).create_infer_request();
        eagle3::log_debug(eagle3::PipelineStep::INIT, m_device + " model compiled successfully", m_verbose);
    }

    // Initialize performance metrics
    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.tokenization_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.detokenization_durations = {MicroSeconds(0.0f)};

    m_sequence_group = nullptr;
}

void Eagle3InferWrapperBase::append_tokens(const std::vector<int64_t>& tokens) {
    if (tokens.empty())
        return;

    auto current_sequence = get_current_sequence();
    OPENVINO_ASSERT(current_sequence, "SequenceGroup not initialized");

    for (auto token : tokens) {
        current_sequence->append_token(token, 0.0f);
    }

    eagle3::log_debug(eagle3::PipelineStep::ITER,
                      "Appended " + std::to_string(tokens.size()) + " tokens: " + eagle3::format_tokens(tokens) +
                          ", new seq_len=" + std::to_string(get_sequence_length()),
                      m_verbose);
}

void Eagle3InferWrapperBase::truncate_sequence(size_t size) {
    auto current_sequence = get_current_sequence();
    OPENVINO_ASSERT(current_sequence, "SequenceGroup not initialized");

    const size_t prompt_len = m_sequence_group->get_prompt_len();
    const size_t current_len = prompt_len + current_sequence->get_generated_len();

    if (size < current_len) {
        OPENVINO_ASSERT(size >= prompt_len, "Cannot truncate prompt tokens");
        const size_t tokens_to_remove = current_len - size;
        current_sequence->remove_last_tokens(tokens_to_remove);

        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Truncated sequence: " + std::to_string(current_len) + " -> " + std::to_string(size) +
                              " (removed " + std::to_string(tokens_to_remove) + " tokens)",
                          m_verbose);
    }
}

void Eagle3InferWrapperBase::trim_kv_cache(size_t tokens_to_remove) {
    const size_t current_len = get_sequence_length();
    if (tokens_to_remove == 0 || current_len == 0) {
        return;
    }

    OPENVINO_ASSERT(tokens_to_remove > 0 && tokens_to_remove < current_len,
                    "Cannot trim ",
                    tokens_to_remove,
                    " tokens from ",
                    current_len,
                    " tokens. Valid range: 0 < tokens_to_remove < current_len");

    if (m_device != "NPU") {
        utils::KVCacheState state;
        state.num_tokens_to_trim = tokens_to_remove;
        state.seq_length_axis = m_kv_axes_pos.seq_len;
        state.reset_mem_state = false;
        utils::trim_kv_cache(m_request, state, {});
    }

    eagle3::log_debug(eagle3::PipelineStep::KV_CACHE,
                      "KV cache trimmed: " + std::to_string(current_len) + " -> " +
                          std::to_string(current_len - tokens_to_remove) + " (removed " +
                          std::to_string(tokens_to_remove) + ")",
                      m_verbose);
}

void Eagle3InferWrapperBase::reset_state() {
    m_sequence_group = nullptr;

    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.m_durations.clear();
    m_raw_perf_metrics.m_batch_sizes.clear();

    eagle3::log_debug(eagle3::PipelineStep::INIT, "State reset", m_verbose);
}

void Eagle3InferWrapperBase::release_memory() {
    m_request.get_compiled_model().release_memory();
}

void Eagle3InferWrapperBase::build_model_inputs(const size_t input_token_count,
                                                ov::Tensor& input_ids,
                                                ov::Tensor& attention_mask,
                                                ov::Tensor& position_ids,
                                                ov::Tensor& eagle_tree_mask,
                                                InferencePhase phase,
                                                size_t iteration_id,
                                                std::shared_ptr<std::vector<int64_t>> iteration_history) {
    OPENVINO_ASSERT(m_sequence_group, "SequenceGroup not initialized");

    // Special handling for DRAFT_ITERATION with multiple sequences
    if (phase == InferencePhase::DRAFT_ITERATION) {
        auto running_sequences = m_sequence_group->get_running_sequences();
        const size_t sequence_numb = running_sequences.size();
        OPENVINO_ASSERT(sequence_numb > 0, "No running sequences");

        const auto& prompt_ids = m_sequence_group->get_prompt_ids();
        const size_t prompt_len = prompt_ids.size();

        // 1. Build input_ids: {1, sequence_numb} - last token from each sequence
        input_ids = ov::Tensor(ov::element::i64, {1, sequence_numb});
        int64_t* input_ids_ptr = input_ids.data<int64_t>();

        for (size_t i = 0; i < sequence_numb; ++i) {
            const auto& generated_ids = running_sequences[i]->get_generated_ids();
            OPENVINO_ASSERT(!generated_ids.empty(), "Sequence ", i, " has no generated tokens");
            input_ids_ptr[i] = generated_ids.back();
        }

        // 2. Build position_ids: {1, sequence_numb} - all same position value
        const auto& first_seq_generated = running_sequences[0]->get_generated_ids();
        const int64_t position_value = static_cast<int64_t>(prompt_len + first_seq_generated.size() - 1);

        position_ids = ov::Tensor(ov::element::i64, {1, sequence_numb});
        int64_t* position_ids_ptr = position_ids.data<int64_t>();
        std::fill_n(position_ids_ptr, sequence_numb, position_value);

        // 3. Build attention_mask: {1, base_len + sequence_numb * iteration_id - 1}
        // iteration_id starts from 1 (0 uses DRAFT_INITIAL phase, not DRAFT_ITERATION)
        const size_t base_attention_mask_len = static_cast<size_t>(position_value + 1);
        const size_t extended_attention_mask_len = base_attention_mask_len + sequence_numb * iteration_id - 1;

        attention_mask = ov::Tensor(ov::element::i64, {1, extended_attention_mask_len});
        std::fill_n(attention_mask.data<int64_t>(), extended_attention_mask_len, 1);

        // 4. Build eagle_tree_mask: {1, 1, sequence_numb, extended_attention_mask_len}
        eagle_tree_mask = ov::Tensor(ov::element::f32, {1, 1, sequence_numb, extended_attention_mask_len});
        float* mask_data = eagle_tree_mask.data<float>();

        // Initialize all to -INF (cannot attend by default)
        std::fill_n(mask_data, sequence_numb * extended_attention_mask_len, -std::numeric_limits<float>::infinity());

        // The tree region starts at base_attention_mask_len - 1
        const size_t tree_start_pos = base_attention_mask_len - 1;

        // Step 1: All sequences can attend to the base region (prompt + initial tokens)
        for (size_t row = 0; row < sequence_numb; ++row) {
            const size_t row_offset = row * extended_attention_mask_len;
            std::fill_n(mask_data + row_offset, tree_start_pos, 0.0f);
        }

        // Step 2: Build precise tree attention mask using iteration history
        if (iteration_history && !iteration_history->empty()) {
            const auto& history = *iteration_history;

            // Calculate how many complete layers we have in history
            // Each layer has sequence_numb tokens
            const size_t completed_layers = history.size() / sequence_numb;

            // For each sequence, find its path through the tree
            for (size_t seq_idx = 0; seq_idx < sequence_numb; ++seq_idx) {
                const auto& generated_ids = running_sequences[seq_idx]->get_generated_ids();
                const size_t row_offset = seq_idx * extended_attention_mask_len;

                // Find which positions in the history match this sequence's path
                // We need to look at the last 'completed_layers' tokens from generated_ids
                // These are the tokens that have already been generated in previous iterations

                OPENVINO_ASSERT(generated_ids.size() >= completed_layers,
                                "Sequence ",
                                seq_idx,
                                " has insufficient tokens: ",
                                generated_ids.size(),
                                " < ",
                                completed_layers);

                // Extract the path: tokens that led to the current position
                std::vector<int64_t> seq_path;
                for (size_t i = generated_ids.size() - completed_layers; i < generated_ids.size(); ++i) {
                    seq_path.push_back(generated_ids[i]);
                }

                // Now match this path against the iteration history
                // For each completed layer in the history, find which position matches this sequence's token
                size_t history_offset = 0;
                for (size_t layer = 0; layer < completed_layers; ++layer) {
                    const int64_t target_token = seq_path[layer];

                    // Search in this layer's history (sequence_numb tokens)
                    for (size_t pos_in_layer = 0; pos_in_layer < sequence_numb; ++pos_in_layer) {
                        const size_t history_idx = history_offset + pos_in_layer;
                        OPENVINO_ASSERT(history_idx < history.size(),
                                        "History index ",
                                        history_idx,
                                        " out of bounds ",
                                        history.size());

                        if (history[history_idx] == target_token) {
                            // This position matches the path - allow attention
                            const size_t abs_pos = tree_start_pos + layer * sequence_numb + pos_in_layer;
                            mask_data[row_offset + abs_pos] = 0.0f;
                            break;  // Found the match for this layer
                        }
                    }

                    history_offset += sequence_numb;
                }

                // For the current layer (layer == iteration_id), allow attention to own position
                // This is the layer we're currently generating
                const size_t current_layer_start = tree_start_pos + iteration_id * sequence_numb;
                mask_data[row_offset + current_layer_start + seq_idx] = 0.0f;
            }
        } else {
            // Fallback: if no history available (shouldn't happen), use simple strategy
            eagle3::log_debug(eagle3::PipelineStep::ITER,
                              "Warning: iteration_history not available, using fallback mask strategy",
                              m_verbose);

            for (size_t row = 0; row < sequence_numb; ++row) {
                const size_t row_offset = row * extended_attention_mask_len;
                // Allow attention to tree region for current sequence's position
                const size_t current_layer_start = tree_start_pos + iteration_id * sequence_numb;
                mask_data[row_offset + current_layer_start + row] = 0.0f;
            }
        }

        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Eagle tree mask (DRAFT_ITERATION): {1, 1, " + std::to_string(sequence_numb) + ", " +
                              std::to_string(extended_attention_mask_len) + "}, iter=" + std::to_string(iteration_id) +
                              ", position=" + std::to_string(position_value) + ", history_size=" +
                              (iteration_history ? std::to_string(iteration_history->size()) : "N/A"),
                          m_verbose);

        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Built model inputs (DRAFT_ITERATION): sequence_numb=" + std::to_string(sequence_numb) +
                              ", iteration_id=" + std::to_string(iteration_id) +
                              ", position=" + std::to_string(position_value) +
                              ", attn_mask_len=" + std::to_string(extended_attention_mask_len),
                          m_verbose);
        return;
    }

    // Standard logic for other phases
    auto current_sequence = get_current_sequence();
    OPENVINO_ASSERT(current_sequence, "SequenceGroup not initialized");

    const auto& prompt_ids = m_sequence_group->get_prompt_ids();
    const auto& generated_ids = current_sequence->get_generated_ids();

    const size_t prompt_len = prompt_ids.size();
    const size_t generated_len = generated_ids.size();
    const size_t total_len = prompt_len + generated_len;
    const size_t start_pos = total_len - input_token_count;

    OPENVINO_ASSERT(input_token_count > 0 && input_token_count <= total_len,
                    "Invalid input_token_count: ",
                    input_token_count,
                    ", total_len: ",
                    total_len);

    // Allocate tensors
    input_ids = ov::Tensor(ov::element::i64, {1, input_token_count});
    position_ids = ov::Tensor(ov::element::i64, {1, input_token_count});

    int64_t* input_ids_ptr = input_ids.data<int64_t>();
    int64_t* position_ids_ptr = position_ids.data<int64_t>();

    // Fill input_ids and position_ids from sequence
    if (start_pos < prompt_len) {
        // Part from prompt
        const size_t prompt_count = std::min(input_token_count, prompt_len - start_pos);
        std::copy_n(prompt_ids.data() + start_pos, prompt_count, input_ids_ptr);
        std::iota(position_ids_ptr, position_ids_ptr + prompt_count, static_cast<int64_t>(start_pos));

        // Part from generated (if any)
        if (input_token_count > prompt_count) {
            const size_t generated_count = input_token_count - prompt_count;
            std::copy_n(generated_ids.data(), generated_count, input_ids_ptr + prompt_count);
            std::iota(position_ids_ptr + prompt_count,
                      position_ids_ptr + prompt_count + generated_count,
                      static_cast<int64_t>(prompt_len));
        }
    } else {
        // All from generated
        const size_t generated_start = start_pos - prompt_len;
        std::copy_n(generated_ids.data() + generated_start, input_token_count, input_ids_ptr);
        std::iota(position_ids_ptr,
                  position_ids_ptr + input_token_count,
                  static_cast<int64_t>(prompt_len + generated_start));
    }

    // Build attention mask (always all 1s)
    const size_t attention_mask_len = static_cast<size_t>(position_ids_ptr[input_token_count - 1] + 1);
    attention_mask = ov::Tensor(ov::element::i64, {1, attention_mask_len});
    std::fill_n(attention_mask.data<int64_t>(), attention_mask_len, 1);

    // Build eagle_tree_mask based on inference phase
    switch (phase) {
    case InferencePhase::TARGET_PREFILL:
    case InferencePhase::DRAFT_INITIAL:
        // During prefill/initial phase: eagle_tree_mask is all zeros
        // Minimal shape: {1, 1, 1, 1}
        eagle_tree_mask = ov::Tensor(ov::element::f32, {1, 1, 1, 1});
        std::fill_n(eagle_tree_mask.data<float>(), 1, 0.0f);
        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Eagle tree mask (TARGET_PREFILL/DRAFT_INITIAL): {1, 1, 1, 1}, all zeros",
                          m_verbose);
        break;

    case InferencePhase::TARGET_VALIDATION: {
        // During validation phase: construct EAGLE tree attention mask
        // Shape: {1, 1, input_token_count, attention_mask_len}
        eagle_tree_mask = ov::Tensor(ov::element::f32, {1, 1, input_token_count, attention_mask_len});
        float* mask_data = eagle_tree_mask.data<float>();

        // Step 1: Get proposed_token_numb from first position_id
        const int64_t proposed_token_numb = position_ids_ptr[0];
        const size_t proposed_token_numb_sz = static_cast<size_t>(proposed_token_numb);

        // Step 2: First proposed_token_numb positions in each row: set first {1, 1, input_token_count,
        // proposed_token_numb} to all zeros
        for (size_t row = 0; row < input_token_count; ++row) {
            const size_t row_offset = row * attention_mask_len;
            std::fill_n(mask_data + row_offset, proposed_token_numb_sz, 0.0f);
        }

        // Step 3: Remaining region [input_token_count, attention_mask_len - proposed_token_numb]
        // This should be a triangular matrix: lower triangle = 0, upper triangle = -INF
        const size_t tree_width = attention_mask_len - proposed_token_numb_sz;

        // Verify: tree_width should theoretically equal input_token_count
        if (tree_width != input_token_count) {
            eagle3::log_debug(eagle3::PipelineStep::ITER,
                              "Warning: tree_width (" + std::to_string(tree_width) + ") != input_token_count (" +
                                  std::to_string(input_token_count) + ")",
                              m_verbose);
        }

        // Build triangular matrix in the remaining region
        for (size_t row = 0; row < input_token_count; ++row) {
            const size_t row_offset = row * attention_mask_len;
            const size_t tree_start = row_offset + proposed_token_numb_sz;

            // For each column in the tree region
            for (size_t col = 0; col < tree_width; ++col) {
                if (col <= row) {
                    // Lower triangle (including diagonal): 0
                    mask_data[tree_start + col] = 0.0f;
                } else {
                    // Upper triangle: -INF
                    mask_data[tree_start + col] = -std::numeric_limits<float>::infinity();
                }
            }
        }

        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Eagle tree mask (TARGET_VALIDATION): {1, 1, " + std::to_string(input_token_count) + ", " +
                              std::to_string(attention_mask_len) + "}, proposed_token_numb=" +
                              std::to_string(proposed_token_numb) + ", tree_width=" + std::to_string(tree_width),
                          m_verbose);
        break;
    }

    case InferencePhase::DRAFT_ITERATION:
        // This case is handled at the beginning of the function
        OPENVINO_ASSERT(false, "DRAFT_ITERATION should be handled in the early return path");
        break;
    }

    // Log input preparation details
    eagle3::log_debug(eagle3::PipelineStep::ITER,
                      "Built model inputs: input_token_count=" + std::to_string(input_token_count) + ", start_pos=" +
                          std::to_string(start_pos) + ", attn_mask_len=" + std::to_string(attention_mask_len),
                      m_verbose);
}

std::vector<int64_t> Eagle3InferWrapperBase::sample_tokens(const ov::Tensor& logits,
                                                           size_t input_token_count,
                                                           size_t sample_count,
                                                           size_t num_tokens_to_validate) {
    const ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Invalid logits shape: ", eagle3::format_shape(shape));
    OPENVINO_ASSERT(sample_count > 0 && sample_count <= shape[1],
                    "Invalid sample_count: ",
                    sample_count,
                    ", logits seq_len: ",
                    shape[1]);
    OPENVINO_ASSERT(input_token_count > 0, "Invalid input_token_count");

    const bool is_validation_mode = num_tokens_to_validate > 0;

    eagle3::log_debug(eagle3::PipelineStep::SAMPLE,
                      "sample_tokens: input_tokens=" + std::to_string(input_token_count) + ", sample_count=" +
                          std::to_string(sample_count) + ", validate=" + std::to_string(num_tokens_to_validate) +
                          ", logits_shape=" + eagle3::format_shape(shape),
                      m_verbose);

    auto sequence_group = get_sequence_group();
    OPENVINO_ASSERT(sequence_group, "SequenceGroup not initialized");

    const size_t logits_seq_len = shape[1];
    const size_t vocab_size = shape[2];

    // Slice logits to last 'sample_count' positions if needed
    ov::Tensor sliced_logits = logits;
    if (sample_count < logits_seq_len) {
        auto [start_coord, end_coord] =
            ov::genai::utils::make_roi(shape, 1, logits_seq_len - sample_count, logits_seq_len);
        sliced_logits = ov::Tensor(logits, start_coord, end_coord);
    }

    // Configure sequence group for sampling
    sequence_group->schedule_tokens(input_token_count);
    sequence_group->set_output_seq_len(sample_count);
    sequence_group->set_num_validated_tokens(num_tokens_to_validate);

    // Execute sampling
    m_sampler.sample({sequence_group}, sliced_logits, is_validation_mode);
    sequence_group->finish_iteration();

    // Extract results from all sequences
    // Note: Get num_sequences AFTER sampling, as sample() may update the sequence group
    auto running_sequences = sequence_group->get_running_sequences();
    const size_t num_sequences = running_sequences.size();
    OPENVINO_ASSERT(num_sequences > 0, "No running sequences after sampling");

    eagle3::log_debug(eagle3::PipelineStep::SAMPLE,
                      "Processing " + std::to_string(num_sequences) + " sequence(s) after sampling",
                      m_verbose);

    std::vector<int64_t> result_tokens;

    if (!is_validation_mode) {
        // Non-validation mode: collect last token from each sequence
        for (size_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
            auto seq = running_sequences[seq_idx];
            OPENVINO_ASSERT(seq, "Invalid sequence at index ", seq_idx);

            const auto& generated_ids = seq->get_generated_ids();
            OPENVINO_ASSERT(!generated_ids.empty(), "Sequence ", seq_idx, " has no generated tokens");

            // Add the last token from this sequence
            result_tokens.push_back(generated_ids.back());
        }

        record_generated_tokens(num_sequences);

        eagle3::log_debug(eagle3::PipelineStep::SAMPLE,
                          "Sampled " + std::to_string(num_sequences) + " token(s) from " +
                              std::to_string(num_sequences) + " sequence(s): " + eagle3::format_tokens(result_tokens),
                          m_verbose);

        return result_tokens;
    } else {
        // Validation mode: collect last token from each sequence
        // In validation mode, sampler has already updated sequences, we just need to collect the last tokens
        for (size_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
            auto seq = running_sequences[seq_idx];
            OPENVINO_ASSERT(seq, "Invalid sequence at index ", seq_idx);

            const auto& generated_ids = seq->get_generated_ids();
            OPENVINO_ASSERT(!generated_ids.empty(), "Sequence ", seq_idx, " has no generated tokens");

            // Add the last token from this sequence
            result_tokens.push_back(generated_ids.back());
        }

        record_generated_tokens(result_tokens.size());

        eagle3::log_debug(eagle3::PipelineStep::VALID,
                          "Validation result: collected " + std::to_string(result_tokens.size()) + " token(s) from " +
                              std::to_string(num_sequences) +
                              " sequence(s), tokens=" + eagle3::format_tokens(result_tokens),
                          m_verbose);

        return result_tokens;
    }
}

ov::Tensor Eagle3InferWrapperBase::get_logits() const {
    return m_request.get_tensor("logits");
}

ov::Tensor Eagle3InferWrapperBase::get_hidden_features() const {
    auto hidden_state = m_request.get_tensor("last_hidden_state");
    const auto shape = hidden_state.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Invalid hidden state shape: ", eagle3::format_shape(shape));

    const size_t output_seq_len = shape[1];
    const size_t hidden_size = shape[2];
    const size_t actual_seq_len = m_request.get_tensor("input_ids").get_shape()[1];

    if (output_seq_len == actual_seq_len)
        return hidden_state;

    OPENVINO_ASSERT(actual_seq_len <= output_seq_len,
                    "Sequence length mismatch: actual=",
                    actual_seq_len,
                    ", output=",
                    output_seq_len);
    auto [start_coord, end_coord] =
        ov::genai::utils::make_roi(shape, 1, output_seq_len - actual_seq_len, output_seq_len);
    return ov::Tensor(hidden_state, start_coord, end_coord);
}

uint64_t Eagle3InferWrapperBase::execute_inference() {
    auto start = std::chrono::steady_clock::now();
    m_request.infer();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    return duration_us;
}

void Eagle3InferWrapperBase::update_inference_time(uint64_t inference_time_us) {
    m_raw_perf_metrics.m_durations.emplace_back(static_cast<float>(inference_time_us));
    m_raw_perf_metrics.m_inference_durations[0] += MicroSeconds(static_cast<float>(inference_time_us));
}

void Eagle3InferWrapperBase::record_generated_tokens(size_t actual_generated_count) {
    m_raw_perf_metrics.m_batch_sizes.emplace_back(actual_generated_count);
}

Eagle3TargetWrapper::Eagle3TargetWrapper(const ov::genai::ModelDesc& model_desc) : Eagle3InferWrapperBase(model_desc) {
    eagle3::log_debug(eagle3::PipelineStep::INIT, "Target model wrapper initialized", m_verbose);
}

void Eagle3TargetWrapper::initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config) {
    const auto shape = input_ids.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] == 1,
                    "Expected input_ids shape [1, seq_len], got ",
                    eagle3::format_shape(shape));

    const int64_t* ids_data = input_ids.data<const int64_t>();
    const size_t seq_len = shape[1];
    OPENVINO_ASSERT(seq_len > 0, "Empty prompt");

    TokenIds prompt_ids(ids_data, ids_data + seq_len);
    m_sequence_group = std::make_shared<SequenceGroup>(0, prompt_ids, config, 0);

    OPENVINO_ASSERT(get_running_sequence_count() == 1,
                    "Expected single sequence after initialization, got ",
                    get_running_sequence_count());

    eagle3::log_debug(eagle3::PipelineStep::INIT,
                      "Target sequence initialized: prompt_len=" + std::to_string(seq_len),
                      m_verbose);
}

InferenceOutput Eagle3TargetWrapper::infer(const ov::Tensor& input_ids,
                                           const ov::Tensor& attention_mask,
                                           const ov::Tensor& position_ids,
                                           const ov::Tensor& eagle_tree_mask) {
    const size_t prompt_len = input_ids.get_shape()[1];

    eagle3::log_debug(eagle3::PipelineStep::ITER,
                      "Target inference: " + std::to_string(prompt_len) + " tokens",
                      m_verbose);
    eagle3::log_model_inputs(input_ids, attention_mask, position_ids, eagle_tree_mask, m_verbose);

    if (m_device == "NPU") {
        OPENVINO_ASSERT(prompt_len <= m_max_prompt_len,
                        "NPU prompt length ",
                        prompt_len,
                        " exceeds max ",
                        m_max_prompt_len);
    }

    // Set model inputs
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);
    m_request.set_tensor("eagle_tree_mask", eagle_tree_mask);

    // Execute inference
    uint64_t time_us = execute_inference();
    update_inference_time(time_us);

    // Collect outputs
    InferenceOutput output;
    output.logits = get_logits();
    output.hidden_features = get_hidden_features();

    eagle3::log_model_outputs(output.logits, output.hidden_features, m_verbose);
    eagle3::log_debug(eagle3::PipelineStep::ITER,
                      "Target inference done: " + std::to_string(time_us / 1000.0) + "ms",
                      m_verbose);

    return output;
}

InferResult Eagle3TargetWrapper::forward(const InferContext& ctx) {
    eagle3::log_debug(eagle3::PipelineStep::ITER,
                      "Target forward: input_token_count=" + std::to_string(ctx.input_token_count) + ", sample_count=" +
                          std::to_string(ctx.sample_count) + ", validate=" + std::to_string(ctx.num_tokens_to_validate),
                      m_verbose);
    // 1. Prepare inputs from sequence state
    ov::Tensor input_ids, attention_mask, position_ids, eagle_tree_mask;
    // Determine phase based on context
    InferencePhase phase =
        (ctx.num_tokens_to_validate > 0) ? InferencePhase::TARGET_VALIDATION : InferencePhase::TARGET_PREFILL;
    build_model_inputs(ctx.input_token_count,
                       input_ids,
                       attention_mask,
                       position_ids,
                       eagle_tree_mask,
                       phase,
                       ctx.iteration_id,
                       ctx.iteration_history);

    // 2. Infer
    auto output = infer(input_ids, attention_mask, position_ids, eagle_tree_mask);

    // 3. Sample (use sample_count for number of positions to sample from)
    auto sampled = sample_tokens(output.logits, ctx.input_token_count, ctx.sample_count, ctx.num_tokens_to_validate);

    // 4. Store hidden states to sequence for draft model to use
    get_current_sequence()->update_hidden_state(output.hidden_features);

    return InferResult{std::move(output), std::move(sampled)};
}

Eagle3DraftWrapper::Eagle3DraftWrapper(const ov::genai::ModelDesc& model_desc) : Eagle3InferWrapperBase(model_desc) {
    eagle3::log_debug(eagle3::PipelineStep::INIT, "Draft model wrapper initialized", m_verbose);
}

void Eagle3DraftWrapper::initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config) {
    const auto shape = input_ids.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] == 1,
                    "Expected input_ids shape [1, seq_len], got ",
                    eagle3::format_shape(shape));

    const int64_t* ids_data = input_ids.data<const int64_t>();
    const size_t total_len = shape[1];
    OPENVINO_ASSERT(total_len >= 2, "Draft model requires at least 2 tokens");

    // Draft model uses tokens[1:] (Eagle3 specific behavior)
    TokenIds draft_prompt_ids(ids_data + 1, ids_data + total_len);
    m_sequence_group = std::make_shared<SequenceGroup>(1, draft_prompt_ids, config, 0);

    OPENVINO_ASSERT(get_running_sequence_count() == 1,
                    "Expected single sequence after initialization, got ",
                    get_running_sequence_count());

    eagle3::log_debug(eagle3::PipelineStep::INIT,
                      "Draft sequence initialized: prompt_len=" + std::to_string(draft_prompt_ids.size()) +
                          " (from original " + std::to_string(total_len) + ")",
                      m_verbose);
}

InferenceOutput Eagle3DraftWrapper::infer(const ov::Tensor& input_ids,
                                          const ov::Tensor& attention_mask,
                                          const ov::Tensor& position_ids,
                                          const ov::Tensor& eagle_tree_mask,
                                          const ov::Tensor& hidden_states) {
    const size_t input_token_count = input_ids.get_shape()[1];

    eagle3::log_debug(eagle3::PipelineStep::ITER,
                      "Draft inference: " + std::to_string(input_token_count) + " tokens",
                      m_verbose);
    eagle3::log_model_inputs(input_ids, attention_mask, position_ids, eagle_tree_mask, hidden_states, m_verbose);

    // Set standard inputs
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);
    m_request.set_tensor("eagle_tree_mask", eagle_tree_mask);

    // Set hidden states (either from target model or internal)
    OPENVINO_ASSERT(hidden_states && hidden_states.get_size() > 0, "hidden_states must be provided");
    auto shape = hidden_states.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "Invalid hidden states shape: ", eagle3::format_shape(shape));

    eagle3::log_debug(eagle3::PipelineStep::ITER, "Using hidden states: " + eagle3::format_shape(shape), m_verbose);
    m_request.set_tensor("hidden_states", hidden_states);

    // Execute inference
    uint64_t time_us = execute_inference();
    update_inference_time(time_us);

    // Collect outputs
    InferenceOutput output;
    output.logits = get_logits();
    output.hidden_features = get_hidden_features();

    eagle3::log_model_outputs(output.logits, output.hidden_features, m_verbose);
    eagle3::log_debug(eagle3::PipelineStep::ITER,
                      "Draft inference done: " + std::to_string(time_us / 1000.0) + "ms",
                      m_verbose);

    return output;
}

InferResult Eagle3DraftWrapper::forward(const InferContext& ctx) {
    eagle3::log_debug(eagle3::PipelineStep::ITER,
                      "Draft forward: input_token_count=" + std::to_string(ctx.input_token_count) +
                          ", use_target_hidden=" + std::to_string(ctx.use_target_hidden) +
                          ", iteration_id=" + std::to_string(ctx.iteration_id),
                      m_verbose);
    // 1. Prepare inputs
    ov::Tensor input_ids, attention_mask, position_ids, eagle_tree_mask;
    InferencePhase phase = ctx.use_target_hidden ? InferencePhase::DRAFT_INITIAL : InferencePhase::DRAFT_ITERATION;
    build_model_inputs(ctx.input_token_count,
                       input_ids,
                       attention_mask,
                       position_ids,
                       eagle_tree_mask,
                       phase,
                       ctx.iteration_id,
                       ctx.iteration_history);

    // 2. Get hidden states from appropriate source
    ov::Tensor hidden_states;
    if (ctx.use_target_hidden) {
        // DRAFT_INITIAL: Use target model's hidden state (single position, will be used for all sequences)
        OPENVINO_ASSERT(ctx.target_sequence, "target_sequence required when use_target_hidden=true");
        hidden_states = ctx.target_sequence->get_hidden_state();
        OPENVINO_ASSERT(hidden_states && hidden_states.get_size() > 0, "Source sequence contains invalid hidden state");
    } else {
        // DRAFT_ITERATION: Simply concatenate hidden states from all sequences
        // Each sequence stores only its current hidden_state {1, 1, hidden_size}
        // We just concatenate them along dim=1: [seq0, seq1, seq2, ...] → {1, sequence_numb, hidden_size}
        auto running_sequences = m_sequence_group->get_running_sequences();
        const size_t sequence_numb = running_sequences.size();
        OPENVINO_ASSERT(sequence_numb > 0, "No running sequences");

        // Collect hidden states from all sequences
        std::vector<ov::Tensor> seq_hidden_states;
        for (size_t i = 0; i < sequence_numb; ++i) {
            auto seq_hidden = running_sequences[i]->get_hidden_state();
            OPENVINO_ASSERT(seq_hidden && seq_hidden.get_size() > 0, "Sequence ", i, " contains invalid hidden state");

            const auto& shape = seq_hidden.get_shape();
            OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && shape[1] == 1,
                            "Expected hidden state shape [1, 1, hidden_size], got: ",
                            eagle3::format_shape(shape));

            seq_hidden_states.push_back(seq_hidden);
        }

        // Concatenate all hidden states along dim=1
        hidden_states = utils::eagle3::concatenate_hidden_states(seq_hidden_states);

        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Concatenated hidden states: " + eagle3::format_shape(hidden_states.get_shape()) + " from " +
                              std::to_string(sequence_numb) + " sequence(s)",
                          m_verbose);
    }

    // 3. Infer
    auto output = infer(input_ids, attention_mask, position_ids, eagle_tree_mask, hidden_states);

    // 4. Sample
    auto sampled = sample_tokens(output.logits, ctx.input_token_count, 1);

    // 5. Store hidden states for next iteration
    // After sampling, update each sequence's hidden_state history
    auto running_sequences = m_sequence_group->get_running_sequences();
    const size_t sequence_numb = running_sequences.size();

    if (ctx.use_target_hidden) {
        // DRAFT_INITIAL: All sequences get the same hidden state (last position)
        auto next_hidden = utils::eagle3::slice_hidden_state_for_last_token(output.hidden_features);

        for (size_t i = 0; i < sequence_numb; ++i) {
            running_sequences[i]->update_hidden_state(next_hidden);
        }

        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Stored same hidden state for " + std::to_string(sequence_numb) + " sequence(s)",
                          m_verbose);
    } else {
        // DRAFT_ITERATION: Each sequence stores only its current iteration's hidden state
        // No history accumulation - just replace with the new hidden state
        // Strategy: Slice from the END (reverse order) because hardware alignment may pad at the front
        const auto& hidden_shape = output.hidden_features.get_shape();
        OPENVINO_ASSERT(hidden_shape.size() == 3 && hidden_shape[1] >= sequence_numb,
                        "Invalid hidden features shape: ",
                        eagle3::format_shape(hidden_shape),
                        ", expected seq_len >= ",
                        sequence_numb);

        const size_t output_seq_len = hidden_shape[1];

        for (size_t i = 0; i < sequence_numb; ++i) {
            // Slice from the END: position = output_seq_len - sequence_numb + i
            // This ensures we get the valid data, skipping any front padding
            const size_t slice_position = output_seq_len - sequence_numb + i;

            // Slice the new hidden state for this sequence
            auto new_hidden = utils::eagle3::slice_hidden_state_at_position(output.hidden_features, slice_position);

            // Simply store the new hidden state (no accumulation)
            running_sequences[i]->update_hidden_state(new_hidden);

            eagle3::log_debug(eagle3::PipelineStep::ITER,
                              "Sequence " + std::to_string(i) + ": stored hidden state from position " +
                                  std::to_string(slice_position),
                              m_verbose);
        }

        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Stored current hidden states for " + std::to_string(sequence_numb) + " sequence(s)",
                          m_verbose);
    }

    return InferResult{std::move(output), std::move(sampled)};
}

StatefulEagle3LLMPipeline::StatefulEagle3LLMPipeline(const ov::genai::ModelDesc& target_model_desc,
                                                     const ov::genai::ModelDesc& draft_model_desc)
    : StatefulSpeculativePipelineBase(target_model_desc.tokenizer, target_model_desc.generation_config) {
    // Initialize draft iterations from generation config
    ensure_num_assistant_tokens_is_set(m_generation_config);
    m_draft_iterations = m_generation_config.num_assistant_tokens;

    // Extract hidden_layers_list from draft model properties
    OPENVINO_ASSERT(draft_model_desc.properties.find("hidden_layers_list") != draft_model_desc.properties.end(),
                    "hidden_layers_list must be present in draft model properties");

    m_hidden_layers_to_abstract = draft_model_desc.properties.at("hidden_layers_list").as<std::vector<int32_t>>();

    OPENVINO_ASSERT(m_hidden_layers_to_abstract.size() == 3,
                    "Eagle3 requires exactly three layers for feature extraction, got: " +
                        std::to_string(m_hidden_layers_to_abstract.size()) +
                        ". Please ensure 'hidden_layers_list' is properly configured in draft model properties.");

    auto target_model = target_model_desc.model;
    auto draft_model = draft_model_desc.model;
    OPENVINO_ASSERT(target_model, "Target model must not be null");
    OPENVINO_ASSERT(draft_model, "Draft model must not be null");

    // Model preparation
    utils::eagle3::share_vocabulary(target_model, draft_model);

    auto d2t_mapping = utils::eagle3::extract_d2t_mapping_table(draft_model);
    OPENVINO_ASSERT(d2t_mapping && d2t_mapping->get_element_type() == ov::element::i64, "Invalid d2t mapping tensor");

    utils::eagle3::apply_eagle3_attention_mask_transform(draft_model);
    utils::eagle3::apply_eagle3_attention_mask_transform(target_model);

    utils::eagle3::transform_hidden_state(target_model, m_hidden_layers_to_abstract);
    utils::eagle3::move_fc_from_draft_to_main(draft_model, target_model);
    utils::eagle3::transform_hidden_state(draft_model, {-1});

    const size_t validation_window = m_draft_iterations + 1;

    // Configure and create draft model
    auto draft_desc = draft_model_desc;
    if (draft_desc.device == "NPU") {
        draft_desc.properties["NPUW_EAGLE"] = "TRUE";
        draft_desc.properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = validation_window;
        draft_desc.properties["NPUW_ONLINE_PIPELINE"] = "NONE";
        draft_desc.properties["NPUW_DEVICES"] = "CPU";
    }
    m_draft = std::make_unique<Eagle3DraftWrapper>(draft_desc);

    m_draft->set_draft_target_mapping(d2t_mapping);

    // Configure and create target model
    auto target_desc = target_model_desc;
    if (target_desc.device == "NPU") {
        target_desc.properties["NPUW_EAGLE"] = "TRUE";
        target_desc.properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = validation_window;
        target_desc.properties["NPUW_SLICE_OUT"] = "NO";
        target_desc.properties["NPUW_DEVICES"] = "CPU";
    }
    m_target = std::make_unique<Eagle3TargetWrapper>(target_desc);

    eagle3::log_info("Pipeline initialized: draft_iterations=" + std::to_string(m_draft_iterations) +
                     ", validation_window=" + std::to_string(validation_window));
}

StatefulEagle3LLMPipeline::~StatefulEagle3LLMPipeline() {
    m_target->release_memory();
    m_draft->release_memory();
}

GenerationConfig StatefulEagle3LLMPipeline::resolve_generation_config(OptionalGenerationConfig generation_config) {
    // Call base class implementation to handle common defaults
    GenerationConfig config = StatefulSpeculativePipelineBase::resolve_generation_config(generation_config);

    // Apply Eagle3 specific validations
    const size_t prev_draft_iterations = m_draft_iterations;
    ensure_num_assistant_tokens_is_set(config);
    m_draft_iterations = config.num_assistant_tokens;

    // Log configuration changes
    if (m_draft_iterations != prev_draft_iterations) {
        if (m_draft_iterations == 0) {
            eagle3::log_info("Speculative decoding DISABLED (num_assistant_tokens=0)");
        } else {
            eagle3::log_debug("Draft iterations: " + std::to_string(prev_draft_iterations) + " -> " +
                                  std::to_string(m_draft_iterations),
                              is_verbose());
        }
    }

    return config;
}

EncodedResults StatefulEagle3LLMPipeline::generate_tokens(const EncodedInputs& inputs,
                                                          const GenerationConfig& config,
                                                          StreamerVariant streamer) {
    ManualTimer generate_timer("StatefulEagle3LLMPipeline::generate(EncodedInputs)");
    generate_timer.start();

    std::shared_ptr<StreamerBase> streamer_ptr = ov::genai::utils::create_streamer(streamer, m_tokenizer);

    // Extract input tensors
    ov::Tensor input_ids, attention_mask;
    if (auto* tensor_input = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *tensor_input;
        attention_mask = ov::genai::utils::init_attention_mask(input_ids);
    } else if (auto* tokenized_input = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = tokenized_input->input_ids;
        attention_mask = tokenized_input->attention_mask;
    }

    OPENVINO_ASSERT(input_ids.get_shape()[0] == 1, "Only batch size 1 supported");
    m_prompt_length = input_ids.get_shape()[1];

    // Initialize position IDs
    ov::Tensor position_ids{ov::element::i64, input_ids.get_shape()};
    utils::initialize_position_ids(position_ids, attention_mask);

    eagle3::log_debug("=== GENERATION START ===", is_verbose());
    eagle3::log_debug("Prompt length: " + std::to_string(m_prompt_length) +
                          ", max_new_tokens: " + std::to_string(config.max_new_tokens) +
                          ", draft_iterations: " + std::to_string(m_draft_iterations),
                      is_verbose());

    // Reset model states
    m_target->reset_state();
    m_draft->reset_state();

    // Prepare sampling config with extended max_new_tokens to prevent premature termination
    // during draft generation. Actual length control is in the generation loop.
    auto sampling_config = config;
    sampling_config.max_new_tokens = config.max_new_tokens + m_draft_iterations + 1;

    m_draft->initialize_sequence(input_ids, sampling_config);

    // For tree search mode, temporarily disable tree expansion during initial token generation
    // We need to modify the GenerationConfig inside the SequenceGroup (not the local sampling_config)
    const bool is_tree_search_mode = sampling_config.is_tree_search();
    size_t original_tree_depth = 0;
    if (is_tree_search_mode) {
        original_tree_depth = sampling_config.eagle_tree_params.tree_depth;
        sampling_config.eagle_tree_params.tree_depth = 0;
        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Prefill: temporarily set tree_depth to 0 (original: " + std::to_string(original_tree_depth) +
                              ") for initial token generation",
                          is_verbose());
    }

    // Initialize sequences with sampling config
    m_target->initialize_sequence(input_ids, sampling_config);

    // Phase 1: Initial Prompt Processing (Prefill)
    eagle3::log_generation_step("PREFILL", 0, is_verbose());

    // Prefill: process all prompt tokens from sequence
    InferContext prefill_ctx;
    prefill_ctx.input_token_count = m_prompt_length;
    auto prefill_result = m_target->forward(prefill_ctx);
    OPENVINO_ASSERT(prefill_result.sampled_tokens.size() == 1, "Expected single token from prefill");
    auto initial_token = prefill_result.sampled_tokens[0];

    // Append initial token to draft model
    m_draft->append_tokens({initial_token});

    // Restore original tree_depth after initial token generation
    if (is_tree_search_mode) {
        auto& target_params =
            const_cast<ov::genai::GenerationConfig&>(m_target->get_sequence_group()->get_sampling_parameters());
        target_params.eagle_tree_params.tree_depth = original_tree_depth;
        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Prefill: restored tree_depth to " + std::to_string(original_tree_depth),
                          is_verbose());
    }

    eagle3::log_debug("Initial token: " + std::to_string(initial_token), is_verbose());
    eagle3::log_sequence_state("after prefill",
                               m_prompt_length,
                               m_target->get_sequence_length(),
                               m_draft->get_sequence_length(),
                               m_target->get_generated_tokens(),
                               m_draft->get_generated_tokens(),
                               is_verbose());

    auto streaming_status = stream_generated_tokens(streamer_ptr, {initial_token});

    // Phase 2: Speculative Decoding Loop
    size_t generated_tokens = 1;
    size_t total_draft_accepted = 0;
    size_t total_draft_generated = 0;
    bool eos_reached = false;

    size_t input_token_count = m_draft->get_sequence_length();

    while (!eos_reached && generated_tokens < config.max_new_tokens &&
           m_target->get_sequence_length() < m_prompt_length + config.max_new_tokens &&
           streaming_status == ov::genai::StreamingStatus::RUNNING) {
        eagle3::log_generation_step("SPECULATIVE ITERATION", generated_tokens, is_verbose());

        auto result = run_speculative_iteration(input_token_count, static_cast<int64_t>(config.eos_token_id));

        streaming_status = stream_generated_tokens(streamer_ptr, result.validated_tokens);

        // Update statistics
        total_draft_generated += m_draft_iterations;
        total_draft_accepted += result.accepted_tokens_count;
        eos_reached = result.eos_reached;
        generated_tokens++;

        // Prepare for next iteration (hidden states are stored in sequence)
        input_token_count = result.next_window_size;

        eagle3::log_debug("Iteration complete: accepted=" + std::to_string(result.accepted_tokens_count) + "/" +
                              std::to_string(m_draft_iterations) + ", eos=" + std::to_string(result.eos_reached),
                          is_verbose());
        eagle3::log_sequence_state("after iteration " + std::to_string(generated_tokens),
                                   m_prompt_length,
                                   m_target->get_sequence_length(),
                                   m_draft->get_sequence_length(),
                                   m_target->get_generated_tokens(),
                                   m_draft->get_generated_tokens(),
                                   is_verbose());
    }

    // Phase 3: Finalization
    m_streaming_was_cancelled = (streaming_status == ov::genai::StreamingStatus::CANCEL);
    if (streamer_ptr)
        streamer_ptr->end();

    // Collect results
    EncodedResults results;
    results.tokens = {m_target->get_generated_tokens()};
    results.scores = {0.0f};

    generate_timer.end();

    // Update performance metrics
    m_sd_perf_metrics.num_input_tokens = m_prompt_length;
    m_sd_perf_metrics.load_time = m_load_time_ms;
    m_sd_perf_metrics.num_accepted_tokens = total_draft_accepted;
    m_sd_perf_metrics.raw_metrics.generate_durations.clear();
    m_sd_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_timer.get_duration_microsec());

    // Reset evaluated flags before updating raw_metrics to ensure statistics are recalculated
    m_sd_perf_metrics.m_evaluated = false;
    m_sd_perf_metrics.main_model_metrics.m_evaluated = false;
    m_sd_perf_metrics.draft_model_metrics.m_evaluated = false;

    m_sd_perf_metrics.main_model_metrics.raw_metrics = m_target->get_raw_perf_metrics();
    m_sd_perf_metrics.draft_model_metrics.raw_metrics = m_draft->get_raw_perf_metrics();

    if (total_draft_generated > 0) {
        float acceptance_rate = static_cast<float>(total_draft_accepted) / total_draft_generated * 100.0f;
        m_sd_metrics.update_acceptance_rate(0, acceptance_rate);
        m_sd_metrics.update_draft_accepted_tokens(0, total_draft_accepted);
        m_sd_metrics.update_draft_generated_len(0, total_draft_generated);
        m_sd_metrics.update_generated_len(generated_tokens);
    }

    m_sd_perf_metrics.evaluate_statistics(generate_timer.get_start_time());
    results.perf_metrics = m_sd_perf_metrics;
    results.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_sd_perf_metrics);

    // Reset timer
    generate_timer.clear();

    eagle3::log_debug("=== GENERATION END ===", is_verbose());
    return results;
}

StatefulEagle3LLMPipeline::SpeculativeResult StatefulEagle3LLMPipeline::run_speculative_iteration(
    size_t input_token_count,
    int64_t eos_token_id) {
    SpeculativeResult result;

    OPENVINO_ASSERT(m_target->get_running_sequence_count() == 1 && m_draft->get_running_sequence_count() == 1,
                    "Eagle3 speculative iteration requires single sequence per model");

    auto target_hidden_states = m_target->get_current_sequence()->get_hidden_state();
    OPENVINO_ASSERT(target_hidden_states && target_hidden_states.get_size() > 0,
                    "Target model contains invalid hidden state for speculation");

    eagle3::log_debug("--- Draft Phase Start ---", is_verbose());
    eagle3::log_debug("Input: input_token_count=" + std::to_string(input_token_count) +
                          ", hidden_shape=" + eagle3::format_shape(target_hidden_states.get_shape()),
                      is_verbose());

    // Record pre-draft sequence lengths for potential rollback
    const size_t pre_draft_token_len = m_draft->get_sequence_length();

    // Create iteration history to track all generated tokens across iterations
    auto iteration_history = std::make_shared<std::vector<int64_t>>();
    iteration_history->reserve(m_draft_iterations * 10);  // Reserve space for efficiency

    // Step 1: Generate first draft token using target hidden states
    InferContext first_ctx;
    first_ctx.input_token_count = input_token_count;
    first_ctx.use_target_hidden = true;
    first_ctx.target_sequence = m_target->get_current_sequence();
    first_ctx.iteration_id = 0;
    first_ctx.iteration_history = iteration_history;
    auto first_result = m_draft->forward(first_ctx);

    // first_result.sampled_tokens contains tokens from all sequences (one token per sequence)
    const auto& first_draft_tokens = first_result.sampled_tokens;
    OPENVINO_ASSERT(!first_draft_tokens.empty(), "Expected at least one token from first draft");

    eagle3::log_debug("First draft tokens (" + std::to_string(first_draft_tokens.size()) +
                          " sequence(s)): " + eagle3::format_tokens(first_draft_tokens),
                      is_verbose());

    // Collect draft candidates - store all tokens from first iteration
    std::vector<int64_t> draft_candidates;
    draft_candidates.reserve(m_draft_iterations * first_draft_tokens.size());
    draft_candidates.insert(draft_candidates.end(), first_draft_tokens.begin(), first_draft_tokens.end());

    // Record first iteration tokens in history
    iteration_history->insert(iteration_history->end(), first_draft_tokens.begin(), first_draft_tokens.end());

    // Append first tokens to target model (draft model already has them from sampler)
    m_target->append_tokens(first_draft_tokens);

    // Step 2: Generate additional draft tokens using internal hidden states
    for (size_t i = 1; i < m_draft_iterations; ++i) {
        InferContext more_ctx;
        more_ctx.input_token_count = 1;  // This will be updated by build_model_inputs based on num sequences
        more_ctx.use_target_hidden = false;
        more_ctx.iteration_id = i;                       // Pass the current iteration index
        more_ctx.iteration_history = iteration_history;  // Share the history
        auto more_result = m_draft->forward(more_ctx);

        const auto& draft_tokens = more_result.sampled_tokens;
        OPENVINO_ASSERT(!draft_tokens.empty(), "Expected tokens from draft iteration ", i);

        eagle3::log_debug("Draft iteration " + std::to_string(i) + " tokens (" + std::to_string(draft_tokens.size()) +
                              " sequence(s)): " + eagle3::format_tokens(draft_tokens),
                          is_verbose());

        // Collect all tokens from this iteration
        draft_candidates.insert(draft_candidates.end(), draft_tokens.begin(), draft_tokens.end());

        // Record this iteration's tokens in history
        iteration_history->insert(iteration_history->end(), draft_tokens.begin(), draft_tokens.end());

        // Append draft tokens to target sequence for validation phase
        // During validation, target model will retrieve tokens from its own sequence
        // so we need to speculatively add draft predictions here
        m_target->append_tokens(draft_tokens);
    }

    eagle3::log_debug("Draft candidates: " + eagle3::format_tokens(draft_candidates), is_verbose());
    eagle3::log_debug("--- Draft Phase End ---", is_verbose());

    // Step 3: Validate draft tokens with target model
    eagle3::log_debug("--- Validation Phase Start ---", is_verbose());

    const size_t validation_window_size = m_draft_iterations + 1;

    InferContext val_ctx;
    val_ctx.input_token_count = validation_window_size;
    val_ctx.sample_count = validation_window_size;
    val_ctx.num_tokens_to_validate = m_draft_iterations;
    auto val_result = m_target->forward(val_ctx);

    // Sampler validates draft tokens and returns accepted + new sampled token
    auto validated_tokens = val_result.sampled_tokens;

    // Result: [accepted_draft_tokens..., new_sampled_token]
    const size_t accepted_count = validated_tokens.size() - 1;
    const int64_t target_predicted_token = validated_tokens.back();
    const size_t tokens_to_remove = m_draft_iterations - accepted_count;
    const size_t total_accepted_tokens = validated_tokens.size();

    eagle3::log_debug("Validation result: accepted=" + std::to_string(accepted_count) + "/" +
                          std::to_string(m_draft_iterations) + ", new_token=" + std::to_string(target_predicted_token),
                      is_verbose());
    eagle3::log_debug("Validated tokens: " + eagle3::format_tokens(validated_tokens), is_verbose());

    // Step 4: Synchronize sequences and KV cache
    // Target model's sequence is already updated by Sampler
    // Sync draft model's sequence
    m_draft->truncate_sequence(pre_draft_token_len);
    m_draft->append_tokens(validated_tokens);

    // Trim KV cache for rejected tokens
    if (tokens_to_remove > 0) {
        m_target->trim_kv_cache(tokens_to_remove);
        m_draft->trim_kv_cache(tokens_to_remove);
        eagle3::log_debug("KV cache trimmed: removed " + std::to_string(tokens_to_remove) + " rejected tokens",
                          is_verbose());
    }

    // Step 5: Update hidden states for next iteration
    // Note: forward() already stored hidden_features to sequence, but we need to slice it
    auto current_hidden = val_result.output.hidden_features;
    OPENVINO_ASSERT(current_hidden && current_hidden.get_size() > 0, "Missing hidden features");

    const auto h_shape = current_hidden.get_shape();
    OPENVINO_ASSERT(h_shape.size() == 3 && h_shape[0] == 1 && h_shape[1] >= total_accepted_tokens,
                    "Invalid hidden state shape: ",
                    eagle3::format_shape(h_shape),
                    ", expected seq_len >= ",
                    total_accepted_tokens);

    // Store sliced hidden states (only accepted tokens) for next iteration
    auto [start_coord, end_coord] = ov::genai::utils::make_roi(h_shape, 1, 0, total_accepted_tokens);
    auto next_hidden = ov::Tensor(current_hidden, start_coord, end_coord);
    m_target->get_current_sequence()->update_hidden_state(next_hidden);

    result.accepted_tokens_count = accepted_count;
    result.next_window_size = accepted_count + 1;
    result.validated_tokens = std::move(validated_tokens);
    result.eos_reached = (target_predicted_token == eos_token_id);

    eagle3::log_debug("--- Validation Phase End ---", is_verbose());
    return result;
}

void StatefulEagle3LLMPipeline::finish_chat() {
    // Eagle3 uses base class implementation directly (no model state reset needed)
    StatefulSpeculativePipelineBase::finish_chat();
}

SpeculativeDecodingMetrics StatefulEagle3LLMPipeline::get_speculative_decoding_metrics() const {
    return m_sd_metrics;
}

}  // namespace ov::genai
