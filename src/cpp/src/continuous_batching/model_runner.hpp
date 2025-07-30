// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdlib>

#include <openvino/runtime/infer_request.hpp>

#include "visual_language/embedding_model.hpp"
#include "sequence_group.hpp"
#include "continuous_batching/scheduler.hpp"
#include "continuous_batching/timer.hpp"

#include "continuous_batching/attention_output.hpp"

namespace ov::genai {

inline std::string get_paged_attention_score_output_for_decoder_layer(size_t decoder_layer_id) {
    std::stringstream ss;
    ss << "scores." << decoder_layer_id;
    return ss.str();
}

/**
 * @brief Runs the LLM infer request, parsing the continuous batching scheduler output into proper inputs in terms of OV API (e.g. token input IDs,
 * KV cache block indices etc.) and returning the logit scores for the next token to be generated for each of the currently scheduled sequences.
 */
class ModelRunner {
    ov::InferRequest m_request;
    AttentionScoresForEachSubsequence m_last_attention_scores;
    size_t m_block_size;
    size_t m_num_decoder_layers;
    bool m_collect_attention_scores;
    bool m_is_use_per_layer_cache_control;
    bool m_is_use_rotation_inputs;
    std::vector<std::map<size_t, std::vector<size_t>>> m_rotated_block_logical_indices_per_sequence_for_each_layer;
    std::vector<ov::Tensor> m_cache_rotation_deltas_for_each_layer;
    ov::Tensor m_cache_rotation_trig_lut;

    bool m_is_aggregate_attention_scores;

    // A model to compute token embeddings.
    // Input shape: [N, conversation length].
    // Output shape: [1, conversation length, hidden_size].
    EmbeddingsModel::Ptr m_embedding;
    bool m_is_hidden_state_export_needed = false; // need to export hidden state after inference
    bool m_is_hidden_state_import_needed = false; // need to import hidden state from another model runner
    bool m_is_hidden_state_internal_needed = false; // need to use internal hidden state, e.g, eagle2
    std::map<std::pair<size_t, size_t>, std::pair<size_t, size_t>> m_sequence_hidden_state_mapping; // pre-requisite: main/draft have same seq group and running seq grouped id
    // a container which use sequence group id and request id as key to store hidden states
    std::map<std::pair<size_t, size_t>, ov::Tensor> m_initial_hidden_states; // shape: [N, seq_len, hidden_size]
public:
    /**
     * Constructs the ModelRunner.
     * @param request The ov::InferRequest for the LLM to be inferred in the continuous batching mode.
     * @param num_decoder_layers Number of decoder attention layers in the LLM corresponding to the request.
     * @param collect_attention_scores If true, then after each `forward` call the ModelRunner will collect and make
     * available the per-token attention scores for each decoder layer, so that these can be used in per-step cache
     * optimizations (such as cache eviction algorithm).
     * @param is_use_per_layer_cache_control If true, then the runner will pass cache control input tensors to the model
     * on a per-attention layer basis.
     * @param is_use_rotation_inputs If true, then the runner will pass cache rotation input tensors to the model
     * on a per-attention layer basis.
     * @param is_aggregate_attention_scores If true, then the runner will pass the input tensors containing per-sequence
     * score aggregation window sizes to the model as requested by the scheduler.
     * on a per-attention layer basis.
     */
    ModelRunner(ov::InferRequest request,
                size_t block_size,
                size_t num_decoder_layers = 1,
                bool collect_attention_scores = false,
                bool is_use_per_layer_cache_control = false,
                bool is_use_rotation_inputs = false,
                bool is_aggregate_attention_scores = false)
        : m_request(std::move(request)),
          m_block_size(block_size),
          m_num_decoder_layers(num_decoder_layers),
          m_collect_attention_scores(collect_attention_scores),
          m_is_use_per_layer_cache_control(is_use_per_layer_cache_control),
          m_is_use_rotation_inputs(is_use_rotation_inputs),
          m_rotated_block_logical_indices_per_sequence_for_each_layer(num_decoder_layers),
          m_is_aggregate_attention_scores(is_aggregate_attention_scores) {
        OPENVINO_ASSERT(m_num_decoder_layers != 0, "num_decoder_layers must be non-zero");
        _reset_cache_rotation_coefficients();
    }

    /**
     * @return The ov::InferRequest this ModelRunner is handling.
     */
    ov::InferRequest get_infer_request() {
        return m_request;
    }

    void set_hidden_state_export_needed(bool is_needed) {
        m_is_hidden_state_export_needed = is_needed;
    }

    void set_hidden_state_import_needed(bool is_needed) {
        m_is_hidden_state_import_needed = is_needed;
    }

    void set_hidden_state_internal_needed(bool is_needed) {
        m_is_hidden_state_internal_needed = is_needed;
    }

    void set_embedding_model(const EmbeddingsModel::Ptr& embedder) {
        m_embedding = embedder;
    }

    /**
     * @return A map of sequence IDs to vectors of ov::Tensor per-token attention scores. Each vector element is associated with its own
     * decoder layer, in order of their execution in the model. Each ov::Tensor has a shape of {N_k}, where N_k is the length of
     * a sequence with ID k processed during the previous `forward` call.
     */
    const AttentionScoresForEachSubsequence& get_last_attention_scores() const {
        return m_last_attention_scores;
    }


    void set_cache_rotation_trig_lut(ov::Tensor&& rotation_trig_lut) {
        m_cache_rotation_trig_lut = std::move(rotation_trig_lut);
    }

    void set_cache_rotation_data(std::vector<std::map<size_t, std::vector<size_t>>>&&
                                     rotated_logical_block_indices_per_sequence_for_each_layer,
                                 std::vector<ov::Tensor>&& rotation_deltas_for_each_layer) {
        m_rotated_block_logical_indices_per_sequence_for_each_layer =
            std::move(rotated_logical_block_indices_per_sequence_for_each_layer);
        m_cache_rotation_deltas_for_each_layer = std::move(rotation_deltas_for_each_layer);
    }

    ov::Tensor get_hidden_state(size_t request_id, size_t seq_grouped_id) const {
        if (m_hidden_states.get_size() == 0) {
            return ov::Tensor();
        }

        auto key = std::make_pair(request_id, seq_grouped_id);
        auto it = m_sequence_hidden_state_mapping.find(key);
        if (it == m_sequence_hidden_state_mapping.end()) {
            return ov::Tensor();
        }

        size_t start_idx = it->second.first;
        size_t length = it->second.second;

        auto shape = m_hidden_states.get_shape();
        if (shape.size() < 2) {
            return ov::Tensor();
        }

        size_t hidden_size = shape[shape.size() - 1];

        ov::Coordinate start_coord(shape.size(), 0);
        ov::Coordinate end_coord(shape.size(), 0);

        start_coord[0] = start_idx;
        end_coord[0] = start_idx + length;

        for (size_t i = 1; i < shape.size(); ++i) {
            start_coord[i] = 0;
            end_coord[i] = shape[i];
        }

        return ov::Tensor(m_hidden_states, start_coord, end_coord);
    }

    void set_initial_hidden_state(size_t request_id, size_t seq_grouped_id, const ov::Tensor& hidden_state) {
        // m_initial_hidden_states.clear();
        auto key = std::make_pair(request_id, seq_grouped_id);
        m_initial_hidden_states[key] = hidden_state;
    }

    /**
     * Runs the forward inference call on the underlying LLM's ov::InferRequest, scheduling for inferencing tokens for given sequences
     * taking into account the supplied scheduler output struct.
     * @param sequence_groups A vector of pointers to sequence groups to be processed during this `forward` call
     * @param scheduler_output The scheduler output struct with information on the specifics of the token scheduling during this forward call
     * @return An ov::Tensor with next-token logit scores for each sequence processed during this `forward` call.
     */
    ov::Tensor forward(const std::vector<SequenceGroup::Ptr> & sequence_groups, const Scheduler::Output& scheduler_output) {
        m_sequence_hidden_state_mapping.clear();

        size_t num_sequence_groups = scheduler_output.m_scheduled_sequence_groups_ids.size();
        size_t batch_size_in_sequences = 0;
        size_t total_num_tokens = 0, total_num_blocks = 0;
        size_t max_context_len_val = 0;
        size_t hidden_size = 0;
        OPENVINO_ASSERT(sequence_groups.size() > 0);
        auto sequence_group_type = sequence_groups[0]->get_sequence_group_type();
        if (sequence_group_type == SequenceGroupType::EMBEDDINGS) {
            hidden_size = sequence_groups[0]->get_hidden_size();
        }

        // compute aggregated values
        for (size_t i = 0; i < num_sequence_groups; ++i) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
            size_t num_sequences = sequence_group->num_running_seqs();
            batch_size_in_sequences += num_sequences;
            total_num_tokens += sequence_group->get_num_scheduled_tokens() * num_sequences;
            total_num_blocks += sequence_group->get_num_blocks() * num_sequences;
            max_context_len_val = std::max(max_context_len_val, sequence_group->get_context_len());
        }

        ov::Tensor
            input_ids(ov::element::i64, {total_num_tokens}),
            inputs_embeds(ov::element::f32, {total_num_tokens, hidden_size}),
            position_ids(ov::element::i64, {total_num_tokens}),
            // PA specific parameters
            past_lens(ov::element::i32, {batch_size_in_sequences}),
            subsequence_begins(ov::element::i32, {batch_size_in_sequences + 1}),
            // block_indices are handled in a special fashion below
            block_indices_begins(ov::element::i32, {batch_size_in_sequences + 1}),
            max_context_len(ov::element::i32, {});
        ov::Tensor hidden_state_input;
        float* hidden_state_data = nullptr;
        if (m_is_hidden_state_import_needed || m_is_hidden_state_internal_needed) {
            if (hidden_size == 0) {
                for (const auto& entry : m_initial_hidden_states) {
                    const auto& stored_hidden_state = entry.second;
                    if (stored_hidden_state.get_size() > 0) {
                        auto shape = stored_hidden_state.get_shape();
                        if (shape.size() >= 2) {
                            hidden_size = shape[shape.size() - 1];
                            break;
                        }
                    }
                }
            }
            if (hidden_size > 0) {
                hidden_state_input = ov::Tensor(ov::element::f32, {total_num_tokens, 1, hidden_size});
                hidden_state_data = hidden_state_input.data<float>();
                std::memset(hidden_state_data, 0, total_num_tokens * hidden_size * sizeof(float));
            }
        }

        ov::Tensor score_aggregation_window(ov::element::i32, {batch_size_in_sequences});

        ov::Tensor generated_ids_embeds;
        float *generated_ids_embeds_data = nullptr;

        max_context_len.data<int32_t>()[0] = max_context_len_val;

        // get raw pointers to copy to
        float *inputs_embeds_data = nullptr;
        int64_t *input_ids_data = nullptr;

        if (sequence_group_type == SequenceGroupType::EMBEDDINGS) {
            inputs_embeds_data = inputs_embeds.data<float>();
        } else if (sequence_group_type == SequenceGroupType::TOKENS) {
            input_ids_data = input_ids.data<int64_t>();
        }

        int64_t
            * position_ids_data = position_ids.data<int64_t>();

        int32_t
            * past_lens_data = past_lens.data<int32_t>(),
            * subsequence_begins_data = subsequence_begins.data<int32_t>(),
            * block_indices_begins_data = block_indices_begins.data<int32_t>(),
            * score_aggregation_window_data = score_aggregation_window.data<int32_t>();

        // sub-sequence data starts with 0
        subsequence_begins_data[0] = 0;
        block_indices_begins_data[0] = 0;

        bool matmul_gathering_is_available = false;
        size_t gathering_current_index = 0;
        std::vector<int64_t> gather_indices_values;
        try {
            std::ignore = m_request.get_tensor("sampled_tokens_indices");
            matmul_gathering_is_available = true;
        } catch (const ov::Exception&) {}

        size_t current_token_idx = 0;

        for (size_t i = 0; i < num_sequence_groups; ++i) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::Ptr sequence_group = sequence_groups[seq_group_id];
            std::vector<Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
            size_t num_running_sequences = running_sequences.size();
            size_t num_scheduled_tokens = sequence_group->get_num_scheduled_tokens();
            size_t group_position_id = sequence_group->get_num_processed_tokens();
            size_t prompt_len = sequence_group->get_prompt_len();

            // Next variables are only for sliced matmul case
            size_t output_seq_len = 0;
            const bool echo_output = sequence_group->get_sampling_parameters().echo;
            const bool sampling_is_required = sequence_group->requires_sampling();
            const size_t tokens_to_sample_per_sequence = 1 + sequence_group->get_num_tokens_to_validate();

            for (size_t seq_idx = 0; seq_idx < num_running_sequences; ++seq_idx) {
                output_seq_len = 0;
                Sequence::CPtr sequence = running_sequences[seq_idx];
                if (m_is_hidden_state_export_needed) {
                    size_t start_token_idx = current_token_idx;
                    size_t sequence_length = num_scheduled_tokens;

                    auto key = std::make_pair(sequence_group->get_request_id(), sequence->get_grouped_id());
                    m_sequence_hidden_state_mapping[key] = std::make_pair(start_token_idx, sequence_length);
                }
                if (m_is_hidden_state_import_needed && hidden_state_data && hidden_size > 0) {
                    auto key = std::make_pair(sequence_group->get_request_id(), sequence->get_grouped_id());
                    auto it = m_initial_hidden_states.find(key);

                    if (it != m_initial_hidden_states.end()) {
                        const auto& stored_hidden_state = it->second;

                        if (stored_hidden_state.get_size() > 0) {
                            auto stored_shape = stored_hidden_state.get_shape();

                            if (stored_shape.size() >= 2) {
                                size_t stored_seq_len = stored_shape[0];
                                size_t stored_hidden_size = stored_shape[stored_shape.size() - 1];

                                if (stored_hidden_size == hidden_size) {
                                    if (stored_seq_len == total_num_tokens) {
                                        hidden_state_input = stored_hidden_state;  // all tokens from eagle is accepted
                                    } else {
                                        size_t copy_length = std::min(stored_seq_len, num_scheduled_tokens);

                                        size_t source_start_idx =
                                            stored_seq_len >= copy_length ? stored_seq_len - copy_length : 0;

                                        const float* source_data = stored_hidden_state.data<float>();
                                        float* target_data = hidden_state_data + current_token_idx * hidden_size;

                                        for (size_t token_offset = 0; token_offset < copy_length; ++token_offset) {
                                            size_t source_offset = (source_start_idx + token_offset) * hidden_size;
                                            size_t target_offset = token_offset * hidden_size;

                                            std::copy_n(source_data + source_offset,
                                                        hidden_size,
                                                        target_data + target_offset);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // fill hidden_state_data with m_hidden_states
                    if (hidden_state_data) {
                        std::memset(hidden_state_data + current_token_idx * hidden_size,
                                    0,
                                    num_scheduled_tokens * hidden_size * sizeof(float));
                        auto hidden_state = running_sequences[seq_idx]->get_hidden_state();
                        if (hidden_state.get_size() > 0) {
                            auto shape = hidden_state.get_shape();
                            if (shape.size() >= 2 && shape[shape.size() - 1] == hidden_size) {
                                size_t seq_len = shape[0];
                                size_t copy_length = std::min(seq_len, num_scheduled_tokens);
                                const float* source_data = hidden_state.data<float>();
                                float* target_data = hidden_state_data + current_token_idx * hidden_size;
                                for (size_t token_offset = 0; token_offset < copy_length; ++token_offset) {
                                    size_t source_offset = (seq_len - token_offset - 1) * hidden_size;
                                    size_t target_offset = token_offset * hidden_size;
                                    std::copy_n(source_data + source_offset, hidden_size, target_data + target_offset);
                                }
                            }
                        }
                    }
                }
                for (size_t token_id = 0, position_id = group_position_id; token_id < num_scheduled_tokens; ++token_id, ++position_id, ++gathering_current_index) {
                    // compute token for current sequence
                    if (sequence_group_type == SequenceGroupType::TOKENS) {
                        input_ids_data[token_id] = position_id < prompt_len ?
                            sequence_group->get_prompt_ids()[position_id] :
                            sequence->get_generated_ids()[position_id - prompt_len];
                    } else if (sequence_group_type == SequenceGroupType::EMBEDDINGS) {
                        const auto& generated_embeds = sequence->get_generated_ids_embeds();
                        const float* src = position_id < prompt_len ? sequence_group->get_input_embeds()[position_id].data() :  generated_embeds[position_id - prompt_len].data();
                        std::copy_n(src, hidden_size, inputs_embeds_data + token_id * hidden_size);
                    } else {
                        OPENVINO_THROW("Unknown model inputs type.");
                    }

                    position_ids_data[token_id] = position_id;

                    // Check if token gathering is required for the entire sequence group
                    if (matmul_gathering_is_available && (sampling_is_required || echo_output)) {
                        // Determine if the current token should be gathered
                        if (echo_output ||
                            // Skip gathering for prompt tokens
                            group_position_id + token_id >= prompt_len - 1 &&
                            // Gather only the last scheduled token or 1 + num_tokens_to_validate tokens for SD
                            // In SD, tokens_to_sample_per_sequence may exceed num_scheduled_tokens
                            token_id + tokens_to_sample_per_sequence >= num_scheduled_tokens) {
                            gather_indices_values.push_back(gathering_current_index);
                            output_seq_len++;
                        }
                    }
                }

                size_t expected_kv_cache_size = sequence_group->get_num_processed_tokens() - sequence_group->get_num_evicted_tokens();
                past_lens_data[0] = expected_kv_cache_size;

                subsequence_begins_data[1] = subsequence_begins_data[0] + num_scheduled_tokens;

                size_t num_blocks = (sequence_group->get_context_len()  - sequence_group->get_num_evicted_tokens() +  m_block_size - 1) / m_block_size;
                block_indices_begins_data[1] = block_indices_begins_data[0] + num_blocks;

                // apply strides to shift to a next sequence
                if (sequence_group_type == SequenceGroupType::TOKENS) {
                    input_ids_data += num_scheduled_tokens;
                } else if (sequence_group_type == SequenceGroupType::EMBEDDINGS) {
                    inputs_embeds_data += num_scheduled_tokens * hidden_size;
                }


                if (m_is_aggregate_attention_scores) {
                    size_t seq_id = sequence->get_id();
                    auto it = scheduler_output.m_score_aggregation_windows.find(seq_id);
                    if (it != scheduler_output.m_score_aggregation_windows.end()) {
                        *score_aggregation_window_data = it->second; // the prompt has reached the SnapKV window, either fully or partially
                    }
                    else {
                        // either the prompt has not reached the SnapKV window yet (in which case we will disregard the scores anyway),
                        // or the sequence is in the generation stage already
                        *score_aggregation_window_data = 1;
                    }
                }
                current_token_idx += num_scheduled_tokens;;
                position_ids_data += num_scheduled_tokens;
                past_lens_data += 1;
                subsequence_begins_data += 1;
                block_indices_begins_data += 1;
                score_aggregation_window_data += 1;
            }
            sequence_group->set_output_seq_len(matmul_gathering_is_available ? output_seq_len : num_scheduled_tokens);
        }

        if (sequence_group_type == SequenceGroupType::TOKENS) {
            m_request.set_tensor("input_ids", input_ids);
        }
        else if (sequence_group_type == SequenceGroupType::EMBEDDINGS) {
            m_request.set_tensor("inputs_embeds", inputs_embeds);
        }
        if (hidden_state_input && hidden_state_input.get_size() > 0) {
            try {
                m_request.set_tensor("target_hidden_state_input", hidden_state_input);
            } catch (const ov::Exception& e) {
            }
        }
        // typical LLM parameters
        m_request.set_tensor("position_ids", position_ids);

        // PA specific parameters
        m_request.set_tensor("past_lens", past_lens);
        m_request.set_tensor("subsequence_begins", subsequence_begins);

        _set_block_indices(sequence_groups, scheduler_output, total_num_blocks);
        m_request.set_tensor("block_indices_begins", block_indices_begins);
        m_request.set_tensor("max_context_len", max_context_len);

        if (m_is_use_rotation_inputs) {
            m_request.set_tensor("rotation_trig_lut", m_cache_rotation_trig_lut);
            _set_cache_rotation_coefficients(sequence_groups, scheduler_output);
        }

        if (matmul_gathering_is_available) {
            ov::Tensor gather_indices(ov::element::i64, {gather_indices_values.size()});
            std::memcpy(gather_indices.data(), gather_indices_values.data(), gather_indices_values.size() * sizeof(int64_t));
            m_request.set_tensor("sampled_tokens_indices", gather_indices);
        }

        if (m_is_aggregate_attention_scores) {
            m_request.set_tensor("score_aggregation_window", score_aggregation_window);
        }

        {
            static ManualTimer timer("pure generate inference");
            timer.start();
            m_request.infer();
            timer.end();
        }

        if (m_collect_attention_scores) {
            _collect_attention_scores(sequence_groups, scheduler_output);
        }
        _reset_cache_rotation_coefficients();
        if (m_is_hidden_state_export_needed) {
            try {
                m_hidden_states = m_request.get_tensor("last_hidden_state");
                for (size_t i = 0; i < num_sequence_groups; ++i) {
                    size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
                    SequenceGroup::Ptr sequence_group = sequence_groups[seq_group_id];
                    std::vector<Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
                    for (size_t seq_idx = 0; seq_idx < running_sequences.size(); ++seq_idx) {
                        Sequence::Ptr sequence = running_sequences[seq_idx];
                        sequence->update_hidden_state(
                            get_hidden_state(sequence_group->get_request_id(), sequence->get_grouped_id()));
                    }
                }
            } catch (const ov::Exception&) {
                m_hidden_states = ov::Tensor();
            }
        }

        // return logits
        return m_request.get_tensor("logits");
    }

    void append_embeddings(const std::vector<SequenceGroup::Ptr> & sequence_groups, const Scheduler::Output& scheduler_output) {
        size_t num_sequence_groups = scheduler_output.m_scheduled_sequence_groups_ids.size();
        size_t num_generated_ids_without_embeddings = 0;
        OPENVINO_ASSERT(sequence_groups.size() > 0);

        // compute aggregated values
        for (size_t i = 0; i < num_sequence_groups; ++i) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
            size_t num_sequences = sequence_group->num_running_seqs();
            OPENVINO_ASSERT(sequence_group->get_sequence_group_type() == SequenceGroupType::EMBEDDINGS);
            for (auto seq: sequence_group->get_running_sequences()) {
                num_generated_ids_without_embeddings += seq->get_generated_len() - seq->get_generated_ids_embeds().size();
            }
        }
        size_t hidden_size = sequence_groups[0]->get_hidden_size();

        ov::Tensor generated_ids_embeds;
        float *generated_ids_embeds_data = nullptr;

        ov::Tensor generated_ids = ov::Tensor(ov::element::i64, {1, num_generated_ids_without_embeddings});
        int64_t *generated_ids_data = generated_ids.data<int64_t>();
        size_t pos = 0;
        for (size_t i = 0; i < num_sequence_groups; ++i) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
            for (auto seq: sequence_group->get_running_sequences()) {
                const auto& generated_ids = seq->get_generated_ids();
                for (size_t token_idx = seq->get_generated_ids_embeds().size(); token_idx < generated_ids.size(); token_idx++) {
                    generated_ids_data[pos] = generated_ids[token_idx];
                    pos++;
                }
            }
        }
        if (pos > 0) {
            CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
            EmbeddingsRequest& req = embeddings_request_guard.get();
            generated_ids_embeds = m_embedding->infer(req, generated_ids);
            generated_ids_embeds_data = generated_ids_embeds.data<float>();
            size_t embeds_pos = 0;
            for (size_t i = 0; i < num_sequence_groups; ++i) {
                size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
                SequenceGroup::Ptr sequence_group = sequence_groups[seq_group_id];
                for (auto seq: sequence_group->get_running_sequences()) {
                    auto generated_ids = seq->get_generated_ids();
                    size_t new_embeds_count = seq->get_generated_len() - seq->get_generated_ids_embeds().size();
                    ov::Coordinate start{0, embeds_pos, 0};
                    ov::Coordinate end{1, embeds_pos + new_embeds_count, hidden_size};
                    ov::Tensor embedding(generated_ids_embeds, start, end);
                    seq->append_generated_ids_embeds(embedding);
                    embeds_pos += new_embeds_count;
                }
            }
        }
    }

private:
    ov::Tensor m_hidden_states;

    void _fill_indices_from_block_tables(
        const std::vector<std::string>& dst_tensor_names,
        const std::vector<SequenceGroup::Ptr>& sequence_groups,
        const Scheduler::Output& scheduler_output,
        const std::vector<std::map<size_t, std::vector<size_t>>>& seq_id_to_select_logical_idx_maps) {
        OPENVINO_ASSERT(seq_id_to_select_logical_idx_maps.size() == dst_tensor_names.size() ||
                        seq_id_to_select_logical_idx_maps.empty());
        bool is_fill_all = seq_id_to_select_logical_idx_maps.empty();
        size_t num_sequence_groups = scheduler_output.m_scheduled_sequence_groups_ids.size();
        std::vector<size_t> filled_blocks_per_layer(dst_tensor_names.size(), 0);


        for (size_t layer_idx = 0; layer_idx < dst_tensor_names.size(); layer_idx++) {
            auto input_tensor = m_request.get_tensor(dst_tensor_names[layer_idx]);
            auto block_indices_data = input_tensor.data<int32_t>();
            if (is_fill_all) {
                for (size_t i = 0; i < num_sequence_groups; ++i) {
                    size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
                    SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
                    std::vector<Sequence::CPtr> running_sequences = sequence_group->get_running_sequences();
                    size_t num_running_sequences = running_sequences.size();

                    for (size_t i = 0; i < num_running_sequences; ++i) {
                        Sequence::CPtr sequence = running_sequences[i];

                        size_t num_blocks = sequence_group->get_num_logical_blocks();
                        const auto& kv_blocks = scheduler_output.m_block_tables.at(sequence->get_id());


                        for (size_t block_id = 0; block_id < num_blocks; ++block_id) {
                            // In case no cache eviction is requested, all per-layer block tables are expected to be
                            // identical at all times
                            block_indices_data[block_id] = kv_blocks[layer_idx][block_id]->get_index();
                        }
                        block_indices_data += num_blocks;
                        filled_blocks_per_layer[layer_idx] += num_blocks;
                    }
                }
            } else {
                    // This branch will fill the block indices in the seq_id order defined by the seq_id_to_select_logical_idx_maps argument
                    auto seq_id_to_select_logical_idx_map = seq_id_to_select_logical_idx_maps[layer_idx];
                    for (const auto& kv : seq_id_to_select_logical_idx_map) {
                        size_t seq_id = kv.first;
                        auto block_table_it = scheduler_output.m_block_tables.find(seq_id);
                        OPENVINO_ASSERT(block_table_it != scheduler_output.m_block_tables.end());
                        const auto& select_logical_idxs = kv.second;
                        const auto& kv_blocks = block_table_it->second;
                        size_t block_table_size = kv_blocks[layer_idx].size();

                        for (size_t block_id = 0; block_id < select_logical_idxs.size(); ++block_id) {
                            size_t logical_block_idx = select_logical_idxs[block_id];
                            OPENVINO_ASSERT(logical_block_idx < block_table_size);

                            block_indices_data[block_id] = kv_blocks[layer_idx][logical_block_idx]->get_index();
                        }
                    block_indices_data += select_logical_idxs.size();
                    filled_blocks_per_layer[layer_idx] += select_logical_idxs.size();
                }
            }
        }
        for (size_t layer_idx = 0; layer_idx < dst_tensor_names.size(); layer_idx++) {
            const auto& target_tensor_name = dst_tensor_names[layer_idx];
            size_t tensor_size = m_request.get_tensor(target_tensor_name).get_size();
            size_t last_filled_element_idx = filled_blocks_per_layer[layer_idx];
            OPENVINO_ASSERT(tensor_size == last_filled_element_idx, "did not fill tensor ", target_tensor_name, " completely, tensor size in elements ", tensor_size, ", last filled idx ", last_filled_element_idx);
        }
    }

    void _set_block_indices(const std::vector<SequenceGroup::Ptr>& sequence_groups,
                            const Scheduler::Output& scheduler_output,
                            size_t total_num_blocks) {
        std::vector<std::string> tensor_names = {"block_indices"};

        if (m_is_use_per_layer_cache_control) {
            tensor_names.resize(m_num_decoder_layers);
            for (size_t i = 0; i < tensor_names.size(); i++) {
                tensor_names[i] = std::string("block_indices.") + std::to_string(i);
            }
        }

        for (auto& name : tensor_names) {
            m_request.get_tensor(name).set_shape({total_num_blocks});
        }

        _fill_indices_from_block_tables(tensor_names, sequence_groups, scheduler_output, {});
    }

    void _set_cache_rotation_coefficients(const std::vector<SequenceGroup::Ptr>& sequence_groups,
                                          const Scheduler::Output& scheduler_output) {
        std::vector<std::string> rotation_indices_tensor_names(m_num_decoder_layers);
        for (size_t i = 0; i < m_num_decoder_layers; i++) {
            auto tensor_name = std::string("rotated_block_indices.") + std::to_string(i);
            rotation_indices_tensor_names[i] = tensor_name;
            size_t num_indices = 0;
            for (const auto& entry : m_rotated_block_logical_indices_per_sequence_for_each_layer[i]) {
                num_indices += entry.second.size();
            }
            auto rotated_block_indices_tensor = m_request.get_tensor(tensor_name);
            rotated_block_indices_tensor.set_shape({num_indices});
        }

        for (size_t i = 0; i < m_num_decoder_layers; i++) {
            auto tensor_name = std::string("rotation_deltas.") + std::to_string(i);
            m_request.set_tensor(tensor_name, m_cache_rotation_deltas_for_each_layer[i]);
        }


        // NB: the order of per-sequence index filling in the function below must be the same
        // as the order of `seq_id`s in which the "rotation_coefficients.N" inputs are filled
        _fill_indices_from_block_tables(rotation_indices_tensor_names,
                                        sequence_groups,
                                        scheduler_output,
                                        m_rotated_block_logical_indices_per_sequence_for_each_layer);
    }

    void _reset_cache_rotation_coefficients() {
        m_cache_rotation_deltas_for_each_layer.clear();
        for (size_t i = 0; i < m_num_decoder_layers; i++) {
            m_cache_rotation_deltas_for_each_layer.push_back(ov::Tensor());
        }
    }

    void _collect_attention_scores(const std::vector<SequenceGroup::Ptr> & sequence_groups, const Scheduler::Output& scheduler_output) {
        m_last_attention_scores.clear();
        size_t num_sequence_groups = scheduler_output.m_scheduled_sequence_groups_ids.size();
        using IndexSpan = std::pair<size_t, size_t>;
        std::list<std::pair<size_t, IndexSpan>> running_sequence_group_ids_and_kvcache_spans;
        size_t offset = 0;
        for (size_t i = 0; i < num_sequence_groups; ++i) { size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i]; SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id]; std::vector<Sequence::CPtr> running_sequences = sequence_group->get_running_sequences();

            for (size_t seq_idx = 0; seq_idx < running_sequences.size(); ++seq_idx) {
                Sequence::CPtr sequence = running_sequences[seq_idx];
                size_t subsequence_length = sequence_group->get_context_len() - sequence_group->get_num_evicted_tokens();
                IndexSpan span = {offset, offset + subsequence_length};
                offset += subsequence_length;

                size_t global_sequence_id = sequence->get_id();

                bool is_prefill_finished = sequence_group->can_generate_tokens();
                bool has_snapkv_scores = (scheduler_output.m_score_aggregation_windows.find(global_sequence_id) != scheduler_output.m_score_aggregation_windows.end());
                if (is_prefill_finished || (!is_prefill_finished && has_snapkv_scores)) {
                    // During prompt phase, will only collect the scores for sequences that have been processed up to their SnapKV window size
                    // (this may happen across multiple scheduling iterations - assuming here that the code using the collected scores does simple aggregation
                    // such as addition and therefore does not need to know which part of the SnapKV window a given score vector belongs to).
                    //
                    // During generation phase, the scores may be either SnapKV-aggregated (if the phase included the very last part of the prompt) or
                    // not (regular non-aggregated single-token-position scores for the newly generated token), but this should also not matter to the simple aggregation
                    // code.
                    running_sequence_group_ids_and_kvcache_spans.emplace_back(global_sequence_id, span);
                }
            }
        }

        for (const auto& seq_id_and_score_span : running_sequence_group_ids_and_kvcache_spans) {
            auto attention_scores_across_decoder_layers_for_current_sequence = AttentionScoresForEachDecoderLayer(m_num_decoder_layers);
            size_t global_sequence_id = seq_id_and_score_span.first;
            IndexSpan span = seq_id_and_score_span.second;
            for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_decoder_layers; decoder_layer_id++) {
                auto attention_score = m_request.get_tensor(get_paged_attention_score_output_for_decoder_layer(decoder_layer_id));
                auto scores_for_cache_of_current_sequence_group = ov::Tensor(attention_score, ov::Coordinate{span.first}, ov::Coordinate{span.second});
                auto copied_tensor = ov::Tensor(scores_for_cache_of_current_sequence_group.get_element_type(), ov::Shape{span.second - span.first});
                scores_for_cache_of_current_sequence_group.copy_to(copied_tensor);
                attention_scores_across_decoder_layers_for_current_sequence[decoder_layer_id] = scores_for_cache_of_current_sequence_group;
            }
            m_last_attention_scores[global_sequence_id] = attention_scores_across_decoder_layers_for_current_sequence;
        }
    }
};
}
