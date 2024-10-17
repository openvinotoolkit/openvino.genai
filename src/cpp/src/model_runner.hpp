// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdlib>

#include <openvino/runtime/infer_request.hpp>

#include "debug_utils.hpp"
#include "sequence_group.hpp"
#include "scheduler.hpp"
#include "timer.hpp"

#include "attention_output.hpp"

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
    SchedulerConfig m_scheduler_config;
    AttentionScoresForEachSubsequence m_last_attention_scores;
    size_t m_num_decoder_layers;
    bool m_collect_attention_scores;
    std::vector<ov::Tensor> m_cache_rotation_coefficients;
public:
    /**
     * Constructs the ModelRunner.
     * @param request The ov::InferRequest for the LLM to be inferred in the continous batching mode.
     * @param scheduler_config Configuration struct for the scheduler that is to be used with this ModelRunner.
     * @param num_decoder_layers Number of decoder attention layers in the LLM corresponding to the request.
     * @param collect_attention_scores If true, then after each `forward` call the ModelRunner will collect and make available the per-token attention
     * scores for each decoder layer, so that these can be used in per-step cache optimizations (such as cache eviction algorithm).
     */
    ModelRunner(ov::InferRequest request, const SchedulerConfig& scheduler_config, size_t num_decoder_layers = 1, bool collect_attention_scores = false) :
        m_request(std::move(request)),
        m_scheduler_config(scheduler_config),
        m_num_decoder_layers(num_decoder_layers),
        m_collect_attention_scores(collect_attention_scores) {
        OPENVINO_ASSERT(m_num_decoder_layers != 0, "num_decoder_layers must be non-zero");
    }

    /**
     * @return The ov::InferRequest this ModelRunner is handling.
     */
    ov::InferRequest get_infer_request() const {
        return m_request;
    }

    /**
     * @return A map of sequence IDs to vectors of ov::Tensor per-token attention scores. Each vector element is associated with its own
     * decoder layer, in order of their execution in the model. Each ov::Tensor has a shape of {N_k}, where N_k is the length of
     * a sequence with ID k processed during the previous `forward` call.
     */
    const AttentionScoresForEachSubsequence& get_last_attention_scores() const {
        return m_last_attention_scores;
    }

    void set_cache_rotation_coefficients(const std::vector<ov::Tensor>& cache_rotation_coefficients_for_each_layer) {
        // TODO (vshampor): avoid vector copy
        m_cache_rotation_coefficients = cache_rotation_coefficients_for_each_layer;
    }

    /**
     * Runs the forward inference call on the underlying LLM's ov::InferRequest, scheduling for inferencing tokens for given sequences
     * taking into account the supplied scheduler output struct.
     * @param sequence_groups A vector of pointers to sequence groups to be processed during this `forward` call
     * @param scheduler_output The scheduler output struct with information on the specifics of the token scheduling during this forward call
     * @return An ov::Tensor with next-token logit scores for each sequence processed during this `forward` call.
     */
    ov::Tensor forward(const std::vector<SequenceGroup::Ptr> & sequence_groups, const Scheduler::Output& scheduler_output) {
        size_t num_sequence_groups = scheduler_output.m_scheduled_sequence_groups_ids.size();
        size_t batch_size_in_sequences = 0;
        size_t total_num_tokens = 0, total_num_blocks = 0;
        size_t max_context_len_val = 0;

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
            position_ids(ov::element::i64, {total_num_tokens}),
            // PA specific parameters
            past_lens(ov::element::i32, {batch_size_in_sequences}),
            subsequence_begins(ov::element::i32, {batch_size_in_sequences + 1}),
            // block_indices are handled in a special fashion below
            block_indices_begins(ov::element::i32, {batch_size_in_sequences + 1}),
            max_context_len(ov::element::i32, {});

        max_context_len.data<int32_t>()[0] = max_context_len_val;

        // get raw pointers to copy to
        int64_t
            * input_ids_data = input_ids.data<int64_t>(),
            * position_ids_data = position_ids.data<int64_t>();
        int32_t 
            * past_lens_data = past_lens.data<int32_t>(),
            * subsequence_begins_data = subsequence_begins.data<int32_t>(),
            * block_indices_begins_data = block_indices_begins.data<int32_t>();

        // sub-sequence data starts with 0
        subsequence_begins_data[0] = 0;
        block_indices_begins_data[0] = 0;

        for (size_t i = 0; i < num_sequence_groups; ++i) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
            std::vector<Sequence::CPtr> running_sequences = sequence_group->get_running_sequences();
            size_t num_running_sequences = running_sequences.size();
            size_t num_scheduled_tokens = sequence_group->get_num_scheduled_tokens();
            size_t group_position_id = sequence_group->get_num_processed_tokens();

            // spec: In case of multiple input tokens for current sequence (prompt_len > 1),
            // context_len corresponds to first token within subgroup of scheduled tokens
            size_t group_context_len = group_position_id;

            for (size_t seq_id = 0; seq_id < num_running_sequences; ++seq_id) {
                Sequence::CPtr sequence = running_sequences[seq_id];

                for (size_t token_id = 0, position_id = group_position_id; token_id < num_scheduled_tokens; ++token_id, ++position_id) {
                    // compute token for current sequence
                    input_ids_data[token_id] = position_id < sequence_group->get_prompt_len() ?
                        sequence_group->get_prompt_ids()[position_id] :
                        sequence->get_generated_ids()[position_id - sequence_group->get_prompt_len()];

                    position_ids_data[token_id] = position_id;
                }

                size_t expected_kv_cache_size = sequence_group->get_num_processed_tokens() - sequence_group->get_num_evicted_tokens();
                past_lens_data[0] = expected_kv_cache_size;

                subsequence_begins_data[1] = subsequence_begins_data[0] + num_scheduled_tokens;

                size_t num_blocks = (sequence_group->get_context_len()  - sequence_group->get_num_evicted_tokens() +  m_scheduler_config.block_size - 1) / m_scheduler_config.block_size;
                block_indices_begins_data[1] = block_indices_begins_data[0] + num_blocks;

                // apply strides to shift to a next sequence
                input_ids_data += num_scheduled_tokens;
                position_ids_data += num_scheduled_tokens;
                past_lens_data += 1;
                subsequence_begins_data += 1;
                block_indices_begins_data += 1;
            }
        }

        // typical LLM parameters
        m_request.set_tensor("input_ids", input_ids);
        m_request.set_tensor("position_ids", position_ids);

        // PA specific parameters
        m_request.set_tensor("past_lens", past_lens);
        m_request.set_tensor("subsequence_begins", subsequence_begins);

        _set_block_indices(sequence_groups, scheduler_output, total_num_blocks);

        if (!m_cache_rotation_coefficients.empty()) {
            _set_cache_rotation_coefficients(sequence_groups, scheduler_output);
        }

        m_request.set_tensor("block_indices_begins", block_indices_begins);
        m_request.set_tensor("max_context_len", max_context_len);

        // print_tensor("input_ids", input_ids);
        // print_tensor("position_ids", position_ids);

        // print_tensor("past_lens", past_lens);
        // print_tensor("subsequence_begins", subsequence_begins);
        // print_tensor("block_indices", block_indices);
        // print_tensor("block_indices_begins", block_indices_begins);
        // print_tensor("max_context_len", max_context_len);

        {
            static ManualTimer timer("pure generate inference");
            timer.start();
            m_request.infer();
            timer.end();
        }

        if (m_collect_attention_scores && m_scheduler_config.use_cache_eviction) {
            _collect_attention_scores(sequence_groups, scheduler_output);
        }

        m_cache_rotation_coefficients.clear();

        // return logits
        return m_request.get_tensor("logits");
    }

private:
    void _fill_indices_from_block_tables(const std::vector<std::string>& dst_tensor_names,
                                        const std::vector<SequenceGroup::Ptr> & sequence_groups,
                                        const Scheduler::Output& scheduler_output,
                                        const std::vector<size_t>& fill_n_last_vec) {
        OPENVINO_ASSERT(fill_n_last_vec.size() == dst_tensor_names.size() || fill_n_last_vec.empty());
        size_t num_sequence_groups = scheduler_output.m_scheduled_sequence_groups_ids.size();
        size_t block_offset = 0;
        for (size_t i = 0; i < num_sequence_groups; ++i) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
            std::vector<Sequence::CPtr> running_sequences = sequence_group->get_running_sequences();
            size_t num_running_sequences = running_sequences.size();

            for (size_t seq_id = 0; seq_id < num_running_sequences; ++seq_id) {
                Sequence::CPtr sequence = running_sequences[seq_id];

                size_t num_blocks = (sequence_group->get_context_len()  - sequence_group->get_num_evicted_tokens() +  m_scheduler_config.block_size - 1) / m_scheduler_config.block_size;
                const auto & kv_blocks = scheduler_output.m_block_tables.at(sequence->get_id());

                for (size_t layer_idx = 0; layer_idx < dst_tensor_names.size(); layer_idx++) {
                    size_t fill_n_last = num_blocks;
                    if (!fill_n_last_vec.empty()) {
                        fill_n_last = fill_n_last_vec[layer_idx];
                    }
                    OPENVINO_ASSERT(num_blocks >= fill_n_last);
                    size_t starting_offset = num_blocks - fill_n_last;
                    auto input_tensor = m_request.get_tensor(dst_tensor_names[layer_idx]);
                    auto block_indices_data = input_tensor.data<int32_t>() + block_offset;
                    for (size_t block_id = 0; block_id < fill_n_last; ++block_id)
                        // In case no cache eviction is requested, all per-layer block tables are expected to be identical
                        // at all times
                        block_indices_data[block_id] = kv_blocks[layer_idx][starting_offset + block_id]->get_index();
                }

                block_offset += num_blocks;
            }
        }
    }
    void _set_block_indices(const std::vector<SequenceGroup::Ptr> & sequence_groups, const Scheduler::Output& scheduler_output,
                            size_t total_num_blocks) {
        std::vector<std::string> tensor_names = {"block_indices"};

        if (m_scheduler_config.use_cache_eviction) {
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

    void _set_cache_rotation_coefficients(const std::vector<SequenceGroup::Ptr> & sequence_groups, const Scheduler::Output& scheduler_output) {
        for (size_t i = 0; i < m_num_decoder_layers; i++) {
            auto tensor_name = std::string("cache_rotation_coefficients.") + std::to_string(i);
            m_request.set_tensor(tensor_name, m_cache_rotation_coefficients[i]);
        }

        std::vector<std::string> rotation_indices_tensor_names(m_num_decoder_layers);
        std::vector<size_t> rotation_indices_sizes_in_blocks(m_num_decoder_layers);
        for (size_t i = 0; i < m_num_decoder_layers; i++) {
            auto tensor_name = std::string("rotated_block_indices.") + std::to_string(i);
            rotation_indices_tensor_names[i] = tensor_name;
            size_t size_in_blocks = m_cache_rotation_coefficients[i].get_size() / m_scheduler_config.block_size;
            m_request.get_tensor(tensor_name).set_shape({size_in_blocks});
            rotation_indices_sizes_in_blocks[i] = size_in_blocks;
        }

        _fill_indices_from_block_tables(rotation_indices_tensor_names, sequence_groups, scheduler_output, rotation_indices_sizes_in_blocks);
    }

    void _collect_attention_scores(const std::vector<SequenceGroup::Ptr> & sequence_groups, const Scheduler::Output& scheduler_output) {
        m_last_attention_scores.clear();
        size_t num_sequence_groups = scheduler_output.m_scheduled_sequence_groups_ids.size();
        using IndexSpan = std::pair<size_t, size_t>;
        std::list<std::pair<size_t, IndexSpan>> running_sequence_group_ids_and_kvcache_spans;
        size_t offset = 0;
        for (size_t i = 0; i < num_sequence_groups; ++i) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
            std::vector<Sequence::CPtr> running_sequences = sequence_group->get_running_sequences();

            for (size_t seq_id = 0; seq_id < running_sequences.size(); ++seq_id) {
                Sequence::CPtr sequence = running_sequences[seq_id];
                size_t subsequence_length = sequence_group->get_context_len() - sequence_group->get_num_evicted_tokens();
                IndexSpan span = {offset, offset + subsequence_length};
                size_t global_sequence_id = sequence->get_id();
                running_sequence_group_ids_and_kvcache_spans.emplace_back(global_sequence_id, span);
                offset += subsequence_length;
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
