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

class ModelRunner {
    ov::InferRequest m_request;
    SchedulerConfig m_scheduler_config;
public:
    ModelRunner(ov::InferRequest request, const SchedulerConfig& scheduler_config) :
        m_request(std::move(request)),
        m_scheduler_config(scheduler_config) { }

    ov::InferRequest get_infer_request() const {
        return m_request;
    }

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
            block_indices(ov::element::i32, {total_num_blocks}),
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
            * block_indices_data = block_indices.data<int32_t>(),
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
            size_t num_blocks = sequence_group->get_num_blocks();
            size_t group_position_id = sequence_group->get_num_processed_tokens(),
                // spec: In case of multiple input tokens for current sequence (prompt_len > 1), context_len corresponds to first token within subgroup of scheduled tokens
                group_context_len = group_position_id;

            for (size_t seq_id = 0; seq_id < num_running_sequences; ++seq_id) {
                Sequence::CPtr sequence = running_sequences[seq_id];

                for (size_t token_id = 0, position_id = group_position_id; token_id < num_scheduled_tokens; ++token_id, ++position_id) {
                    // compute token for current sequence
                    input_ids_data[token_id] = position_id < sequence_group->get_prompt_len() ?
                        sequence_group->get_prompt_ids()[position_id] :
                        sequence->get_generated_ids()[position_id - sequence_group->get_prompt_len()];

                    position_ids_data[token_id] = position_id;
                }

                past_lens_data[0] = group_context_len;

                subsequence_begins_data[1] = subsequence_begins_data[0] + num_scheduled_tokens;
                block_indices_begins_data[1] = block_indices_begins_data[0] + num_blocks;

                const std::vector<KVCacheBlock::Ptr> & kv_blocks = scheduler_output.m_block_tables.at(sequence->get_id());
                for (size_t block_id = 0; block_id < num_blocks; ++block_id)
                    block_indices_data[block_id] = kv_blocks[block_id]->get_index();

                // apply strides to shift to a next sequence
                input_ids_data += num_scheduled_tokens;
                position_ids_data += num_scheduled_tokens;
                past_lens_data += 1;
                subsequence_begins_data += 1;
                block_indices_data += num_blocks;
                block_indices_begins_data += 1;
            }
        }

        // typical LLM parameters
        m_request.set_tensor("input_ids", input_ids);
        m_request.set_tensor("position_ids", position_ids);

        // PA specific parameters
        m_request.set_tensor("past_lens", past_lens);
        m_request.set_tensor("subsequence_begins", subsequence_begins);
        m_request.set_tensor("block_indices", block_indices);
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

        // return logits
        return m_request.get_output_tensor();
    }
};
