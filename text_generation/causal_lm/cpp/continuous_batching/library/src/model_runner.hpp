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
        m_request(request),
        m_scheduler_config(scheduler_config) { }

    ov::Tensor forward(const std::vector<SequenceGroup::Ptr> & sequence_groups, const Scheduler::Output& scheduler_output) {
        return scheduler_output.is_prompt ?
            _forward_prompt(sequence_groups, scheduler_output) :
            _forward_generate(sequence_groups, scheduler_output);
    }

private:
    ov::Tensor _forward_prompt(const std::vector<SequenceGroup::Ptr> & sequence_groups, const Scheduler::Output& scheduler_output) {
        OPENVINO_ASSERT(scheduler_output.is_prompt, "Internal error: current function can only be called when 'is_prompt' is 'true'");

        size_t batch_size = scheduler_output.m_scheduled_sequence_groups_ids.size(), max_num_blocks = 0, max_seq_len /* the same as context_len */ = 0;

        // compute aggregated values
        for (size_t i = 0; i < scheduler_output.m_scheduled_sequence_groups_ids.size(); ++i) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
            OPENVINO_ASSERT(sequence_group->num_running_seqs() == 1, "Internal error: prompt phase can contain a single sequence within a group");

            max_num_blocks = std::max(max_num_blocks, sequence_group->get_num_blocks());
            max_seq_len = std::max(max_seq_len, sequence_group->get_context_len());
        }

        ov::Tensor
            input_ids(ov::element::i64, {batch_size, max_seq_len}),
            position_ids(ov::element::i64, {batch_size, max_seq_len}),
            is_prompt(ov::element::boolean, {}),
            max_context_len(ov::element::i64, {}),
            slot_mapping(ov::element::i64, {batch_size, max_seq_len}),
            context_lens(ov::element::i64, {batch_size}),
            block_tables(ov::element::i32, {batch_size, max_num_blocks});

        max_context_len.data<int64_t>()[0] = max_seq_len;
        // we dedicated prefill stage
        is_prompt.data<bool>()[0] = true;

        // get raw pointers to copy to
        int64_t
            * input_ids_data = input_ids.data<int64_t>(),
            * position_ids_data = position_ids.data<int64_t>(),
            * slot_mapping_data = slot_mapping.data<int64_t>(),
            * context_lens_data = context_lens.data<int64_t>();
        int32_t
            * block_tables_data = block_tables.data<int32_t>();

        for (size_t i = 0; i < scheduler_output.m_scheduled_sequence_groups_ids.size(); ++i) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
            size_t context_len = sequence_group->get_num_scheduled_tokens();
            Sequence::CPtr sequence = (*sequence_group)[0];
            const TokenIds& input_ids = sequence_group->get_prompt_ids(); 
            const std::vector<KVCacheBlock::Ptr> & kv_blocks = scheduler_output.m_block_tables.at(sequence->get_id());

            // fill information about current sequence group's prompt
            // note: we don't fill padding values with some values like -1 for slot_mapping data
            {
                for (size_t block_id = 0; block_id < max_num_blocks; ++block_id)
                    block_tables_data[block_id] = block_id < kv_blocks.size() ?
                        kv_blocks[block_id]->get_index() :
                        -1;

                std::iota(position_ids_data, position_ids_data + context_len, 0);
                std::fill_n(position_ids_data + context_len, max_seq_len - context_len, -1);

                std::copy_n(input_ids.data(), context_len, input_ids_data);
                context_lens_data[0] = context_len;

                // compute slot_id
                for (size_t token_id = 0; token_id < max_seq_len; ++token_id) {
                    size_t logical_block_id = token_id / m_scheduler_config.block_size, block_offset = token_id % m_scheduler_config.block_size;
                    int64_t slot_id = token_id < context_len ?
                        m_scheduler_config.block_size * kv_blocks[logical_block_id]->get_index() + block_offset :
                        -1 /* invalid slot ID */ ;
                    slot_mapping_data[token_id] = slot_id;
                }

                // apply strides to shift to next sequence
                input_ids_data += max_seq_len;
                position_ids_data += max_seq_len;
                slot_mapping_data += max_seq_len;
                block_tables_data += max_num_blocks;
                context_lens_data += 1;
            }
        }

        // typical LLM parameters
        m_request.set_tensor("input_ids", input_ids);
        m_request.set_tensor("position_ids", position_ids);

        // PagedAttention specific parameters
        m_request.set_tensor("is_prompt", is_prompt);
        m_request.set_tensor("slot_mapping", slot_mapping);
        m_request.set_tensor("max_context_len", max_context_len);
        m_request.set_tensor("context_lens", context_lens);
        m_request.set_tensor("block_tables", block_tables);

        // print_tensor("input_ids", input_ids);
        // print_tensor("position_ids", position_ids);

        // print_tensor("is_prompt", is_prompt);
        // print_tensor("slot_mapping", slot_mapping);
        // print_tensor("max_context_len", max_context_len);
        // print_tensor("context_lens", context_lens);
        // print_tensor("block_tables", block_tables);

        {
            static ScopedTimer timer("pure prompt inference");
            timer.start();
            m_request.infer();
            timer.end();
        }

        // return logits
        return m_request.get_output_tensor();
    }

    ov::Tensor _forward_generate(const std::vector<SequenceGroup::Ptr> & sequence_groups, const Scheduler::Output& scheduler_output) {
        OPENVINO_ASSERT(!scheduler_output.is_prompt, "Internal error: current function can only be called when 'is_prompt' is 'false'");

        size_t batch_size = 0, max_num_blocks = 0, max_context_len_value = 0;
        // since we merge sequence_len and batch to avoid ragged dimensions => batch dimension contains all tokens, while seq len is 1
        const size_t seq_len = 1;

        // compute aggregated values
        for (size_t i = 0; i < scheduler_output.m_scheduled_sequence_groups_ids.size(); ++i) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
            batch_size += sequence_group->get_num_scheduled_tokens() * sequence_group->num_running_seqs();
            max_num_blocks = std::max(max_num_blocks, sequence_group->get_num_blocks());
            max_context_len_value = std::max(max_context_len_value, sequence_group->get_context_len());
        }

        ov::Tensor
            input_ids(ov::element::i64, {batch_size, seq_len}),
            position_ids(ov::element::i64, {batch_size, seq_len}),
            is_prompt(ov::element::boolean, {}),
            max_context_len(ov::element::i64, {}),
            slot_mapping(ov::element::i64, {batch_size, seq_len}),
            context_lens(ov::element::i64, {batch_size}),
            block_tables(ov::element::i32, {batch_size, max_num_blocks});

        max_context_len.data<int64_t>()[0] = max_context_len_value;
        // we don't differentiate prefill and generate phases
        is_prompt.data<bool>()[0] = false;

        // get raw pointers to copy to
        int64_t
            * input_ids_data = input_ids.data<int64_t>(),
            * position_ids_data = position_ids.data<int64_t>(),
            * slot_mapping_data = slot_mapping.data<int64_t>(),
            * context_lens_data = context_lens.data<int64_t>();
        int32_t
            * block_tables_data = block_tables.data<int32_t>();

        for (size_t i = 0; i < scheduler_output.m_scheduled_sequence_groups_ids.size(); ++i) {
            size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
            SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
            std::vector<Sequence::CPtr> running_sequences = sequence_group->get_running_sequences();
            size_t num_scheduled_tokens = sequence_group->get_num_scheduled_tokens();
            size_t group_position_id = sequence_group->get_num_processed_tokens(), group_context_len = group_position_id + 1;

            for (size_t seq_id = 0; seq_id < running_sequences.size(); ++seq_id) {
                Sequence::CPtr sequence = running_sequences[seq_id];
                size_t position_id = group_position_id, context_len = group_context_len;

                for (size_t token_id = 0; token_id < num_scheduled_tokens; ++token_id, ++position_id, ++context_len) {
                    const std::vector<KVCacheBlock::Ptr> & kv_blocks = scheduler_output.m_block_tables.at(sequence->get_id());
                    const size_t num_blocks = kv_blocks.size();
                    for (size_t block_id = 0; block_id < kv_blocks.size(); ++block_id)
                        block_tables_data[block_id] = kv_blocks[block_id]->get_index();
                    block_tables_data += max_num_blocks;

                    position_ids_data[token_id] = position_id;
                    context_lens_data[token_id] = context_len;

                    // compute token for current sequence
                    input_ids_data[token_id] = position_id < sequence_group->get_prompt_len() ?
                        sequence_group->get_prompt_ids()[position_id] :
                        sequence->get_generated_ids()[position_id - sequence_group->get_prompt_len()];

                    // compute slot_id
                    size_t physical_block_id = position_id / m_scheduler_config.block_size, block_offset = position_id % m_scheduler_config.block_size;
                    int64_t slot_id = m_scheduler_config.block_size * kv_blocks[physical_block_id]->get_index() + block_offset;
                    slot_mapping_data[token_id] = slot_id;
                }

                // apply strides to shift to next sequence
                input_ids_data += num_scheduled_tokens;
                position_ids_data += num_scheduled_tokens;
                slot_mapping_data += num_scheduled_tokens;
                context_lens_data += num_scheduled_tokens;
            }
        }

        // typical LLM parameters
        m_request.set_tensor("input_ids", input_ids);
        m_request.set_tensor("position_ids", position_ids);

        // PagedAttention specific parameters
        m_request.set_tensor("is_prompt", is_prompt);
        m_request.set_tensor("slot_mapping", slot_mapping);
        m_request.set_tensor("max_context_len", max_context_len);
        m_request.set_tensor("context_lens", context_lens);
        m_request.set_tensor("block_tables", block_tables);

        // print_tensor("input_ids", input_ids);
        // print_tensor("position_ids", position_ids);

        // print_tensor("is_prompt", is_prompt);
        // print_tensor("slot_mapping", slot_mapping);
        // print_tensor("max_context_len", max_context_len);
        // print_tensor("context_lens", context_lens);
        // print_tensor("block_tables", block_tables);

        {
            static ScopedTimer timer("pure generate inference");
            timer.start();
            m_request.infer();
            timer.end();
        }

        // return logits
        return m_request.get_output_tensor();
    }
};
