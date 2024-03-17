
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <vector>
#include <unordered_set>

#include "model_config.hpp"
#include "block_manager.hpp"
#include "sequence_group.hpp"
#include "block_manager.hpp"

struct SchedulerConfig {
    // a maximum number of tokens to batch
    // (in constrast to max_batch_size which combines independent sequences, we consider total amount of tokens in a batch)
    // TODO: benchmark this value and understand a required value to ensure inference is not memory bound
    const size_t max_tokens_to_batch = 16;

    // total number of KV blocks available to scheduler logic
    const size_t num_kv_blocks = NUM_BLOCKS;
};

class Scheduler {
    SchedulerConfig m_config;
    BlockManager m_block_manager;

public:
    struct Output {
        // IDs of scheduled groups
        std::vector<uint64_t> m_scheduled_sequence_groups_ids;
        // map of src -> dst blocks copies, which need to be performed by CacheManager
        std::map<size_t, std::unordered_set<size_t>> m_block_copy_map;
        // block tables for scheduled sequences
        std::map<uint64_t, std::vector<KVCacheBlock>> m_block_tables;
        // total number of scheduled tokens
        size_t m_total_num_scheduled_tokens = 0;
    };

    explicit Scheduler(const SchedulerConfig & config = {}) :
        m_config(config), m_block_manager(m_config.num_kv_blocks) { }

    Output schedule(std::vector<SequenceGroup>& sequence_groups) {
        Output scheduler_output;

        // 1. perform balancing of running groups first to ensure we have some free KV blocks for generation
        _apply_preemption(sequence_groups);

        // 2. schedule groups in generation phase
        _schedule_generate_phase(sequence_groups, scheduler_output);

        // 3. schedule groups in prompt phase
        _schedule_prompt_phase(sequence_groups, scheduler_output);

        return scheduler_output;
    }

    const std::vector<KVCacheBlock>& get_block_table(const Sequence& seq) {
        return m_block_manager.get_block_table(seq.get_id());
    }

private:
    void _preempt_by_recompute(SequenceGroup & sequence_group) {
        // currently, we support only preemption by (TODO: implement "partial") recompute
        for (size_t s = 0; s < sequence_group.num_running_seqs(); ++s) {
            // so, let's fully drop a sequence(s) from block_manager
            m_block_manager.free_sequence(sequence_group[s].get_id());
        }
        // update computed part of each sequence
        sequence_group.preempt_tokens(sequence_group.get_num_processed_tokens());
    }

    size_t _get_low_priority_sequence_group_id(std::vector<SequenceGroup>& sequence_groups) {
        for (size_t seq_group_id = 0, num_groups = sequence_groups.size(); seq_group_id < num_groups; ++seq_group_id) {
            SequenceGroup & sequence_group = sequence_groups[num_groups - seq_group_id - 1];
            if (sequence_group.get_num_processed_tokens() > 0) {
                // we are here, because current sequence group has some reserved KV blocks in block manager
                // which can be freed
                return seq_group_id;
            }
        }

        return std::numeric_limits<size_t>::max();
    }

    // current function iterates over all sequence groups and understands whether some KV blocks of existing 
    // sequences (either on generate or prompt phase) need to be freed to generation phase of higher priority
    // sequence groups (priority here is index within vector of sequence groups)
    void _apply_preemption(std::vector<SequenceGroup>& sequence_groups) {
        for (size_t sequence_group_id = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
            SequenceGroup & high_priority_sequence_group = sequence_groups[sequence_group_id];

            // we don't consider sequence which is not in "generation" or "preempted" phases as high priority
            if (!high_priority_sequence_group.can_generate_tokens())
                continue;

            // check whether current sequence requires a new slot / block
            // TODO: we use simple heuristic like in vLLM (see impl. of "can_append_slot"), but it can be implemented more precise
            while (!m_block_manager.can_append_slot(high_priority_sequence_group)) {
                // let's run a sequence for eviction
                size_t evicted_sequence_group_id = _get_low_priority_sequence_group_id(sequence_groups);
            
                if (evicted_sequence_group_id >= sequence_group_id) {
                    // we have a cycle when current group need to evict itself to be in a running state
                    return;
                }

                _preempt_by_recompute(sequence_groups[evicted_sequence_group_id]);
            }
        }
    }

    void _schedule_prompt_phase(std::vector<SequenceGroup>& sequence_groups, Output& scheduler_output) {
        // in the current method we need to balance multiple prompts (or parts of prompts) between
        // available amount of tokens in megabatch
        // Considerations:
        // 1. To reduce discrepancy between ragged dimensions (context lengths) in Attention module
        //    we can slice prompt on chunks and schedule only portion of each prompt instead of
        //    greedy scheduling of prompt with higher priority
        // 2. The machanism below performs greedy scheduling of high priority prompts

        for (size_t sequence_group_id = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
            SequenceGroup & sequence_group = sequence_groups[sequence_group_id];
            if (!sequence_group.can_generate_tokens()) {
                size_t num_running_seqs = sequence_group.num_running_seqs();
                // prompt phases can have a single running sequence
                OPENVINO_ASSERT(num_running_seqs == 1);

                size_t num_tokens_in_megabatch = m_config.max_tokens_to_batch - scheduler_output.m_total_num_scheduled_tokens;
                size_t num_available_tokens = sequence_group.get_num_available_tokens_for_batching();

                // apply megabatch limitations
                size_t num_scheduled_tokens = std::min(num_tokens_in_megabatch, num_available_tokens);

                // apply KV cache limitations
                size_t context_len = sequence_group.get_context_len();
                size_t available_slots = context_len % BLOCK_SIZE, required_slots = context_len - available_slots;
                size_t num_required_blocks = (required_slots + BLOCK_SIZE - 1) / BLOCK_SIZE, num_free_blocks = m_block_manager.num_free_blocks();
                size_t num_scheduled_blocks = std::min(num_required_blocks, num_free_blocks);
                num_scheduled_tokens = available_slots + num_scheduled_blocks * BLOCK_SIZE;

                // allocate KV blocks
                m_block_manager.allocate(sequence_group[0], num_scheduled_blocks);
                // and schedule tokens
                sequence_group.schedule_tokens(num_scheduled_tokens);

                // add information to scheduler_output
                {
                    auto request_id = sequence_group.get_request_id();
                    scheduler_output.m_scheduled_sequence_groups_ids.push_back(request_id);
                    scheduler_output.m_block_tables[request_id] = m_block_manager.get_block_table(request_id);
                    scheduler_output.m_total_num_scheduled_tokens += num_scheduled_tokens * num_running_seqs;
                }

                // if we added maximum amount of tokens to compute
                if (scheduler_output.m_total_num_scheduled_tokens == m_config.max_tokens_to_batch)
                    break;
            }
        }
    }

    void _schedule_generate_phase(std::vector<SequenceGroup>& sequence_groups, Output& scheduler_output) {
        for (size_t sequence_group_id = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
            SequenceGroup & sequence_group = sequence_groups[sequence_group_id];
            // Note, that can_generate_tokens will mix preempted sequence groups
            // and real generate ones
            // Question: do we need to schedule preeempted first as it's done in vLLM?
            // Answer: preempted sequences have low priority, so they should be after "running" ones. So, here we
            //         keep latencies for sequence groups of high priority
            if (sequence_group.can_generate_tokens()) {
                OPENVINO_ASSERT(!sequence_group.has_finished());
                size_t num_running_seqs = sequence_group.num_running_seqs();
                size_t num_tokens_in_megabatch = m_config.max_tokens_to_batch - scheduler_output.m_total_num_scheduled_tokens;
                size_t available_tokens_per_seq_in_megabatch = num_tokens_in_megabatch / num_running_seqs;

                // we cannot schedule even a single token per each sequence in a group
                if (!available_tokens_per_seq_in_megabatch)
                    continue;

                // Note: current function can return more than 1 token even for generation phase in case of some tokens
                // of current sequence group were evicted before
                size_t num_available_tokens_per_seq = sequence_group.get_num_available_tokens_for_batching();

                size_t num_scheduled_tokens_per_seq = std::min(available_tokens_per_seq_in_megabatch, num_available_tokens_per_seq);
                sequence_group.schedule_tokens(num_scheduled_tokens_per_seq);

                // TODO: below functions can_append_slot / append_slot can allocate just a single slot, while we require multiple ones in generic case
                // So, let's state this as current limitation of scheduler logic
                OPENVINO_ASSERT(num_scheduled_tokens_per_seq == 1);

                // we can ensure that preemption stage freed some KV blocks to allocate new slots for us
                OPENVINO_ASSERT(m_block_manager.can_append_slot(sequence_group));
                // allocate new slots
                auto copy_blocks_map = m_block_manager.append_slot(sequence_group);

                // add information to scheduler_output
                {
                    auto request_id = sequence_group.get_request_id();
                    scheduler_output.m_scheduled_sequence_groups_ids.push_back(request_id);
                    scheduler_output.m_block_tables[request_id] = m_block_manager.get_block_table(request_id);
                    scheduler_output.m_total_num_scheduled_tokens += num_scheduled_tokens_per_seq * num_running_seqs;

                    // merge copy_blocks
                    for (const auto& src_dst : copy_blocks_map) {
                        size_t src_indx = src_dst.first, dst_index = src_dst.second;
                        scheduler_output.m_block_copy_map[src_indx].insert(dst_index);
                    }
                }

                // if we added maximum amount of tokens to compute
                if (scheduler_output.m_total_num_scheduled_tokens == m_config.max_tokens_to_batch)
                    break;
            }
        }
    }
};
