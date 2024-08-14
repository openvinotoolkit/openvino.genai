
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <vector>

#include "openvino/genai/scheduler_config.hpp"
#include "block_manager.hpp"
#include "sequence_group.hpp"

namespace ov::genai {
class Scheduler {
    SchedulerConfig m_config;
    BlockManager m_block_manager;

public:
    struct Output {
        // IDs of scheduled groups
        std::vector<uint64_t> m_scheduled_sequence_groups_ids;
        // map of src -> dst blocks copies, which need to be performed by CacheManager
        std::map<size_t, std::list<size_t>> m_block_copy_map;
        // block tables for scheduled sequences
        std::map<uint64_t, std::vector<KVCacheBlock::Ptr>> m_block_tables;
        // total number of scheduled tokens
        size_t m_total_num_scheduled_tokens = 0;
        // dedicated prompt phase
        bool is_prompt = false;
        // current cache usage
        float m_cache_usage = 0.0;
    };

    explicit Scheduler(const SchedulerConfig & config = {}) :
        m_config(config), m_block_manager(m_config.num_kv_blocks, m_config.enable_prefix_caching, m_config.block_size) { }

    Output schedule(std::vector<SequenceGroup::Ptr>& sequence_groups) {
        Output scheduler_output;

        if (m_config.enable_prefix_caching)
            _restore_cached_blocks(sequence_groups);

        if (m_config.dynamic_split_fuse) {
            // deepspeed-mii case

            // generation phase is always scheduled first
            _schedule_generate_phase_dynamic_split_fuse(sequence_groups, scheduler_output);
            // some tokens from prompt phase are also scheduled
            _schedule_prompt_phase_dynamic_split_fuse(sequence_groups, scheduler_output);
        } else {
            // vLLM case

            // schedule prompt phase using whole prompt's input_ids
            _schedule_prompt_phase_vllm(sequence_groups, scheduler_output);

            if (!scheduler_output.is_prompt) {
                // prompt sequences are not scheduler => scheduler generation phase by dynamic_split_fuse implementation
                _schedule_generate_phase_dynamic_split_fuse(sequence_groups, scheduler_output);
            }
        }

        // reset preemption logic
        _reset_preempted_sequences(sequence_groups);

        scheduler_output.m_cache_usage = m_block_manager.get_used_percentage();

        return scheduler_output;
    }

    const std::vector<KVCacheBlock::Ptr>& get_block_table(const Sequence& seq) {
        return m_block_manager.get_block_table(seq.get_id());
    }

    const bool has_block_table(uint64_t seq_id) {
        return m_block_manager.has_block_table(seq_id);
    }

    void free_sequence(uint64_t seq_id) {
        m_block_manager.free_sequence(seq_id);
    }

    void fork_sequence(uint64_t parent_id, uint64_t child_id) {
        m_block_manager.fork_sequence(parent_id, child_id);
    }

    const SchedulerConfig& get_config() const {
        return m_config;
    }

private:
    static size_t _num_running_sequences(const std::vector<SequenceGroup::Ptr>& sequence_groups) {
        size_t num_running = 0;
        for (const SequenceGroup::CPtr& seq_group : sequence_groups) {
            if (seq_group->can_generate_tokens()) {
                const GenerationConfig& sampling_params = seq_group->get_sampling_parameters();
                num_running += std::max(sampling_params.num_beams, sampling_params.num_return_sequences);
            }
        }

        return num_running;
    }


    bool _preempt_by_recompute(SequenceGroup::Ptr sequence_group, size_t blocks_needed) {
        size_t context_len = sequence_group->get_context_len();
        size_t block_size = m_config.block_size;
        size_t prev_blocks_count = m_block_manager.num_free_blocks();
        size_t preempted_tokens = 0;
        size_t blocks_in_sequence_group = m_block_manager.get_number_of_used_blocks(sequence_group);

        if (blocks_in_sequence_group <= blocks_needed) {
            auto running_sequences = sequence_group->get_running_sequences();
            for (size_t s = 0; s < running_sequences.size(); ++s) {
                auto seq_id = running_sequences[s]->get_id();
                m_block_manager.free_sequence(seq_id);
            }
            sequence_group->preempt_tokens(context_len);
            return m_block_manager.num_free_blocks() > prev_blocks_count;
        }

        size_t logical_blocks_released = m_block_manager.free_sequence_group_partially(sequence_group, blocks_needed);

        // calculate the number of preempted tokens
        auto tokens_in_last_block = context_len % block_size;
        if (tokens_in_last_block == 0) {
            tokens_in_last_block = block_size;
        }
        preempted_tokens = tokens_in_last_block + std::max<size_t>((int)logical_blocks_released - 1, 0) * block_size;

        // case when preemption requires preempt prompt tokens
        if (!m_config.dynamic_split_fuse && context_len - preempted_tokens < sequence_group->get_prompt_len()) {
            // preempt prompt fully to not leave partially generated prompt
            preempted_tokens = context_len;
            // TODO: ilavrenov - what is we have multiple sequences within a group? we need to drop them all
            auto seq_id = (*sequence_group)[0]->get_id();
            m_block_manager.free_sequence(seq_id);
        }
        sequence_group->preempt_tokens(preempted_tokens);

        return logical_blocks_released > 0;
    }

    static size_t _get_low_priority_sequence_group_id(const std::vector<SequenceGroup::Ptr>& sequence_groups) {
        for (size_t seq_group_id = 0, num_groups = sequence_groups.size(); seq_group_id < num_groups; ++seq_group_id) {
            size_t group_idx = num_groups - seq_group_id - 1;
            SequenceGroup::CPtr sequence_group = sequence_groups[group_idx];
            if (sequence_group->get_context_len() > 0) {
                // we are here, because current sequence group has some reserved KV blocks in block manager
                // which can be freed
                return group_idx;
            }
        }

        return std::numeric_limits<size_t>::max();
    }

    // performs restoring of KV cache blocks using Evictor
    // it's so-called "prefix caching"
    void _restore_cached_blocks(const std::vector<SequenceGroup::Ptr>& sequence_groups) {
        for (size_t sequence_group_id = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
            SequenceGroup::Ptr sequence_group = sequence_groups[sequence_group_id];
            if (sequence_group->can_generate_tokens() || sequence_group->num_running_seqs() != 1)
                continue;
            m_block_manager._restore_cached_blocks(sequence_group, m_config.block_size);
        }
    }

    // apply preemption of low priority sequence groups to be able to scheduler a device sequence group
    // with `sequence_group_id` sequence group ID
    void _apply_preemption(size_t sequence_group_id, const std::vector<SequenceGroup::Ptr>& sequence_groups) {
        SequenceGroup::Ptr sequence_group = sequence_groups[sequence_group_id];

        // check whether current sequence requires a new slot / block
        while (!m_block_manager.can_append_slots(sequence_group)) {
            // let's run a sequence for eviction
            size_t preempted_sequence_group_id = _get_low_priority_sequence_group_id(sequence_groups);
        
            if (preempted_sequence_group_id <= sequence_group_id) {
                // we have a cycle when current group need to preempt itself to be in a running state
                break;
            }
            size_t blocks_needed = m_block_manager.get_number_of_required_blocks(sequence_group);
            if (!_preempt_by_recompute(sequence_groups[preempted_sequence_group_id], blocks_needed)){
                break;
            }
        }
    }

    void _schedule_prompt_phase_dynamic_split_fuse(std::vector<SequenceGroup::Ptr>& sequence_groups, Output& scheduler_output) {
        // in the current method we need to balance multiple prompts (or parts of prompts) between
        // available amount of tokens in megabatch
        // Considerations:
        // 1. To reduce discrepancy between ragged dimensions (context lengths) in Attention module
        //    we can slice prompt on chunks and schedule only portion of each prompt instead of
        //    greedy scheduling of prompt with higher priority
        // 2. The machanism below performs greedy scheduling of high priority prompts

        for (size_t sequence_group_id = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
            SequenceGroup::Ptr sequence_group = sequence_groups[sequence_group_id];
            if (!sequence_group->can_generate_tokens()) {
                size_t num_running_seqs = sequence_group->num_running_seqs();
                // prompt phases can have a single running sequence
                OPENVINO_ASSERT(num_running_seqs == 1);
                Sequence::Ptr sequence = (*sequence_group)[0];
                uint64_t seq_id = sequence->get_id();

                size_t num_tokens_in_megabatch = m_config.max_num_batched_tokens - scheduler_output.m_total_num_scheduled_tokens;
                size_t num_available_tokens = sequence_group->get_num_available_tokens_for_batching();

                // apply megabatch limitations
                size_t num_scheduled_tokens = std::min(num_tokens_in_megabatch, num_available_tokens);

                // apply KV cache limitations
                size_t available_slots = sequence_group->get_num_blocks() * m_config.block_size - sequence_group->get_context_len(),
                       required_slots = num_scheduled_tokens > available_slots ? num_scheduled_tokens - available_slots : 0;
                size_t num_required_blocks = (required_slots + m_config.block_size - 1) / m_config.block_size, num_free_blocks = m_block_manager.num_free_blocks();
                size_t num_scheduled_blocks = std::min(num_required_blocks, num_free_blocks);
                // some scheduled blocks can be no fully occupied, so we need to take min between num_scheduled_blocks
                // and total "scheduled capacity"
                num_scheduled_tokens = std::min(num_scheduled_tokens, available_slots + num_scheduled_blocks * m_config.block_size);

                if (num_scheduled_tokens > 0) {
                    // schedule tokens
                    sequence_group->schedule_tokens(num_scheduled_tokens);
                    // and allocate KV blocks if required
                    if (num_scheduled_blocks > 0)
                        m_block_manager.allocate(sequence, num_scheduled_blocks, sequence_group->get_prompt_ids());

                    // add information to scheduler_output
                    {
                        scheduler_output.m_scheduled_sequence_groups_ids.push_back(sequence_group_id);
                        scheduler_output.m_block_tables[seq_id] = m_block_manager.get_block_table(seq_id);
                        scheduler_output.m_total_num_scheduled_tokens += num_scheduled_tokens * num_running_seqs;
                    }
                }

                // if we added maximum amount of tokens to compute
                if (scheduler_output.m_total_num_scheduled_tokens == m_config.max_num_batched_tokens)
                    break;
            }
        }
    }

    void _schedule_generate_phase_dynamic_split_fuse(const std::vector<SequenceGroup::Ptr>& sequence_groups, Output& scheduler_output) {
        for (size_t sequence_group_id = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
            SequenceGroup::Ptr sequence_group = sequence_groups[sequence_group_id];
            // Note, that can_generate_tokens will mix preempted sequence groups
            // and real generate ones
            // Question: do we need to schedule preeempted first as it's done in vLLM?
            // Answer: preempted sequences have low priority, so they should be after "running" ones. So, here we
            //         keep latencies for sequence groups of high priority
            if (sequence_group->can_generate_tokens()) {
                OPENVINO_ASSERT(!sequence_group->has_finished());
                size_t num_running_seqs = sequence_group->num_running_seqs();
                size_t num_tokens_in_megabatch = m_config.max_num_batched_tokens - scheduler_output.m_total_num_scheduled_tokens;
                size_t available_tokens_per_seq_in_megabatch = num_tokens_in_megabatch / num_running_seqs;

                // we cannot schedule even a single token per each sequence in a group
                if (!available_tokens_per_seq_in_megabatch)
                    continue;

                // Note: current function can return more than 1 token even for generation phase in case of some tokens
                // of current sequence group were preempted before
                size_t num_available_tokens_per_seq = sequence_group->get_num_available_tokens_for_batching();

                size_t num_scheduled_tokens_per_seq = std::min(available_tokens_per_seq_in_megabatch, num_available_tokens_per_seq);
                sequence_group->schedule_tokens(num_scheduled_tokens_per_seq);

                _apply_preemption(sequence_group_id, sequence_groups);

                // if we can't preempt any other sequence, clear scheduled tokens and move to next sequence
                if (!m_block_manager.can_append_slots(sequence_group)){
                    sequence_group->reset_scheduled_tokens();
                    continue;
                }

                // allocate new slots
                std::map<size_t, std::list<size_t>> copy_blocks_map = m_block_manager.append_slots(sequence_group);

                // add information to scheduler_output
                {
                    auto request_id = sequence_group->get_request_id();
                    scheduler_output.m_scheduled_sequence_groups_ids.push_back(sequence_group_id);
                    scheduler_output.m_total_num_scheduled_tokens += num_scheduled_tokens_per_seq * num_running_seqs;

                    // block tables for each running sequence within a group
                    std::vector<Sequence::Ptr> running_seqs = sequence_group->get_running_sequences();
                    for (const auto & seq : sequence_group->get_running_sequences()) {
                        scheduler_output.m_block_tables[seq->get_id()] = m_block_manager.get_block_table(seq->get_id());
                    }

                    // merge copy_blocks
                    for (const auto& src_dst : copy_blocks_map) {
                        size_t src_index = src_dst.first;
                        const std::list<size_t>& dst_indexes = src_dst.second;
                        for (const auto dst_index : dst_indexes)
                            scheduler_output.m_block_copy_map[src_index].push_back(dst_index);
                    }
                }

                // if we added maximum amount of tokens to compute
                if (scheduler_output.m_total_num_scheduled_tokens == m_config.max_num_batched_tokens)
                    break;
            }
        }
    }

    void _schedule_prompt_phase_vllm(std::vector<SequenceGroup::Ptr>& sequence_groups, Output& scheduler_output) {
        // Current scheduling method schedules prompts only in a manner similar to vLLM:
        // - Limits max batch size by:
        //   - max_num_seqs (256 in vLLM's defaults)
        //   - max_num_batched_tokens (max_model_length (and at least 2048) in vLLM's defaults)

        OPENVINO_ASSERT(!m_config.dynamic_split_fuse, "Internal error: we are in vLLM scheduling");
        OPENVINO_ASSERT(m_config.max_num_seqs <= m_config.max_num_batched_tokens, "Max num batched tokens (", m_config.max_num_batched_tokens,
            ") must be greater or equal to max num sequences (", m_config.max_num_seqs, ")");
        OPENVINO_ASSERT(scheduler_output.m_scheduled_sequence_groups_ids.empty(), "Internal error: in vLLM scheduling, prompt phase is always first one");

        size_t num_running_sequences = _num_running_sequences(sequence_groups);

        for (size_t sequence_group_id = 0, num_scheduled_tokens = 0, max_sequence_len = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
            SequenceGroup::Ptr sequence_group = sequence_groups[sequence_group_id];
            if (!sequence_group->can_generate_tokens()) {
                size_t num_running_seqs = sequence_group->num_running_seqs();
                // prompt phases can have a single running sequence
                OPENVINO_ASSERT(num_running_seqs == 1);
                // here we also assume that sequence must be scheduled in a single shot and has no already generated context
                if (!m_config.enable_prefix_caching)
                    OPENVINO_ASSERT(sequence_group->get_context_len() == 0);

                size_t num_available_tokens_in_megabatch = m_config.max_num_batched_tokens - scheduler_output.m_total_num_scheduled_tokens;
                size_t sequence_len = sequence_group->get_num_available_tokens_for_batching();
                max_sequence_len = std::max(max_sequence_len, sequence_len);

                // TODO: better handling
                // e.g. return status that sequence is ignored and cannot be processed by current scheduling algorigthm
                OPENVINO_ASSERT(m_config.max_num_batched_tokens >= sequence_len, "Sequence length (", sequence_len, ") is longer than max number of tokens in batch (", m_config.max_num_batched_tokens, ")");

                // if we limited by max_num_seqs condition
                if (num_running_sequences >= m_config.max_num_seqs)
                    break;

                // apply max num batched tokens limitation
                if (num_available_tokens_in_megabatch < max_sequence_len)
                    break;

                // apply KV cache limitations
                const size_t num_required_blocks = (sequence_len + m_config.block_size - 1) / m_config.block_size;
                if (!m_block_manager.can_allocate_blocks(num_required_blocks))
                    break;

                // add scheduling information
                {
                    Sequence::Ptr sequence = (*sequence_group)[0];
                    uint64_t seq_id = sequence->get_id();

                    // and schedule tokens
                    sequence_group->schedule_tokens(sequence_len);

                    // allocate KV blocks
                    if (sequence_group->get_context_len() == 0)
                        m_block_manager.allocate(sequence, num_required_blocks, sequence_group->get_prompt_ids());
                    else 
                        m_block_manager.append_slots(sequence_group);

                    // add information to scheduler_output
                    {
                        scheduler_output.m_scheduled_sequence_groups_ids.push_back(sequence_group_id);
                        scheduler_output.m_block_tables[seq_id] = m_block_manager.get_block_table(seq_id);
                        scheduler_output.m_total_num_scheduled_tokens = max_sequence_len * scheduler_output.m_scheduled_sequence_groups_ids.size();
                    }

                    // update "is_prompt" flag
                    scheduler_output.is_prompt = true;
                }

                num_scheduled_tokens += sequence_len;
                num_running_sequences += 1;
            }
        }
    }

    void _reset_preempted_sequences(const std::vector<SequenceGroup::Ptr>& sequence_groups) {
        for (size_t sequence_group_id = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) { 
            sequence_groups[sequence_group_id]->reset_preempted_sequences();
        }
    }
};
}
