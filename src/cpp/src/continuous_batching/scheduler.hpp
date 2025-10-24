
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <vector>

#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/genai/scheduler_config.hpp"
#include "continuous_batching/block_manager.hpp"
#include "sequence_group.hpp"
#include "continuous_batching/cache_manager.hpp"
#include "continuous_batching/timer.hpp"
#include "continuous_batching/sparse_attention.hpp"
#include "utils.hpp"
#include "continuous_batching/cache_eviction.hpp"

namespace ov::genai {
class Scheduler {
    bool m_can_use_partial_preemption;

    SchedulerConfig m_config;
    std::shared_ptr<BlockManager> m_block_manager;
    friend class CacheStateDumper;

    bool m_dynamic_memory_allocation = false;

    // Dynamic KV-cache allocation params
    size_t m_kv_blocks_initial_multiplier = 2;
    const float m_cache_growth_num_tokens = 256; // Number of tokens by which KV-cache is increased

    std::shared_ptr<CacheManager> m_cache_manager;

    size_t m_snapkv_window_size = 1;
public:
    struct Output {
        // IDs of scheduled groups
        std::vector<uint64_t> m_scheduled_sequence_groups_ids;
        // block tables for scheduled sequences per each attention layer in the model
        std::map<uint64_t, std::vector<BlocksPerLayer>> m_block_tables;
        // how many previous token scores to aggregate in the paged attention score output, per sequence
        std::map<uint64_t, size_t> m_score_aggregation_windows;

        bool m_apply_sparse_attention_mask = false;
        std::map<uint64_t, std::set<size_t>> m_sparse_attention_skipped_logical_blocks;

        // XAttention thresholds per-sequence, a value of 0.0 means that XAttention is not to be applied
        // to this sequence
        std::map<uint64_t, float> m_xattention_thresholds;
        size_t m_xattention_block_size = 0;
        size_t m_xattention_stride = 0;


        // total number of scheduled tokens
        size_t m_total_num_scheduled_tokens = 0;
        // dedicated prompt phase
        bool is_prompt = false;
        // current cache usage
        float m_cache_usage = 0.0;
    };

    Scheduler(size_t block_size, std::shared_ptr<CacheManager> cache_manager, const SchedulerConfig & config = {}, size_t num_layers = 1, bool can_use_partial_preemption = true, size_t snapkv_window_size = 1) :
        m_can_use_partial_preemption(can_use_partial_preemption),
        m_config(config),
        m_cache_manager(cache_manager),
        m_snapkv_window_size(snapkv_window_size) {
        m_block_manager = std::make_shared<BlockManager>(m_config.num_kv_blocks, m_config.enable_prefix_caching, block_size, num_layers);
        OPENVINO_ASSERT(num_layers != 0, "num_layers must be non-zero");
    }

    void release() {
        m_cache_manager.reset();
        m_block_manager.reset();
    }

    Output schedule(std::vector<SequenceGroup::Ptr>& sequence_groups) {
        Output scheduler_output;
        // map of src -> dst blocks copies, which need to be performed by CacheManager
        std::map<size_t, std::list<size_t>> block_copy_map;

        // free some blocks taken by non-confirmed condidates in SD / prompt look-up
        clean_empty_blocks(sequence_groups);

        if (m_block_manager->get_total_number_of_kv_blocks() == 0) {
            _initialize_cache(sequence_groups);
        }

        if (m_config.dynamic_split_fuse) {
            // deepspeed-mii case
            // generation phase is always scheduled first
            _schedule_generate_phase_dynamic_split_fuse(sequence_groups, scheduler_output, block_copy_map);
            // some tokens from generation prompt are also scheduled
            _schedule_prompt_phase_dynamic_split_fuse(sequence_groups, scheduler_output);
        } else {
            // vLLM case
            // schedule prompt phase using whole prompt's input_ids

            _schedule_prompt_phase_vllm(sequence_groups, scheduler_output);

            if (!scheduler_output.is_prompt) {
                // prompt sequences are not scheduler => scheduler generation phase by dynamic_split_fuse implementation
                _schedule_generate_phase_dynamic_split_fuse(sequence_groups, scheduler_output, block_copy_map);
            }
        }

        m_cache_manager->allocate_cache_if_needed(m_block_manager->get_total_number_of_kv_blocks());
        _clear_waiting_sequences(sequence_groups);
        scheduler_output.m_cache_usage = m_block_manager->get_used_percentage();

        static ManualTimer copy_blocks_timer("copy block");
        copy_blocks_timer.start();
        m_cache_manager->copy_blocks(block_copy_map);
        copy_blocks_timer.end();

        return scheduler_output;
    }

    /**
     * Some requests can contain empty blocks after prompt look-up or speculative decoding
     * when candidates are not confirmed by main model and we need to free blocks, taken by these candidates
     */
    void clean_empty_blocks(std::vector<SequenceGroup::Ptr>& seq_groups) {
        for (const auto& seq_group : seq_groups)
            m_block_manager->free_empty_physical_blocks(seq_group);
    }

    const std::vector<BlocksPerLayer>& get_block_tables(const Sequence& seq) const {
        return m_block_manager->get_block_tables(seq.get_id());
    }

    const size_t get_block_size() const {
        return m_block_manager->get_block_size();
    }

    const std::vector<BlocksPerLayer>& get_block_tables(size_t seq_id) const {
        return m_block_manager->get_block_tables(seq_id);
    }

    const bool has_block_table(uint64_t seq_id) {
        return m_block_manager->has_block_table(seq_id);
    }

    void free_sequence(uint64_t seq_id) {
        m_block_manager->free_sequence(seq_id);
    }

    void fork_sequence(uint64_t parent_id, uint64_t child_id) {
        m_block_manager->fork_sequence(parent_id, child_id);
    }

    void restore_cached_blocks(const SequenceGroup::Ptr& sequence_group) {
        m_block_manager->restore_cached_blocks(sequence_group);
    }

    const SchedulerConfig& get_config() const {
        return m_config;
    }

    void free_blocks_from_sequence(size_t seq_id, const std::vector<std::set<size_t>>& per_layer_logical_block_indices_to_free) {
        m_block_manager->free_blocks_from_sequence(seq_id, per_layer_logical_block_indices_to_free);
    }

    void clear_kv_cache() {
        OPENVINO_ASSERT(m_config.enable_prefix_caching == false, "KV-cache should not be cleared if prefix caching is enabled.");
        m_cache_manager->clear();
        m_block_manager->clear();
    }

private:
    static size_t _num_running_sequence_groups(const std::vector<SequenceGroup::Ptr>& sequence_groups) {
        size_t num_running = 0;
        for (const SequenceGroup::CPtr& seq_group : sequence_groups) {
            if (seq_group->can_generate_tokens())
                ++num_running;
        }

        return num_running;
    }


    bool _preempt_by_recompute(SequenceGroup::Ptr sequence_group, size_t blocks_needed) {
        size_t processed_tokens = sequence_group->get_num_processed_tokens();
        size_t prev_blocks_count = m_block_manager->num_free_blocks();
        size_t preempted_tokens = 0;
        size_t num_blocks_occupied_by_sequence = m_block_manager->get_number_of_blocks_occupied_by_sequence(sequence_group);
        bool was_evicted_from = (sequence_group->get_num_evicted_tokens() != 0);

        if (num_blocks_occupied_by_sequence <= blocks_needed || !m_can_use_partial_preemption || was_evicted_from) {
            auto sequences = sequence_group->get_not_finished_sequences();
            for (size_t s = 0; s < sequences.size(); ++s) {
                auto seq_id = sequences[s]->get_id();
                m_block_manager->free_sequence(seq_id);
            }
            sequence_group->preempt_tokens(processed_tokens);
            if (was_evicted_from) {
                sequence_group->reset_eviction_token_count();
            }
            sequence_group->set_waiting();
            return m_block_manager->num_free_blocks() > prev_blocks_count;
        }

        size_t logical_blocks_released;
        if (sequence_group->get_sampling_parameters().is_beam_search()) {
            logical_blocks_released = m_block_manager->free_partially_beam_search_group(sequence_group, blocks_needed);
        }
        else {
            logical_blocks_released = m_block_manager->free_group_partially(sequence_group, blocks_needed);
        }

        size_t block_size = get_block_size();
        // calculate the number of preempted tokens
        auto tokens_in_last_block = processed_tokens % block_size;
        if (tokens_in_last_block == 0) {
            tokens_in_last_block = block_size;
        }
        preempted_tokens = tokens_in_last_block + (logical_blocks_released == 0 ? 0 : logical_blocks_released - 1) * block_size;

        // case when preemption requires preempt prompt tokens
        if (!m_config.dynamic_split_fuse && processed_tokens - preempted_tokens < sequence_group->get_prompt_len()) {
            // preempt prompt fully to not leave partially generated prompt
            preempted_tokens = processed_tokens;
            for (auto sequence: sequence_group->get_not_finished_sequences()) {
                auto seq_id = sequence->get_id();
                if (m_block_manager->has_block_table(seq_id)) {
                    m_block_manager->free_sequence(seq_id);
                }
            }
        }
        sequence_group->preempt_tokens(preempted_tokens);
        sequence_group->set_waiting();
        return m_block_manager->num_free_blocks() > prev_blocks_count;
    }

    static size_t _get_low_priority_sequence_group_id(const std::vector<SequenceGroup::Ptr>& sequence_groups) {
        for (size_t seq_group_id = 0, num_groups = sequence_groups.size(); seq_group_id < num_groups; ++seq_group_id) {
            size_t group_idx = num_groups - seq_group_id - 1;
            SequenceGroup::CPtr sequence_group = sequence_groups[group_idx];
            if (sequence_group->get_num_processed_tokens() > 0) {
                // we are here, because current sequence group has some reserved KV blocks in block manager
                // which can be freed
                return group_idx;
            }
        }

        return std::numeric_limits<size_t>::max();
    }

    void _apply_preemption(size_t sequence_group_id, const std::vector<SequenceGroup::Ptr>& sequence_groups) {
        SequenceGroup::Ptr sequence_group = sequence_groups[sequence_group_id];

        // check whether current sequence requires a new slot / block
        while (!m_block_manager->can_append_slots(sequence_group)) {
            // let's run a sequence for eviction
            size_t evicted_sequence_group_id = _get_low_priority_sequence_group_id(sequence_groups);

            if (evicted_sequence_group_id <= sequence_group_id) {
                // we have a cycle when current group need to evict itself to be in a running state
                break;
            }
            size_t blocks_needed = m_block_manager->required_blocks_count(sequence_group);
            if (!_preempt_by_recompute(sequence_groups[evicted_sequence_group_id], blocks_needed)){
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
        // 2. The mechanism below performs greedy scheduling of high priority prompts

        for (size_t sequence_group_id = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
            SequenceGroup::Ptr sequence_group = sequence_groups[sequence_group_id];
            if (!sequence_group->can_generate_tokens() && !sequence_group->is_waiting() && !sequence_group->handle_stopped() && !sequence_group->handle_cancelled()) {
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
                size_t block_size = get_block_size();
                size_t currently_allocated_token_slots = sequence_group->get_num_blocks() * block_size;
                size_t occupied_token_slots = sequence_group->get_num_processed_tokens() - sequence_group->get_num_evicted_tokens();
                OPENVINO_ASSERT(currently_allocated_token_slots >= occupied_token_slots, "internal error");
                size_t available_slots = currently_allocated_token_slots - occupied_token_slots,
                       required_slots = num_scheduled_tokens > available_slots ? num_scheduled_tokens - available_slots : 0;
                size_t num_required_blocks = (required_slots + block_size - 1) / block_size;
                while (num_required_blocks > m_block_manager->num_free_blocks()) {
                    if (!_try_increase_cache()) {
                        break;
                    }
                }
                size_t num_scheduled_blocks = std::min(num_required_blocks, m_block_manager->num_free_blocks());
                // some scheduled blocks can be no fully occupied, so we need to take min between num_scheduled_blocks
                // and total "scheduled capacity"
                num_scheduled_tokens = std::min(num_scheduled_tokens, available_slots + num_scheduled_blocks * block_size);

                if (num_scheduled_tokens > 0) {
                    // allocate KV blocks if required
                    if (num_scheduled_blocks > 0)
                        m_block_manager->allocate(sequence, num_scheduled_blocks, sequence_group->get_prompt_len());
                    // and schedule tokens
                    sequence_group->schedule_tokens(num_scheduled_tokens);

                    // add information to scheduler_output
                    {
                        scheduler_output.m_scheduled_sequence_groups_ids.push_back(sequence_group_id);
                        scheduler_output.m_block_tables[seq_id] = m_block_manager->get_block_tables(seq_id);
                        scheduler_output.m_total_num_scheduled_tokens += num_scheduled_tokens * num_running_seqs;


                        scheduler_output.m_score_aggregation_windows[seq_id] = _schedule_scores_to_aggregate(sequence_group);
                        scheduler_output.m_apply_sparse_attention_mask = m_config.use_sparse_attention && m_config.sparse_attention_config.mode == SparseAttentionMode::TRISHAPE;
                        if (scheduler_output.m_apply_sparse_attention_mask) {
                            TriShapeSparseAttentionTokenSkipper skipper(block_size,
                                    m_config.sparse_attention_config.num_last_dense_tokens_in_prefill,
                                    m_config.sparse_attention_config.num_retained_start_tokens_in_cache,
                                    m_config.sparse_attention_config.num_retained_recent_tokens_in_cache);
                            scheduler_output.m_sparse_attention_skipped_logical_blocks[seq_id] = skipper.get_skipped_blocks(sequence_group);
                        }
                        scheduler_output.m_xattention_thresholds[seq_id] = _schedule_xattention_threshold(sequence_group);
                        scheduler_output.m_xattention_block_size = m_config.sparse_attention_config.xattention_block_size;
                        scheduler_output.m_xattention_stride = m_config.sparse_attention_config.xattention_stride;
                    }
                }

                // if we added maximum amount of tokens to compute
                if (scheduler_output.m_total_num_scheduled_tokens == m_config.max_num_batched_tokens)
                    break;
            }
        }
    }

    void _schedule_generate_phase_dynamic_split_fuse(const std::vector<SequenceGroup::Ptr>& sequence_groups,
                                                     Output& scheduler_output,
                                                     std::map<size_t, std::list<size_t>>& block_copy_map) {
        for (size_t sequence_group_id = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
            SequenceGroup::Ptr sequence_group = sequence_groups[sequence_group_id];
            // Note, that can_generate_tokens will mix preempted sequence groups
            // and real generate ones
            // Question: do we need to schedule preeempted first as it's done in vLLM?
            // Answer: preempted sequences have low priority, so they should be after "running" ones. So, here we
            //         keep latencies for sequence groups of high priority
            if (sequence_group->can_generate_tokens() && !sequence_group->is_waiting() && !sequence_group->handle_stopped() && !sequence_group->handle_cancelled()) {
                OPENVINO_ASSERT(!sequence_group->has_finished());
                size_t num_running_seqs = sequence_group->num_running_seqs();
                OPENVINO_ASSERT(num_running_seqs);
                size_t num_tokens_in_megabatch = m_config.max_num_batched_tokens - scheduler_output.m_total_num_scheduled_tokens;
                size_t available_tokens_per_seq_in_megabatch = num_tokens_in_megabatch / num_running_seqs;

                // we cannot schedule even a single token per each sequence in a group
                if (!available_tokens_per_seq_in_megabatch)
                    continue;

                // Note: current function can return more than 1 token even for generation phase in case of some tokens
                // of current sequence group were evicted before
                size_t num_available_tokens_per_seq = sequence_group->get_num_available_tokens_for_batching();

                size_t num_scheduled_tokens_per_seq = std::min(available_tokens_per_seq_in_megabatch, num_available_tokens_per_seq);
                sequence_group->schedule_tokens(num_scheduled_tokens_per_seq);

                while (!m_block_manager->can_append_slots(sequence_group)){
                    if (!_try_increase_cache()) {
                        break;
                    }
                }

                _apply_preemption(sequence_group_id, sequence_groups);

                // if we can't preemt any more sequences, clear scheduled tokens and move to next sequence
                if (!m_block_manager->can_append_slots(sequence_group)) {
                    sequence_group->clear_scheduled_tokens();
                    continue;
                }

                // allocate new slots
                std::map<size_t, std::list<size_t>> copy_blocks_map = m_block_manager->append_slots(sequence_group);

                // add information to scheduler_output
                {
                    auto request_id = sequence_group->get_request_id();
                    scheduler_output.m_scheduled_sequence_groups_ids.push_back(sequence_group_id);
                    scheduler_output.m_total_num_scheduled_tokens += num_scheduled_tokens_per_seq * num_running_seqs;

                    std::vector<Sequence::Ptr> running_seqs = sequence_group->get_running_sequences();
                    for (const auto & seq : sequence_group->get_running_sequences()) {
                        size_t seq_id = seq->get_id();
                        // block tables for each running sequence within a group
                        scheduler_output.m_block_tables[seq_id] = m_block_manager->get_block_tables(seq_id);

                        scheduler_output.m_score_aggregation_windows[seq_id] = _schedule_scores_to_aggregate(sequence_group);
                        scheduler_output.m_xattention_block_size = m_config.sparse_attention_config.xattention_block_size;
                        scheduler_output.m_xattention_stride = m_config.sparse_attention_config.xattention_stride;
                    }



                    // merge copy_blocks
                    for (const auto& src_dst : copy_blocks_map) {
                        size_t src_index = src_dst.first;
                        const std::list<size_t>& dst_indexes = src_dst.second;
                        for (const auto dst_index : dst_indexes)
                            block_copy_map[src_index].push_back(dst_index);
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

        // TODO: it currently does not handle beam search, where beam width should contribute to total number of "num running sequences"
        size_t num_running_sequence_groups = _num_running_sequence_groups(sequence_groups);

        for (size_t sequence_group_id = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
            SequenceGroup::Ptr sequence_group = sequence_groups[sequence_group_id];
            const bool recompute_evicted_sequences = sequence_group->get_num_processed_tokens() == 0 && !m_can_use_partial_preemption;
            if ((!sequence_group->can_generate_tokens() || recompute_evicted_sequences) && !sequence_group->is_waiting() && !sequence_group->handle_stopped() && !sequence_group->handle_cancelled()) {
                size_t num_running_seqs = sequence_group->num_running_seqs();
                // prompt phases can have a single running sequence
                OPENVINO_ASSERT(num_running_seqs == 1);
                // here we also assume that sequence must be scheduler in a single shot and has no already generated context
                if (!m_config.enable_prefix_caching)
                    OPENVINO_ASSERT(sequence_group->get_context_len() == 0);
                size_t num_available_tokens_in_megabatch = m_config.max_num_batched_tokens - scheduler_output.m_total_num_scheduled_tokens;
                size_t sequence_len = sequence_group->get_num_available_tokens_for_batching();

                // TODO: better handling
                // e.g. return status that sequence is ignored and cannot be processed by current scheduling algorigthm
                OPENVINO_ASSERT(m_config.max_num_batched_tokens >= sequence_len, "Sequence length (", sequence_len, ") is longer than max number of tokens in batch (", m_config.max_num_batched_tokens, ")");

                // if we limited by max_num_seqs condition
                if (num_running_sequence_groups >= m_config.max_num_seqs)
                    break;

                // apply max num batched tokens limitation
                if (num_available_tokens_in_megabatch < sequence_len)
                    break;

                // apply KV cache limitations
                size_t block_size = get_block_size();
                const size_t num_required_blocks = (sequence_len + block_size - 1) / block_size;
                while (!m_block_manager->can_allocate_blocks(num_required_blocks)){
                    if (!_try_increase_cache()) {
                        break;
                    }
                }
                if (!m_block_manager->can_allocate_blocks(num_required_blocks))
                    break;

                // add scheduling information
                {
                    Sequence::Ptr sequence = (*sequence_group)[0];
                    uint64_t seq_id = sequence->get_id();
                    // and schedule tokens
                    sequence_group->schedule_tokens(sequence_len);

                    // allocate KV blocks
                    m_block_manager->append_slots(sequence_group);

                    // add information to scheduler_output
                    {
                        scheduler_output.m_scheduled_sequence_groups_ids.push_back(sequence_group_id);
                        uint64_t seq_id = sequence_group->get_running_sequences()[0]->get_id();
                        scheduler_output.m_block_tables[seq_id] = m_block_manager->get_block_tables(seq_id);
                        scheduler_output.m_total_num_scheduled_tokens += sequence_len;
                        scheduler_output.m_score_aggregation_windows[seq_id] = _schedule_scores_to_aggregate(sequence_group);
                        scheduler_output.m_xattention_thresholds[seq_id] = _schedule_xattention_threshold(sequence_group);
                        scheduler_output.m_xattention_block_size = m_config.sparse_attention_config.xattention_block_size;
                        scheduler_output.m_xattention_stride = m_config.sparse_attention_config.xattention_stride;
                    }

                    // update "is_prompt" flag
                    scheduler_output.is_prompt = true;
                }

                num_running_sequence_groups += 1;
            }
        }
    }

    void _clear_waiting_sequences(const std::vector<SequenceGroup::Ptr>& sequence_groups) {
        for (size_t sequence_group_id = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
            sequence_groups[sequence_group_id]->clear_waiting_sequences();
        }
    }

    void _initialize_cache(const std::vector<SequenceGroup::Ptr>& sequence_groups) {
        size_t blocks_sum = 0;
        for (auto idx = 0; idx < sequence_groups.size(); idx++) {
            auto seq_length = sequence_groups[idx]->get_prompt_len() * m_kv_blocks_initial_multiplier;
            const auto& gen_config = sequence_groups[idx]->get_sampling_parameters();
            seq_length = std::min(seq_length, sequence_groups[idx]->get_prompt_len() + sequence_groups[idx]->get_max_new_tokens());
            size_t blocks_num = std::ceil(static_cast<float>(seq_length) / m_block_manager->get_block_size());
            if (gen_config.is_beam_search()) {
                blocks_num *= gen_config.num_beams;
            } else if (gen_config.is_multinomial()) {
                blocks_num *= gen_config.num_return_sequences;
            }
            blocks_sum += blocks_num;
        }
        m_block_manager->increase_kv_blocks_number(blocks_sum);
        m_dynamic_memory_allocation = true;
    }

    bool _try_increase_cache() {
        if (!m_dynamic_memory_allocation) {
            return false;
        }
        auto device = m_cache_manager->get_device();
        size_t current_num_of_kv_blocks = m_block_manager->get_total_number_of_kv_blocks();
        size_t new_blocks_num = current_num_of_kv_blocks + std::ceil(m_cache_growth_num_tokens / get_block_size());

        if (device.find("GPU") == std::string::npos) {
            m_block_manager->increase_kv_blocks_number(new_blocks_num);
        } else {
            const size_t available_gpu_memory = utils::get_available_gpu_memory(m_cache_manager->get_device(), m_cache_manager->get_num_decoder_layers());
            const size_t block_size_in_bytes = m_cache_manager->get_block_size_in_bytes();
            size_t required_memory = (new_blocks_num - current_num_of_kv_blocks) * block_size_in_bytes;
            if (required_memory <= available_gpu_memory) {
                m_block_manager->increase_kv_blocks_number(new_blocks_num);
            } else {
                size_t possible_blocks_to_add = available_gpu_memory / block_size_in_bytes;
                if (possible_blocks_to_add > 0) {
                    m_block_manager->increase_kv_blocks_number(current_num_of_kv_blocks + possible_blocks_to_add);
                } else {
                    return false;
                }
            }
        }
        return true;
    }

    size_t _schedule_scores_to_aggregate(SequenceGroup::Ptr sequence_group) {
        auto calculator = SnapKVScoreAggregationCalculator(m_snapkv_window_size);

        size_t num_scheduled_tokens = sequence_group->get_num_scheduled_tokens();
        size_t num_processed_tokens = sequence_group->get_num_processed_tokens();
        size_t prompt_len = sequence_group->get_prompt_len();

        return calculator.get_num_token_scores_to_aggregate(prompt_len, num_scheduled_tokens, num_processed_tokens);
    }

    float _schedule_xattention_threshold(SequenceGroup::Ptr sequence_group) {
        if (!(m_config.use_sparse_attention && m_config.sparse_attention_config.mode == SparseAttentionMode::XATTENTION)) {
            return 0.0;
        }
        if (sequence_group->can_generate_tokens() && sequence_group->get_num_scheduled_tokens() == 1) {
            // generation phase, excluding last prompt chunk
            return 0.0;
        }

        size_t prompt_len = sequence_group->get_prompt_len();
        size_t num_processed_tokens_after_this_chunk = sequence_group->get_num_processed_tokens() + sequence_group->get_num_scheduled_tokens();

        if (num_processed_tokens_after_this_chunk < prompt_len && (prompt_len - num_processed_tokens_after_this_chunk) < m_config.sparse_attention_config.num_last_dense_tokens_in_prefill) {
            // dense attention chunk, potentially overspilling to an extra chunk before that
            return 0.0;
        }

        return m_config.sparse_attention_config.xattention_threshold;
    }

};

}
