
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/genai/scheduler_config.hpp"
#include "continuous_batching/cache/cache_orchestrator.hpp"
#include "sequence_group.hpp"
#include "continuous_batching/sparse_attention.hpp"
#include "utils.hpp"
#include "continuous_batching/cache/cache_eviction.hpp"

namespace ov::genai {
class Scheduler {
public:
    // Stable data that doesn't change across scheduling calls
    struct KVPagedAttentionGlobalData {
        KVPagedAttentionGlobalData() = default;

        explicit KVPagedAttentionGlobalData(const SchedulerConfig& config) :
            apply_sparse_attention_mask(config.use_sparse_attention && config.sparse_attention_config.mode == SparseAttentionMode::TRISHAPE),
            xattention_block_size(config.sparse_attention_config.xattention_block_size),
            xattention_stride(config.sparse_attention_config.xattention_stride),
            adaptive_rkv_start_size(config.cache_eviction_config.get_start_size()) {
        }

        bool apply_sparse_attention_mask = false;
        size_t xattention_block_size = 0;
        size_t xattention_stride = 0;
        size_t adaptive_rkv_start_size = 0;
    };

private:
    bool m_can_use_partial_preemption;

    SchedulerConfig m_config;
    std::shared_ptr<const KVPagedAttentionGlobalData> m_kv_paged_attention_global_data;
    std::shared_ptr<CacheOrchestrator> m_cache_orchestrator;
    friend class CacheStateDumper;

    bool m_dynamic_memory_allocation = false;

    // Dynamic KV-cache allocation params
    size_t m_kv_blocks_initial_multiplier = 2;
    const float m_cache_growth_num_tokens = 256; // Number of tokens by which KV-cache is increased

    size_t m_snapkv_window_size = 1;
    std::map<uint64_t, size_t> m_expected_num_scheduled_tokens;

public:
    struct Output {
        struct KVPagedAttentionData {
            std::vector<BlocksPerLayer> block_tables;
            size_t score_aggregation_window = 0;
            bool has_score_aggregation_window = false;
            std::set<size_t> sparse_attention_skipped_logical_blocks;
            float xattention_threshold = 0.0f;
            size_t adaptive_rkv_evictable_size = 0;
            bool has_adaptive_rkv_evictable_size = false;
            std::vector<std::vector<size_t>> adaptive_rkv_diversity_block_sets;
        };

        struct LinearAttentionPagingData {
            std::vector<int32_t> block_indices;
            int32_t past_length = 0;
            int32_t cache_interval = 0;
        };

        // IDs of scheduled groups
        std::vector<uint64_t> m_scheduled_sequence_groups_ids;
        std::map<uint64_t, KVPagedAttentionData> m_kv_paged_attention_data;
        std::map<uint64_t, LinearAttentionPagingData> m_linear_attention_paging_data;
        std::shared_ptr<const KVPagedAttentionGlobalData> m_kv_paged_attention_global_data;

        // total number of scheduled tokens
        size_t m_total_num_scheduled_tokens = 0;
        // dedicated prompt phase
        bool is_prompt = false;
        // maximum cache usage across registered cache types
        float m_cache_usage = 0.0;
        // total allocated cache size in bytes across registered cache types
        size_t m_cache_size_in_bytes = 0;

        void set_kv_block_tables(uint64_t seq_id, const std::vector<BlocksPerLayer>& block_tables) {
            m_kv_paged_attention_data[seq_id].block_tables = block_tables;
        }

        void set_score_aggregation_window(uint64_t seq_id, size_t score_aggregation_window) {
            KVPagedAttentionData& kv_data = m_kv_paged_attention_data[seq_id];
            kv_data.score_aggregation_window = score_aggregation_window;
            kv_data.has_score_aggregation_window = true;
        }

        void set_kv_paged_attention_global_data(const std::shared_ptr<const KVPagedAttentionGlobalData>& global_data) {
            m_kv_paged_attention_global_data = global_data;
        }

        void set_sparse_attention_skipped_logical_blocks(uint64_t seq_id, const std::set<size_t>& skipped_logical_blocks) {
            KVPagedAttentionData& kv_data = m_kv_paged_attention_data[seq_id];
            kv_data.sparse_attention_skipped_logical_blocks = skipped_logical_blocks;
        }

        void set_xattention_threshold(uint64_t seq_id, float threshold) {
            m_kv_paged_attention_data[seq_id].xattention_threshold = threshold;
        }

        void set_adaptive_rkv_evictable_size(uint64_t seq_id, size_t evictable_size) {
            KVPagedAttentionData& kv_data = m_kv_paged_attention_data[seq_id];
            kv_data.adaptive_rkv_evictable_size = evictable_size;
            kv_data.has_adaptive_rkv_evictable_size = evictable_size > 0;
        }

        void set_linear_attention_paging_data(uint64_t seq_id, LinearAttentionPagingData&& paging_data) {
            m_linear_attention_paging_data[seq_id] = std::move(paging_data);
        }

        const std::vector<BlocksPerLayer>& get_kv_block_tables(uint64_t seq_id) const {
            return get_kv_paged_attention_data(seq_id).block_tables;
        }

        const KVPagedAttentionData& get_kv_paged_attention_data(uint64_t seq_id) const {
            auto it = m_kv_paged_attention_data.find(seq_id);
            OPENVINO_ASSERT(it != m_kv_paged_attention_data.end(),
                            "KV paged attention data was not scheduled for sequence ", seq_id);
            return it->second;
        }

        const KVPagedAttentionGlobalData& get_kv_paged_attention_global_data() const {
            static const KVPagedAttentionGlobalData default_global_data;
            return m_kv_paged_attention_global_data != nullptr ? *m_kv_paged_attention_global_data : default_global_data;
        }

        bool has_score_aggregation_window(uint64_t seq_id) const {
            auto it = m_kv_paged_attention_data.find(seq_id);
            return it != m_kv_paged_attention_data.end() && it->second.has_score_aggregation_window;
        }

        size_t get_score_aggregation_window(uint64_t seq_id) const {
            const KVPagedAttentionData& kv_data = get_kv_paged_attention_data(seq_id);
            OPENVINO_ASSERT(kv_data.has_score_aggregation_window, "Score aggregation window was not scheduled for sequence ", seq_id);
            return kv_data.score_aggregation_window;
        }

        const std::set<size_t>& get_sparse_attention_skipped_logical_blocks(uint64_t seq_id) const {
            return get_kv_paged_attention_data(seq_id).sparse_attention_skipped_logical_blocks;
        }

        float get_xattention_threshold(uint64_t seq_id) const {
            auto it = m_kv_paged_attention_data.find(seq_id);
            return it == m_kv_paged_attention_data.end() ? 0.0f : it->second.xattention_threshold;
        }

        size_t get_adaptive_rkv_evictable_size(uint64_t seq_id) const {
            auto it = m_kv_paged_attention_data.find(seq_id);
            return it == m_kv_paged_attention_data.end() ? 0 : it->second.adaptive_rkv_evictable_size;
        }

        bool has_adaptive_rkv_evictable_size(uint64_t seq_id) const {
            auto it = m_kv_paged_attention_data.find(seq_id);
            return it != m_kv_paged_attention_data.end() && it->second.has_adaptive_rkv_evictable_size;
        }

        const std::vector<std::vector<size_t>>& get_adaptive_rkv_diversity_block_sets(uint64_t seq_id) const {
            return get_kv_paged_attention_data(seq_id).adaptive_rkv_diversity_block_sets;
        }

        bool has_linear_attention_paging_data() const {
            return !m_linear_attention_paging_data.empty();
        }

        bool has_linear_attention_paging_data(uint64_t seq_id) const {
            return m_linear_attention_paging_data.find(seq_id) != m_linear_attention_paging_data.end();
        }

        const LinearAttentionPagingData& get_linear_attention_paging_data(uint64_t seq_id) const {
            auto it = m_linear_attention_paging_data.find(seq_id);
            OPENVINO_ASSERT(it != m_linear_attention_paging_data.end(),
                            "Linear attention paging data was not scheduled for sequence ", seq_id);
            return it->second;
        }
    };

    Scheduler(std::shared_ptr<CacheOrchestrator> cache_orchestrator, const SchedulerConfig & config = {}, bool can_use_partial_preemption = true, size_t snapkv_window_size = 1) :
        m_can_use_partial_preemption(can_use_partial_preemption),
        m_config(config),
        m_kv_paged_attention_global_data(std::make_shared<const KVPagedAttentionGlobalData>(config)),
        m_cache_orchestrator(std::move(cache_orchestrator)),
        m_snapkv_window_size(snapkv_window_size) {
    }

    void release() {
        m_cache_orchestrator.reset();
    }

    Output schedule(std::vector<SequenceGroup::Ptr>& sequence_groups) {
        Output scheduler_output;
        scheduler_output.set_kv_paged_attention_global_data(m_kv_paged_attention_global_data);
        // map of src -> dst blocks copies per cache type
        std::map<CacheType, std::map<size_t, std::list<size_t>>> typed_block_copy_map;

        // free some blocks taken by non-confirmed candidates in SD / prompt look-up
        clean_empty_blocks(sequence_groups);

        if (!m_cache_orchestrator->has_token_capacity()) {
            _initialize_cache(sequence_groups);
        } else if (m_dynamic_memory_allocation) {
            // Cache already initialized and growing dynamically: make sure a newly-arrived prompt
            // has its capacity (on top of running sequences) reserved in one reallocation rather
            // than growing in small chunks across the prefill.
            _reserve_dynamic_capacity_for_active_sequences(sequence_groups);
        }

        if (m_config.dynamic_split_fuse) {
            // deepspeed-mii case
            // generation phase is always scheduled first
            _schedule_generate_phase_dynamic_split_fuse(sequence_groups, scheduler_output, typed_block_copy_map);
            // some tokens from generation prompt are also scheduled
            _schedule_prompt_phase_dynamic_split_fuse(sequence_groups, scheduler_output);
        } else {
            // vLLM case
            // schedule prompt phase using whole prompt's input_ids

            _schedule_prompt_phase_vllm(sequence_groups, scheduler_output);

            if (!scheduler_output.is_prompt) {
                // prompt sequences are not scheduler => scheduler generation phase by dynamic_split_fuse implementation
                _schedule_generate_phase_dynamic_split_fuse(sequence_groups, scheduler_output, typed_block_copy_map);
            }
        }

        m_cache_orchestrator->allocate_cache_if_needed();
        _clear_waiting_sequences(sequence_groups);
        scheduler_output.m_cache_usage = m_cache_orchestrator->get_used_percentage();
        scheduler_output.m_cache_size_in_bytes = m_cache_orchestrator->get_total_cache_size_in_bytes();

        m_cache_orchestrator->copy_blocks(typed_block_copy_map);

        return scheduler_output;
    }

    /**
     * Some requests can contain empty blocks after prompt look-up or speculative decoding
     * when candidates are not confirmed by main model and we need to free blocks, taken by these candidates
     */
    void clean_empty_blocks(std::vector<SequenceGroup::Ptr>& seq_groups) {
        for (const auto& seq_group : seq_groups)
            m_cache_orchestrator->free_empty_physical_blocks(seq_group);
    }

    const std::vector<BlocksPerLayer>& get_kv_block_tables(const Sequence& seq) const {
        return m_cache_orchestrator->get_kv_block_tables(seq.get_id());
    }

    size_t get_block_size(CacheType type) const {
        return m_cache_orchestrator->get_block_size(type);
    }

    size_t get_num_kv_logical_blocks(SequenceGroup::CPtr seq_group) const {
        return m_cache_orchestrator->get_num_kv_logical_blocks(seq_group);
    }

    const std::vector<BlocksPerLayer>& get_kv_block_tables(size_t seq_id) const {
        return m_cache_orchestrator->get_kv_block_tables(seq_id);
    }

    const bool has_block_table(uint64_t seq_id) {
        return m_cache_orchestrator->has_block_table(seq_id);
    }

    void free_sequence(uint64_t seq_id) {
        m_cache_orchestrator->free_sequence(seq_id);
    }

    void fork_sequence(uint64_t parent_id, uint64_t child_id) {
        m_cache_orchestrator->fork_sequence(parent_id, child_id);
    }

    void restore_cached_blocks(const SequenceGroup::Ptr& sequence_group) {
        m_cache_orchestrator->restore_cached_blocks(sequence_group);
    }

    const SchedulerConfig& get_config() const {
        return m_config;
    }

    void free_blocks_from_sequence(size_t seq_id, const std::vector<std::set<size_t>>& per_layer_logical_block_indices_to_free, CacheType cache_type) {
        m_cache_orchestrator->free_blocks_from_sequence(seq_id, per_layer_logical_block_indices_to_free, cache_type);
    }

    void clear_cache() {
        OPENVINO_ASSERT(m_config.enable_prefix_caching == false, "Cache should not be cleared if prefix caching is enabled.");
        m_cache_orchestrator->clear();
    }

    void set_expected_num_scheduled_tokens(uint64_t request_id, size_t num_tokens) {
        m_expected_num_scheduled_tokens[request_id] = num_tokens;
    }

    size_t get_expected_num_scheduled_tokens(uint64_t request_id) const {
        auto it = m_expected_num_scheduled_tokens.find(request_id);
        if (it != m_expected_num_scheduled_tokens.end()) {
            return it->second;
        }
        return 0;
    }

    void clear_expected_num_scheduled_tokens(uint64_t request_id) {
        m_expected_num_scheduled_tokens.erase(request_id);
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


    bool _preempt_by_recompute(SequenceGroup::Ptr victim, SequenceGroup::CPtr target) {
        size_t processed_tokens = victim->get_num_processed_tokens();
        size_t prev_blocks_count = m_cache_orchestrator->num_free_blocks();
        size_t preempted_tokens = 0;
        bool was_evicted_from = (victim->get_num_evicted_tokens() != 0);

        if (!m_cache_orchestrator->can_partially_preempt(victim, target) || !m_can_use_partial_preemption || was_evicted_from) {
            auto sequences = victim->get_not_finished_sequences();
            for (size_t s = 0; s < sequences.size(); ++s) {
                auto seq_id = sequences[s]->get_id();
                m_cache_orchestrator->free_sequence(seq_id);
            }
            victim->preempt_tokens(processed_tokens);
            if (was_evicted_from) {
                victim->reset_eviction_token_count();
            }
            victim->set_waiting();
            return m_cache_orchestrator->num_free_blocks() > prev_blocks_count;
        }

        if (victim->get_sampling_parameters().is_beam_search()) {
            preempted_tokens = m_cache_orchestrator->free_partially_beam_search_group_for_target(victim, target);
        }
        else {
            preempted_tokens = m_cache_orchestrator->free_group_partially_for_target(victim, target);
        }

        preempted_tokens = std::min(preempted_tokens, processed_tokens);

        // case when preemption requires preempt prompt tokens
        if (!m_config.dynamic_split_fuse && processed_tokens - preempted_tokens < victim->get_prompt_len()) {
            // preempt prompt fully to not leave partially generated prompt
            preempted_tokens = processed_tokens;
            for (auto sequence: victim->get_not_finished_sequences()) {
                auto seq_id = sequence->get_id();
                if (m_cache_orchestrator->has_block_table(seq_id)) {
                    m_cache_orchestrator->free_sequence(seq_id);
                }
            }
        }
        victim->preempt_tokens(preempted_tokens);
        victim->set_waiting();
        return m_cache_orchestrator->num_free_blocks() > prev_blocks_count;
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
        while (!m_cache_orchestrator->can_append_slots(sequence_group)) {
            // let's run a sequence for eviction
            size_t evicted_sequence_group_id = _get_low_priority_sequence_group_id(sequence_groups);

            if (evicted_sequence_group_id <= sequence_group_id) {
                // we have a cycle when current group need to evict itself to be in a running state
                break;
            }
            if (!_preempt_by_recompute(sequence_groups[evicted_sequence_group_id], sequence_group)){
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

                // use externally expected scheduling size when an external expectation is provided.
                auto it_expected_scheduled_tokens =
                    m_expected_num_scheduled_tokens.find(sequence_group->get_request_id());
                if (it_expected_scheduled_tokens != m_expected_num_scheduled_tokens.end()) {
                    const size_t expected_num_scheduled_tokens = it_expected_scheduled_tokens->second;
                    if (expected_num_scheduled_tokens > 0 && expected_num_scheduled_tokens < num_scheduled_tokens) {
                        num_scheduled_tokens = expected_num_scheduled_tokens;
                    }
                }

                // apply KV cache limitations
                while (m_cache_orchestrator->available_token_slots(sequence_group) < num_scheduled_tokens) {
                    if (!_try_increase_cache(sequence_group)) {
                        break;
                    }
                }
                num_scheduled_tokens = std::min(num_scheduled_tokens, m_cache_orchestrator->available_token_slots(sequence_group));

                if (num_scheduled_tokens > 0) {
                    // allocate KV blocks if required
                    m_cache_orchestrator->allocate_tokens(sequence, sequence_group, num_scheduled_tokens, sequence_group->get_prompt_len());
                    // and schedule tokens
                    sequence_group->schedule_tokens(num_scheduled_tokens);

                    // add information to scheduler_output
                    {
                        scheduler_output.m_scheduled_sequence_groups_ids.push_back(sequence_group_id);
                        _set_kv_paged_attention_data(scheduler_output, sequence_group, seq_id);
                        scheduler_output.m_total_num_scheduled_tokens += num_scheduled_tokens * num_running_seqs;

                        // fill linear attention block tables if registered
                        if (m_cache_orchestrator->has_linear_attention_cache()) {
                            const auto& la_blocks = m_cache_orchestrator->get_linear_attention_block_table(seq_id);
                            const size_t la_block_logical_start = m_cache_orchestrator->get_linear_attention_block_table_logical_start(seq_id);
                            _set_linear_attention_paging_data(scheduler_output, sequence_group, seq_id, la_blocks, la_block_logical_start);
                        }
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
                                                     std::map<CacheType, std::map<size_t, std::list<size_t>>>& typed_block_copy_map) {
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

                while (!m_cache_orchestrator->can_append_slots(sequence_group)) {
                    if (!_try_increase_cache(sequence_group)) {
                        break;
                    }
                }

                _apply_preemption(sequence_group_id, sequence_groups);

                // if we can't preemt any more sequences, clear scheduled tokens and move to next sequence
                if (!m_cache_orchestrator->can_append_slots(sequence_group)) {
                    sequence_group->clear_scheduled_tokens();
                    continue;
                }

                // allocate new slots
                std::map<CacheType, std::map<size_t, std::list<size_t>>> per_type_copy_map = m_cache_orchestrator->append_slots(sequence_group);

                // add information to scheduler_output
                {
                    auto request_id = sequence_group->get_request_id();
                    scheduler_output.m_scheduled_sequence_groups_ids.push_back(sequence_group_id);
                    scheduler_output.m_total_num_scheduled_tokens += num_scheduled_tokens_per_seq * num_running_seqs;

                    std::vector<Sequence::Ptr> running_seqs = sequence_group->get_running_sequences();
                    for (const auto & seq : sequence_group->get_running_sequences()) {
                        size_t seq_id = seq->get_id();
                        // block tables for each running sequence within a group
                        _set_kv_paged_attention_data(scheduler_output, sequence_group, seq_id);
                    }

                    for (auto& [type, copy_map] : per_type_copy_map) {
                        auto& accumulated_copy_map = typed_block_copy_map[type];
                        for (auto& [src_index, dst_indexes] : copy_map) {
                            auto& accumulated_dst_indexes = accumulated_copy_map[src_index];
                            accumulated_dst_indexes.splice(accumulated_dst_indexes.end(), dst_indexes);
                        }
                    }

                    // fill linear attention block tables if registered
                    if (m_cache_orchestrator->has_linear_attention_cache()) {
                        for (const auto& seq : sequence_group->get_running_sequences()) {
                            size_t sid = seq->get_id();
                            const auto& la_blocks = m_cache_orchestrator->get_linear_attention_block_table(sid);
                            const size_t la_block_logical_start = m_cache_orchestrator->get_linear_attention_block_table_logical_start(sid);
                            _set_linear_attention_paging_data(scheduler_output, sequence_group, sid, la_blocks, la_block_logical_start);
                        }
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
                while (!m_cache_orchestrator->can_allocate_tokens(sequence_group, sequence_len)){
                    if (!_try_increase_cache(sequence_group)) {
                        break;
                    }
                }
                if (!m_cache_orchestrator->can_allocate_tokens(sequence_group, sequence_len))
                    break;

                // add scheduling information
                {
                    // and schedule tokens
                    sequence_group->schedule_tokens(sequence_len);

                    // allocate KV blocks
                    m_cache_orchestrator->append_slots(sequence_group);

                    // add information to scheduler_output
                    {
                        scheduler_output.m_scheduled_sequence_groups_ids.push_back(sequence_group_id);
                        uint64_t seq_id = sequence_group->get_running_sequences()[0]->get_id();
                        _set_kv_paged_attention_data(scheduler_output, sequence_group, seq_id);
                        scheduler_output.m_total_num_scheduled_tokens += sequence_len;

                        // fill linear attention block tables if registered
                        if (m_cache_orchestrator->has_linear_attention_cache()) {
                            const auto& la_blocks = m_cache_orchestrator->get_linear_attention_block_table(seq_id);
                            const size_t la_block_logical_start = m_cache_orchestrator->get_linear_attention_block_table_logical_start(seq_id);
                            _set_linear_attention_paging_data(scheduler_output, sequence_group, seq_id, la_blocks, la_block_logical_start);
                        }
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

    // Build (tokens-per-sequence, num-sequences) capacity targets for the given sequence groups,
    // sized to each group's expected total length. Also accumulates the total concurrent sequence
    // count (for fixed-size-per-sequence pools).
    std::vector<std::pair<size_t, size_t>> _build_sequence_token_targets(
            const std::vector<SequenceGroup::Ptr>& sequence_groups,
            size_t& total_concurrent_seqs) {
        std::vector<std::pair<size_t, size_t>> sequence_token_targets;
        sequence_token_targets.reserve(sequence_groups.size());
        total_concurrent_seqs = 0;
        for (size_t idx = 0; idx < sequence_groups.size(); idx++) {
            size_t seq_length = sequence_groups[idx]->get_prompt_len() * m_kv_blocks_initial_multiplier;
            const auto& gen_config = sequence_groups[idx]->get_sampling_parameters();
            seq_length = std::min(seq_length, sequence_groups[idx]->get_prompt_len() + sequence_groups[idx]->get_max_new_tokens());
            size_t concurrent_seqs = 1;
            if (gen_config.is_beam_search()) {
                concurrent_seqs = gen_config.num_beams;
            } else if (gen_config.is_multinomial()) {
                concurrent_seqs = gen_config.num_return_sequences;
            }
            total_concurrent_seqs += concurrent_seqs;
            sequence_token_targets.emplace_back(seq_length, concurrent_seqs);
        }
        return sequence_token_targets;
    }

    void _initialize_cache(const std::vector<SequenceGroup::Ptr>& sequence_groups) {
        size_t total_concurrent_seqs = 0;
        auto sequence_token_targets = _build_sequence_token_targets(sequence_groups, total_concurrent_seqs);
        m_cache_orchestrator->ensure_sequence_token_capacity(sequence_token_targets);
        // Fixed-size-per-sequence managers (e.g. linear attention) are not covered by
        // ensure_sequence_token_capacity.  Pre-grow their pool to the number of arriving sequences
        // so the prompt phase can allocate without triggering _try_increase_cache.
        m_cache_orchestrator->grow_fixed_size_capacity(total_concurrent_seqs);
        m_dynamic_memory_allocation = true;
    }

    // In dynamic-allocation mode, pre-grow the variable-size caches so a newly-arriving prompt
    // fits in a single reallocation. Without this, a long prompt that arrives after the cache was
    // sized for a smaller one grows m_cache_growth_num_tokens (256) at a time across the whole
    // prefill; every growth reallocates and copies the entire cache, which is O(n^2) copy work and
    // produces the unstable memory spikes observed for hybrid models.
    //
    // The reservation targets ALL active sequences (both prompt- and generate-phase), not just the
    // arriving prompt: ensure_sequence_token_capacity grows to the absolute sum of the supplied
    // targets, so it must include the footprint already held by running sequences, otherwise a new
    // prompt arriving alongside running ones would be under-reserved and fall back to chunked
    // growth. It is idempotent (grows only when the required level exceeds the current one), so the
    // work is skipped entirely on pure generation steps and is a no-op once capacity is sufficient.
    void _reserve_dynamic_capacity_for_active_sequences(const std::vector<SequenceGroup::Ptr>& sequence_groups) {
        auto is_active = [](const SequenceGroup::Ptr& sg) {
            return !sg->is_waiting() && !sg->has_finished() && !sg->handle_stopped() && !sg->handle_cancelled();
        };

        // Only (re)reserve when a prompt-phase sequence is entering this round; pure generation
        // steps need no capacity change.
        const bool has_pending_prompt = std::any_of(sequence_groups.begin(), sequence_groups.end(),
            [&](const SequenceGroup::Ptr& sg) { return is_active(sg) && !sg->can_generate_tokens(); });
        if (!has_pending_prompt) {
            return;
        }

        std::vector<SequenceGroup::Ptr> active_sequences;
        active_sequences.reserve(sequence_groups.size());
        for (const auto& sequence_group : sequence_groups) {
            if (is_active(sequence_group)) {
                active_sequences.push_back(sequence_group);
            }
        }
        if (active_sequences.empty()) {
            return;
        }
        size_t total_concurrent_seqs = 0;
        auto sequence_token_targets = _build_sequence_token_targets(active_sequences, total_concurrent_seqs);
        m_cache_orchestrator->ensure_sequence_token_capacity(sequence_token_targets);
    }

    bool _try_increase_cache(SequenceGroup::CPtr sequence_group = nullptr) {
        if (!m_dynamic_memory_allocation) {
            return false;
        }
        bool grew_capacity = false;
        if (sequence_group) {
            grew_capacity = m_cache_orchestrator->ensure_sequence_capacity(sequence_group);
        }

        auto device = m_cache_orchestrator->get_device();
        const size_t growth_tokens = static_cast<size_t>(m_cache_growth_num_tokens);
        if (growth_tokens == 0) {
            return grew_capacity;
        }

        if (device.find("GPU") == std::string::npos) {
            grew_capacity = m_cache_orchestrator->grow_capacity_by_tokens(growth_tokens) || grew_capacity;
        } else {
            const size_t available_gpu_memory = utils::get_available_gpu_memory(
                m_cache_orchestrator->get_device(),
                m_cache_orchestrator->get_num_cache_tensors());
            size_t required_memory = m_cache_orchestrator->memory_cost_for_additional_tokens(growth_tokens);
            if (required_memory <= available_gpu_memory) {
                grew_capacity = m_cache_orchestrator->grow_capacity_by_tokens(growth_tokens) || grew_capacity;
            } else {
                size_t possible_tokens = m_cache_orchestrator->max_additional_tokens_for_memory(available_gpu_memory);
                if (possible_tokens > 0) {
                    grew_capacity = m_cache_orchestrator->grow_capacity_by_tokens(possible_tokens) || grew_capacity;
                } else {
                    return grew_capacity;
                }
            }
        }
        return grew_capacity;
    }

    void _set_linear_attention_paging_data(Output& scheduler_output,
                                           SequenceGroup::CPtr sequence_group,
                                           uint64_t seq_id,
                                           const BlocksPerLayer& la_blocks,
                                           size_t block_table_logical_start) {
        OPENVINO_ASSERT(!la_blocks.empty(), "Linear attention block table empty for sequence ", seq_id);

        Output::LinearAttentionPagingData paging_data;
        const size_t num_processed_tokens = sequence_group->get_num_processed_tokens();
        const size_t num_scheduled_tokens = sequence_group->get_num_scheduled_tokens();

        paging_data.past_length = checked_size_to_int32(num_processed_tokens, "past length", seq_id);
        if (!m_config.enable_prefix_caching) {
            const int32_t block_index = checked_block_index_to_int32(la_blocks[0]->get_index(), seq_id);
            paging_data.block_indices.push_back(block_index);
            paging_data.block_indices.push_back(block_index);
            paging_data.cache_interval = 0;
            scheduler_output.set_linear_attention_paging_data(seq_id, std::move(paging_data));
            return;
        }

        OPENVINO_ASSERT(num_scheduled_tokens > 0, "Linear attention paging requires scheduled tokens for sequence ", seq_id);

        const size_t cache_interval = m_cache_orchestrator->get_block_size(CacheType::LINEAR_ATTENTION_CACHE);
        OPENVINO_ASSERT(cache_interval > 0,
            "Internal error: linear attention cache interval must be greater than 0 when prefix caching is enabled");
        paging_data.cache_interval = checked_size_to_int32(cache_interval, "cache interval", seq_id);
        const size_t read_block_position = num_processed_tokens == 0 ? 0 : (num_processed_tokens - 1) / cache_interval;
        const size_t write_block_begin = num_processed_tokens / cache_interval;
        const size_t write_blocks_count = (num_processed_tokens % cache_interval + num_scheduled_tokens + cache_interval - 1) / cache_interval;
        const size_t write_block_end = write_block_begin + write_blocks_count;

        OPENVINO_ASSERT(read_block_position >= block_table_logical_start,
                        "Linear attention read block precedes restored block table for sequence ", seq_id,
                        ": read position ", read_block_position, ", table starts at ", block_table_logical_start);
        OPENVINO_ASSERT(write_block_begin >= block_table_logical_start,
                        "Linear attention write blocks precede restored block table for sequence ", seq_id,
                        ": write position ", write_block_begin, ", table starts at ", block_table_logical_start);
        const size_t read_block_table_position = read_block_position - block_table_logical_start;
        const size_t write_block_table_begin = write_block_begin - block_table_logical_start;
        const size_t write_block_table_end = write_block_end - block_table_logical_start;

        OPENVINO_ASSERT(write_block_table_end <= la_blocks.size(),
                        "Linear attention block table has insufficient blocks for sequence ", seq_id,
                        ": expected at least ", write_block_table_end, " blocks from logical start ",
                        block_table_logical_start, ", got ", la_blocks.size());

        paging_data.block_indices.reserve(1 + write_blocks_count);
        paging_data.block_indices.push_back(checked_block_index_to_int32(la_blocks[read_block_table_position]->get_index(), seq_id));
        for (size_t block_position = write_block_table_begin; block_position < write_block_table_end; ++block_position) {
            paging_data.block_indices.push_back(checked_block_index_to_int32(la_blocks[block_position]->get_index(), seq_id));
        }
        scheduler_output.set_linear_attention_paging_data(seq_id, std::move(paging_data));
    }

    void _set_kv_paged_attention_data(Output& scheduler_output,
                                      SequenceGroup::Ptr sequence_group,
                                      uint64_t seq_id) {
        if (!m_cache_orchestrator->has_kv_cache()) {
            return;
        }

        scheduler_output.set_kv_block_tables(seq_id, m_cache_orchestrator->get_kv_block_tables(seq_id));
        scheduler_output.set_score_aggregation_window(seq_id, _schedule_scores_to_aggregate(sequence_group));
        const size_t num_processed_tokens_after_chunk = sequence_group->get_num_processed_tokens() +
                                                        sequence_group->get_num_scheduled_tokens();
        if (m_kv_paged_attention_global_data->apply_sparse_attention_mask &&
            num_processed_tokens_after_chunk <= sequence_group->get_prompt_len()) {
            TriShapeSparseAttentionTokenSkipper skipper(get_block_size(CacheType::KV_CACHE),
                    m_config.sparse_attention_config.num_last_dense_tokens_in_prefill,
                    m_config.sparse_attention_config.num_retained_start_tokens_in_cache,
                    m_config.sparse_attention_config.num_retained_recent_tokens_in_cache);
            scheduler_output.set_sparse_attention_skipped_logical_blocks(seq_id, skipper.get_skipped_blocks(sequence_group));
        }
        scheduler_output.set_xattention_threshold(seq_id, _schedule_xattention_threshold(sequence_group));
        scheduler_output.set_adaptive_rkv_evictable_size(seq_id, _schedule_adaptive_rkv_evictable_size(sequence_group));
    }

    static int32_t checked_size_to_int32(size_t value, const char* value_name, uint64_t seq_id) {
        OPENVINO_ASSERT(value <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                        "Linear attention paging ", value_name, " for sequence ", seq_id,
                        " exceeds int32_t maximum: ", value);
        return static_cast<int32_t>(value);
    }

    static int32_t checked_block_index_to_int32(int block_index, uint64_t seq_id) {
        OPENVINO_ASSERT(block_index >= 0,
                        "Linear attention paging block index for sequence ", seq_id,
                        " must be non-negative: ", block_index);
        OPENVINO_ASSERT(static_cast<uint64_t>(block_index) <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
                        "Linear attention paging block index for sequence ", seq_id,
                        " exceeds int32_t maximum: ", block_index);
        return static_cast<int32_t>(block_index);
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

    size_t _schedule_adaptive_rkv_evictable_size(SequenceGroup::Ptr sequence_group) {
        if (!(m_config.use_cache_eviction && m_config.cache_eviction_config.aggregation_mode == AggregationMode::ADAPTIVE_RKV)) {
            return 0;
        }
        if (!sequence_group->can_generate_tokens()) {
            // Won't evict during prefill
            return 0;
        }

        // First similarity/diversity calculation will be scheduled when at least `max_cache_size` tokens are filled
        if (sequence_group->get_num_processed_tokens() < m_config.cache_eviction_config.get_max_cache_size()) {
            return 0;
        }

        const size_t kv_block_size = get_block_size(CacheType::KV_CACHE);
        if (sequence_group->get_num_cached_tokens() % kv_block_size != 0) {
            // Only request similarity computation once every block since eviction can only occur with a block granularity
            return 0;
        }

        size_t non_evictable_size = m_config.cache_eviction_config.get_max_cache_size() - m_config.cache_eviction_config.get_evictable_size();
        OPENVINO_ASSERT(get_num_kv_logical_blocks(sequence_group) * kv_block_size >= non_evictable_size);

        return get_num_kv_logical_blocks(sequence_group) * kv_block_size - non_evictable_size;
    }
};

}
