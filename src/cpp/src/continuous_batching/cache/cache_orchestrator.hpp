// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>

#include "openvino/runtime/infer_request.hpp"
#include "openvino/genai/scheduler_config.hpp"
#include "continuous_batching/cache/cache_type.hpp"
#include "continuous_batching/cache/i_cache_manager.hpp"
#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/linear_attention_cache_manager.hpp"

namespace ov::genai {

/**
 * @brief Aggregates multiple cache type managers and block managers, presenting a unified,
 *        cache-type-agnostic interface.
 *
 * Callers (e.g. Scheduler) interact with the orchestrator without knowing which cache types
 * are registered.  The orchestrator routes every operation to the appropriate per-type
 * manager(s) internally.
 *
 * Adding a new cache type requires:
 *   1. Implementing ICacheManager for the new type.
 *   2. Calling register_cache_type() with the new type, its manager and block manager.
 *      Layer IDs are assigned contiguously in registration order, starting from 0.
 */
class CacheOrchestrator {
public:
    CacheOrchestrator() = default;

    /**
     * @brief Detect model cache types, create managers and block managers, normalize config,
     *        and return a fully populated CacheOrchestrator.
     *
     * @param infer_request         The inference request (provides compiled model info).
     * @param[in,out] config        Scheduler configuration.  num_kv_blocks is derived from
     *                              cache_size when it is zero.
     * @param get_available_memory  Returns available device memory in bytes given the device
     *                              string and number of cache tensors across all cache types.
     */
    static std::shared_ptr<CacheOrchestrator> create(
            ov::InferRequest& infer_request,
            SchedulerConfig& config,
            std::function<size_t(const std::string&, size_t)> get_available_memory) {
        config.validate();

        auto orchestrator = std::make_shared<CacheOrchestrator>();

        auto [kv_mgr, la_mgr] = detect_cache_managers(infer_request, config);

        const size_t num_cache_tensors =
            (kv_mgr ? kv_mgr->get_num_cache_tensors() : 0) +
            (la_mgr ? la_mgr->get_num_cache_tensors() : 0);
        const std::string allocation_device = kv_mgr ? kv_mgr->get_device()
                                            : (la_mgr ? la_mgr->get_device() : std::string{});
        const size_t total_available_memory = allocation_device.empty()
                                                  ? std::numeric_limits<size_t>::max()
                                                  : get_available_memory(allocation_device, num_cache_tensors);

        auto [num_kv_blocks, num_la_blocks] = normalize_block_counts(kv_mgr.get(), la_mgr.get(), config, total_available_memory);
        config.num_kv_blocks = num_kv_blocks;
        config.num_linear_attention_blocks = num_la_blocks;

        if (kv_mgr) {
            orchestrator->register_kv_cache(std::move(kv_mgr), config);
        }
        if (la_mgr) {
            orchestrator->register_linear_attention_cache(std::move(la_mgr), config);
        }

        OPENVINO_ASSERT(orchestrator->has_registered_types(), "No supported cache types detected in the model");

        return orchestrator;
    }

    /**
     * @brief Register a cache type with its managers.
     *
     * @param type              Cache type identifier.
     * @param cache_mgr         Physical cache manager for this type.
     * @param block_mgr         Block manager for this type.
     * @param per_layer_control If true, the model was compiled with per-layer block index
     *                          inputs for this cache type (e.g. for cache eviction).
     */
    void register_cache_type(
            CacheType type,
            std::unique_ptr<ICacheManager> cache_mgr,
            std::unique_ptr<BlockManager> block_mgr,
            bool per_layer_control = false) {
        OPENVINO_ASSERT(cache_mgr, "Cache manager must not be null");
        OPENVINO_ASSERT(block_mgr, "Block manager must not be null");
        OPENVINO_ASSERT(m_cache_managers.find(type) == m_cache_managers.end(),
                "Cache type is already registered");
        const size_t num_layers = block_mgr->get_num_layers();
        OPENVINO_ASSERT(num_layers > 0, "Cache type must register at least one block-table layer");
        const size_t layer_start = m_layer_to_cache_type.size();
        m_type_layer_start[type] = layer_start;
        m_cache_managers[type] = std::move(cache_mgr);
        m_block_managers[type] = std::move(block_mgr);
        m_per_layer_control[type] = per_layer_control;
        for (size_t local_idx = 0; local_idx < num_layers; ++local_idx) {
            m_layer_to_cache_type[layer_start + local_idx] = type;
        }
    }

    // -----------------------------------------------------------------------
    //  Physical cache management  (applies to all registered types)
    // -----------------------------------------------------------------------

    void allocate_cache_if_needed() {
        for (auto& [type, block_mgr] : m_block_managers) {
            m_cache_managers.at(type)->allocate_cache_if_needed(block_mgr->get_total_number_of_kv_blocks());
        }
    }

    void copy_blocks(const std::map<CacheType, std::map<size_t, std::list<size_t>>>& per_type_copy_map) {
        for (const auto& [type, copy_map] : per_type_copy_map) {
            m_cache_managers.at(type)->copy_blocks(copy_map);
        }
    }

    void clear() {
        for (auto& [type, cache_mgr] : m_cache_managers) {
            cache_mgr->clear();
        }
        for (auto& [type, block_mgr] : m_block_managers) {
            block_mgr->clear();
        }
    }

    // -----------------------------------------------------------------------
    //  Block management  (applies to all registered types)
    // -----------------------------------------------------------------------

    /**
     * @brief Compose a unified per-layer block table for a sequence by merging block tables
     *        from all registered block managers.
     *
     * Each block manager stores block tables using local (0-based) layer indices.
     * This method maps them back to global layer positions so the returned vector
     * is indexed by global layer ID.
     */
    std::vector<BlocksPerLayer> get_block_tables(uint64_t seq_id) const {
        const size_t total_layers = m_layer_to_cache_type.size();
        std::vector<BlocksPerLayer> merged(total_layers);
        for (const auto& [global_layer_id, type] : m_layer_to_cache_type) {
            const size_t local_idx = m_per_layer_control.at(type) ? global_layer_id - m_type_layer_start.at(type) : 0;
            const auto& local_tables = m_block_managers.at(type)->get_block_tables(seq_id);
            OPENVINO_ASSERT(local_idx < local_tables.size(), "Block table layer index is out of range");
            merged[global_layer_id] = local_tables[local_idx];
        }
        return merged;
    }

    bool has_block_table(uint64_t seq_id) const {
        return std::any_of(m_block_managers.begin(), m_block_managers.end(),
            [seq_id](const auto& pair) { return pair.second->has_block_table(seq_id); });
    }

    void allocate_tokens(Sequence::Ptr sequence, SequenceGroup::CPtr seq_group, size_t num_tokens, size_t prompt_size = 0) {
        for (auto& [type, block_mgr] : m_block_managers) {
            block_mgr->allocate_tokens(sequence, seq_group, num_tokens, prompt_size);
        }
    }

    size_t available_token_slots(SequenceGroup::CPtr seq_group) const {
        size_t min_slots = std::numeric_limits<size_t>::max();
        for (const auto& [type, block_mgr] : m_block_managers) {
            min_slots = std::min(min_slots, block_mgr->available_token_slots(seq_group));
        }
        return min_slots;
    }

    bool can_allocate_tokens(SequenceGroup::CPtr seq_group, size_t num_tokens) const {
        return std::all_of(m_block_managers.begin(), m_block_managers.end(),
            [&seq_group, num_tokens](const auto& pair) { return pair.second->can_allocate_tokens(seq_group, num_tokens); });
    }

    std::map<CacheType, std::map<size_t, std::list<size_t>>> append_slots(SequenceGroup::Ptr seq_group) {
        std::map<CacheType, std::map<size_t, std::list<size_t>>> per_type;
        for (auto& [type, block_mgr] : m_block_managers) {
            auto copy_map = block_mgr->append_slots(seq_group);
            if (!copy_map.empty()) {
                per_type[type] = std::move(copy_map);
            }
        }
        return per_type;
    }

    bool can_append_slots(SequenceGroup::CPtr seq_group) const {
        return std::all_of(m_block_managers.begin(), m_block_managers.end(),
            [&seq_group](const auto& pair) { return pair.second->can_append_slots(seq_group); });
    }

    /**
     * @brief Returns the maximum token deficit for the target across all cache types.
     * Each type independently computes required_blocks * block_size.
     * @param seq_group The target sequence group.
     * @return Maximum number of tokens needed across all cache types.
     */
    size_t required_tokens_count(SequenceGroup::CPtr seq_group) const {
        size_t max_tokens = 0;
        for (const auto& [type, block_mgr] : m_block_managers) {
            max_tokens = std::max(max_tokens, block_mgr->required_tokens_count(seq_group));
        }
        return max_tokens;
    }

    size_t num_free_blocks() const {
        size_t min_free = std::numeric_limits<size_t>::max();
        for (const auto& [type, block_mgr] : m_block_managers) {
            min_free = std::min(min_free, block_mgr->num_free_blocks());
        }
        return min_free;
    }

    void free_sequence(uint64_t seq_id) {
        for (auto& [type, block_mgr] : m_block_managers) {
            if (block_mgr->has_block_table(seq_id)) {
                block_mgr->free_sequence(seq_id);
            }
        }
    }

    void fork_sequence(uint64_t parent_id, uint64_t child_id) {
        for (auto& [type, block_mgr] : m_block_managers) {
            block_mgr->fork_sequence(parent_id, child_id);
        }
    }

    void restore_cached_blocks(const SequenceGroup::Ptr& sequence_group) {
        const size_t initial_processed_tokens = sequence_group->get_num_processed_tokens();
        size_t common_processed_tokens = std::numeric_limits<size_t>::max();
        for (auto& [type, block_mgr] : m_block_managers) {
            sequence_group->update_processed_tokens_num(initial_processed_tokens);
            block_mgr->restore_cached_blocks(sequence_group);
            common_processed_tokens = std::min(common_processed_tokens, sequence_group->get_num_processed_tokens());
        }
        if (common_processed_tokens == std::numeric_limits<size_t>::max()) {
            return;
        }

        sequence_group->update_processed_tokens_num(common_processed_tokens);
        for (auto& [type, block_mgr] : m_block_managers) {
            block_mgr->free_empty_physical_blocks(sequence_group);
        }
    }

    void free_blocks_from_sequence(size_t seq_id,
                                   const std::vector<std::set<size_t>>& per_layer_logical_block_indices,
                                   CacheType cache_type) {
        auto it = m_block_managers.find(cache_type);
        OPENVINO_ASSERT(it != m_block_managers.end(), "Cache type not registered");
        it->second->free_blocks_from_sequence(seq_id, per_layer_logical_block_indices);
    }

    void free_empty_physical_blocks(SequenceGroup::Ptr seq_group) {
        for (auto& [type, block_mgr] : m_block_managers) {
            block_mgr->free_empty_physical_blocks(seq_group);
        }
    }

    float get_used_percentage() const {
        float max_usage = 0.0f;
        for (const auto& [type, block_mgr] : m_block_managers) {
            max_usage = std::max(max_usage, block_mgr->get_used_percentage());
        }
        return max_usage;
    }

    size_t get_total_number_of_kv_blocks() const {
        size_t min_total = std::numeric_limits<size_t>::max();
        for (const auto& [type, block_mgr] : m_block_managers) {
            min_total = std::min(min_total, block_mgr->get_total_number_of_kv_blocks());
        }
        return min_total;
    }

    /// @return Block size in tokens for the given cache type.
    size_t get_block_size(CacheType type) const {
        return m_block_managers.at(type)->get_block_size();
    }

    /// @return Number of KV cache blocks currently allocated for the given sequence group or 0 if model does not have KV cache.
    size_t get_num_logical_blocks(SequenceGroup::CPtr seq_group) const {
        auto it = m_block_managers.find(CacheType::KV_CACHE);
        return it != m_block_managers.end() ? it->second->get_num_logical_blocks(seq_group) : 0;
    }

    // -----------------------------------------------------------------------
    //  Token-level API  (cache-type-agnostic interface for the Scheduler)
    // -----------------------------------------------------------------------

    /// @return Approximate aggregate memory cost per token across all variable-size cache types.
    size_t get_bytes_per_token() const {
        size_t total = 0;
        for (const auto& [type, cache_mgr] : m_cache_managers) {
            if (m_block_managers.at(type)->is_fixed_size_per_sequence())
                continue;
            total += cache_mgr->get_block_size_in_bytes() / m_block_managers.at(type)->get_block_size();
        }
        return total;
    }

    /// @return Exact memory needed to grow all variable-size caches by num_tokens tokens (accounting for block rounding).
    size_t memory_cost_for_additional_tokens(size_t num_tokens) const {
        size_t total = 0;
        for (const auto& [type, block_mgr] : m_block_managers) {
            if (block_mgr->is_fixed_size_per_sequence())
                continue;
            const size_t bs = block_mgr->get_block_size();
            const size_t blocks = (num_tokens + bs - 1) / bs;
            total += blocks * m_cache_managers.at(type)->get_block_size_in_bytes();
        }
        return total;
    }

    /// @return Maximum additional tokens that fit within the memory budget across all cache types.
    size_t max_additional_tokens_for_memory(size_t available_memory) const {
        if (m_block_managers.empty() || available_memory == 0) {
            return 0;
        }
        const size_t bpt = get_bytes_per_token();
        if (bpt == 0) {
            return 0;
        }
        size_t lo = 0, hi = available_memory / bpt;
        while (lo < hi) {
            const size_t mid = lo + (hi - lo + 1) / 2;
            if (memory_cost_for_additional_tokens(mid) <= available_memory) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    /**
     * @brief Partially preempts victim for target using per-cache-type deficits.
     *
     * Fixed-size-per-sequence cache types are skipped: their capacity is sequence-level,
     * not token-level. For token accounting, the return value uses KV cache released tokens
     * when KV cache is present; otherwise it falls back to the minimum released tokens across
     * variable-size cache types.
     */
    size_t free_group_partially_for_target(SequenceGroup::Ptr victim, SequenceGroup::CPtr target) {
        size_t tokens_to_release = 0;

        for (const auto& [type, block_mgr] : m_block_managers) {
            if (block_mgr->is_fixed_size_per_sequence()) {
                continue;
            }
            tokens_to_release = std::max(tokens_to_release, block_mgr->required_tokens_count(target));
        }

        if (tokens_to_release == 0) {
            return 0;
        }

        size_t min_tokens_released = std::numeric_limits<size_t>::max();
        size_t kv_tokens_released = 0;
        bool has_kv = false;
        bool has_variable_types = false;

        for (auto& [type, block_mgr] : m_block_managers) {
            if (block_mgr->is_fixed_size_per_sequence()) {
                continue;
            }

            has_variable_types = true;
            const size_t released = block_mgr->free_group_partially_by_tokens(victim, tokens_to_release);
            min_tokens_released = std::min(min_tokens_released, released);
            if (type == CacheType::KV_CACHE) {
                kv_tokens_released = released;
                has_kv = true;
            }
        }

        if (has_kv) {
            return kv_tokens_released;
        }
        return has_variable_types ? min_tokens_released : 0;
    }

    /**
     * @brief Partially preempts a beam-search victim for target using per-cache-type deficits.
     *
     * Fixed-size-per-sequence cache types are skipped: their capacity is sequence-level,
     * not token-level. For token accounting, the return value uses KV cache released tokens
     * when KV cache is present; otherwise it falls back to the minimum released tokens across
     * variable-size cache types.
     */
    size_t free_partially_beam_search_group_for_target(SequenceGroup::Ptr victim, SequenceGroup::CPtr target) {
        size_t tokens_to_release = 0;

        for (const auto& [type, block_mgr] : m_block_managers) {
            if (block_mgr->is_fixed_size_per_sequence()) {
                continue;
            }
            tokens_to_release = std::max(tokens_to_release, block_mgr->required_tokens_count(target));
        }

        if (tokens_to_release == 0) {
            return 0;
        }

        size_t min_tokens_released = std::numeric_limits<size_t>::max();
        size_t kv_tokens_released = 0;
        bool has_kv = false;
        bool has_variable_types = false;

        for (auto& [type, block_mgr] : m_block_managers) {
            if (block_mgr->is_fixed_size_per_sequence()) {
                continue;
            }

            has_variable_types = true;
            const size_t released = block_mgr->free_partially_beam_search_group_by_tokens(victim, tokens_to_release);
            min_tokens_released = std::min(min_tokens_released, released);
            if (type == CacheType::KV_CACHE) {
                kv_tokens_released = released;
                has_kv = true;
            }
        }

        if (has_kv) {
            return kv_tokens_released;
        }
        return has_variable_types ? min_tokens_released : 0;
    }

    /**
     * @brief Checks whether partially preempting victim can free enough blocks in every
     * cache type's pool to satisfy the target sequence group's deficit.
     * @param victim The sequence group to potentially free blocks from.
     * @param target The sequence group that needs blocks allocated.
     * @return Whether partial preemption of victim is sufficient for target in all cache types.
     */
    bool can_partially_preempt(SequenceGroup::Ptr victim, SequenceGroup::CPtr target) {
        return std::all_of(m_block_managers.begin(), m_block_managers.end(),
            [&](const auto& pair) {
                const auto& block_mgr = pair.second;
                if (block_mgr->is_fixed_size_per_sequence()) {
                    return block_mgr->required_blocks_count(target) == 0 && block_mgr->can_partially_preempt_victim(victim);
                }
                return block_mgr->can_partially_preempt_victim(victim) && block_mgr->can_partially_preempt(victim, target);
            });
    }

    /**
     * @return Whether every registered cache type has an allocated block pool.
     */
    bool has_token_capacity() const {
        return std::all_of(m_block_managers.begin(), m_block_managers.end(),
            [](const auto& pair) {
                return pair.second->has_token_capacity();
            });
    }

    /**
     * @return Total token capacity (minimum across all cache types).
     */
    size_t total_token_capacity() const {
        size_t min_capacity = std::numeric_limits<size_t>::max();
        for (const auto& [type, block_mgr] : m_block_managers) {
            min_capacity = std::min(min_capacity, block_mgr->total_token_capacity());
        }
        return min_capacity;
    }

    /**
     * @brief Grows each variable-size cache type's block pool to accommodate the given number of additional tokens.
     * Fixed-size-per-sequence managers (e.g. linear attention state) are skipped: their capacity
     * is sequence-count-driven, not token-count-driven.
     * @param num_tokens Number of additional tokens to accommodate.
     */
    bool grow_capacity_by_tokens(size_t num_tokens) {
        bool grew_capacity = false;
        for (auto& [type, block_mgr] : m_block_managers) {
            if (!block_mgr->is_fixed_size_per_sequence()) {
                block_mgr->grow_capacity_by_tokens(num_tokens);
                grew_capacity = true;
            }
        }
        return grew_capacity;
    }

    /**
     * @brief Ensures each variable-size cache type's block pool has capacity for at least the given total number of tokens.
     * Fixed-size-per-sequence managers (e.g. linear attention state) are skipped: their pool
     * grows on demand via grow_fixed_size_capacity(), not by token count.
     * @param num_tokens Total number of tokens the pools should accommodate.
     */
    void ensure_token_capacity(size_t num_tokens) {
        for (auto& [type, block_mgr] : m_block_managers) {
            if (!block_mgr->is_fixed_size_per_sequence())
                block_mgr->ensure_token_capacity(num_tokens);
        }
    }

    /**
     * @brief Grows each fixed-size-per-sequence cache type's block pool by the given number of
     *        additional sequences worth of blocks.  Variable-size managers are skipped.
     * @param num_seqs Number of additional concurrent sequences to accommodate.
     */
    void grow_fixed_size_capacity(size_t num_seqs) {
        for (auto& [type, block_mgr] : m_block_managers) {
            if (block_mgr->is_fixed_size_per_sequence()) {
                const size_t additional_blocks = num_seqs * block_mgr->get_fixed_blocks_per_sequence();
                block_mgr->increase_kv_blocks_number(
                    block_mgr->get_total_number_of_kv_blocks() + additional_blocks);
            }
        }
    }

    /**
     * @brief Ensures each fixed-size-per-sequence cache type has enough free blocks for
     *        the sequence group. Variable-size managers are skipped.
     * @return Whether any fixed-size block pool was grown.
     */
    bool ensure_sequence_capacity(SequenceGroup::CPtr seq_group) {
        bool grew_capacity = false;
        for (auto& [type, block_mgr] : m_block_managers) {
            if (!block_mgr->is_fixed_size_per_sequence()) {
                continue;
            }

            const size_t required_blocks = block_mgr->required_blocks_count(seq_group);
            const size_t free_blocks = block_mgr->num_free_blocks();
            if (required_blocks > free_blocks) {
                block_mgr->increase_kv_blocks_number(
                    block_mgr->get_total_number_of_kv_blocks() + required_blocks - free_blocks);
                grew_capacity = true;
            }
        }
        return grew_capacity;
    }

    // -----------------------------------------------------------------------
    //  Aggregate queries
    // -----------------------------------------------------------------------

    size_t get_total_cache_size_in_bytes() const {
        size_t total = 0;
        for (const auto& [type, block_mgr] : m_block_managers) {
            total += block_mgr->get_total_number_of_kv_blocks() * m_cache_managers.at(type)->get_block_size_in_bytes();
        }
        return total;
    }

    std::string get_device() const {
        OPENVINO_ASSERT(!m_cache_managers.empty(), "No cache types registered");
        const auto& first_cache_manager = *m_cache_managers.begin()->second;
        return first_cache_manager.get_device();
    }

    size_t get_num_layers() const {
        size_t total = 0;
        for (const auto& [type, cache_mgr] : m_cache_managers) {
            total += cache_mgr->get_num_layers();
        }
        return total;
    }

    size_t get_num_cache_tensors() const {
        size_t total = 0;
        for (const auto& [type, cache_mgr] : m_cache_managers) {
            total += cache_mgr->get_num_cache_tensors();
        }
        return total;
    }

    size_t get_block_size_in_bytes() const {
        size_t total = 0;
        for (const auto& [type, cache_mgr] : m_cache_managers) {
            total += cache_mgr->get_block_size_in_bytes();
        }
        return total;
    }

    // -----------------------------------------------------------------------
    //  Low-level accessors (for debugging / type-specific edge cases)
    // -----------------------------------------------------------------------

    const ICacheManager& get_cache_manager(CacheType type) const {
        return *m_cache_managers.at(type);
    }

    BlockManager& get_block_manager(CacheType type) {
        return *m_block_managers.at(type);
    }

    const BlockManager& get_block_manager(CacheType type) const {
        return *m_block_managers.at(type);
    }

    const std::map<size_t, CacheType>& get_layer_to_cache_type_map() const {
        return m_layer_to_cache_type;
    }

    CacheType get_cache_type_for_layer(size_t layer_id) const {
        return m_layer_to_cache_type.at(layer_id);
    }

    // -----------------------------------------------------------------------
    //  Linear attention cache helpers
    // -----------------------------------------------------------------------

    /// @return Whether a linear attention cache type is registered.
    bool has_linear_attention_cache() const {
        return m_cache_managers.count(CacheType::LINEAR_ATTENTION_CACHE) > 0;
    }

    /**
     * @brief Returns the linear attention block table for a sequence (first layer).
     * All linear attention layers share the same block allocation, so a single layer suffices.
     */
    BlocksPerLayer get_linear_attention_block_table(uint64_t seq_id) const {
        OPENVINO_ASSERT(has_linear_attention_cache(), "No linear attention cache registered");
        return m_block_managers.at(CacheType::LINEAR_ATTENTION_CACHE)->get_block_tables(seq_id)[0];
    }

    /// @return Number of KV attention layers only (excluding other cache types).
    size_t get_num_kv_layers() const {
        auto it = m_cache_managers.find(CacheType::KV_CACHE);
        return it != m_cache_managers.end() ? it->second->get_num_layers() : 0;
    }

    /**
     * @brief Whether the model requires per-layer KV block index inputs.
     *
     * Returns true when any registered cache type with shared block_indices inputs
     * has per-layer control enabled. Cache types with their own dedicated inputs
     * (e.g. LINEAR_ATTENTION_CACHE with paged_conv_ / paged_gdn. inputs) do not contribute.
     */
    bool needs_per_layer_block_indices() const {
        return std::any_of(m_per_layer_control.begin(), m_per_layer_control.end(),
            [](const auto& pair) {
                // Only KV_CACHE uses block_indices / block_indices.N inputs
                return pair.first == CacheType::KV_CACHE && pair.second;
            });
    }

private:
    bool has_registered_types() const {
        return !m_cache_managers.empty();
    }

    /**
     * @brief Detect and create KV and linear attention cache managers from the compiled model.
     *        Validates that LA-specific config fields are not set when no LA cache is present.
     *        Returns a tuple of (kv_manager, la_manager).
     */
    static std::tuple<std::unique_ptr<KVCacheManager>, std::unique_ptr<LinearAttentionCacheManager>>
    detect_cache_managers(ov::InferRequest& infer_request, const SchedulerConfig& config) {
        ov::CompiledModel compiled_model = infer_request.get_compiled_model();

        std::unique_ptr<KVCacheManager> kv_manager;
        if (KVCacheManager::has_cache_inputs(compiled_model)) {
            kv_manager = std::make_unique<KVCacheManager>(infer_request);
        }

        std::unique_ptr<LinearAttentionCacheManager> la_manager;
        if (LinearAttentionCacheManager::has_cache_inputs(compiled_model)) {
            la_manager = std::make_unique<LinearAttentionCacheManager>(infer_request);
        }

        OPENVINO_ASSERT(la_manager || config.num_linear_attention_blocks == 0,
                        "SchedulerConfig num_linear_attention_blocks can be set only for models with linear attention cache inputs");

        if (!la_manager) {
            OPENVINO_ASSERT(config.cache_interval == DEFAULT_LINEAR_ATTENTION_CACHE_INTERVAL || config.cache_interval == 0,
                            "SchedulerConfig cache_interval can be set only for models with linear attention cache inputs");
        }

        return {std::move(kv_manager), std::move(la_manager)};
    }

    /**
     * @brief Compute the number of KV and LA blocks to allocate, honouring the memory budget
     *        expressed either as cache_size (GiB) or explicit block counts in config.
     *        Returns a tuple of (num_kv_blocks, num_la_blocks).
     */
    static std::tuple<size_t, size_t>
    normalize_block_counts(const KVCacheManager* kv_manager,
                          const LinearAttentionCacheManager* la_manager,
                          const SchedulerConfig& config,
                          size_t total_available_memory) {
        const size_t kv_block_size = kv_manager ? kv_manager->get_block_size() : 0;
        const size_t kv_block_size_in_bytes = kv_manager ? kv_manager->get_block_size_in_bytes() : 0;
        const size_t la_block_size_in_bytes = la_manager ? la_manager->get_block_size_in_bytes() : 0;

        auto bytes_for_token_target = [&](size_t token_target) {
            size_t total_bytes = 0;
            if (kv_manager) {
                const size_t kv_blocks = (token_target + kv_block_size - 1) / kv_block_size;
                total_bytes += kv_blocks * kv_block_size_in_bytes;
            }
            if (la_manager && config.enable_prefix_caching) {
                const size_t la_blocks = (token_target + config.cache_interval - 1) / config.cache_interval;
                total_bytes += la_blocks * la_block_size_in_bytes;
            }
            return total_bytes;
        };

        size_t normalized_num_kv_blocks = config.num_kv_blocks;
        size_t normalized_num_la_blocks = config.num_linear_attention_blocks;

        if (la_manager && !config.enable_prefix_caching && normalized_num_la_blocks == 0 &&
            (normalized_num_kv_blocks > 0 || config.cache_size > 0)) {
            normalized_num_la_blocks = config.max_num_batched_tokens == std::numeric_limits<size_t>::max()
                                           ? 1
                                           : config.max_num_seqs;
        }

        if (config.cache_size > 0 && normalized_num_kv_blocks == 0) {
            const size_t budget_in_bytes = config.cache_size * 1024ULL * 1024ULL * 1024ULL;
            OPENVINO_ASSERT(budget_in_bytes <= total_available_memory,
                            "Requested cache size is larger than available memory size on the system.");

            if (la_manager && !config.enable_prefix_caching) {
                const size_t reserved_la_bytes = normalized_num_la_blocks * la_block_size_in_bytes;
                OPENVINO_ASSERT(reserved_la_bytes <= budget_in_bytes,
                                "Requested linear attention cache allocation exceeds the configured cache size.");
                if (kv_manager) {
                    normalized_num_kv_blocks = (budget_in_bytes - reserved_la_bytes) / kv_block_size_in_bytes;
                }
            } else {
                size_t low = 0;
                size_t high = 0;
                if (kv_manager) {
                    high = (budget_in_bytes / kv_block_size_in_bytes) * kv_block_size;
                } else if (la_manager) {
                    high = (budget_in_bytes / la_block_size_in_bytes) * config.cache_interval;
                }

                while (low < high) {
                    const size_t mid = low + (high - low + 1) / 2;
                    if (bytes_for_token_target(mid) <= budget_in_bytes) {
                        low = mid;
                    } else {
                        high = mid - 1;
                    }
                }

                if (kv_manager) {
                    normalized_num_kv_blocks = (low + kv_block_size - 1) / kv_block_size;
                }
                if (la_manager) {
                    normalized_num_la_blocks = (low + config.cache_interval - 1) / config.cache_interval;
                }
            }
        } else if (normalized_num_kv_blocks > 0) {
            if (la_manager && config.enable_prefix_caching && normalized_num_la_blocks == 0) {
                const size_t token_target = normalized_num_kv_blocks * kv_block_size;
                normalized_num_la_blocks = (token_target + config.cache_interval - 1) / config.cache_interval;
            }

            size_t total_requested_bytes = 0;
            if (kv_manager) {
                total_requested_bytes += normalized_num_kv_blocks * kv_block_size_in_bytes;
            }
            if (la_manager) {
                total_requested_bytes += normalized_num_la_blocks * la_block_size_in_bytes;
            }
            OPENVINO_ASSERT(total_requested_bytes <= total_available_memory,
                            "Requested cache blocks require more memory than available on the system.");
        }

        return {normalized_num_kv_blocks, normalized_num_la_blocks};
    }

    /**
     * @brief Create a BlockManager for KV cache and register it with this orchestrator.
     */
    void register_kv_cache(std::unique_ptr<KVCacheManager> kv_manager,
                           const SchedulerConfig& config) {
        auto block_manager = std::make_unique<BlockManager>(
            config.num_kv_blocks,
            config.enable_prefix_caching,
            kv_manager->get_block_size(),
            kv_manager->get_num_layers());

        register_cache_type(CacheType::KV_CACHE, std::move(kv_manager), std::move(block_manager),
                            config.use_cache_eviction);
    }

    /**
     * @brief Create a BlockManager for linear attention cache and register it with this orchestrator.
     *        Layer IDs are assigned contiguously after any previously registered types.
     */
    void register_linear_attention_cache(std::unique_ptr<LinearAttentionCacheManager> la_manager,
                                         const SchedulerConfig& config) {
        std::unique_ptr<BlockManager> la_block_manager;
        if (config.enable_prefix_caching) {
            OPENVINO_ASSERT(config.cache_interval > 0,
                            "Internal error: SchedulerConfig cache_interval must be greater than 0 when prefix caching is enabled");
            la_block_manager = std::make_unique<BlockManager>(
                config.num_linear_attention_blocks,
                true,
                config.cache_interval,
                1);
        } else {
            la_block_manager = std::make_unique<BlockManager>(
                config.num_linear_attention_blocks,
                false,
                1,
                1,
                1);
        }

        // Linear-attention state tensors are per physical layer/group, but share one logical block table.
        register_cache_type(CacheType::LINEAR_ATTENTION_CACHE, std::move(la_manager), std::move(la_block_manager));
    }

    std::map<CacheType, std::unique_ptr<ICacheManager>> m_cache_managers;
    std::map<CacheType, std::unique_ptr<BlockManager>> m_block_managers;
    std::map<size_t, CacheType> m_layer_to_cache_type;
    std::map<CacheType, size_t> m_type_layer_start;  ///< first global layer ID for each registered cache type
    std::map<CacheType, bool> m_per_layer_control;   ///< per-type flag: layers managed individually or as one
};

}  // namespace ov::genai
