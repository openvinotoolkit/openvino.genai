// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

#include "openvino/runtime/infer_request.hpp"
#include "openvino/genai/scheduler_config.hpp"
#include "continuous_batching/cache/cache_type.hpp"
#include "continuous_batching/cache/i_cache_manager.hpp"
#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"

namespace ov::genai {

/**
 * @brief Aggregates multiple cache type managers and block managers, presenting a unified,
 *        cache-type-agnostic interface.
 *
 * Callers (e.g. Scheduler) interact with the orchestrator without knowing which cache types
 * are registered.  The orchestrator routes every operation to the appropriate per-type
 * manager(s) internally.
 *
 * Currently only KV_CACHE is registered.  Adding a new cache type requires:
 *   1. Implementing ICacheManager for the new type.
 *   2. Calling register_cache_type() with the new type, its manager, block manager, and layer IDs.
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
     *                              string and number of decoder layers.
     */
    static std::shared_ptr<CacheOrchestrator> create(
            ov::InferRequest& infer_request,
            SchedulerConfig& config,
            std::function<size_t(const std::string&, size_t)> get_available_memory) {
        ov::CompiledModel compiled_model = infer_request.get_compiled_model();

        auto orchestrator = std::make_shared<CacheOrchestrator>();

        size_t aggregate_block_size_in_bytes = 0;
        size_t total_num_layers = 0;

        // KV Cache detection
        if (KVCacheManager::has_cache_inputs(compiled_model)) {
            auto kv_manager = std::make_shared<KVCacheManager>(infer_request);

            total_num_layers += kv_manager->get_num_decoder_layers();
            aggregate_block_size_in_bytes += kv_manager->get_block_size_in_bytes();

            std::vector<size_t> layer_ids(kv_manager->get_num_decoder_layers());
            std::iota(layer_ids.begin(), layer_ids.end(), 0);

            size_t total_available_memory = get_available_memory(kv_manager->get_device(), total_num_layers);
            if (config.num_kv_blocks == 0 && config.cache_size > 0) {
                size_t size_in_bytes = config.cache_size * 1024 * 1024 * 1024;
                OPENVINO_ASSERT(size_in_bytes <= total_available_memory,
                                "Requested KV-cache size is larger than available memory size on the system.");
                config.num_kv_blocks = size_in_bytes / aggregate_block_size_in_bytes;
            }
            if (config.num_kv_blocks > 0) {
                size_t size_in_bytes = aggregate_block_size_in_bytes * config.num_kv_blocks;
                OPENVINO_ASSERT(size_in_bytes <= total_available_memory,
                                "Requested number of KV-blocks require more memory than available on the system.");
            }

            auto block_manager = std::make_shared<BlockManager>(
                config.num_kv_blocks,
                config.enable_prefix_caching,
                kv_manager->get_block_size(),
                kv_manager->get_num_decoder_layers());

            orchestrator->register_cache_type(CacheType::KV_CACHE, kv_manager, block_manager, layer_ids);
        }

        // Future cache types (e.g. LINEAR_ATTENTION, SLIDING_WINDOW) follow the same pattern.

        OPENVINO_ASSERT(!orchestrator->get_registered_types().empty(),
                        "No supported cache types detected in the model");

        return orchestrator;
    }

    /**
     * @brief Register a cache type with its managers and the model layers it handles.
     * @param type         Cache type identifier.
     * @param cache_mgr    Physical cache manager for this type.
     * @param block_mgr    Block manager for this type.
     * @param layer_ids    Decoder layer indices handled by this cache type.
     */
    void register_cache_type(CacheType type,
                             std::shared_ptr<ICacheManager> cache_mgr,
                             std::shared_ptr<BlockManager> block_mgr,
                             const std::vector<size_t>& layer_ids) {
        m_cache_managers[type] = std::move(cache_mgr);
        m_block_managers[type] = std::move(block_mgr);
        for (size_t local_idx = 0; local_idx < layer_ids.size(); ++local_idx) {
            size_t global_id = layer_ids[local_idx];
            m_layer_to_cache_type[global_id] = type;
            m_global_to_local_layer_id[global_id] = local_idx;
        }
        m_types_ordered.push_back(type);
    }

    // -----------------------------------------------------------------------
    //  Physical cache management  (applies to all registered types)
    // -----------------------------------------------------------------------

    void allocate_cache_if_needed() {
        for (auto& [type, block_mgr] : m_block_managers) {
            m_cache_managers.at(type)->allocate_cache_if_needed(block_mgr->get_total_number_of_kv_blocks());
        }
    }

    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) {
        for (auto& [type, cache_mgr] : m_cache_managers) {
            cache_mgr->copy_blocks(block_copy_map);
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
            size_t local_idx = m_global_to_local_layer_id.at(global_layer_id);
            const auto& local_tables = m_block_managers.at(type)->get_block_tables(seq_id);
            merged[global_layer_id] = local_tables[local_idx];
        }
        return merged;
    }

    bool has_block_table(uint64_t seq_id) const {
        return std::any_of(m_block_managers.begin(), m_block_managers.end(),
            [seq_id](const auto& pair) { return pair.second->has_block_table(seq_id); });
    }

    void allocate_tokens(Sequence::Ptr sequence, size_t num_tokens, size_t prompt_size = 0) {
        for (auto& [type, block_mgr] : m_block_managers) {
            block_mgr->allocate_tokens(sequence, num_tokens, prompt_size);
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

    std::map<size_t, std::list<size_t>> append_slots(SequenceGroup::Ptr seq_group) {
        std::map<size_t, std::list<size_t>> merged;
        for (auto& [type, block_mgr] : m_block_managers) {
            auto copy_map = block_mgr->append_slots(seq_group);
            for (auto& [src, dst_list] : copy_map) {
                merged[src].insert(merged[src].end(), dst_list.begin(), dst_list.end());
            }
        }
        return merged;
    }

    bool can_append_slots(SequenceGroup::CPtr seq_group) const {
        return std::all_of(m_block_managers.begin(), m_block_managers.end(),
            [&seq_group](const auto& pair) { return pair.second->can_append_slots(seq_group); });
    }

    size_t required_blocks_count(SequenceGroup::CPtr seq_group) const {
        size_t max_required = 0;
        for (const auto& [type, block_mgr] : m_block_managers) {
            max_required = std::max(max_required, block_mgr->required_blocks_count(seq_group));
        }
        return max_required;
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
        for (auto& [type, block_mgr] : m_block_managers) {
            block_mgr->restore_cached_blocks(sequence_group);
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

    void increase_kv_blocks_number(size_t new_num) {
        for (auto& [type, block_mgr] : m_block_managers) {
            block_mgr->increase_kv_blocks_number(new_num);
        }
    }

    size_t get_block_size() const {
        return first_block_manager()->get_block_size();
    }

    size_t get_number_of_blocks_occupied_by_sequence(SequenceGroup::Ptr seq_group) const {
        size_t max_occupied = 0;
        for (const auto& [type, block_mgr] : m_block_managers) {
            max_occupied = std::max(max_occupied, block_mgr->get_number_of_blocks_occupied_by_sequence(seq_group));
        }
        return max_occupied;
    }

    /**
     * @return Number of tokens freed (minimum across all cache managers).
     */
    size_t free_group_partially(SequenceGroup::Ptr seq_group, size_t num_required_blocks) {
        size_t min_tokens_released = std::numeric_limits<size_t>::max();
        for (auto& [type, block_mgr] : m_block_managers) {
            min_tokens_released = std::min(min_tokens_released, block_mgr->free_group_partially(seq_group, num_required_blocks));
        }
        return min_tokens_released;
    }

    /**
     * @return Number of tokens freed (minimum across all cache managers).
     */
    size_t free_partially_beam_search_group(SequenceGroup::Ptr seq_group, size_t num_required_blocks) {
        size_t min_tokens_released = std::numeric_limits<size_t>::max();
        for (auto& [type, block_mgr] : m_block_managers) {
            min_tokens_released = std::min(min_tokens_released, block_mgr->free_partially_beam_search_group(seq_group, num_required_blocks));
        }
        return min_tokens_released;
    }

    // -----------------------------------------------------------------------
    //  Token-level API  (cache-type-agnostic interface for the Scheduler)
    // -----------------------------------------------------------------------

    /**
     * @brief Frees enough blocks from victim to release at least num_tokens tokens.
     * Each cache type independently converts tokens to blocks and frees from its own pool.
     * @param victim The sequence group to free from.
     * @param num_tokens Minimum number of tokens to free from the group total.
     * @return Number of tokens actually freed (minimum across all cache types).
     */
    size_t free_group_partially_by_tokens(SequenceGroup::Ptr victim, size_t num_tokens) {
        size_t min_tokens_released = std::numeric_limits<size_t>::max();
        for (auto& [type, block_mgr] : m_block_managers) {
            min_tokens_released = std::min(min_tokens_released, block_mgr->free_group_partially_by_tokens(victim, num_tokens));
        }
        return min_tokens_released;
    }

    /**
     * @brief Frees enough blocks from a beam search victim to release at least num_tokens tokens.
     * Each cache type independently converts tokens to blocks and frees from its own pool.
     * @param victim The sequence group to free from.
     * @param num_tokens Minimum number of tokens to free from the group total.
     * @return Number of tokens actually freed (minimum across all cache types).
     */
    size_t free_partially_beam_search_group_by_tokens(SequenceGroup::Ptr victim, size_t num_tokens) {
        size_t min_tokens_released = std::numeric_limits<size_t>::max();
        for (auto& [type, block_mgr] : m_block_managers) {
            min_tokens_released = std::min(min_tokens_released, block_mgr->free_partially_beam_search_group_by_tokens(victim, num_tokens));
        }
        return min_tokens_released;
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
            [&](const auto& pair) { return pair.second->can_partially_preempt(victim, target); });
    }

    /**
     * @return Whether any token capacity has been allocated (at least one block in every type).
     */
    bool has_token_capacity() const {
        return std::all_of(m_block_managers.begin(), m_block_managers.end(),
            [](const auto& pair) { return pair.second->has_token_capacity(); });
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
     * @brief Grows each cache type's block pool to accommodate the given number of additional tokens.
     * Each type converts tokens to blocks using its own block size.
     * @param num_tokens Number of additional tokens to accommodate.
     */
    void grow_capacity_by_tokens(size_t num_tokens) {
        for (auto& [type, block_mgr] : m_block_managers) {
            block_mgr->grow_capacity_by_tokens(num_tokens);
        }
    }

    /**
     * @brief Ensures each cache type's block pool has capacity for at least the given total number of tokens.
     * Each type converts tokens to blocks using its own block size.
     * @param num_tokens Total number of tokens the pools should accommodate.
     */
    void ensure_token_capacity(size_t num_tokens) {
        for (auto& [type, block_mgr] : m_block_managers) {
            block_mgr->ensure_token_capacity(num_tokens);
        }
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
        return first_cache_manager()->get_device();
    }

    size_t get_num_layers() const {
        size_t total = 0;
        for (const auto& [type, cache_mgr] : m_cache_managers) {
            total += cache_mgr->get_num_layers();
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

    std::shared_ptr<ICacheManager> get_cache_manager(CacheType type) const {
        return m_cache_managers.at(type);
    }

    std::shared_ptr<BlockManager> get_block_manager(CacheType type) const {
        return m_block_managers.at(type);
    }

    const std::map<size_t, CacheType>& get_layer_to_cache_type_map() const {
        return m_layer_to_cache_type;
    }

    CacheType get_cache_type_for_layer(size_t layer_id) const {
        return m_layer_to_cache_type.at(layer_id);
    }

    const std::vector<CacheType>& get_registered_types() const {
        return m_types_ordered;
    }

private:
    const std::shared_ptr<BlockManager>& first_block_manager() const {
        OPENVINO_ASSERT(!m_block_managers.empty(), "No cache types registered");
        return m_block_managers.begin()->second;
    }

    const std::shared_ptr<ICacheManager>& first_cache_manager() const {
        OPENVINO_ASSERT(!m_cache_managers.empty(), "No cache types registered");
        return m_cache_managers.begin()->second;
    }

    std::map<CacheType, std::shared_ptr<ICacheManager>> m_cache_managers;
    std::map<CacheType, std::shared_ptr<BlockManager>> m_block_managers;
    std::map<size_t, CacheType> m_layer_to_cache_type;
    std::map<size_t, size_t> m_global_to_local_layer_id;  ///< global layer ID -> local index within its block manager
    std::vector<CacheType> m_types_ordered;
};

}  // namespace ov::genai
