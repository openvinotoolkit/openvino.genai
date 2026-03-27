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
        for (size_t layer_id : layer_ids) {
            m_layer_to_cache_type[layer_id] = type;
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

    const std::vector<BlocksPerLayer>& get_block_tables(uint64_t seq_id) const {
        return first_block_manager()->get_block_tables(seq_id);
    }

    bool has_block_table(uint64_t seq_id) const {
        return std::any_of(m_block_managers.begin(), m_block_managers.end(),
            [seq_id](const auto& pair) { return pair.second->has_block_table(seq_id); });
    }

    void allocate(Sequence::Ptr sequence, size_t num_blocks, size_t prompt_size = 0) {
        for (auto& [type, block_mgr] : m_block_managers) {
            block_mgr->allocate(sequence, num_blocks, prompt_size);
        }
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

    bool can_allocate_blocks(size_t num_blocks) const {
        return std::all_of(m_block_managers.begin(), m_block_managers.end(),
            [num_blocks](const auto& pair) { return pair.second->can_allocate_blocks(num_blocks); });
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

    size_t free_group_partially(SequenceGroup::Ptr seq_group, size_t num_required_blocks) {
        size_t min_released = std::numeric_limits<size_t>::max();
        for (auto& [type, block_mgr] : m_block_managers) {
            min_released = std::min(min_released, block_mgr->free_group_partially(seq_group, num_required_blocks));
        }
        return min_released;
    }

    size_t free_partially_beam_search_group(SequenceGroup::Ptr seq_group, size_t num_required_blocks) {
        size_t min_released = std::numeric_limits<size_t>::max();
        for (auto& [type, block_mgr] : m_block_managers) {
            min_released = std::min(min_released, block_mgr->free_partially_beam_search_group(seq_group, num_required_blocks));
        }
        return min_released;
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
    std::vector<CacheType> m_types_ordered;
};

}  // namespace ov::genai
