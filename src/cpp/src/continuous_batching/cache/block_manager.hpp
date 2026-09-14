// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

#include "logger.hpp"
#include "sequence_group.hpp"

namespace ov::genai {

class CacheBlock {
    int m_ref_count;
    int m_live_ref_count;
    int m_index;
    size_t m_hash;
    bool m_has_published_hash;
    std::chrono::time_point<std::chrono::steady_clock> m_timestamp;
public:
    using Ptr = std::shared_ptr<CacheBlock>;
    explicit CacheBlock(int index)
        : m_ref_count(0),
          m_live_ref_count(0),
          m_index(index),
          m_hash(0),
          m_has_published_hash(false),
          m_timestamp(std::chrono::steady_clock::now()) { }

    int get_index() const {
        return m_index;
    }

    bool is_free() const {
        return m_ref_count == 0;
    }

    void increment() {
        ++m_ref_count;
    }

    void release() {
        OPENVINO_ASSERT(m_ref_count > 0);
        --m_ref_count;
    }

    void release_prevalidated() noexcept {
        --m_ref_count;
    }

    void increment_live() noexcept {
        ++m_ref_count;
        ++m_live_ref_count;
    }

    void release_live() noexcept {
        --m_ref_count;
        --m_live_ref_count;
    }

    bool copy_on_write() const {
        return m_ref_count - m_live_ref_count > 1;
    }

    int get_references_count() const {
        return m_ref_count;
    }

    size_t get_hash() const {
        OPENVINO_ASSERT(m_has_published_hash, "Unpublished cache block has no prefix hash");
        return m_hash;
    }

    void set_hash(size_t hash) {
        m_hash = hash;
        m_has_published_hash = true;
    }

    bool has_published_hash() const {
        return m_has_published_hash;
    }

    void clear_published_hash() {
        m_has_published_hash = false;
    }

    void set_timestamp(const std::chrono::time_point<std::chrono::steady_clock>& timestamp) {
        m_timestamp = timestamp;
    }

    std::chrono::time_point<std::chrono::steady_clock> get_timestamp() {
        return m_timestamp;
    }
};

using BlocksPerLayer = std::vector<CacheBlock::Ptr>;

/**
 * @brief Allows to store and retrieve KV-cache blocks based on their content- and position-based hash.
 * Blocks with the same prefix in the generated sequence will have the same hash. Blocks within this store
 * are not owned by any sequence (but had been once) and may be either selected for overwriting, if the allocator
 * runs out of fresh blocks, or reused if their contents match to the prefix-based requested hash.
 */
class OverwritableBlocksHashStore {
    std::map<size_t, BlocksPerLayer> m_blocks;
    friend class BlockAllocator;
    size_t m_num_layers;
    public:
    /**
     * Constructs the BlockHashStore.
     * @param num_layers The number of separate attention layers with KV caches in the LLM associated with the pipeline.
     */
    explicit OverwritableBlocksHashStore(size_t num_layers = 1) : m_num_layers(num_layers) { OPENVINO_ASSERT(num_layers != 0, "num_layers must be non-zero"); }

    /**
     * Registers allocated KV cache blocks as overwritable. The blocks must not be owned by any sequence.
     * @param blocks_for_all_layers A vector of KV cache blocks (one for each decoder layer) to be added to the store.
     * The hash of each block across the vector must be identical.
     */
    void add(const BlocksPerLayer& blocks_for_all_layers) {
        OPENVINO_ASSERT(blocks_for_all_layers.size() == m_num_layers);
        bool is_all_free = std::all_of(blocks_for_all_layers.begin(), blocks_for_all_layers.end(), [](const CacheBlock::Ptr& block_ptr) { return block_ptr->is_free(); });
        OPENVINO_ASSERT(is_all_free);
        size_t hash = blocks_for_all_layers[0]->get_hash();
        for (const auto& block : blocks_for_all_layers) {
            if (block->get_hash() != hash) {
                OPENVINO_THROW("internal error - block hashes for all layers must be equal");
            }
        }
        OPENVINO_ASSERT(m_blocks.count(hash) == 0);
        m_blocks[hash] = blocks_for_all_layers;
    }


    /**
      * Retrieves KV cache blocks from storage by their hash (expected to be identical for all layers) for their contents
      * to be reused by another sequence. Returned blocks will have reference counters equal to 1.
      * @param hash The hash value to look up in the store.
      * @return A vector of KV cache blocks (one for each decoder layer) previously stored under this hash.
      */
    BlocksPerLayer get_block_to_restore(size_t hash) {
        auto it = m_blocks.find(hash);
        if (it == m_blocks.end())
        {
            return {};
        }
        BlocksPerLayer blocks_for_all_layers = it->second;
        for (auto& block_ptr : blocks_for_all_layers) {

            block_ptr->set_timestamp(std::chrono::steady_clock::now());
            block_ptr->increment();
        }
        m_blocks.erase(it);
        return blocks_for_all_layers;
    }

    /**
     * Pops the least recently used blocks from the store to be used and overwritten by another sequence.
     * Returned blocks will have reference counters equal to 1.
     * @return A vector of KV cache blocks (one for each decoder layer) that has least recently been added to the store
     * based on the timestamp.
     */
    BlocksPerLayer get_lru_block_to_overwrite() {
        if (m_blocks.empty()) {
            return {};
        }
        auto hash_and_blocks_for_all_layers = std::min_element(std::begin(m_blocks), std::end(m_blocks), [](const auto& lhs, const auto& rhs) -> bool { return lhs.second[0]->get_timestamp() < rhs.second[0]->get_timestamp(); });
        auto blocks_for_all_layers = hash_and_blocks_for_all_layers->second;
        for (auto& block_ptr : blocks_for_all_layers) {
            block_ptr->increment();
        }
        m_blocks.erase(hash_and_blocks_for_all_layers->first);
        return blocks_for_all_layers;
    }

    /**
     *
     * @return Number of blocks (per layer) currently in the store.
     */
    size_t num_blocks() const {
        return m_blocks.size();
    }

    bool has_block(size_t hash) const {
        return m_blocks.count(hash) > 0;
    }

    /**
     * @brief Removes blocks matching to the supplied hashes from the store
     * @param hashes_to_discard A set of hashes. For each hash, if it is present in the store, the corresponding block will be discarded
     *   and the block added to the returned vector. If a hash is not present in the store, it is silently ignored.
     * @return A vector of blocks, each element corresponding to a removed hash.
     */
    std::vector<BlocksPerLayer> clean_store(const std::set<uint64_t>& hashes_to_discard) {
        std::vector<BlocksPerLayer> retval;
        retval.reserve(hashes_to_discard.size());
        for (uint64_t hash : hashes_to_discard) {
            auto it = m_blocks.find(hash);
            if (it != m_blocks.end()) {
                retval.push_back(it->second);
                m_blocks.erase(it);
            }
        }
        return retval;
    }

    void clear() {
        m_blocks.clear();
    }
};

class CacheStateDumper;

/**
 * @brief Maintains a layered pool of cache block descriptors, freeing or allocating them as requested.
 */
class BlockAllocator {
    std::vector<std::list<CacheBlock::Ptr>> m_free_blocks;
    // We keep m_free_blocks_num instead of m_free_blocks[X].size() to WA old CXX library implementation issue for std::list::size()
    // see https://stackoverflow.com/questions/13157164/why-isnt-stdlist-size-constant-time
    std::vector<size_t> m_free_blocks_num;
    size_t m_total_num_blocks;
    friend class CacheStateDumper;
    size_t m_num_layers;
    bool m_enable_prefix_caching;
    ov::genai::OverwritableBlocksHashStore m_overwriteable_blocks;
public:
    struct UncachedBlockAllocation {
        BlocksPerLayer blocks;
        std::vector<std::list<CacheBlock::Ptr>> free_nodes;
    };

    struct PreparedBlockRelease {
        UncachedBlockAllocation allocation;
        std::map<size_t, BlocksPerLayer> cached_node;
    };

    PreparedBlockRelease prepare_block_release(const BlocksPerLayer& blocks,
                                               const std::map<uint64_t, BlocksPerLayer>& registry) const {
        PreparedBlockRelease prepared;
        prepared.allocation.blocks = blocks;
        prepared.allocation.free_nodes.resize(m_num_layers);
        for (auto& nodes : prepared.allocation.free_nodes) {
            nodes.emplace_back();
        }
        if (blocks.front()->has_published_hash()) {
            const auto found = registry.find(blocks.front()->get_hash());
            if (found != registry.end() && found->second == blocks) {
                prepared.cached_node.emplace(found->first, blocks);
            }
        }
        return prepared;
    }

    void release_prepared_block(PreparedBlockRelease& prepared) noexcept {
        auto& allocation = prepared.allocation;
        for (const auto& block : allocation.blocks) {
            block->release_prevalidated();
        }
        if (!allocation.blocks.front()->is_free()) {
            return;
        }
        if (!prepared.cached_node.empty()) {
            m_overwriteable_blocks.m_blocks.merge(prepared.cached_node);
            return;
        }
        for (size_t layer = 0; layer < m_num_layers; ++layer) {
            allocation.blocks[layer]->clear_published_hash();
            allocation.free_nodes[layer].front() = allocation.blocks[layer];
            m_free_blocks[layer].splice(m_free_blocks[layer].end(), allocation.free_nodes[layer]);
            ++m_free_blocks_num[layer];
        }
    }

    struct CacheBlockAllocationResult {
        BlocksPerLayer blocks;
        std::optional<uint64_t> evicted_hash;
        std::optional<std::chrono::time_point<std::chrono::steady_clock>> evicted_timestamp;
        std::optional<uint64_t> erased_hash;

        operator BlocksPerLayer&() {
            return blocks;
        }

        operator const BlocksPerLayer&() const {
            return blocks;
        }

        CacheBlock::Ptr& operator[](size_t index) {
            return blocks[index];
        }

        const CacheBlock::Ptr& operator[](size_t index) const {
            return blocks[index];
        }

        CacheBlock::Ptr& at(size_t index) {
            return blocks.at(index);
        }

        const CacheBlock::Ptr& at(size_t index) const {
            return blocks.at(index);
        }
    };

private:
    static bool erase_registration_if_owned(uint64_t hash,
                                            const BlocksPerLayer& blocks,
                                            std::map<uint64_t, BlocksPerLayer>& cached_blocks) {
        const auto cached_blocks_it = cached_blocks.find(hash);
        if (cached_blocks_it == cached_blocks.end() || cached_blocks_it->second != blocks) {
            return false;
        }
        cached_blocks.erase(cached_blocks_it);
        return true;
    }

public:

    /**
     * Constructs the BlockAllocator.
     * @param num_blocks Number of cache blocks in the free block pool to be owned by this allocator.
     * @param enable_prefix_caching Whether prefix caching should be enabled for this allocator.
     * See also the equivalent parameter in ov::genai::ContinuousBatchingPipeline
     * @param num_layers The number of separate block-table layers associated with the allocator.
     * Blocks returned will be vectors with this size, each vector entry to be associated with a separate cache layer.
     */
    BlockAllocator(size_t num_blocks, bool enable_prefix_caching, size_t num_layers = 1) :
            m_total_num_blocks(num_blocks), m_num_layers(num_layers), m_enable_prefix_caching(enable_prefix_caching), m_overwriteable_blocks(num_layers) {
        OPENVINO_ASSERT(num_layers != 0, "num_layers must be non-zero");
        OPENVINO_ASSERT(num_blocks <= static_cast<size_t>(std::numeric_limits<int>::max()),
                        "num_blocks ", num_blocks, " exceeds the maximum allowed block index (", std::numeric_limits<int>::max(), ")");
        m_free_blocks.resize(m_num_layers);
        if (num_blocks > 0) {
            m_free_blocks_num = std::vector<size_t>(num_layers, num_blocks);
            for (auto& per_layer_block_list : m_free_blocks) {
                for (size_t block_id = 0; block_id < m_total_num_blocks; ++block_id) {
                    per_layer_block_list.push_back(std::make_shared<CacheBlock>(static_cast<int>(block_id)));
                }
            }
        } else {
            m_free_blocks_num = std::vector<size_t>(m_num_layers, 0);
        }
    }

    ~BlockAllocator() {
        // sanity check to validate that all blocks are freed
        for (auto& free_block : m_free_blocks_num) {
            const size_t free_and_overwritable_block_cnt = free_block + num_overwriteable_blocks();
            if (m_total_num_blocks != free_and_overwritable_block_cnt) {
                try {
                    GENAI_ERR("BlockAllocator leaked blocks. Expected num free blocks: %zu, actual: %zu",
                              m_total_num_blocks,
                              free_and_overwritable_block_cnt);
                } catch (...) {
                    std::fprintf(stderr,
                                 "BlockAllocator leaked blocks. Expected: %zu, actual: %zu\n",
                                 m_total_num_blocks,
                                 free_and_overwritable_block_cnt);
                }
            }
        }
    }

    void increase_block_count(size_t new_block_count) {
        OPENVINO_ASSERT(new_block_count <= static_cast<size_t>(std::numeric_limits<int>::max()),
                        "new_block_count ", new_block_count, " exceeds the maximum allowed block index (", std::numeric_limits<int>::max(), ")");
        OPENVINO_ASSERT(new_block_count > m_total_num_blocks, "New blocks number should be more than previous blocks number.");
        size_t added_blocks = new_block_count - m_total_num_blocks;
        for (size_t idx = 0; idx < m_free_blocks_num.size(); ++idx) {
            m_free_blocks_num[idx] += added_blocks;
        }
        for (auto& per_layer_block_list : m_free_blocks) {
            for (size_t block_id = m_total_num_blocks; block_id < new_block_count; ++block_id) {
                per_layer_block_list.push_back(std::make_shared<CacheBlock>(static_cast<int>(block_id)));
            }
        }
        m_total_num_blocks = new_block_count;
    }


    /**
     * Returns the number of free blocks for a given layer.
     * @param layer_idx Index of the layer.
     * @return Number of free blocks for this layer.
     */
    size_t num_free_blocks(size_t layer_idx) const {
        return m_free_blocks_num[layer_idx] + num_overwriteable_blocks();
    }

    /**
     * Returns the number of overwritable blocks (in a prefix caching scenario).
     * @return Number of overwritable blocks for this layer.
     */
    size_t num_overwriteable_blocks() const {
        return m_overwriteable_blocks.num_blocks();
    }

    /**
     * Returns a boolean describing whether a given number of blocks can be allocated, based on the number of currently
     * available free blocks.
     * @param num_blocks The number of blocks requested to be allocated.
     * @return Whether `num_blocks` can be allocated at this time.
     */
    bool can_allocate_blocks(size_t num_blocks) const {
        bool retval = true;
        for (size_t i = 0; i < m_num_layers; i++) retval &= can_allocate_blocks(num_blocks, i);
        return retval;
    }

    /**
     * Returns a boolean describing whether a given number of blocks can be allocated for a given layer,
     * based on the number of currently available free blocks for the same layer.
     * @param num_blocks The number of blocks requested to be allocated.
     * @param layer_idx The index of the layer for which the allocation should occur.
     * @return Whether `num_blocks` can be allocated at this time for this layer.
     */
    bool can_allocate_blocks(size_t num_blocks, size_t layer_idx) const {
        return num_blocks <= num_free_blocks(layer_idx);
    }

    bool can_allocate_uncached_blocks(size_t num_blocks) const {
        return can_allocate_blocks(num_blocks);
    }

    /**
     * Frees a given block for a given layer. If no sequence is associated with the block after freeing, the block
     * is returned to the "free" pool.
     * @param block_ptr The block to be freed.
     * @param layer_idx The index of the layer with which the block is associated.
     */
    void free(CacheBlock::Ptr& block_ptr, size_t layer_idx) {
        OPENVINO_ASSERT(!m_enable_prefix_caching);
        OPENVINO_ASSERT(layer_idx < m_num_layers);
        block_ptr->release();
        if (block_ptr->is_free()) {
            m_free_blocks[layer_idx].push_back(block_ptr);
            ++m_free_blocks_num[layer_idx];
        }
    }

    std::vector<UncachedBlockAllocation> allocate_temporary_blocks(
        size_t num_blocks,
        std::map<uint64_t, BlocksPerLayer>& cached_blocks,
        std::vector<int>& block_indices,
        std::vector<uint64_t>& erased_hashes) {
        OPENVINO_ASSERT(num_blocks > 0 && can_allocate_blocks(num_blocks),
                        "Not enough cache capacity for temporary reservation");
        const size_t fresh_count = std::min(num_blocks, m_free_blocks_num.front());
        const size_t eviction_count = num_blocks - fresh_count;
        using StoreIterator = decltype(m_overwriteable_blocks.m_blocks)::iterator;
        std::vector<StoreIterator> candidates;
        if (eviction_count > 0) {
            candidates.reserve(m_overwriteable_blocks.m_blocks.size());
            for (auto iterator = m_overwriteable_blocks.m_blocks.begin();
                 iterator != m_overwriteable_blocks.m_blocks.end(); ++iterator) {
                candidates.push_back(iterator);
            }
            std::partial_sort(candidates.begin(), candidates.begin() + eviction_count, candidates.end(),
                              [](const StoreIterator& left, const StoreIterator& right) {
                                  return left->second.front()->get_timestamp() < right->second.front()->get_timestamp();
                              });
        }
        std::vector<UncachedBlockAllocation> allocations(num_blocks);
        block_indices.reserve(num_blocks);
        erased_hashes.reserve(eviction_count);
        for (auto& allocation : allocations) {
            allocation.blocks.resize(m_num_layers);
            allocation.free_nodes.resize(m_num_layers);
        }
        for (size_t layer = 0; layer < m_num_layers; ++layer) {
            auto fresh = m_free_blocks[layer].begin();
            for (size_t row = 0; row < num_blocks; ++row) {
                auto& allocation = allocations[row];
                if (row < fresh_count) {
                    allocation.blocks[layer] = *fresh++;
                } else {
                    allocation.blocks[layer] = candidates[row - fresh_count]->second[layer];
                    allocation.free_nodes[layer].emplace_back();
                }
                OPENVINO_ASSERT(allocation.blocks[layer]->is_free(),
                                "Temporary reservation cannot evict a referenced cache row");
            }
        }
        for (const auto& allocation : allocations) {
            block_indices.push_back(allocation.blocks.front()->get_index());
        }
        for (size_t row = 0; row < num_blocks; ++row) {
            auto& allocation = allocations[row];
            if (row >= fresh_count) {
                const auto candidate = candidates[row - fresh_count];
                if (erase_registration_if_owned(candidate->first, allocation.blocks, cached_blocks)) {
                    erased_hashes.push_back(candidate->first);
                }
                m_overwriteable_blocks.m_blocks.erase(candidate);
            }
            for (size_t layer = 0; layer < m_num_layers; ++layer) {
                if (row < fresh_count) {
                    allocation.free_nodes[layer].splice(allocation.free_nodes[layer].end(),
                                                         m_free_blocks[layer], m_free_blocks[layer].begin());
                    --m_free_blocks_num[layer];
                }
                allocation.blocks[layer]->increment();
                allocation.blocks[layer]->clear_published_hash();
            }
        }
        return allocations;
    }

    void free_uncached(UncachedBlockAllocation& allocation) noexcept {
        for (size_t layer_idx = 0; layer_idx < m_num_layers; ++layer_idx) {
            CacheBlock::Ptr& block = allocation.blocks[layer_idx];
            allocation.free_nodes[layer_idx].front() = block;
            block->release_prevalidated();
            if (block->is_free()) {
                m_free_blocks[layer_idx].splice(m_free_blocks[layer_idx].end(),
                                                allocation.free_nodes[layer_idx]);
                ++m_free_blocks_num[layer_idx];
            } else {
                allocation.free_nodes[layer_idx].clear();
            }
        }
        allocation.blocks.clear();
    }

    void free_prepared(CacheBlock::Ptr& block,
                       std::list<CacheBlock::Ptr>& free_node,
                       size_t layer_index) noexcept {
        block->release_prevalidated();
        if (block->is_free()) {
            free_node.front() = block;
            m_free_blocks[layer_index].splice(m_free_blocks[layer_index].end(), free_node);
            ++m_free_blocks_num[layer_index];
        }
    }

    CacheBlockAllocationResult acquire_unpublished_block(std::map<uint64_t, BlocksPerLayer>& cached_blocks) {
        CacheBlockAllocationResult result;
        if (m_free_blocks_num[0] > 0) {
            result.blocks.reserve(m_num_layers);
            for (size_t layer_idx = 0; layer_idx < m_num_layers; ++layer_idx) {
                CacheBlock::Ptr allocated_block = m_free_blocks[layer_idx].front();
                allocated_block->increment();
                allocated_block->clear_published_hash();
                result.blocks.push_back(allocated_block);
                m_free_blocks[layer_idx].pop_front();
                --m_free_blocks_num[layer_idx];
            }
            return result;
        }
        if (m_overwriteable_blocks.num_blocks() == 0) {
            return result;
        }
        result.blocks = m_overwriteable_blocks.get_lru_block_to_overwrite();
        result.evicted_hash = result.blocks.front()->get_hash();
        result.evicted_timestamp = result.blocks.front()->get_timestamp();
        if (erase_registration_if_owned(*result.evicted_hash, result.blocks, cached_blocks)) {
            result.erased_hash = result.evicted_hash;
        }
        for (auto& block : result.blocks) {
            block->clear_published_hash();
        }
        return result;
    }

    void rollback_uncached_block(CacheBlockAllocationResult allocation,
                                 std::map<uint64_t, BlocksPerLayer>& cached_blocks) {
        OPENVINO_ASSERT(!allocation.blocks.empty());
        if (!allocation.evicted_hash.has_value()) {
            free_uncached(allocation.blocks);
            return;
        }
        OPENVINO_ASSERT(allocation.evicted_timestamp.has_value());
        for (auto& block : allocation.blocks) {
            block->set_hash(*allocation.evicted_hash);
            block->set_timestamp(*allocation.evicted_timestamp);
            block->release();
        }
        if (allocation.erased_hash.has_value()) {
            cached_blocks[*allocation.erased_hash] = allocation.blocks;
        }
        m_overwriteable_blocks.add(allocation.blocks);
    }

    /**
     * Releases an uncached block set directly to the free pool. This bypasses
     * prefix-cache overwrite storage because transient checkpoints do not have
     * stable prefix hashes.
     */
    void free_uncached(const BlocksPerLayer& blocks_for_all_layers) {
        OPENVINO_ASSERT(blocks_for_all_layers.size() == m_num_layers);
        for (size_t layer_idx = 0; layer_idx < m_num_layers; ++layer_idx) {
            auto& block_ptr = blocks_for_all_layers[layer_idx];
            block_ptr->release();
            if (block_ptr->is_free()) {
                m_free_blocks[layer_idx].push_back(block_ptr);
                ++m_free_blocks_num[layer_idx];
            }
        }
    }

    /**
     * Frees a block for each layer. If no sequence is associated with the blocks after freeing, the blocks
     * are either returned to the "free" pool, or, if prefix caching is enabled, stored internally for its contents
     * to be potentially reused if a prefix of a new sequence matches to the prefix with which the currently freed blocks
     * were computed.
     * @param blocks_for_all_layers The blocks to be freed (one for each layer).
     */
    void free(const BlocksPerLayer& blocks_for_all_layers,
              std::map<uint64_t, BlocksPerLayer>& cached_blocks) {
        OPENVINO_ASSERT(blocks_for_all_layers.size() == m_num_layers);
        for (size_t i = 0; i < m_num_layers; i++) {
            auto& block_ptr = blocks_for_all_layers[i];
            block_ptr->release();
        }

        auto free_predicate = [](const CacheBlock::Ptr& block_ptr) { return block_ptr->is_free(); };
        bool is_any_free = std::any_of(blocks_for_all_layers.begin(), blocks_for_all_layers.end(), free_predicate);
        bool is_all_free = false;
        if (is_any_free && m_num_layers > 1) {
            is_all_free = std::all_of(blocks_for_all_layers.begin(), blocks_for_all_layers.end(), free_predicate);
            OPENVINO_ASSERT(is_all_free, "blocks across layers must be freed simultaneously");
        }

        if (is_any_free) {
            // is_all_free == true due to assert above
            if (m_enable_prefix_caching && !blocks_for_all_layers[0]->has_published_hash()) {
                for (size_t layer_idx = 0; layer_idx < blocks_for_all_layers.size(); ++layer_idx) {
                    OPENVINO_ASSERT(!blocks_for_all_layers[layer_idx]->has_published_hash(),
                                    "Cache blocks across layers must share publication state");
                    m_free_blocks[layer_idx].push_back(blocks_for_all_layers[layer_idx]);
                    ++m_free_blocks_num[layer_idx];
                }
            } else if (m_enable_prefix_caching)
            {
                std::set<uint64_t> hashes_across_blocks;
                for (const auto& block : blocks_for_all_layers) {
                    hashes_across_blocks.insert(block->get_hash());
                }
                bool is_all_have_same_hash = (hashes_across_blocks.size() == 1);
                if (is_all_have_same_hash) {
                    // guard against hash collision
                    auto colliding_blocks = m_overwriteable_blocks.clean_store(hashes_across_blocks);
                    if (!colliding_blocks.empty()) {
                        OPENVINO_ASSERT(colliding_blocks.size() == 1);
                        BlocksPerLayer& colliding_blocks_per_layer = colliding_blocks[0];
                        bool is_same_block = true;
                        for (size_t layer_idx = 0; layer_idx < colliding_blocks_per_layer.size(); layer_idx++) {
                            if (colliding_blocks_per_layer[layer_idx]->get_index() != blocks_for_all_layers[layer_idx]->get_index()) {
                                is_same_block = false;
                                break;
                            }
                        }

                        if (is_same_block) {
                            OPENVINO_THROW("internal error - double free when prefix caching");
                        }

                        // actual collision case
                        for (size_t layer_idx = 0; layer_idx < colliding_blocks_per_layer.size(); layer_idx++) {
                            m_free_blocks[layer_idx].push_back(colliding_blocks_per_layer[layer_idx]);
                            ++m_free_blocks_num[layer_idx];
                        }

                        // As block returns to free memory, cached_blocks should stop referring to the evicted block.
                        // Keep the hash registered because blocks_for_all_layers will remain cached under the same hash.
                        auto cached_it = cached_blocks.find(colliding_blocks_per_layer[0]->get_hash());
                        if (cached_it != cached_blocks.end()) {
                            OPENVINO_ASSERT(!cached_it->second.empty(), "cached_blocks entry must not be empty");
                            const auto colliding_block_idx = colliding_blocks_per_layer[0]->get_index();
                            const auto cached_block_idx = cached_it->second[0]->get_index();
                            if (colliding_block_idx == cached_block_idx) {
                                cached_it->second = blocks_for_all_layers;
                            }
                        }
                    }
                    m_overwriteable_blocks.add(blocks_for_all_layers);
                } else {
                    // This set of blocks to be freed corresponds to blocks from different time steps, and thus not eligible for caching
                    // TODO (vshampor): more fine-grained hash store control
                    for (size_t layer_idx = 0; layer_idx < blocks_for_all_layers.size(); layer_idx++) {
                        m_free_blocks[layer_idx].push_back(blocks_for_all_layers[layer_idx]);
                        ++m_free_blocks_num[layer_idx];
                    }
                }
            }
            else {
                for (size_t layer_idx = 0; layer_idx < blocks_for_all_layers.size(); layer_idx++) {
                    m_free_blocks[layer_idx].push_back(blocks_for_all_layers[layer_idx]);
                    ++m_free_blocks_num[layer_idx];
                }
            }
        }
    }

    /**
     * Allocates and returns one block for each layer. Can only be used if prefix caching is disabled.
     * @return A vector of blocks allocated (one for each layer).
     */
    BlocksPerLayer allocate_block() {
        BlocksPerLayer retval;
        retval.reserve(m_num_layers);
        for (size_t i = 0; i < m_num_layers; i++) {
            retval.push_back(allocate_block(i));
        }
        return retval;
    }

    /**
     * Allocates and returns one block for a given layer. Can only be used if prefix caching is disabled.
     * @return The block allocated for this layer.
     */
    CacheBlock::Ptr allocate_block(size_t layer_idx) {
        OPENVINO_ASSERT(layer_idx < m_free_blocks.size());
        OPENVINO_ASSERT(!m_enable_prefix_caching);
        OPENVINO_ASSERT(can_allocate_blocks(1, layer_idx));
        CacheBlock::Ptr allocated_block = m_free_blocks[layer_idx].front();
        allocated_block->increment();
        m_free_blocks[layer_idx].pop_front();
        --m_free_blocks_num[layer_idx];
        return allocated_block;
    }

    /**
     * Returns one block for each layer, either by allocating new blocks if the allocator's initial "free" pool is not
     * exhausted, or by selecting a least recently used block from the hash store (so that its contents would be overwritten) otherwise.
     * Can only be used if prefix caching is enabled.
     * @param[in] hash The expected hash of the new block (based on the current sequence prefix).
     * @param[in,out] cached_blocks The map of known hashes to already allocated and filled blocks. If the blocks are freshly allocated,
     * it is added to this map under `hash`. If the blocks are reused from the internal overwritable block store,
     * the previous hash entry for these is deleted and the reused blocks are likewise stored in the map under the (new) `hash`.
     * @return A vector of blocks (one for each layer), either freshly allocated or reused for overwriting,
     * or an empty vector if cache is exhausted.
     */
    CacheBlockAllocationResult allocate_block(size_t hash,
                                              std::map<uint64_t, BlocksPerLayer>& cached_blocks) {
        OPENVINO_ASSERT(m_enable_prefix_caching);
        OPENVINO_ASSERT(can_allocate_blocks(1));

        CacheBlockAllocationResult result;
        if (m_free_blocks_num[0] > 0) {
            // allocate new empty block
            result.blocks.reserve(m_num_layers);
            for (size_t i = 0; i < m_num_layers; i++) {
                CacheBlock::Ptr allocated_block = m_free_blocks[i].front();
                allocated_block->increment();
                allocated_block->set_hash(hash);
                result.blocks.push_back(allocated_block);
                m_free_blocks[i].pop_front();
                --m_free_blocks_num[i];
            }
            cached_blocks[hash] = result.blocks;
            return result;
        }
        if (m_overwriteable_blocks.num_blocks() > 0) {
            // get least recently used block from store and reuse it
            result.blocks = m_overwriteable_blocks.get_lru_block_to_overwrite();
            const uint64_t previous_hash = result.blocks[0]->get_hash();
            if (erase_registration_if_owned(previous_hash, result.blocks, cached_blocks)) {
                result.erased_hash = previous_hash;
            }

            // update block with new hash
            const auto timestamp = std::chrono::steady_clock::now();
            for (auto& block : result.blocks) {
                block->set_hash(hash);
                block->set_timestamp(timestamp);
            }
            cached_blocks[hash] = result.blocks;
            return result;
        }
        // should not be reachable due to the can_allocate_blocks assert in the beginning
        return result;
    }

    /**
     * Returns the blocks corresponding to a given hash either from the internal allocator store,
     * or from the supplied storage map, or nothing if there are no blocks corresponding to this hash.
     *
     * @param hash The hash of the blocks to be looked up.
     * @param cached_blocks The map of known hashes to already allocated and filled blocks.
     * @return A vector of blocks (one for each layer) corresponding to this hash, or an empty vector if the hash is not found in the map.
     */
    BlocksPerLayer get_cached_block(size_t hash, std::map<uint64_t, BlocksPerLayer>& cached_blocks) {
        auto blocks_for_all_layers = m_overwriteable_blocks.get_block_to_restore(hash);
        if (!blocks_for_all_layers.empty()) {
            // use cached block from internal store
            return blocks_for_all_layers;
        }
        auto it = cached_blocks.find(hash);
        if (it != cached_blocks.end()) {
            // use cached block from cached_blocks
            // TODO: add tokens validation in case of hash collision
            blocks_for_all_layers = it->second;
            for (auto& block_ptr : cached_blocks[hash]) {
                block_ptr->increment();
            }
            return blocks_for_all_layers;
        }
        return {};
    }

    bool has_cached_block(size_t hash, const std::map<uint64_t, BlocksPerLayer>& cached_blocks) const {
        return m_overwriteable_blocks.has_block(hash) || cached_blocks.count(hash) > 0;
    }

    BlocksPerLayer peek_cached_block(uint64_t hash, const std::map<uint64_t, BlocksPerLayer>& cached_blocks) const {
        const auto stored = m_overwriteable_blocks.m_blocks.find(hash);
        if (stored != m_overwriteable_blocks.m_blocks.end()) {
            return stored->second;
        }
        const auto cached = cached_blocks.find(hash);
        return cached == cached_blocks.end() ? BlocksPerLayer{} : cached->second;
    }

    void acquire_prepared_cached_block(uint64_t hash, const BlocksPerLayer& blocks) noexcept {
        m_overwriteable_blocks.m_blocks.erase(hash);
        for (const auto& block : blocks) {
            block->increment();
            block->set_timestamp(std::chrono::steady_clock::now());
        }
    }

    /**
     * @return The percentage of the allocator's free block pool utilization.
     */
    float get_used_percentage() const {
        size_t sum = 0;
        for (size_t layer_idx = 0; layer_idx < m_num_layers; layer_idx++) sum += num_free_blocks(layer_idx);
        return static_cast<float>(m_num_layers * m_total_num_blocks - sum) / (m_num_layers * m_total_num_blocks) * 100;
    }

    /**
     * @return The total number of blocks.
     */
    size_t get_total_block_count() const {
        return m_total_num_blocks;
    }

    void clear() {
        m_total_num_blocks = 0;
        m_free_blocks_num = std::vector<size_t>(m_num_layers, 0);
        for (auto& per_layer_block_list : m_free_blocks) {
            per_layer_block_list.clear();
        }
        m_overwriteable_blocks.clear();
    }
};

/**
 * @brief Works with `ov::genai::SequenceGroup`s and individual `ov::genai::Sequence`s to assign cache blocks to these
 * at each pipeline generation step. A block table is kept for each sequence, storing the indices of "physical"
 * cache blocks currently allocated to a given sequence. Each block table defines a linear "logical" block space, with positions of
 * blocks within the block table being associated with "logical" block indices.
 */
class BlockManager {
public:
    struct LinearAttentionLiveState {
        size_t endpoint = 0;
        BlocksPerLayer rows;
        size_t generation = 0;
        bool is_empty = true;
    };

private:
    using TemporaryBlockTable = std::vector<BlockAllocator::UncachedBlockAllocation>;

    void replace_linear_attention_live_rows(LinearAttentionLiveState& live_state,
                                            const BlocksPerLayer& rows) noexcept {
        for (const auto& row : rows) {
            row->increment_live();
        }
        for (auto& row : live_state.rows) {
            row->release_live();
        }
        live_state.rows = rows;
    }

    void release_linear_attention_live_rows(uint64_t seq_id) noexcept {
        const auto live_state_it = m_linear_attention_live_states.find(seq_id);
        if (live_state_it == m_linear_attention_live_states.end()) {
            return;
        }
        for (auto& row : live_state_it->second.rows) {
            row->release_live();
        }
        m_linear_attention_live_states.erase(live_state_it);
    }
    friend class CacheStateDumper;
    BlockAllocator m_allocator;
    bool m_enable_prefix_caching;
    size_t m_block_size;
    size_t m_num_layers;
    size_t m_fixed_blocks_per_sequence = 0;  /// When > 0, each sequence gets exactly this many blocks.
    size_t m_max_total_blocks = 0;  /// When > 0, the pool may never grow beyond this many blocks.
    bool m_restore_latest_prefix_block_only = false;
    // TODO: caching time can probably be improved if we use the prefix tree
    std::map<uint64_t, BlocksPerLayer> m_prefix_hash_to_cached_blocks;
    std::map<size_t, size_t> m_cached_content_length_ref_counts;
    std::map<uint64_t, size_t> m_cached_hash_to_content_length;

    // stores blocks for each sequence (not sequence group)
    // the same block can be seen in multiple block_tables for different sequences
    std::map<uint64_t, std::vector<BlocksPerLayer>> m_block_table;
    std::map<uint64_t, TemporaryBlockTable> m_temporary_block_table;
    std::map<uint64_t, TemporaryBlockTable> m_linear_attention_headroom;
    std::map<uint64_t, size_t> m_temporary_window_sizes;
    std::map<uint64_t, size_t> m_block_table_logical_start;

    std::map<uint64_t, LinearAttentionLiveState> m_linear_attention_live_states;

    std::mutex m_cached_blocks_map_mutex;
public:
    struct PrefixRestorePlan {
        std::vector<size_t> block_content_lengths;
        size_t cache_token_position = 0;
        size_t processed_tokens = 0;
        size_t logical_block_start = 0;

        bool empty() const {
            return block_content_lengths.empty();
        }
    };

    class PreparedPrefixRestore {
    public:
        PreparedPrefixRestore(const PreparedPrefixRestore&) = delete;
        PreparedPrefixRestore& operator=(const PreparedPrefixRestore&) = delete;
        PreparedPrefixRestore(PreparedPrefixRestore&&) = default;
        PreparedPrefixRestore& operator=(PreparedPrefixRestore&&) = delete;

        void apply() {
            OPENVINO_ASSERT(m_lock.owns_lock() && !m_applied,
                            "Prepared prefix restore was already applied or moved");
            for (const auto& [hash, blocks] : m_blocks) {
                m_owner.m_allocator.acquire_prepared_cached_block(hash, blocks);
            }
            for (const auto& [seq_id, live] : m_live_states) {
                for (const auto& row : live.rows) {
                    row->increment_live();
                }
            }
            m_owner.m_block_table.merge(m_tables);
            m_owner.m_block_table_logical_start.merge(m_logical_starts);
            m_owner.m_linear_attention_live_states.merge(m_live_states);
            m_applied = true;
        }

    private:
        friend class BlockManager;
        PreparedPrefixRestore(BlockManager& owner, std::unique_lock<std::mutex> lock)
            : m_owner(owner), m_lock(std::move(lock)) {}

        BlockManager& m_owner;
        std::unique_lock<std::mutex> m_lock;
        std::vector<std::pair<uint64_t, BlocksPerLayer>> m_blocks;
        std::map<uint64_t, std::vector<BlocksPerLayer>> m_tables;
        std::map<uint64_t, size_t> m_logical_starts;
        std::map<uint64_t, LinearAttentionLiveState> m_live_states;
        bool m_applied = false;
    };

    std::optional<PreparedPrefixRestore> prepare_prefix_restore(const SequenceGroup::Ptr& group,
                                                               const PrefixRestorePlan& plan) {
        std::unique_lock<std::mutex> lock(m_cached_blocks_map_mutex);
        if (plan.empty()) {
            return std::nullopt;
        }
        const auto sequences = group->get_not_finished_sequences();
        OPENVINO_ASSERT(sequences.size() == 1, "Prefix restore requires one sequence");
        const auto& sequence = sequences.front();
        const uint64_t seq_id = sequence->get_id();
        OPENVINO_ASSERT(m_block_table.count(seq_id) == 0 && m_linear_attention_live_states.count(seq_id) == 0 &&
                            m_temporary_block_table.count(seq_id) == 0,
                        "Prefix restore requires an unallocated sequence");
        const auto current = get_prefix_restore_plan_unlocked(group, plan.cache_token_position);
        if (current.block_content_lengths != plan.block_content_lengths ||
            current.cache_token_position != plan.cache_token_position ||
            current.processed_tokens != plan.processed_tokens || current.logical_block_start != plan.logical_block_start) {
            return std::nullopt;
        }
        PreparedPrefixRestore prepared(*this, std::move(lock));
        auto& tables = prepared.m_tables.try_emplace(seq_id, m_num_layers).first->second;
        prepared.m_blocks.reserve(plan.block_content_lengths.size());
        for (auto& table : tables) {
            table.reserve(plan.block_content_lengths.size());
        }
        for (size_t content_length : plan.block_content_lengths) {
            const uint64_t hash = sequence->get_hash(content_length, m_block_size);
            auto rows = m_allocator.peek_cached_block(hash, m_prefix_hash_to_cached_blocks);
            if (rows.size() != m_num_layers) {
                return std::nullopt;
            }
            for (size_t layer = 0; layer < m_num_layers; ++layer) {
                OPENVINO_ASSERT(rows[layer]->has_published_hash() && rows[layer]->get_hash() == hash,
                                "Restored prefix row identity is inconsistent");
                tables[layer].push_back(rows[layer]);
            }
            prepared.m_blocks.emplace_back(hash, std::move(rows));
        }
        if (m_restore_latest_prefix_block_only) {
            prepared.m_logical_starts.emplace(seq_id, plan.logical_block_start);
            prepared.m_live_states.emplace(seq_id,
                LinearAttentionLiveState{plan.cache_token_position, prepared.m_blocks.back().second, 1, false});
        }
        return prepared;
    }

    struct PreparedTemporaryBlocks {
        size_t endpoint;
        size_t generation;
        std::vector<int> block_indices;
    };

    struct TemporaryPromotionRequest {
        uint64_t seq_id;
        size_t checkpoint_slot;
        size_t expected_endpoint;
        size_t expected_generation;
    };

    struct TailReleaseTarget {
        uint64_t seq_id;
        size_t endpoint;
    };

    class PreparedTailReleases {
    public:
        PreparedTailReleases(const PreparedTailReleases&) = delete;
        PreparedTailReleases& operator=(const PreparedTailReleases&) = delete;
        PreparedTailReleases(PreparedTailReleases&&) noexcept = default;
        PreparedTailReleases& operator=(PreparedTailReleases&&) = delete;

        void apply() {
            OPENVINO_ASSERT(m_lock.owns_lock(), "Prepared KV tail releases were already applied or moved");
            for (PreparedRelease& release : m_releases) {
                for (size_t block_index = 0; block_index < release.blocks.size(); ++block_index) {
                    for (size_t layer_index = 0; layer_index < m_owner->m_num_layers; ++layer_index) {
                        CacheBlock::Ptr& block = release.blocks[block_index][layer_index];
                        m_owner->m_allocator.free_prepared(
                            block, release.free_nodes[block_index][layer_index], layer_index);
                    }
                }
                for (auto& layer_table : release.table_it->second) {
                    layer_table.resize(layer_table.size() - release.blocks.size());
                }
                if (release.table_it->second.front().empty()) {
                    m_owner->m_block_table_logical_start.erase(release.seq_id);
                    m_owner->m_block_table.erase(release.table_it);
                }
            }
            m_lock.unlock();
        }

    private:
        using TableIterator = std::map<uint64_t, std::vector<BlocksPerLayer>>::iterator;
        struct PreparedRelease {
            uint64_t seq_id;
            TableIterator table_it;
            std::vector<BlocksPerLayer> blocks;
            std::vector<std::vector<std::list<CacheBlock::Ptr>>> free_nodes;
        };

        friend class BlockManager;
        PreparedTailReleases(BlockManager& owner,
                             std::unique_lock<std::mutex> lock,
                             std::vector<PreparedRelease> releases)
            : m_owner(&owner), m_lock(std::move(lock)), m_releases(std::move(releases)) {}

        BlockManager* m_owner;
        std::unique_lock<std::mutex> m_lock;
        std::vector<PreparedRelease> m_releases;
    };

    class PreparedTemporaryPromotions {
    public:
        PreparedTemporaryPromotions(const PreparedTemporaryPromotions&) = delete;
        PreparedTemporaryPromotions& operator=(const PreparedTemporaryPromotions&) = delete;
                PreparedTemporaryPromotions(PreparedTemporaryPromotions&& other) noexcept
                        : m_owner(std::exchange(other.m_owner, nullptr)),
                            m_lock(std::move(other.m_lock)),
                            m_promotions(std::move(other.m_promotions)),
                            m_promoted_indices(std::move(other.m_promoted_indices)) {}
                PreparedTemporaryPromotions& operator=(PreparedTemporaryPromotions&&) = delete;

                const std::vector<size_t>& apply() {
                        OPENVINO_ASSERT(m_owner != nullptr && m_lock.owns_lock(),
                                                        "Temporary promotions were already applied or moved");
            for (PreparedPromotion& promotion : m_promotions) {
                auto& temporary_blocks = promotion.temporary_it->second;
                auto& selected_allocation = temporary_blocks[promotion.selected_index];
                BlocksPerLayer& selected_blocks = selected_allocation.blocks;
                auto& block_table = promotion.table_it->second;
                auto& live_state = promotion.live_state_it->second;
                if (m_owner->m_restore_latest_prefix_block_only) {
                    for (auto& row : live_state.rows) {
                        row->release_live();
                    }
                    for (size_t layer = 0; layer < m_owner->m_num_layers; ++layer) {
                        for (const auto& row : promotion.prefix_table[layer]) {
                            row->increment();
                        }
                        selected_blocks[layer]->increment_live();
                        live_state.rows[layer] = selected_blocks[layer];
                    }
                    block_table.swap(promotion.prefix_table);
                    auto& headroom = promotion.headroom_node.begin()->second;
                    for (size_t position = 0; position < promotion.prefix_releases.size(); ++position) {
                        auto& release = promotion.prefix_releases[position];
                        if (position == promotion.reusable_position) {
                            headroom.push_back(std::move(release.allocation));
                        } else {
                            m_owner->m_allocator.release_prepared_block(release);
                        }
                    }
                    m_owner->m_block_table_logical_start.merge(promotion.logical_start_node);
                    m_owner->m_block_table_logical_start.find(promotion.table_it->first)->second =
                        promotion.logical_start;
                    live_state.endpoint = promotion.endpoint;
                    ++live_state.generation;
                    live_state.is_empty = false;
                    for (auto& allocation : temporary_blocks) {
                        if (headroom.size() < promotion.window_size &&
                            allocation.blocks.front()->get_references_count() == 1) {
                            headroom.push_back(std::move(allocation));
                        } else {
                            m_owner->m_allocator.free_uncached(allocation);
                        }
                    }
                    m_owner->m_linear_attention_headroom.erase(promotion.table_it->first);
                    m_owner->m_linear_attention_headroom.merge(promotion.headroom_node);
                    m_owner->m_temporary_window_sizes.erase(promotion.table_it->first);
                    m_owner->m_temporary_block_table.erase(promotion.temporary_it);
                    continue;
                }
                for (const auto& block : selected_blocks) {
                    block->increment_live();
                }
                for (size_t layer_idx = 0; layer_idx < m_owner->m_num_layers; ++layer_idx) {
                    live_state.rows[layer_idx]->release_live();
                    std::swap(block_table[layer_idx][0], selected_blocks[layer_idx]);
                    live_state.rows[layer_idx] = block_table[layer_idx][0];
                }
                live_state.endpoint = promotion.endpoint;
                ++live_state.generation;
                live_state.is_empty = false;
                for (auto& allocation : temporary_blocks) {
                    m_owner->m_allocator.free_uncached(allocation);
                }
                m_owner->m_temporary_block_table.erase(promotion.temporary_it);
            }
            m_owner = nullptr;
            m_lock.unlock();
            return m_promoted_indices;
        }

    private:
        using TemporaryIterator = std::map<uint64_t, TemporaryBlockTable>::iterator;
        using TableIterator = std::map<uint64_t, std::vector<BlocksPerLayer>>::iterator;
        using LiveStateIterator = std::map<uint64_t, LinearAttentionLiveState>::iterator;

        struct PreparedPromotion {
            TemporaryIterator temporary_it;
            TableIterator table_it;
            LiveStateIterator live_state_it;
            size_t selected_index;
            size_t endpoint;
            std::vector<BlocksPerLayer> prefix_table;
            std::vector<BlockAllocator::PreparedBlockRelease> prefix_releases;
            size_t logical_start = 0;
            std::map<uint64_t, size_t> logical_start_node;
            std::map<uint64_t, TemporaryBlockTable> headroom_node;
            size_t window_size = 0;
            size_t reusable_position = std::numeric_limits<size_t>::max();
        };

        friend class BlockManager;
        PreparedTemporaryPromotions(BlockManager& owner,
                                    std::unique_lock<std::mutex> lock,
                                    std::vector<PreparedPromotion> promotions,
                                    std::vector<size_t> promoted_indices)
            : m_owner(&owner),
              m_lock(std::move(lock)),
              m_promotions(std::move(promotions)),
              m_promoted_indices(std::move(promoted_indices)) {}

        BlockManager* m_owner;
        std::unique_lock<std::mutex> m_lock;
        std::vector<PreparedPromotion> m_promotions;
        std::vector<size_t> m_promoted_indices;
    };

    /**
     * Constructs the BlockManager.
     * @param num_blocks Number of cache blocks available for assignment to the sequences.
     * @param enable_prefix_caching Whether prefix caching should be enabled for this allocator.
     * See also the equivalent parameter in ov::genai::ContinuousBatchingPipeline
     * @param block_size The size of an individual cache block in tokens.
     * @param num_layers The number of separate block-table layers associated with the manager.
     * In current implementation each layer must have the same number of logical blocks allocated at all times.
     * @param fixed_blocks_per_sequence When > 0, each sequence is allocated exactly this many blocks
     *        regardless of context length. Used for fixed-size caches (e.g. CausalConv1D state).
     *        When 0 (default), blocks grow with context length.
     * @param restore_latest_prefix_block_only When true, prefix-cache restore keeps only the latest matching
     *        block and tracks its logical block-table offset. Used for linear-attention state cache.
     * @param max_total_blocks Hard pool-size ceiling, or 0 for unbounded growth.
     */
    BlockManager(int num_blocks, bool enable_prefix_caching, size_t block_size, size_t num_layers = 1,
                 size_t fixed_blocks_per_sequence = 0, bool restore_latest_prefix_block_only = false,
                 size_t max_total_blocks = 0)
        : m_allocator(num_blocks, enable_prefix_caching, num_layers), m_enable_prefix_caching(enable_prefix_caching), m_block_size(block_size),
        m_num_layers(num_layers), m_fixed_blocks_per_sequence(fixed_blocks_per_sequence),
        m_max_total_blocks(max_total_blocks),
        m_restore_latest_prefix_block_only(restore_latest_prefix_block_only) {
        OPENVINO_ASSERT(num_layers != 0, "num_layers must be non-zero");
        OPENVINO_ASSERT(!restore_latest_prefix_block_only || enable_prefix_caching,
                        "Latest prefix block restore requires prefix caching to be enabled");
        OPENVINO_ASSERT(max_total_blocks == 0 || static_cast<size_t>(num_blocks) <= max_total_blocks,
                        "Block pool is constructed with ", num_blocks,
                        " blocks, which already exceeds its own ceiling of ", max_total_blocks, " blocks");
    }

    ~BlockManager() {
        // sanity check that all sequences are freed
        const size_t leaked_tables = m_block_table.size();
        const uint64_t first_leaked_seq_id = leaked_tables > 0 ? m_block_table.begin()->first : 0;
        if (!m_block_table.empty()) {
            try {
                GENAI_ERR("BlockManager leaked sequence block tables: %zu, first leaked sequence id: %llu",
                          leaked_tables,
                          static_cast<unsigned long long>(first_leaked_seq_id));
            } catch (...) {
                std::fprintf(stderr,
                             "BlockManager leaked sequence block tables: %zu, first leaked sequence id: %llu\n",
                             leaked_tables,
                             static_cast<unsigned long long>(first_leaked_seq_id));
            }
        }
    }

    /**
     * Gets the block table for a given sequence.
     * @param seq_id The identifier of an ov::genai::Sequence.
     * @return A vector of per-layer block tables occupied by this sequence.
     * Each per-layer table contains cache blocks for the corresponding block-table layer.
     */
    const std::vector<BlocksPerLayer>& get_block_tables(uint64_t seq_id) const {
        return m_block_table.at(seq_id);
    }

    const LinearAttentionLiveState& get_linear_attention_live_state(uint64_t seq_id) const {
        const auto live_state_it = m_linear_attention_live_states.find(seq_id);
        OPENVINO_ASSERT(live_state_it != m_linear_attention_live_states.end(),
                        "No linear-attention live-state handle for sequence ", seq_id);
        return live_state_it->second;
    }

    void set_linear_attention_live_state(uint64_t seq_id, size_t endpoint, const BlocksPerLayer& rows) {
        OPENVINO_ASSERT(rows.size() == m_num_layers,
                        "Linear-attention live-state handle requires one row per layer for sequence ", seq_id,
                        ": expected ", m_num_layers, ", got ", rows.size());
        auto& live_state = m_linear_attention_live_states[seq_id];
        replace_linear_attention_live_rows(live_state, rows);
        live_state.endpoint = endpoint;
        ++live_state.generation;
        live_state.is_empty = endpoint == 0;
    }

    void set_empty_linear_attention_live_state(uint64_t seq_id, const BlocksPerLayer& rows) {
        OPENVINO_ASSERT(rows.size() == m_num_layers,
                        "Empty linear-attention live-state handle requires one row per layer for sequence ", seq_id);
        auto& live_state = m_linear_attention_live_states[seq_id];
        replace_linear_attention_live_rows(live_state, rows);
        live_state.endpoint = 0;
        ++live_state.generation;
        live_state.is_empty = true;
    }

    void advance_linear_attention_live_state(uint64_t seq_id, size_t endpoint) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        auto live_state_it = m_linear_attention_live_states.find(seq_id);
        OPENVINO_ASSERT(live_state_it != m_linear_attention_live_states.end(),
                        "No linear-attention live-state handle for sequence ", seq_id);
        auto table_it = m_block_table.find(seq_id);
        OPENVINO_ASSERT(table_it != m_block_table.end() && table_it->second.size() == m_num_layers,
                        "No complete linear-attention block table for sequence ", seq_id);
        auto& live_state = live_state_it->second;
        OPENVINO_ASSERT(live_state.rows.size() == m_num_layers,
                        "Linear-attention live-state row count changed for sequence ", seq_id);

        size_t physical_position = 0;
        if (m_enable_prefix_caching && endpoint > 0) {
            const size_t logical_position = (endpoint - 1) / m_block_size;
            const size_t logical_start = get_block_table_logical_start_unlocked(seq_id);
            OPENVINO_ASSERT(logical_position >= logical_start,
                            "Linear-attention endpoint precedes the retained table for sequence ", seq_id);
            physical_position = logical_position - logical_start;
        }
        for (size_t layer_idx = 0; layer_idx < m_num_layers; ++layer_idx) {
            OPENVINO_ASSERT(physical_position < table_it->second[layer_idx].size(),
                            "Linear-attention endpoint has no represented row for sequence ", seq_id);
            table_it->second[layer_idx][physical_position]->increment_live();
        }
        for (size_t layer_idx = 0; layer_idx < m_num_layers; ++layer_idx) {
            live_state.rows[layer_idx]->release_live();
            live_state.rows[layer_idx] = table_it->second[layer_idx][physical_position];
        }
        live_state.endpoint = endpoint;
        ++live_state.generation;
        live_state.is_empty = endpoint == 0;
    }

    size_t get_block_table_logical_start(uint64_t seq_id) const {
        auto it = m_block_table_logical_start.find(seq_id);
        return it == m_block_table_logical_start.end() ? 0 : it->second;
    }

    bool is_prefix_caching_enabled() const {
        return m_enable_prefix_caching;
    }

    /// Maps physical = logical - m_block_table_logical_start for an append-only checkpoint table.
    /// Precondition: the table may only be truncated from the back while its logical start remains tracked.
    const CacheBlock::Ptr& get_block_at_logical_position(uint64_t seq_id,
                                                         size_t layer_idx,
                                                         size_t logical_block_position) const {
        const auto table_it = m_block_table.find(seq_id);
        OPENVINO_ASSERT(table_it != m_block_table.end(), "No block table for sequence ", seq_id);
        OPENVINO_ASSERT(layer_idx < table_it->second.size(),
                        "Block table layer ", layer_idx, " is out of range for sequence ", seq_id);
        const size_t logical_start = get_block_table_logical_start(seq_id);
        OPENVINO_ASSERT(logical_block_position >= logical_start,
                        "Logical block position ", logical_block_position,
                        " precedes block table start ", logical_start, " for sequence ", seq_id);
        const size_t physical_position = logical_block_position - logical_start;
        OPENVINO_ASSERT(physical_position < table_it->second[layer_idx].size(),
                        "Logical block position ", logical_block_position,
                        " is out of range for sequence ", seq_id);
        return table_it->second[layer_idx][physical_position];
    }

    /// Returns the latest row represented by an append-only table.
    const CacheBlock::Ptr& get_latest_represented_block(uint64_t seq_id, size_t layer_idx) const {
        const auto live_state_it = m_linear_attention_live_states.find(seq_id);
        if (live_state_it != m_linear_attention_live_states.end() && !live_state_it->second.rows.empty()) {
            OPENVINO_ASSERT(layer_idx < live_state_it->second.rows.size(),
                            "Linear-attention live-state layer ", layer_idx,
                            " is out of range for sequence ", seq_id);
            return live_state_it->second.rows[layer_idx];
        }
        const auto table_it = m_block_table.find(seq_id);
        OPENVINO_ASSERT(table_it != m_block_table.end(), "No block table for sequence ", seq_id);
        OPENVINO_ASSERT(layer_idx < table_it->second.size(),
                        "Block table layer ", layer_idx, " is out of range for sequence ", seq_id);
        const BlocksPerLayer& blocks = table_it->second[layer_idx];
        OPENVINO_ASSERT(!blocks.empty(), "Block table layer ", layer_idx, " is empty for sequence ", seq_id);
        const size_t latest_logical_position = get_block_table_logical_start(seq_id) + blocks.size() - 1;
        return get_block_at_logical_position(seq_id, layer_idx, latest_logical_position);
    }

    /// Returns whether the latest represented row is referenced by more than one block table.
    bool is_latest_represented_block_shared(uint64_t seq_id, size_t layer_idx) const {
        return get_latest_represented_block(seq_id, layer_idx)->copy_on_write();
    }

    size_t get_num_layers() const {
        return m_num_layers;
    }

    /**
     * Gets the block table for a given sequence and given layer.
     * @param seq_id The identifier of an ov::genai::Sequence.
     * @param layer_idx The index of a layer.
     * @return A vector of blocks (one for each layer) occupied by this sequence for this layer.
     */
    const std::vector<CacheBlock::Ptr>& get_block_table(uint64_t seq_id, size_t layer_idx) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        OPENVINO_ASSERT(m_block_table.count(seq_id) == 1);
        return m_block_table[seq_id][layer_idx];
    }

    /**
     * Gets the block size.
     * @return Block size.
     */
    const size_t get_block_size() const {
        return m_block_size;
    }

    /**
     * Frees a number of blocks with highest logical index from all sequences within a sequence group.
     * @param sequence_group The sequence group to free blocks from.
     * @param num_required_blocks The number of blocks to be freed. Will free an equal
     * number of blocks from each sequence in the group so that at least this number of blocks is freed in total.
     * @return Number of tokens freed.
     */
    size_t free_group_partially(SequenceGroup::Ptr sequence_group, size_t num_required_blocks) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        const auto not_finished_sequences = sequence_group->get_not_finished_sequences();
        const size_t num_not_finished_sequences = not_finished_sequences.size();
        // ceil(num_required_blocks / num_not_finished_sequences) with integer arithmetic
        const size_t blocks_num = (num_required_blocks + num_not_finished_sequences - 1) / num_not_finished_sequences;
        for (size_t idx = 0; idx < not_finished_sequences.size(); ++idx) {
            const auto seq_id = not_finished_sequences[idx]->get_id();
            OPENVINO_ASSERT(m_block_table.count(seq_id) > 0, "Invalid sequence group.");
            free_sequence_partially(seq_id, blocks_num);
        }
        return _blocks_released_to_tokens(sequence_group, blocks_num);
    }

    /**
     * Frees enough blocks from a sequence group to release at least the given number of tokens.
     * @param sequence_group The sequence group to free blocks from.
     * @param num_tokens The minimum number of tokens to free from the group.
     * @return Number of tokens actually freed.
     */
    size_t free_group_partially_by_tokens(SequenceGroup::Ptr sequence_group, size_t num_tokens) {
        size_t num_required_blocks = (num_tokens + m_block_size - 1) / m_block_size;
        return free_group_partially(sequence_group, num_required_blocks);
    }

    const size_t free_last_block_from_each_sequence(SequenceGroup::Ptr sequence_group) {
        size_t blocks_released = 0;
        auto not_finished_sequences = sequence_group->get_not_finished_sequences();
        for (size_t idx = 0; idx < not_finished_sequences.size(); ++idx) {
            const auto seq_id = not_finished_sequences[idx]->get_id();
            OPENVINO_ASSERT(m_block_table.count(seq_id) > 0, "Invalid sequence group.");
            if (free_last_block(seq_id)) {
                blocks_released++;
            }
        }
        return blocks_released;
    }

    bool free_last_block(size_t seq_id) {
        auto& block_table = m_block_table[seq_id];
        OPENVINO_ASSERT(block_table[0].size() >= 1);
        BlocksPerLayer blocks_to_free;
        blocks_to_free.reserve(m_num_layers);
        for (size_t layer_idx = 0; layer_idx < m_num_layers; layer_idx++) {
            blocks_to_free.push_back(block_table[layer_idx].back());
        }
        free_cached_blocks(blocks_to_free);
        for (size_t layer_idx = 0; layer_idx < m_num_layers; layer_idx++) {
            block_table[layer_idx].resize(block_table[layer_idx].size() - 1);
        }

        if (block_table[0].size() == 0) {
            m_block_table_logical_start.erase(seq_id);
            OPENVINO_ASSERT(m_block_table.erase(seq_id) == 1);
         }
        return blocks_to_free[0]->is_free();
    }

    size_t free_partially_beam_search_group(SequenceGroup::Ptr sequence_group, size_t num_required_blocks) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        size_t physical_blocks_released = 0;
        size_t logical_blocks_released = 0;
        while (num_required_blocks > physical_blocks_released) {
            size_t released_count = free_last_block_from_each_sequence(sequence_group);
            logical_blocks_released ++;
            if ((int)sequence_group->get_context_len() - logical_blocks_released * m_block_size <= 0) {
                break;
            }
            physical_blocks_released += released_count;
        }
        return _blocks_released_to_tokens(sequence_group, logical_blocks_released);
    }

    /**
     * Frees enough blocks from a beam search sequence group to release at least the given number of tokens.
     * @param sequence_group The sequence group to free blocks from.
     * @param num_tokens The minimum number of tokens to free from the group.
     * @return Number of tokens actually freed.
     */
    size_t free_partially_beam_search_group_by_tokens(SequenceGroup::Ptr sequence_group, size_t num_tokens) {
        size_t num_required_blocks = (num_tokens + m_block_size - 1) / m_block_size;
        return free_partially_beam_search_group(sequence_group, num_required_blocks);
    }

    /**
     * Returns the total number of distinct physical blocks occupied by a given sequence group.
     * @param sequence_group The sequence group.
     * @return The number of distinct physical blocks occupied by this sequence group.
     */
    const size_t get_number_of_blocks_occupied_by_sequence(SequenceGroup::Ptr sequence_group) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        auto running_sequences = sequence_group->get_not_finished_sequences();
        std::set<size_t> indices;
        for (size_t idx = 0; idx < running_sequences.size(); ++idx) {
            auto seq_id = running_sequences[idx]->get_id();
            if (m_block_table.count(seq_id) == 0) {
                continue;
            }
            auto block_table = m_block_table[seq_id][0];  // assuming all layers always have equal sets of blocks
            for (const auto& block : block_table) {
                indices.insert(block->get_index());
            }
        }
        return indices.size();
    }

    /**
     * @param seq_group Pointer to a sequence group.
     * @return The number of logical blocks required by the sequence group.
     */
    size_t get_num_logical_blocks(SequenceGroup::CPtr seq_group) const {
        if (m_fixed_blocks_per_sequence > 0) {
            return m_fixed_blocks_per_sequence;
        }
        return (seq_group->get_context_len() - seq_group->get_num_evicted_tokens() + m_block_size - 1) / m_block_size;
    }

    /**
     * Checks whether partially preempting victim can free enough blocks to satisfy
     * the target sequence group's deficit in this block manager's pool.
     * @param victim The sequence group to potentially free blocks from.
     * @param target The sequence group that needs blocks allocated.
     * @return Whether the victim has enough occupied blocks to cover target's deficit.
     */
    bool can_partially_preempt(SequenceGroup::Ptr victim, SequenceGroup::CPtr target) {
        size_t needed = required_blocks_count(target);
        if (needed == 0) {
            return true;
        }
        size_t victim_occupied = get_number_of_blocks_occupied_by_sequence(victim);
        return victim_occupied > needed;
    }

    /**
     * Checks whether a victim sequence group can be partially preempted by this block manager.
     * @param victim The sequence group to potentially free blocks from.
     * @return Whether partial preemption is allowed for the victim.
     */
    bool can_partially_preempt_victim(SequenceGroup::Ptr victim) {
        if (!m_fixed_blocks_per_sequence) {
            return true;
        }

        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        for (const auto& sequence : victim->get_not_finished_sequences()) {
            if (m_block_table.find(sequence->get_id()) != m_block_table.end()) {
                return false;
            }
        }
        return true;
    }

    /**
     * @return Whether any blocks have been allocated (capacity > 0).
     */
    bool has_token_capacity() const {
        return get_total_block_count() > 0;
    }

    /**
     * @return Whether this manager uses a fixed block count per sequence (e.g. linear attention state).
     *         When true, token-count-based capacity growth is not meaningful and should be skipped.
     */
    bool is_fixed_size_per_sequence() const {
        return m_fixed_blocks_per_sequence > 0;
    }

    /**
     * @return Number of blocks allocated per sequence for fixed-size managers; 0 for variable-size.
     */
    size_t get_fixed_blocks_per_sequence() const {
        return m_fixed_blocks_per_sequence;
    }

    /**
     * Grows the block pool to accommodate at least the given number of additional tokens.
     * @param num_tokens Number of additional tokens to accommodate.
     * @return Whether the pool actually grew.
     */
    bool grow_capacity_by_tokens(size_t num_tokens) {
        size_t additional_blocks = (num_tokens + m_block_size - 1) / m_block_size;
        return increase_block_count_up_to(get_total_block_count() + additional_blocks);
    }

    /**
     * Ensures the block pool has capacity for sequence-aware token targets.
     * @param sequence_token_targets Pairs of tokens per sequence and number of sequences with that target.
     */
    void ensure_sequence_token_capacity(const std::vector<std::pair<size_t, size_t>>& sequence_token_targets) {
        size_t required_blocks = 0;
        for (const auto& [tokens_per_sequence, num_sequences] : sequence_token_targets) {
            if (tokens_per_sequence == 0 || num_sequences == 0) {
                continue;
            }
            const size_t blocks_per_sequence = (tokens_per_sequence + m_block_size - 1) / m_block_size;
            OPENVINO_ASSERT(blocks_per_sequence <= std::numeric_limits<size_t>::max() / num_sequences,
                            "Requested sequence token capacity exceeds size_t range");
            OPENVINO_ASSERT(required_blocks <= std::numeric_limits<size_t>::max() - blocks_per_sequence * num_sequences,
                            "Requested sequence token capacity exceeds size_t range");
            required_blocks += blocks_per_sequence * num_sequences;
        }
        if (required_blocks > get_total_block_count()) {
            increase_block_count_up_to(required_blocks);
        }
    }

    /**
     * @param seq_id The identifier of an ov::genai::Sequence
     * @return Whether or not this BlockManager is managing this sequence group.
     */
    const bool has_block_table(uint64_t seq_id) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        return m_block_table.count(seq_id) > 0;
    }

    /**
     * @return The number of cache blocks available to be assigned to new sequences.
     */
    size_t num_free_blocks() const {
        return m_allocator.num_free_blocks(0); // relying on the invariant that all layers have identical number of blocks
    }

    /// @return Number of blocks currently taken out of the pool.
    size_t get_num_blocks_in_use() const {
        return get_total_block_count() - num_free_blocks();
    }

    size_t get_num_linear_attention_headroom_blocks() const {
        size_t num_blocks = 0;
        for (const auto& [seq_id, headroom] : m_linear_attention_headroom) {
            num_blocks += headroom.size();
        }
        return num_blocks;
    }

    /// @return Number of sequences currently holding temporary blocks.
    size_t get_num_sequences_with_temporary_blocks() {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        return m_temporary_block_table.size();
    }

    /// @return Whether the sequence currently holds temporary (borrowed) blocks.
    bool has_temporary_blocks(uint64_t seq_id) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        return m_temporary_block_table.count(seq_id) > 0;
    }

    /// @return Whether reserve_temporary_blocks(seq_id, num_blocks) would succeed.
    bool can_reserve_temporary_blocks(uint64_t seq_id, size_t num_blocks) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        if (m_block_table.count(seq_id) == 0) {
            return false;
        }
        // Avoid operator[] so this predicate does not create an empty reservation.
        const auto temporary_it = m_temporary_block_table.find(seq_id);
        if (temporary_it != m_temporary_block_table.end() && !temporary_it->second.empty()) {
            return false;
        }
        return num_blocks > 0 && m_allocator.can_allocate_blocks(num_blocks);
    }

    std::vector<int> reserve_temporary_blocks(uint64_t seq_id, size_t num_blocks) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        OPENVINO_ASSERT(m_block_table.count(seq_id) > 0,
                        "Cannot reserve temporary cache blocks for unknown sequence ", seq_id);
        const auto existing_temporary_it = m_temporary_block_table.find(seq_id);
        OPENVINO_ASSERT(existing_temporary_it == m_temporary_block_table.end() ||
                            existing_temporary_it->second.empty(),
                        "Temporary cache blocks are already reserved for sequence ", seq_id);
        OPENVINO_ASSERT(num_blocks > 0 && m_allocator.can_allocate_blocks(num_blocks),
                        "Not enough cache blocks to reserve ", num_blocks,
                        " temporary checkpoints for sequence ", seq_id);

        std::vector<int> block_indices;
        const auto [temporary_it, inserted] = m_temporary_block_table.try_emplace(seq_id);
        try {
            temporary_it->second =
                acquire_temporary_blocks(num_blocks, block_indices);
        } catch (...) {
            if (inserted) {
                m_temporary_block_table.erase(temporary_it);
            }
            throw;
        }
        return block_indices;
    }

    bool can_prepare_temporary_blocks(uint64_t seq_id, size_t num_blocks) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        const auto live = m_linear_attention_live_states.find(seq_id);
        if (num_blocks == 0 || live == m_linear_attention_live_states.end() ||
            m_temporary_block_table.count(seq_id) != 0) {
            return false;
        }
        const auto retained = m_linear_attention_headroom.find(seq_id);
        const size_t headroom = retained == m_linear_attention_headroom.end() ? 0 : retained->second.size();
        const bool reusable = !live->second.rows.front()->has_published_hash() &&
                              live->second.rows.front()->get_references_count() == 2;
        return headroom + m_allocator.num_free_blocks(0) >=
               num_blocks + (m_restore_latest_prefix_block_only && !reusable ? 1 : 0);
    }

    PreparedTemporaryBlocks prepare_temporary_blocks(uint64_t seq_id, size_t num_blocks) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        OPENVINO_ASSERT(num_blocks > 0, "Cannot prepare zero temporary cache blocks");
        const auto table_it = m_block_table.find(seq_id);
        OPENVINO_ASSERT(table_it != m_block_table.end(),
                        "Cannot prepare temporary cache blocks for unknown sequence ", seq_id);
        const auto& block_table = table_it->second;
        OPENVINO_ASSERT((m_fixed_blocks_per_sequence == 1 || m_restore_latest_prefix_block_only) && block_table.size() == m_num_layers,
                        "Linear-attention scratch preparation requires a fixed one-row table for sequence ", seq_id);
        const auto live_state_it = m_linear_attention_live_states.find(seq_id);
        OPENVINO_ASSERT(live_state_it != m_linear_attention_live_states.end(),
                        "No linear-attention live-state handle for sequence ", seq_id);
        const auto& live_state = live_state_it->second;
        OPENVINO_ASSERT(live_state.rows.size() == m_num_layers,
                        "Linear-attention live-state row count changed for sequence ", seq_id);
        for (size_t layer_idx = 0; layer_idx < m_num_layers; ++layer_idx) {
            OPENVINO_ASSERT(m_restore_latest_prefix_block_only || block_table[layer_idx].size() == 1,
                            "Linear-attention scratch preparation requires one table row per layer for sequence ",
                            seq_id);
            OPENVINO_ASSERT(live_state.rows[layer_idx] == block_table[layer_idx].back(),
                            "Linear-attention live-state row does not match the committed table for sequence ", seq_id,
                            ", layer ", layer_idx);
        }
        OPENVINO_ASSERT(num_blocks <= std::numeric_limits<size_t>::max() - live_state.endpoint,
                        "Linear-attention scratch endpoint exceeds size_t range for sequence ", seq_id);
        const auto existing_temporary_it = m_temporary_block_table.find(seq_id);
        OPENVINO_ASSERT(existing_temporary_it == m_temporary_block_table.end() ||
                            existing_temporary_it->second.empty(),
                        "Temporary cache blocks are already reserved for sequence ", seq_id);
        const auto headroom_it = m_linear_attention_headroom.find(seq_id);
        const size_t retained = headroom_it == m_linear_attention_headroom.end() ? 0 : headroom_it->second.size();
        const bool reusable_live = m_restore_latest_prefix_block_only &&
            !live_state.rows.front()->has_published_hash() && live_state.rows.front()->get_references_count() == 2;
        const size_t minimum_rows = num_blocks + (m_restore_latest_prefix_block_only && !reusable_live ? 1 : 0);
        const size_t optional_rows = m_restore_latest_prefix_block_only
            ? std::min(num_blocks, (live_state.endpoint % m_block_size + num_blocks + m_block_size - 1) / m_block_size)
            : 0;
        const size_t desired_rows = m_restore_latest_prefix_block_only
            ? num_blocks + optional_rows - (reusable_live ? 1 : 0) : num_blocks;
        const size_t capacity = retained + m_allocator.num_free_blocks(0);
        OPENVINO_ASSERT(capacity >= minimum_rows,
                        "Not enough cache blocks to reserve ", num_blocks,
                        " temporary checkpoints for sequence ", seq_id);

        std::vector<int> block_indices;
        block_indices.reserve(num_blocks);
        const size_t total_rows = capacity >= desired_rows ? std::max(desired_rows, retained) : std::max(minimum_rows, retained);
        TemporaryBlockTable allocations;
        allocations.reserve(total_rows);
        std::map<uint64_t, size_t> window_node{{seq_id, num_blocks}};
        const auto [temporary_it, inserted] = m_temporary_block_table.try_emplace(seq_id);
        try {
            std::vector<int> acquired_indices;
            auto acquired = total_rows > retained ? acquire_temporary_blocks(total_rows - retained, acquired_indices)
                                                  : TemporaryBlockTable{};
            if (headroom_it != m_linear_attention_headroom.end()) {
                for (auto& allocation : headroom_it->second) {
                    allocations.push_back(std::move(allocation));
                }
                m_linear_attention_headroom.erase(headroom_it);
            }
            for (auto& allocation : acquired) {
                allocations.push_back(std::move(allocation));
            }
            for (size_t position = 0; position < num_blocks; ++position) {
                block_indices.push_back(allocations[position].blocks.front()->get_index());
            }
            temporary_it->second = std::move(allocations);
            m_temporary_window_sizes.merge(window_node);
        } catch (...) {
            if (inserted) {
                m_temporary_block_table.erase(temporary_it);
            }
            throw;
        }
        return {live_state.endpoint, live_state.generation, std::move(block_indices)};
    }

    void release_temporary_blocks(uint64_t seq_id) noexcept {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        auto it = m_temporary_block_table.find(seq_id);
        if (it == m_temporary_block_table.end()) {
            return;
        }
        for (auto& allocation : it->second) {
            m_allocator.free_uncached(allocation);
        }
        m_temporary_block_table.erase(it);
        m_temporary_window_sizes.erase(seq_id);
    }

    /// Promotes a one-based temporary slot and frees the old committed row and losing temporaries.
    /// @return Physical index of the promoted block.
    size_t promote_temporary_block(uint64_t seq_id,
                                   size_t checkpoint_slot,
                                   size_t expected_endpoint,
                                   size_t expected_generation) {
        std::vector<TemporaryPromotionRequest> requests{
            {seq_id, checkpoint_slot, expected_endpoint, expected_generation}};
        auto prepared = prepare_temporary_promotions(requests);
        return prepared.apply().front();
    }

    PreparedTemporaryPromotions prepare_temporary_promotions(
        const std::vector<TemporaryPromotionRequest>& requests) {
        std::unique_lock<std::mutex> lock(m_cached_blocks_map_mutex);
        std::vector<PreparedTemporaryPromotions::PreparedPromotion> promotions;
        std::vector<size_t> promoted_indices;
        promotions.reserve(requests.size());
        promoted_indices.reserve(requests.size());
        std::set<uint64_t> sequence_ids;
        for (const TemporaryPromotionRequest& request : requests) {
            OPENVINO_ASSERT(sequence_ids.insert(request.seq_id).second,
                            "Duplicate temporary promotion for sequence ", request.seq_id);
            OPENVINO_ASSERT(request.checkpoint_slot > 0,
                            "Checkpoint slot is one-based and must be greater than zero");
            auto temporary_it = m_temporary_block_table.find(request.seq_id);
            OPENVINO_ASSERT(temporary_it != m_temporary_block_table.end(),
                            "No temporary cache blocks reserved for sequence ", request.seq_id);
            auto& temporary_blocks = temporary_it->second;
            OPENVINO_ASSERT(request.checkpoint_slot <= temporary_blocks.size(),
                            "Checkpoint slot ", request.checkpoint_slot, " is out of range for sequence ", request.seq_id,
                            ", reserved checkpoints: ", temporary_blocks.size());
            auto table_it = m_block_table.find(request.seq_id);
            OPENVINO_ASSERT(table_it != m_block_table.end(),
                            "Cannot promote temporary cache block for unknown sequence ", request.seq_id);
            auto& block_table = table_it->second;
            OPENVINO_ASSERT(block_table.size() == m_num_layers,
                            "Temporary cache promotion expects one block table per layer");
            for (size_t layer_idx = 0; layer_idx < m_num_layers; ++layer_idx) {
                OPENVINO_ASSERT(m_restore_latest_prefix_block_only || block_table[layer_idx].size() == 1,
                                "Temporary cache promotion supports fixed one-block sequence state only");
            }
            auto live_state_it = m_linear_attention_live_states.find(request.seq_id);
            OPENVINO_ASSERT(live_state_it != m_linear_attention_live_states.end(),
                            "No linear-attention live-state handle for sequence ", request.seq_id);
            auto& live_state = live_state_it->second;
            OPENVINO_ASSERT(live_state.rows.size() == m_num_layers,
                            "Linear-attention live-state row count changed for sequence ", request.seq_id);
            OPENVINO_ASSERT(live_state.endpoint == request.expected_endpoint &&
                                live_state.generation == request.expected_generation,
                            "Stale linear-attention scratch lease for sequence ", request.seq_id);
            OPENVINO_ASSERT(request.checkpoint_slot <=
                                std::numeric_limits<size_t>::max() - request.expected_endpoint,
                            "Linear-attention checkpoint endpoint exceeds size_t range for sequence ", request.seq_id);
            for (const auto& allocation : temporary_blocks) {
                OPENVINO_ASSERT(allocation.blocks.size() == m_num_layers,
                                "Temporary cache row count changed for sequence ", request.seq_id);
                OPENVINO_ASSERT(allocation.free_nodes.size() == m_num_layers,
                                "Temporary cache release metadata changed for sequence ", request.seq_id);
                for (const auto& free_node : allocation.free_nodes) {
                    OPENVINO_ASSERT(free_node.size() == 1,
                                    "Temporary cache release node changed for sequence ", request.seq_id);
                }
            }
            const size_t selected_index = request.checkpoint_slot - 1;
            const int promoted_index = temporary_blocks[selected_index].blocks.front()->get_index();
            OPENVINO_ASSERT(promoted_index >= 0,
                            "Promoted temporary cache block for sequence ", request.seq_id,
                            " has a negative physical index: ", promoted_index);
            promotions.push_back({temporary_it,
                                  table_it,
                                  live_state_it,
                                  selected_index,
                                  request.expected_endpoint + request.checkpoint_slot});
            if (m_restore_latest_prefix_block_only) {
                auto& promotion = promotions.back();
                const auto window_it = m_temporary_window_sizes.find(request.seq_id);
                OPENVINO_ASSERT(window_it != m_temporary_window_sizes.end(), "Prefix promotion requires prepared scratch");
                promotion.window_size = window_it->second;
                OPENVINO_ASSERT(request.checkpoint_slot <= promotion.window_size, "Accepted depth exceeds verification window");
                auto& headroom = promotion.headroom_node[request.seq_id];
                headroom.reserve(promotion.window_size);
                if (!live_state.rows.front()->has_published_hash() && live_state.rows.front()->get_references_count() == 2) {
                    promotion.reusable_position = block_table.front().size() - 1;
                }
                promotion.prefix_table.resize(m_num_layers);
                promotion.logical_start = request.expected_endpoint / m_block_size;
                const size_t last_logical = (promotion.endpoint - 1) / m_block_size;
                const size_t retained_count = last_logical - promotion.logical_start + 1;
                if (temporary_blocks.size() + (promotion.reusable_position != std::numeric_limits<size_t>::max() ? 1 : 0) <
                    promotion.window_size + retained_count) {
                    promotion.logical_start = last_logical;
                }
                for (size_t logical = promotion.logical_start; logical <= last_logical; ++logical) {
                    const size_t endpoint = std::min((logical + 1) * m_block_size, promotion.endpoint);
                    const auto& rows = temporary_blocks[endpoint - request.expected_endpoint - 1].blocks;
                    for (size_t layer = 0; layer < m_num_layers; ++layer) {
                        promotion.prefix_table[layer].push_back(rows[layer]);
                    }
                }
                promotion.prefix_releases.reserve(block_table.front().size());
                for (size_t position = 0; position < block_table.front().size(); ++position) {
                    BlocksPerLayer rows;
                    rows.reserve(m_num_layers);
                    for (size_t layer = 0; layer < m_num_layers; ++layer) {
                        rows.push_back(block_table[layer][position]);
                    }
                    promotion.prefix_releases.push_back(m_allocator.prepare_block_release(rows, m_prefix_hash_to_cached_blocks));
                }
                promotion.logical_start_node.emplace(request.seq_id, promotion.logical_start);
            }
            promoted_indices.push_back(static_cast<size_t>(promoted_index));
        }
        return PreparedTemporaryPromotions{
            *this, std::move(lock), std::move(promotions), std::move(promoted_indices)};
    }

    /**
     * Returns the number of additional tokens that can be accommodated for a given sequence group,
     * accounting for unused slots in already-allocated blocks and free blocks.
     * @param seq_group The sequence group to check available capacity for.
     * @return The number of additional tokens that can be stored.
     */
    size_t available_token_slots(SequenceGroup::CPtr seq_group) const {
        if (m_fixed_blocks_per_sequence > 0) {
            // Fixed block count: once allocated, unlimited tokens can be served.
            // If not yet allocated, capacity depends on available blocks.
            size_t required_additional_blocks = 0;
            for (auto seq : seq_group->get_running_sequences()) {
                auto it = m_block_table.find(seq->get_id());
                size_t current_blocks = 0;
                if (it != m_block_table.end()) {
                    current_blocks = it->second[0].size();
                }
                if (current_blocks < m_fixed_blocks_per_sequence) {
                    required_additional_blocks += (m_fixed_blocks_per_sequence - current_blocks);
                }
            }
            if (required_additional_blocks == 0) {
                return std::numeric_limits<size_t>::max();
            }
            return can_allocate_blocks(required_additional_blocks) ? std::numeric_limits<size_t>::max() : 0;
        }
        const size_t copy_on_write_blocks = get_required_copy_on_write_blocks(seq_group);
        if (!m_allocator.can_allocate_uncached_blocks(copy_on_write_blocks)) {
            return 0;
        }
        return get_num_unused_tokens(seq_group) +
               (num_free_blocks() - copy_on_write_blocks) * m_block_size;
    }

    /**
     * Checks whether enough cache blocks can be allocated to accommodate the given number of tokens,
     * accounting for unused slots in already-allocated blocks.
     * @param seq_group The sequence group to check allocation feasibility for.
     * @param num_tokens The number of additional tokens to accommodate.
     * @return Whether the allocation is feasible.
     */
    bool can_allocate_tokens(SequenceGroup::CPtr seq_group, size_t num_tokens) const {
        if (m_fixed_blocks_per_sequence > 0) {
            // Fixed block count: check if we can allocate the fixed number of blocks
            // for each running sequence that does not yet have them.
            size_t required_additional_blocks = 0;
            for (auto seq : seq_group->get_running_sequences()) {
                auto it = m_block_table.find(seq->get_id());
                size_t current_blocks = 0;
                if (it != m_block_table.end()) {
                    current_blocks = it->second[0].size();
                }
                if (current_blocks < m_fixed_blocks_per_sequence) {
                    required_additional_blocks += (m_fixed_blocks_per_sequence - current_blocks);
                }
            }
            return can_allocate_blocks(required_additional_blocks);
        }
        const size_t tokens_needing_new_blocks = num_tokens > get_num_unused_tokens(seq_group) ? num_tokens - get_num_unused_tokens(seq_group) : 0;
        size_t blocks_needed = (tokens_needing_new_blocks + m_block_size - 1) / m_block_size;
        const size_t copy_on_write_blocks = get_required_copy_on_write_blocks(seq_group);
        return m_allocator.can_allocate_uncached_blocks(copy_on_write_blocks) &&
               can_allocate_blocks(blocks_needed + copy_on_write_blocks);
    }

    /**
     * Allocates cache blocks to accommodate the given number of additional tokens for a sequence,
     * accounting for unused slots in already-allocated blocks.
     * @param sequence The sequence to allocate blocks for.
     * @param seq_group The sequence group to which this sequence belongs.
     * @param num_tokens The number of additional tokens to accommodate.
     * @param prompt_size Prompt size for this sequence.
     */
    std::map<size_t, std::list<size_t>> allocate_tokens(ov::genai::Sequence::Ptr sequence,
                                                        SequenceGroup::CPtr seq_group,
                                                        size_t num_tokens,
                                                        size_t prompt_size = 0) {
        std::map<size_t, std::list<size_t>> copy_blocks_map;
        if (m_fixed_blocks_per_sequence > 0) {
            auto seq_id = sequence->get_id();
            size_t current_blocks = 0;
            if (m_block_table.find(seq_id) != m_block_table.end()) {
                current_blocks = m_block_table[seq_id][0].size();
            }
            size_t blocks_needed = m_fixed_blocks_per_sequence > current_blocks
                                       ? m_fixed_blocks_per_sequence - current_blocks
                                       : 0;
            if (blocks_needed > 0) {
                allocate(sequence, blocks_needed, prompt_size);
            }
            return copy_blocks_map;
        }
        const size_t unused_tokens = get_num_unused_tokens(seq_group);
        const size_t tokens_needing_new_blocks = num_tokens > unused_tokens ? num_tokens - unused_tokens : 0;
        const size_t num_blocks = (tokens_needing_new_blocks + m_block_size - 1) / m_block_size;
        if (m_enable_prefix_caching) {
            copy_last_block_if_shared(sequence,
                                      seq_group->get_num_processed_tokens(),
                                      copy_blocks_map);
        }
        if (num_blocks > 0) {
            allocate(sequence, num_blocks, prompt_size);
        }
        return copy_blocks_map;
    }

    /**
     * @return Percentage of cache blocks used by all sequences.
     */
    float get_used_percentage() const {
        return m_allocator.get_used_percentage();
    }

    /**
     * @return The hard ceiling on this pool's block count, or 0 when the pool may grow without bound.
     */
    size_t get_max_total_block_count() const {
        return m_max_total_blocks;
    }

    /// @return Whether the configured ceiling permits @p num_blocks.
    bool can_increase_block_count_to(size_t num_blocks) const {
        return m_max_total_blocks == 0 || num_blocks <= m_max_total_blocks;
    }

    /// @brief Grows towards @p num_blocks, clamped to the configured ceiling.
    /// @return Whether the pool actually grew.
    bool increase_block_count_up_to(size_t num_blocks) {
        const size_t target = m_max_total_blocks > 0 ? std::min(num_blocks, m_max_total_blocks) : num_blocks;
        if (target <= get_total_block_count()) {
            return false;
        }
        increase_block_count(target);
        return true;
    }

    /**
     * Increases the number of blocks.
     * @param num_blocks The new number of blocks.
     */
    void increase_block_count(size_t num_blocks) {
        OPENVINO_ASSERT(can_increase_block_count_to(num_blocks),
                        "Cache block pool cannot grow to ", num_blocks,
                        " blocks: the configured budget allows at most ", m_max_total_blocks,
                        " blocks (currently ", get_total_block_count(),
                        "). Raise the configured block count or cache_size, or lower the concurrency "
                        "that requires the growth.");
        m_allocator.increase_block_count(num_blocks);
    }

    /**
     * @return The total number of blocks.
     */
    size_t get_total_block_count() const {
        return m_allocator.get_total_block_count();
    }

    /**
     * @brief Forks a sequence, establishing a new sequence from an existing one, reusing
     * currently allocated blocks of the existing sequence.
     * @param parent_id Parent sequence identifier
     * @param child_id Sequence identifier for the new, forked sequence. Must be unique across
     * other sequences tracked by this BlockManager.
     */
    void fork_sequence(uint64_t parent_id, uint64_t child_id) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        OPENVINO_ASSERT(m_temporary_block_table.count(parent_id) == 0,
                        "Cannot fork linear-attention sequence ", parent_id, " while a scratch lease is active");
        OPENVINO_ASSERT(m_block_table.count(child_id) == 0);
        m_block_table[child_id].resize(m_num_layers);
        m_block_table_logical_start[child_id] = get_block_table_logical_start(parent_id);
        for (size_t layer_idx = 0; layer_idx < m_num_layers; layer_idx++) {
            m_block_table[child_id][layer_idx].reserve(m_block_table[parent_id][layer_idx].size());
            for (CacheBlock::Ptr &block: m_block_table[parent_id][layer_idx]) {
                block->increment();
                m_block_table[child_id][layer_idx].push_back(block);
            }
        }
        const auto live_state_it = m_linear_attention_live_states.find(parent_id);
        if (live_state_it != m_linear_attention_live_states.end()) {
            for (const auto& row : live_state_it->second.rows) {
                row->increment_live();
            }
            m_linear_attention_live_states.emplace(child_id, live_state_it->second);
        }
    }

    /**
     * @brief Frees all blocks for a given sequence.
     * @param seq_id Identifier of the sequence to free.
     */
    void free_sequence(size_t seq_id) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        OPENVINO_ASSERT(m_temporary_block_table.count(seq_id) == 0,
                        "Cannot free linear-attention sequence ", seq_id, " while a scratch lease is active");
        OPENVINO_ASSERT(m_block_table.find(seq_id) != m_block_table.end(), "sequence with id ", seq_id,
                        " not found in BlockManager, but requested to free");
        const auto headroom = m_linear_attention_headroom.find(seq_id);
        if (headroom != m_linear_attention_headroom.end()) {
            for (auto& allocation : headroom->second) {
                m_allocator.free_uncached(allocation);
            }
            m_linear_attention_headroom.erase(headroom);
        }
        release_linear_attention_live_rows(seq_id);
        auto& block_table = m_block_table[seq_id];
        size_t effective_num_layers = block_table.size();
        size_t num_allocated_blocks = block_table[0].size();
        for (size_t i = 0; i < num_allocated_blocks; i++) {
            BlocksPerLayer blocks_to_free;
            blocks_to_free.reserve(effective_num_layers);
            for (size_t layer_idx = 0; layer_idx < effective_num_layers; layer_idx++) {
               blocks_to_free.push_back(block_table[layer_idx][i]);
            }
            free_cached_blocks(blocks_to_free);
        }

        OPENVINO_ASSERT(m_block_table.erase(seq_id) == 1);
        m_block_table_logical_start.erase(seq_id);
    }

    /**
     * Frees a specified number of blocks from the end of a given sequence.
     * If a sequence is freed completely, it is removed from this BlockManager.
     * @param seq_id Sequence identifier
     * @param block_num Number of blocks to be freed from the sequence, starting at
     * the highest logical block.
     */
    void free_sequence_partially(size_t seq_id, size_t block_num) {
        size_t effective_num_layers = m_block_table[seq_id].size();
        for (size_t layer_idx = 0; layer_idx < effective_num_layers; layer_idx++) {
            auto& layer_block_table = m_block_table[seq_id][layer_idx];
            OPENVINO_ASSERT(layer_block_table.size() >= block_num);
        }

        for (size_t idx = 0; idx < block_num; idx++) {
            BlocksPerLayer blocks_to_free;
            blocks_to_free.reserve(effective_num_layers);
            for (size_t layer_idx = 0; layer_idx < effective_num_layers; layer_idx++) {
                auto &layer_block_table = m_block_table[seq_id][layer_idx];
                size_t block_idx = layer_block_table.size() - idx - 1;
                blocks_to_free.push_back(layer_block_table[block_idx]);
            }
            free_cached_blocks(blocks_to_free);
        }

        for (size_t layer_idx = 0; layer_idx < effective_num_layers; layer_idx++) {
            auto& layer_block_table = m_block_table[seq_id][layer_idx];
            layer_block_table.resize(layer_block_table.size() - block_num);
        }

        auto empty_predicate = [](const BlocksPerLayer& v) { return v.empty(); };
        bool any_freed_completely = std::any_of(m_block_table[seq_id].begin(), m_block_table[seq_id].end(), empty_predicate);
        if (any_freed_completely) {
            bool all_freed_completely = std::all_of(m_block_table[seq_id].begin(), m_block_table[seq_id].end(), empty_predicate);
            // The invariant must hold at BlockManager level that all per-layer block tables
            // must have the same size
            OPENVINO_ASSERT(all_freed_completely, "block tables across layers should only be empty all at once");
            OPENVINO_ASSERT(m_block_table.erase(seq_id) == 1);
            m_block_table_logical_start.erase(seq_id);
        }
    }

    /**
     * Frees specific blocks layer-wise from a given sequence.
     * @param seq_id Sequence identifier for the blocks to be freed from.
     * @param logical_block_index_sets_to_free Sets (one for each layer) of logical block indices to be freed from this sequence.
     */
    void free_blocks_from_sequence(size_t seq_id, const std::vector<std::set<size_t>>& logical_block_index_sets_to_free) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        std::vector<std::vector<size_t>> logical_block_indices_to_free(logical_block_index_sets_to_free.size());
        for (size_t i = 0; i < logical_block_index_sets_to_free.size(); i++) {
            const auto& index_set = logical_block_index_sets_to_free[i];
            auto& index_vector = logical_block_indices_to_free[i];
            index_vector.resize(index_set.size());
            std::copy(index_set.begin(), index_set.end(), index_vector.begin());
        }

        size_t presumed_num_layers = logical_block_indices_to_free.size();
        OPENVINO_ASSERT(m_num_layers == presumed_num_layers);
        for (size_t i = 0; i < presumed_num_layers; i++) {
            OPENVINO_ASSERT(logical_block_indices_to_free[i].size() == logical_block_indices_to_free[0].size(), "must free the same amount of blocks per each layer at once");
        }

        if (logical_block_indices_to_free[0].empty()) {
            return;
        }

        size_t num_blocks_to_free = logical_block_indices_to_free[0].size();

        // free blocks at the allocator level
        for (size_t block_idx = 0; block_idx < num_blocks_to_free; block_idx++) {
            BlocksPerLayer per_layer_cache_blocks_to_free;
            per_layer_cache_blocks_to_free.reserve(presumed_num_layers);
            for (size_t layer_idx = 0; layer_idx < presumed_num_layers; layer_idx++) {
                auto& per_layer_block_table = m_block_table[seq_id][layer_idx];
                size_t block_table_size = per_layer_block_table.size();
                size_t logical_block_idx = *(logical_block_indices_to_free[layer_idx].begin() + block_idx);
                OPENVINO_ASSERT(logical_block_idx < block_table_size,
                                "cannot free logical block ", logical_block_idx,
                                " from sequence ", seq_id, " since it only has ", block_table_size, " logical blocks");
                auto block = per_layer_block_table[logical_block_idx];
                per_layer_cache_blocks_to_free.push_back(block);
            }
            free_cached_blocks(per_layer_cache_blocks_to_free);
        }

        // remove freed entries from the block table at this BlockManager's level
        for (size_t layer_idx = 0; layer_idx < presumed_num_layers; layer_idx++) {
            auto& per_layer_block_table = m_block_table[seq_id][layer_idx];
            size_t block_table_size = per_layer_block_table.size();
            const auto& per_layer_block_indices_to_free = logical_block_index_sets_to_free[layer_idx];
            BlocksPerLayer new_sequence_blocks;
            OPENVINO_ASSERT(per_layer_block_indices_to_free.size() <= block_table_size, "too many blocks to free");
            new_sequence_blocks.reserve(block_table_size - per_layer_block_indices_to_free.size());
            for (size_t logical_block_idx = 0; logical_block_idx < block_table_size; logical_block_idx++) {
                if (per_layer_block_indices_to_free.find(logical_block_idx) == per_layer_block_indices_to_free.end()) {
                    // idx NOT in the requested set to free, need to keep this block
                    new_sequence_blocks.push_back(per_layer_block_table[logical_block_idx]);
                }
            }

            per_layer_block_table = new_sequence_blocks;
        }
    }

    /**
     * @param seq_group Pointer to a sequence group.
     * @return Whether enough cache blocks are available to host the sequences in the group.
     */
    bool can_append_slots(SequenceGroup::CPtr seq_group) {
        return required_blocks_count(std::move(seq_group)) <= m_allocator.num_free_blocks(0);
    }

    /**
     * @param seq_group Pointer to a sequence group.
     * @return The number of blocks necessary to host the sequences in the group, excluding the already
     * allocated ones.
     */
    size_t required_blocks_count(SequenceGroup::CPtr seq_group) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        return required_blocks_count_unlocked(std::move(seq_group));
    }

private:
    size_t required_blocks_count_unlocked(SequenceGroup::CPtr seq_group) const {
        std::vector<Sequence::CPtr> running_sequences = seq_group->get_running_sequences();
        size_t blocks_count = 0; // total number of needed blocks for sequence group
        std::set<size_t> last_block_ids; // unique last block indices

        for (auto seq: running_sequences) {
            auto seq_id = seq->get_id();
            const auto table_it = m_block_table.find(seq_id);
            if (table_it == m_block_table.end()) {
                // the block table is empty, so we need to allocate the number of blocks equal to number of logical blocks
                blocks_count += get_num_logical_blocks(seq_group);
                continue;
            }
            const auto& block_table = table_it->second[0];
            size_t num_physical_blocks = block_table.size();
            const size_t num_required_blocks = get_num_required_stored_blocks(seq_group, seq_id);
            OPENVINO_ASSERT(num_physical_blocks > 0);

            if (num_physical_blocks > num_required_blocks)
                // new blocks are not required
                // Case when num_physical_blocks == get_num_logical_blocks(seq_group) may still need block allocation
                // (such as when a sequence with an incomplete last block was forked) and is handled further in the
                // iteration
                continue;

            const size_t needed_blocks_per_sequence = num_required_blocks - num_physical_blocks;
            if (m_enable_prefix_caching) {
                blocks_count += needed_blocks_per_sequence;
                continue;
            }

            size_t last_block_id = block_table.back()->get_index();

            if (last_block_ids.find(last_block_id) != last_block_ids.end())
                // this block was already processed
                continue;
            last_block_ids.insert(last_block_id);

            CacheBlock::Ptr last_block = block_table.back();
            if (last_block->copy_on_write()) {
                // block is used only by multiple sequences
                auto references_count = last_block->get_references_count();

                if (needed_blocks_per_sequence == 0) {
                    // case when last block is not completely filled and needs to be copied n - 1 times, where n - references count
                    blocks_count += references_count - 1;
                }
                else {
                    blocks_count += needed_blocks_per_sequence * references_count;
                }
            }
            else {
                // block is used only by one sequence
                blocks_count += needed_blocks_per_sequence;
            }
        }
        if (m_enable_prefix_caching) {
            blocks_count += get_required_copy_on_write_blocks(seq_group);
        }
        return blocks_count;
    }

public:

    /**
     * @param seq_group Pointer to a sequence group.
     * @return The number of tokens corresponding to the block deficit for the group (required_blocks * block_size).
     */
    size_t required_tokens_count(SequenceGroup::CPtr seq_group) {
        return required_blocks_count(seq_group) * m_block_size;
    }

    /**
     * Clean up not busy physical cache blocks in a sequence group.
     * @param seq_group Pointer to a sequence group.
     */
    void free_empty_physical_blocks(SequenceGroup::Ptr seq_group) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        const size_t num_logical_blocks = seq_group->get_context_len() == 0 ? 0 : get_num_logical_blocks(seq_group);
        for (const auto& sequence : seq_group->get_running_sequences()) {
            auto seq_id = sequence->get_id();
            auto it = m_block_table.find(seq_id);
            if (it == m_block_table.end() || it->second.empty() || it->second[0].empty()) {
                if (num_logical_blocks == 0 && it != m_block_table.end()) {
                    m_block_table_logical_start.erase(seq_id);
                    OPENVINO_ASSERT(m_block_table.erase(seq_id) == 1);
                }
                continue;
            }
            const size_t num_physical_blocks = it->second[0].size();
            const size_t num_required_blocks = get_num_required_stored_blocks(seq_group, seq_id);
            if (num_physical_blocks > num_required_blocks) {
                free_sequence_partially(seq_id, num_physical_blocks - num_required_blocks);
            }
            invalidate_rolled_back_last_block(seq_id, seq_group->get_context_len());
        }
    }

    PreparedTailReleases prepare_tail_releases(const std::vector<TailReleaseTarget>& targets) {
        std::unique_lock<std::mutex> lock(m_cached_blocks_map_mutex);
        OPENVINO_ASSERT(m_fixed_blocks_per_sequence == 0,
                        "Prepared KV tail release is supported only without fixed cache blocks");
        std::set<uint64_t> target_ids;
        for (const TailReleaseTarget& target : targets) {
            OPENVINO_ASSERT(target_ids.insert(target.seq_id).second,
                            "Duplicate prepared KV tail release target for sequence ", target.seq_id);
            const auto table_it = m_block_table.find(target.seq_id);
            OPENVINO_ASSERT(table_it != m_block_table.end(),
                            "Missing KV block table for prepared tail release sequence ", target.seq_id);
            OPENVINO_ASSERT(table_it->second.size() == m_num_layers,
                            "KV block table layer count mismatch for prepared tail release sequence ", target.seq_id);
            const size_t allocated_blocks = table_it->second.front().size();
            for (const BlocksPerLayer& layer_table : table_it->second) {
                OPENVINO_ASSERT(layer_table.size() == allocated_blocks,
                                "KV block tables across layers must have equal size");
            }
            const size_t required_blocks = target.endpoint == 0 ? 0 : 1 + (target.endpoint - 1) / m_block_size;
            OPENVINO_ASSERT(required_blocks <= allocated_blocks,
                            "Prepared KV tail release endpoint exceeds allocated slots for sequence ", target.seq_id);
        }

        std::vector<PreparedTailReleases::PreparedRelease> releases;
        releases.reserve(targets.size());
        for (const TailReleaseTarget& target : targets) {
            auto table_it = m_block_table.find(target.seq_id);
            const size_t allocated_blocks = table_it->second.front().size();
            const size_t required_blocks = target.endpoint == 0 ? 0 : 1 + (target.endpoint - 1) / m_block_size;
            if (allocated_blocks <= required_blocks) {
                continue;
            }
            const size_t release_count = allocated_blocks - required_blocks;
            PreparedTailReleases::PreparedRelease release{target.seq_id, table_it, {}, {}};
            release.blocks.reserve(release_count);
            release.free_nodes.resize(release_count);
            for (size_t block_offset = 0; block_offset < release_count; ++block_offset) {
                BlocksPerLayer blocks;
                blocks.reserve(m_num_layers);
                release.free_nodes[block_offset].resize(m_num_layers);
                for (size_t layer_index = 0; layer_index < m_num_layers; ++layer_index) {
                    const CacheBlock::Ptr& block =
                        table_it->second[layer_index][allocated_blocks - block_offset - 1];
                    OPENVINO_ASSERT(block->get_references_count() > 0,
                                    "Prepared KV tail block must have a live reference");
                    OPENVINO_ASSERT(!m_enable_prefix_caching || !block->has_published_hash(),
                                    "Rejected KV tail rows must not be published");
                    blocks.push_back(block);
                    release.free_nodes[block_offset][layer_index].emplace_back();
                }
                release.blocks.push_back(std::move(blocks));
            }
            releases.push_back(std::move(release));
        }
        return PreparedTailReleases{*this, std::move(lock), std::move(releases)};
    }


    /**
     * Allocates just enough physical cache blocks to a sequence group to be enough for the sequences in it. If the sequences
     * in the group were forked before and their last block is a copy-on-write, then the block contents will have to be copied separately
     * into the freshly allocated block copies as reported in the returned map.
     * @param seq_group Pointer to a sequence group.
     * @return A map where each key is an index of a source *physical* block, and the corresponding value is a list of newly allocated *physical* block
     * indices into which the source block contents should be copied into separately.
     */
    std::map<size_t, std::list<size_t>> append_slots(SequenceGroup::Ptr seq_group, bool private_new_blocks = false) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        if (m_enable_prefix_caching) {
            OPENVINO_ASSERT(required_blocks_count_unlocked(seq_group) <= m_allocator.num_free_blocks(0),
                            "Insufficient cache blocks to append sequence group slots");
        }
        // Will always allocate the identical number of new blocks (if any) to each of the "layers" to keep the
        // number of blocks occupied by each "layer" identical at all times.
        const size_t num_logical_blocks = get_num_logical_blocks(seq_group);
        std::vector<Sequence::Ptr> running_sequences = seq_group->get_running_sequences();

        std::map<size_t, std::list<size_t>> copy_blocks_map;
        if (m_enable_prefix_caching) {
            copy_last_blocks_if_shared(running_sequences,
                                       seq_group->get_num_processed_tokens(),
                                       copy_blocks_map);
        }
        for (size_t i = 0; i < running_sequences.size(); ++i) {
            Sequence::Ptr sequence = running_sequences[i];
            auto seq_id = sequence->get_id();
            size_t num_physical_blocks = 0;
            size_t num_required_blocks = get_num_logical_blocks(seq_group);

            if (m_block_table.find(seq_id) != m_block_table.end())
            {
                num_physical_blocks = m_block_table[seq_id][0].size();
                num_required_blocks = get_num_required_stored_blocks(seq_group, seq_id);
            }

            if (!m_enable_prefix_caching && num_required_blocks <= num_physical_blocks) {
                copy_last_block_if_shared(sequence,
                                          seq_group->get_num_processed_tokens(),
                                          copy_blocks_map);
            }

            if (num_required_blocks > num_physical_blocks) {
                OPENVINO_ASSERT(can_allocate_blocks(num_required_blocks - num_physical_blocks));
                allocate(sequence, num_required_blocks - num_physical_blocks, seq_group->get_prompt_len(), private_new_blocks);
            } else {
                OPENVINO_ASSERT(num_required_blocks == num_physical_blocks, "A number of physical and logic blocks must be the same in this code path");

                size_t effective_num_layers = m_block_table[seq_id].size();
                BlocksPerLayer last_blocks;
                last_blocks.reserve(m_block_table[seq_id].size());
                for (size_t i = 0; i < effective_num_layers; i++) {
                    last_blocks.push_back(m_block_table[seq_id][i].back());
                }

                if (!last_blocks[0]->copy_on_write()) {
                    // we are the only users of this block
                    if (m_enable_prefix_caching && last_blocks[0]->has_published_hash()) {
                        // update hash of block
                        const auto prev_hash = last_blocks[0]->get_hash();
                        const size_t content_length = seq_group->get_context_len();
                        const auto hash = sequence->get_hash(content_length, m_block_size);
                        for (size_t i = 0; i < effective_num_layers; i++) {
                            auto& last_block = last_blocks[i];
                            last_block->set_hash(hash);
                        }
                        m_prefix_hash_to_cached_blocks.erase(prev_hash);
                        m_prefix_hash_to_cached_blocks[hash] = last_blocks;
                        unregister_cached_hash(prev_hash);
                        register_cached_content_length(hash, content_length);
                    }
                }
            }
        }

        // it returns information which blocks should be forked by ICacheManager
        return copy_blocks_map;
    }

    void publish_completed_block(const Sequence::Ptr& sequence, size_t content_length) {
        if (!m_enable_prefix_caching || content_length == 0 || content_length % m_block_size != 0) {
            return;
        }
        publish_completed_blocks(sequence, content_length - m_block_size, content_length);
    }

    void publish_completed_blocks(const Sequence::Ptr& sequence, size_t processed_before, size_t processed_after) {
        OPENVINO_ASSERT(processed_after >= processed_before,
                        "Cache block publication cannot precede the scheduled forward pass");
        processed_after = std::min(processed_after, sequence->get_prefix_publication_limit());
        if (processed_after <= processed_before) {
            return;
        }
        if (!m_enable_prefix_caching || processed_before / m_block_size == processed_after / m_block_size) {
            return;
        }

        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        const uint64_t seq_id = sequence->get_id();
        const auto block_table_it = m_block_table.find(seq_id);
        if (block_table_it == m_block_table.end() || block_table_it->second.empty()) {
            return;
        }

        const auto headroom = m_linear_attention_headroom.find(seq_id);
        const auto live_state = m_linear_attention_live_states.find(seq_id);
        if (headroom != m_linear_attention_headroom.end() && live_state != m_linear_attention_live_states.end() &&
            live_state->second.endpoint == processed_after && processed_after % m_block_size == 0) {
            return;
        }

        const size_t logical_start = get_block_table_logical_start_unlocked(seq_id);
        std::map<uint64_t, BlocksPerLayer> prepared_blocks;
        std::map<uint64_t, size_t> prepared_hash_lengths;
        std::map<size_t, size_t> prepared_length_counts;
        for (size_t logical_block_idx = processed_before / m_block_size;
             logical_block_idx < processed_after / m_block_size; ++logical_block_idx) {
            if (logical_block_idx < logical_start) {
                continue;
            }
            const size_t physical_block_idx = logical_block_idx - logical_start;
            if (physical_block_idx >= block_table_it->second[0].size()) {
                break;
            }
            BlocksPerLayer completed_blocks;
            completed_blocks.reserve(m_num_layers);
            for (size_t layer_idx = 0; layer_idx < m_num_layers; ++layer_idx) {
                CacheBlock::Ptr block = block_table_it->second[layer_idx][physical_block_idx];
                OPENVINO_ASSERT(block->has_published_hash() ==
                                    block_table_it->second[0][physical_block_idx]->has_published_hash(),
                                "Cache blocks across layers must share publication state");
                completed_blocks.push_back(std::move(block));
            }
            if (completed_blocks[0]->has_published_hash()) {
                continue;
            }
            const size_t content_length = (logical_block_idx + 1) * m_block_size;
            const uint64_t hash = sequence->get_hash(content_length, m_block_size);
            const auto content_length_it = m_cached_hash_to_content_length.find(hash);
            OPENVINO_ASSERT(content_length_it == m_cached_hash_to_content_length.end() ||
                                content_length_it->second == content_length,
                            "Cached prefix identity has a different content length");
            const auto cached_blocks_it = m_prefix_hash_to_cached_blocks.find(hash);
            if (cached_blocks_it != m_prefix_hash_to_cached_blocks.end()) {
                OPENVINO_ASSERT(cached_blocks_it->second.size() == m_num_layers,
                                "Cached prefix identity must contain one block per layer");
                for (const CacheBlock::Ptr& cached_block : cached_blocks_it->second) {
                    OPENVINO_ASSERT(cached_block->has_published_hash() && cached_block->get_hash() == hash,
                                    "Cached prefix identity must refer to verified published blocks");
                }
                continue;
            }
            const auto [prepared_it, inserted] = prepared_blocks.emplace(hash, std::move(completed_blocks));
            OPENVINO_ASSERT(inserted, "Crossed cache boundaries must have distinct prefix identities");
            if (content_length_it == m_cached_hash_to_content_length.end()) {
                prepared_hash_lengths.emplace(prepared_it->first, content_length);
                ++prepared_length_counts[content_length];
            }
        }

        for (const auto& [hash, blocks] : prepared_blocks) {
            for (const CacheBlock::Ptr& block : blocks) {
                block->set_hash(hash);
            }
        }
        m_prefix_hash_to_cached_blocks.merge(prepared_blocks);
        for (const auto& [content_length, count] : prepared_length_counts) {
            const auto existing = m_cached_content_length_ref_counts.find(content_length);
            if (existing != m_cached_content_length_ref_counts.end()) {
                existing->second += count;
            }
        }
        m_cached_content_length_ref_counts.merge(prepared_length_counts);
        m_cached_hash_to_content_length.merge(prepared_hash_lengths);
    }

    bool restore_cached_blocks(SequenceGroup::Ptr group) {
        if (!m_enable_prefix_caching) {
            return false;
        }

        const std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        return restore_cached_blocks_unlocked(group, get_prefix_restore_plan_unlocked(group, group->get_prompt_len()));
    }

    PrefixRestorePlan get_prefix_restore_plan(SequenceGroup::Ptr group) {
        return get_prefix_restore_plan(group, group->get_prompt_len());
    }

    PrefixRestorePlan get_prefix_restore_plan(SequenceGroup::Ptr group, size_t max_cache_token_position) {
        PrefixRestorePlan plan;
        if (!m_enable_prefix_caching || max_cache_token_position == 0) {
            return plan;
        }

        const std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        return get_prefix_restore_plan_unlocked(group, max_cache_token_position);
    }

    bool restore_cached_blocks(SequenceGroup::Ptr group, const PrefixRestorePlan& plan) {
        if (!m_enable_prefix_caching || plan.empty()) {
            return false;
        }

        const std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        return restore_cached_blocks_unlocked(group, plan);
    }

    void clear() {
        const std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        // KV-cache should not be cleared if prefix caching is enabled
        OPENVINO_ASSERT(m_enable_prefix_caching == false);

        // Block tables should be cleared when generation is finished
        OPENVINO_ASSERT(m_block_table.empty());

        m_temporary_block_table.clear();
        m_allocator.clear();
        m_prefix_hash_to_cached_blocks.clear();
        m_cached_content_length_ref_counts.clear();
        m_cached_hash_to_content_length.clear();
    }

private:
    PrefixRestorePlan get_prefix_restore_plan_unlocked(SequenceGroup::Ptr group, size_t max_cache_token_position) const {
        PrefixRestorePlan plan;
        if (!m_enable_prefix_caching || max_cache_token_position == 0) {
            return plan;
        }

        const size_t prompt_len = group->get_prompt_len();
        const size_t capped_token_position = std::min(prompt_len, max_cache_token_position);
        auto sequences = group->get_not_finished_sequences();
        OPENVINO_ASSERT(sequences.size() == 1);
        auto sequence = sequences[0];

        if (m_restore_latest_prefix_block_only) {
            return get_latest_prefix_restore_plan(sequence, capped_token_position, prompt_len);
        }

        return get_full_prefix_restore_plan(sequence, capped_token_position, prompt_len);
    }

    bool restore_cached_blocks_unlocked(SequenceGroup::Ptr group, const PrefixRestorePlan& plan) {
        if (plan.empty()) {
            return false;
        }

        const auto sequences = group->get_not_finished_sequences();
        OPENVINO_ASSERT(sequences.size() == 1);
        const auto sequence = sequences[0];
        const auto seq_id = sequence->get_id();

        for (size_t content_len : plan.block_content_lengths) {
            if (!m_allocator.has_cached_block(sequence->get_hash(content_len, m_block_size),
                                              m_prefix_hash_to_cached_blocks)) {
                return false;
            }
        }

        if (m_block_table.find(seq_id) == m_block_table.end()) {
            m_block_table[seq_id].resize(m_num_layers);
        }
        auto& block_table = m_block_table[seq_id];

        for (size_t content_len : plan.block_content_lengths) {
            auto blocks = m_allocator.get_cached_block(sequence->get_hash(content_len, m_block_size),
                                                       m_prefix_hash_to_cached_blocks);
            OPENVINO_ASSERT(!blocks.empty(), "Prefix restore plan became unavailable for token position ", content_len);
            const auto timestamp = std::chrono::steady_clock::now();
            for (size_t layer_idx = 0; layer_idx < block_table.size(); layer_idx++) {
                auto& block = blocks[layer_idx];
                block->set_timestamp(timestamp);
                block_table[layer_idx].push_back(block);
            }
        }

        if (m_restore_latest_prefix_block_only) {
            m_block_table_logical_start[seq_id] = plan.logical_block_start;
            BlocksPerLayer live_rows;
            live_rows.reserve(m_num_layers);
            for (size_t layer_idx = 0; layer_idx < m_num_layers; ++layer_idx) {
                live_rows.push_back(block_table[layer_idx].back());
            }
            set_linear_attention_live_state(seq_id, plan.cache_token_position, live_rows);
        } else {
            m_block_table_logical_start.erase(seq_id);
        }
        group->update_processed_tokens_num(plan.processed_tokens);
        return true;
    }

    PrefixRestorePlan get_full_prefix_restore_plan(const Sequence::Ptr& sequence,
                                                   size_t capped_token_position,
                                                   size_t prompt_len) const {
        PrefixRestorePlan plan;
        size_t content_len = 0;
        while (content_len < capped_token_position) {
            size_t prev_iteration_content_len = content_len;
            content_len += m_block_size;
            if (content_len > capped_token_position) {
                content_len = capped_token_position;
            }
            const auto full_block_hash = sequence->get_hash(content_len, m_block_size);
            if (m_allocator.has_cached_block(full_block_hash, m_prefix_hash_to_cached_blocks)) {
                plan.block_content_lengths.push_back(content_len);
                plan.cache_token_position = content_len;
                plan.processed_tokens = get_processed_tokens_after_restore(content_len, prompt_len);
            } else {
                for (size_t i = 1; i < m_block_size; i++) {
                    if (prev_iteration_content_len + i >= content_len) {
                        break;
                    }
                    const size_t partial_content_len = prev_iteration_content_len + i;
                    const auto hash = sequence->get_hash(partial_content_len, m_block_size);
                    if (m_allocator.has_cached_block(hash, m_prefix_hash_to_cached_blocks)) {
                        plan.block_content_lengths.push_back(partial_content_len);
                        plan.cache_token_position = partial_content_len;
                        plan.processed_tokens = get_processed_tokens_after_restore(partial_content_len, prompt_len);
                        break;
                    }
                }
                break;
            }
        }
        return plan;
    }

    PrefixRestorePlan get_latest_prefix_restore_plan(const Sequence::Ptr& sequence,
                                                     size_t capped_token_position,
                                                     size_t prompt_len) const {
        PrefixRestorePlan plan;
        size_t latest_content_len = 0;
        size_t interval_end = capped_token_position;
        while (interval_end > 0) {
            size_t content_len = 0;
            if (!find_latest_cached_content_len(sequence, interval_end, content_len)) {
                if (!plan.empty()) {
                    break;
                }
                interval_end = get_interval_start(interval_end);
                continue;
            }
            if (latest_content_len == 0) {
                latest_content_len = content_len;
            }
            plan.block_content_lengths.push_back(content_len);
            if (content_len % m_block_size != 0) {
                break;
            }
            interval_end = get_interval_start(content_len);
        }

        if (plan.empty()) {
            return plan;
        }

        std::reverse(plan.block_content_lengths.begin(), plan.block_content_lengths.end());
        plan.cache_token_position = latest_content_len;
        plan.processed_tokens = get_processed_tokens_after_restore(latest_content_len, prompt_len);
        plan.logical_block_start = (plan.block_content_lengths.front() - 1) / m_block_size;
        return plan;
    }

    bool find_latest_cached_content_len(const Sequence::Ptr& sequence, size_t interval_end, size_t& content_len) const {
        const size_t interval_start = get_interval_start(interval_end);
        auto candidate_it = m_cached_content_length_ref_counts.upper_bound(interval_end);
        while (candidate_it != m_cached_content_length_ref_counts.begin()) {
            --candidate_it;
            const size_t candidate_len = candidate_it->first;
            if (candidate_len <= interval_start) {
                break;
            }
            if (!m_allocator.has_cached_block(sequence->get_hash(candidate_len, m_block_size),
                                              m_prefix_hash_to_cached_blocks)) {
                continue;
            }
            content_len = candidate_len;
            return true;
        }
        return false;
    }

    void register_cached_content_length(uint64_t hash, size_t content_length) {
        auto hash_it = m_cached_hash_to_content_length.find(hash);
        if (hash_it != m_cached_hash_to_content_length.end()) {
            unregister_cached_content_length(hash_it->second);
        }
        m_cached_hash_to_content_length[hash] = content_length;
        ++m_cached_content_length_ref_counts[content_length];
    }

    void unregister_cached_hash(uint64_t hash) {
        auto hash_it = m_cached_hash_to_content_length.find(hash);
        if (hash_it == m_cached_hash_to_content_length.end()) {
            return;
        }
        unregister_cached_content_length(hash_it->second);
        m_cached_hash_to_content_length.erase(hash_it);
    }

    void unregister_cached_content_length(size_t content_length) {
        auto content_length_it = m_cached_content_length_ref_counts.find(content_length);
        OPENVINO_ASSERT(content_length_it != m_cached_content_length_ref_counts.end(),
                        "Cached content length ", content_length, " is not registered");
        OPENVINO_ASSERT(content_length_it->second > 0,
                        "Cached content length ", content_length, " has zero reference count");
        --content_length_it->second;
        if (content_length_it->second == 0) {
            m_cached_content_length_ref_counts.erase(content_length_it);
        }
    }

    BlocksPerLayer find_alternative_published_blocks(uint64_t hash, const BlocksPerLayer& excluded_blocks) const {
        for (const auto& [sequence_id, block_tables] : m_block_table) {
            (void)sequence_id;
            if (block_tables.empty()) {
                continue;
            }
            const size_t num_blocks = block_tables[0].size();
            for (const auto& block_table : block_tables) {
                OPENVINO_ASSERT(block_table.size() == num_blocks,
                                "Cache block tables across layers must have equal sizes");
            }
            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
                BlocksPerLayer candidate_blocks;
                candidate_blocks.reserve(m_num_layers);
                for (size_t layer_idx = 0; layer_idx < m_num_layers; ++layer_idx) {
                    const CacheBlock::Ptr& block = block_tables[layer_idx][block_idx];
                    if (!block->has_published_hash() || block->get_hash() != hash) {
                        candidate_blocks.clear();
                        break;
                    }
                    candidate_blocks.push_back(block);
                }
                if (!candidate_blocks.empty() && candidate_blocks != excluded_blocks) {
                    return candidate_blocks;
                }
            }
        }
        return {};
    }

    void invalidate_rolled_back_last_block(uint64_t seq_id, size_t retained_content_length) {
        if (!m_enable_prefix_caching || retained_content_length == 0 ||
            retained_content_length % m_block_size == 0) {
            return;
        }
        const auto block_table_it = m_block_table.find(seq_id);
        if (block_table_it == m_block_table.end() || block_table_it->second.empty() ||
            block_table_it->second[0].empty()) {
            return;
        }
        const size_t retained_logical_position = (retained_content_length - 1) / m_block_size;
        const size_t latest_logical_position =
            get_block_table_logical_start_unlocked(seq_id) + block_table_it->second[0].size() - 1;
        if (retained_logical_position != latest_logical_position) {
            return;
        }

        BlocksPerLayer latest_blocks;
        latest_blocks.reserve(m_num_layers);
        for (size_t layer_idx = 0; layer_idx < m_num_layers; ++layer_idx) {
            const CacheBlock::Ptr& block = block_table_it->second[layer_idx].back();
            if (!block->has_published_hash() || block->get_references_count() > 1) {
                return;
            }
            latest_blocks.push_back(block);
        }

        const uint64_t published_hash = latest_blocks[0]->get_hash();
        const auto cached_blocks_it = m_prefix_hash_to_cached_blocks.find(published_hash);
        const bool owns_registration =
            cached_blocks_it != m_prefix_hash_to_cached_blocks.end() && cached_blocks_it->second == latest_blocks;
        const auto content_length_it = m_cached_hash_to_content_length.find(published_hash);
        OPENVINO_ASSERT(content_length_it != m_cached_hash_to_content_length.end() || !owns_registration,
                        "Registered prefix blocks have no content length");
        if (content_length_it == m_cached_hash_to_content_length.end()) {
            for (const CacheBlock::Ptr& block : latest_blocks) {
                block->clear_published_hash();
            }
            return;
        }
        if (content_length_it->second <= retained_content_length) {
            return;
        }

        if (owns_registration) {
            m_prefix_hash_to_cached_blocks.erase(cached_blocks_it);
            BlocksPerLayer alternative_blocks = find_alternative_published_blocks(published_hash, latest_blocks);
            if (!alternative_blocks.empty()) {
                m_prefix_hash_to_cached_blocks.emplace(published_hash, std::move(alternative_blocks));
            }
        }
        if (!m_allocator.has_cached_block(published_hash, m_prefix_hash_to_cached_blocks)) {
            unregister_cached_hash(published_hash);
        }
        for (const CacheBlock::Ptr& block : latest_blocks) {
            block->clear_published_hash();
        }
    }

    size_t get_interval_start(size_t interval_end) const {
        return ((interval_end - 1) / m_block_size) * m_block_size;
    }

    static size_t get_processed_tokens_after_restore(size_t content_len, size_t prompt_len) {
        return content_len == prompt_len ? content_len - 1 : content_len;
    }

    size_t get_block_table_logical_start_unlocked(uint64_t seq_id) const {
        auto it = m_block_table_logical_start.find(seq_id);
        return it == m_block_table_logical_start.end() ? 0 : it->second;
    }

    size_t get_num_required_stored_blocks(SequenceGroup::CPtr seq_group, uint64_t seq_id) const {
        const size_t num_logical_blocks = get_num_logical_blocks(seq_group);
        const size_t logical_start = get_block_table_logical_start_unlocked(seq_id);
        return num_logical_blocks > logical_start ? num_logical_blocks - logical_start : 0;
    }

    bool last_block_requires_copy_on_write(const Sequence::CPtr& sequence,
                                           size_t num_processed_tokens) const {
        auto table_it = m_block_table.find(sequence->get_id());
        if (table_it == m_block_table.end() || table_it->second.empty() || table_it->second[0].empty()) {
            return false;
        }
        if (m_fixed_blocks_per_sequence == 0 && num_processed_tokens % m_block_size == 0) {
            return false;
        }
        const auto& last_block = table_it->second[0].back();
        const auto live_state_it = m_linear_attention_live_states.find(sequence->get_id());
        const bool is_published_linear_attention_live_row =
            m_restore_latest_prefix_block_only && last_block->has_published_hash() &&
            live_state_it != m_linear_attention_live_states.end() && !live_state_it->second.rows.empty() &&
            live_state_it->second.rows[0] == last_block;
        return last_block->copy_on_write() || is_published_linear_attention_live_row;
    }

    size_t get_required_copy_on_write_blocks(SequenceGroup::CPtr seq_group) const {
        if (!m_enable_prefix_caching || seq_group->get_num_processed_tokens() % m_block_size == 0) {
            return 0;
        }
        std::map<size_t, size_t> required_copies_per_block;
        for (const auto& sequence : seq_group->get_running_sequences()) {
            auto table_it = m_block_table.find(sequence->get_id());
            if (table_it == m_block_table.end() || table_it->second.empty() || table_it->second[0].empty()) {
                continue;
            }
            const auto& last_block = table_it->second[0].back();
            if (!last_block_requires_copy_on_write(sequence,
                                                   seq_group->get_num_processed_tokens())) {
                continue;
            }
            const size_t block_index = static_cast<size_t>(last_block->get_index());
            const size_t maximum_shared_copies = last_block->copy_on_write()
                                                     ? static_cast<size_t>(last_block->get_references_count() - 1)
                                                     : 1;
            required_copies_per_block[block_index] = std::min(required_copies_per_block[block_index] + 1,
                                                               maximum_shared_copies);
        }
        size_t required_blocks = 0;
        for (const auto& [block_index, required_copies] : required_copies_per_block) {
            required_blocks += required_copies;
        }
        return required_blocks;
    }

    struct CopyOnWriteReplacement {
        uint64_t sequence_id;
        BlocksPerLayer previous_blocks;
        BlockAllocator::CacheBlockAllocationResult allocation;
        std::optional<size_t> erased_content_length;
    };

    std::vector<BlockAllocator::UncachedBlockAllocation> acquire_temporary_blocks(
        size_t num_blocks,
        std::vector<int>& block_indices) {
        std::vector<uint64_t> erased_hashes;
        auto allocations = m_allocator.allocate_temporary_blocks(
            num_blocks, m_prefix_hash_to_cached_blocks, block_indices, erased_hashes);
        for (uint64_t hash : erased_hashes) {
            unregister_cached_hash(hash);
        }
        return allocations;
    }

    BlockAllocator::CacheBlockAllocationResult acquire_unpublished_block(
        std::optional<size_t>& erased_content_length) {
        BlockAllocator::CacheBlockAllocationResult allocation =
            m_allocator.acquire_unpublished_block(m_prefix_hash_to_cached_blocks);
        if (allocation.erased_hash.has_value()) {
            try {
                const auto content_length_it = m_cached_hash_to_content_length.find(*allocation.erased_hash);
                OPENVINO_ASSERT(content_length_it != m_cached_hash_to_content_length.end(),
                                "Evicted prefix hash has no registered content length");
                erased_content_length = content_length_it->second;
                unregister_cached_hash(*allocation.erased_hash);
            } catch (...) {
                m_allocator.rollback_uncached_block(std::move(allocation),
                                                    m_prefix_hash_to_cached_blocks);
                throw;
            }
        }
        return allocation;
    }

    void rollback_copy_on_write_replacement(CopyOnWriteReplacement& replacement) {
        for (size_t layer_idx = 0; layer_idx < m_num_layers; ++layer_idx) {
            m_block_table[replacement.sequence_id][layer_idx].back() = replacement.previous_blocks[layer_idx];
        }
        const auto live_state_it = m_linear_attention_live_states.find(replacement.sequence_id);
        if (live_state_it != m_linear_attention_live_states.end()) {
            set_linear_attention_live_state(replacement.sequence_id,
                                            live_state_it->second.endpoint,
                                            replacement.previous_blocks);
        }
        const std::optional<uint64_t> erased_hash = replacement.allocation.erased_hash;
        const std::optional<size_t> erased_content_length = replacement.erased_content_length;
        m_allocator.rollback_uncached_block(std::move(replacement.allocation),
                                            m_prefix_hash_to_cached_blocks);
        if (erased_hash.has_value()) {
            OPENVINO_ASSERT(erased_content_length.has_value());
            register_cached_content_length(*erased_hash, *erased_content_length);
        }
    }

    void copy_last_blocks_if_shared(const std::vector<Sequence::Ptr>& sequences,
                                    size_t num_processed_tokens,
                                    std::map<size_t, std::list<size_t>>& copy_blocks_map) {
        std::vector<CopyOnWriteReplacement> replacements;
        replacements.reserve(sequences.size());
        std::map<size_t, std::list<size_t>> staged_copy_blocks_map;
        std::map<size_t, size_t> staged_replacements_per_source;
        try {
            for (const auto& sequence : sequences) {
                if (!last_block_requires_copy_on_write(sequence,
                                                       num_processed_tokens)) {
                    continue;
                }
                const uint64_t sequence_id = sequence->get_id();
                const auto& source_block = m_block_table[sequence_id][0].back();
                const size_t source_index = static_cast<size_t>(source_block->get_index());
                const size_t maximum_replacements = source_block->copy_on_write()
                                                        ? static_cast<size_t>(source_block->get_references_count() - 1)
                                                        : 1;
                if (staged_replacements_per_source[source_index] >= maximum_replacements) {
                    continue;
                }
                BlocksPerLayer previous_blocks;
                previous_blocks.reserve(m_num_layers);
                for (size_t layer_idx = 0; layer_idx < m_num_layers; ++layer_idx) {
                    previous_blocks.push_back(m_block_table[sequence_id][layer_idx].back());
                }

                std::optional<size_t> erased_content_length;
                BlockAllocator::CacheBlockAllocationResult allocation =
                    acquire_unpublished_block(erased_content_length);
                OPENVINO_ASSERT(!allocation.blocks.empty(),
                                "Shared cache row copy-on-write exhausted free and overwriteable blocks");
                replacements.push_back({sequence_id,
                                        std::move(previous_blocks),
                                        std::move(allocation),
                                        erased_content_length});
                auto& replacement = replacements.back();
                for (size_t layer_idx = 0; layer_idx < m_num_layers; ++layer_idx) {
                    m_block_table[sequence_id][layer_idx].back() = replacement.allocation.blocks[layer_idx];
                }
                const auto live_state_it = m_linear_attention_live_states.find(sequence_id);
                if (live_state_it != m_linear_attention_live_states.end()) {
                    set_linear_attention_live_state(sequence_id,
                                                    live_state_it->second.endpoint,
                                                    replacement.allocation.blocks);
                }
                staged_copy_blocks_map[replacement.previous_blocks[0]->get_index()].push_back(
                    replacement.allocation.blocks[0]->get_index());
                ++staged_replacements_per_source[source_index];
            }

            auto updated_copy_blocks_map = copy_blocks_map;
            for (auto& [source_index, destination_indices] : staged_copy_blocks_map) {
                updated_copy_blocks_map[source_index].splice(
                    updated_copy_blocks_map[source_index].end(),
                    destination_indices);
            }
            copy_blocks_map.swap(updated_copy_blocks_map);
        } catch (...) {
            for (auto replacement_it = replacements.rbegin(); replacement_it != replacements.rend(); ++replacement_it) {
                rollback_copy_on_write_replacement(*replacement_it);
            }
            throw;
        }

        for (auto& replacement : replacements) {
            free_cached_blocks(replacement.previous_blocks);
        }
    }

    void copy_last_block_if_shared(const Sequence::Ptr& sequence,
                                   size_t num_processed_tokens,
                                   std::map<size_t, std::list<size_t>>& copy_blocks_map) {
        copy_last_blocks_if_shared({sequence}, num_processed_tokens, copy_blocks_map);
    }

    size_t get_num_unused_tokens(SequenceGroup::CPtr seq_group) const {
        const size_t occupied = seq_group->get_context_len() - seq_group->get_num_evicted_tokens();
        // equivalent to: occupied % m_block_size == 0 ? 0 : m_block_size - (occupied % m_block_size)
        return (m_block_size - occupied % m_block_size) % m_block_size;
    }

    /**
     * Converts a number of blocks released to the corresponding number of tokens freed,
     * accounting for a potentially partial last block.
     * @param seq_group The sequence group the blocks were released from.
     * @param blocks_released The number of logical blocks released.
     * @return The number of tokens freed.
     */
    size_t _blocks_released_to_tokens(SequenceGroup::CPtr seq_group, size_t blocks_released) const {
        if (blocks_released == 0) {
            return 0;
        }
        const size_t processed_tokens = seq_group->get_num_processed_tokens();
        size_t tokens_in_last_block = processed_tokens % m_block_size;
        if (tokens_in_last_block == 0) {
            tokens_in_last_block = m_block_size;
        }
        return tokens_in_last_block + (blocks_released - 1) * m_block_size;
    }

    bool can_allocate_blocks(size_t num_blocks) const {
        for (size_t layer_idx = 0; layer_idx < m_num_layers; layer_idx++) {
            if (!m_allocator.can_allocate_blocks(num_blocks, layer_idx)) return false;
        }
        return true;
    }

    void free_cached_blocks(const BlocksPerLayer& blocks_for_all_layers) {
        m_allocator.free(blocks_for_all_layers, m_prefix_hash_to_cached_blocks);
    }

    BlocksPerLayer allocate_cached_block(uint64_t hash, size_t content_length) {
        BlockAllocator::CacheBlockAllocationResult allocation_result = m_allocator.allocate_block(hash, m_prefix_hash_to_cached_blocks);
        if (allocation_result.erased_hash.has_value()) {
            unregister_cached_hash(*allocation_result.erased_hash);
        }
        register_cached_content_length(hash, content_length);
        return allocation_result.blocks;
    }

    void allocate(ov::genai::Sequence::Ptr sequence, size_t num_blocks, size_t prompt_size = 0,
                  bool private_new_blocks = false) {
        OPENVINO_ASSERT(num_blocks > 0 && can_allocate_blocks(num_blocks));

        auto sequence_id = sequence->get_id();
        if (m_block_table.find(sequence_id) == m_block_table.end()) {
            m_block_table[sequence_id].resize(m_num_layers);
        }

        auto& block_table = m_block_table[sequence_id][0];
        auto content_length = sequence->get_generated_len() + prompt_size;
        const size_t logical_start = get_block_table_logical_start_unlocked(sequence_id);
        size_t allocated_blocks = block_table.size();
        size_t num_hashed_tokens = (logical_start + allocated_blocks) * m_block_size;

        if (m_enable_prefix_caching && (private_new_blocks ||
            sequence->get_prefix_publication_limit() != std::numeric_limits<size_t>::max())) {
            for (auto& table : m_block_table.at(sequence_id)) {
                table.reserve(table.size() + num_blocks);
            }
            std::vector<int> indices;
            auto allocations = acquire_temporary_blocks(num_blocks, indices);
            for (const auto& allocation : allocations) {
                for (size_t layer = 0; layer < m_num_layers; ++layer) {
                    m_block_table.at(sequence_id)[layer].push_back(allocation.blocks[layer]);
                }
            }
            if (m_restore_latest_prefix_block_only && m_linear_attention_live_states.count(sequence_id) == 0) {
                BlocksPerLayer live_rows;
                live_rows.reserve(m_num_layers);
                for (size_t layer_idx = 0; layer_idx < m_num_layers; ++layer_idx) {
                    live_rows.push_back(m_block_table[sequence_id][layer_idx].front());
                }
                set_empty_linear_attention_live_state(sequence_id, live_rows);
            }
        } else if (!m_enable_prefix_caching) {
            for (size_t layer_idx = 0; layer_idx < m_block_table[sequence_id].size(); layer_idx++) {
                auto& block_table = m_block_table[sequence_id][layer_idx];
                for (size_t i = 0; i < num_blocks; ++i) {
                    ov::genai::CacheBlock::Ptr block = m_allocator.allocate_block(layer_idx);
                    OPENVINO_ASSERT(block != nullptr);
                    m_block_table[sequence_id][layer_idx].push_back(block);
                }
            }
            if (m_fixed_blocks_per_sequence > 0) {
                BlocksPerLayer live_rows;
                live_rows.reserve(m_num_layers);
                for (size_t layer_idx = 0; layer_idx < m_num_layers; ++layer_idx) {
                    live_rows.push_back(m_block_table[sequence_id][layer_idx].back());
                }
                set_empty_linear_attention_live_state(sequence_id, live_rows);
            }
        } else {
            if (block_table.size() > 0 && block_table.back()->has_published_hash()) {
                CacheBlock::Ptr last_block = block_table.back();
                auto hash = sequence->get_hash((logical_start + block_table.size()) * m_block_size, m_block_size);
                auto prev_hash = last_block->get_hash();
                if (prev_hash != hash) {
                    const size_t content_length = (logical_start + block_table.size()) * m_block_size;
                    BlocksPerLayer last_blocks_vec;
                    last_blocks_vec.reserve(m_num_layers);
                    for (size_t layer_idx = 0; layer_idx < m_num_layers; layer_idx++) {
                        auto& lst_blk = m_block_table[sequence_id][layer_idx].back();
                        lst_blk->set_hash(hash);
                        m_prefix_hash_to_cached_blocks.erase(prev_hash);
                        last_blocks_vec.push_back(lst_blk);
                    }
                    m_prefix_hash_to_cached_blocks[hash] = last_blocks_vec;
                    unregister_cached_hash(prev_hash);
                    register_cached_content_length(hash, content_length);
                }
            }
            for (size_t i = 0; i < num_blocks; ++i) {
                num_hashed_tokens += m_block_size;
                if (num_hashed_tokens > content_length) {
                    num_hashed_tokens = content_length;
                }
                auto hash = sequence->get_hash(num_hashed_tokens, m_block_size);
                auto blocks_for_all_layers = allocate_cached_block(hash, num_hashed_tokens);
                for (size_t layer_idx = 0; layer_idx < blocks_for_all_layers.size(); layer_idx++) {
                    m_block_table[sequence_id][layer_idx].push_back(blocks_for_all_layers[layer_idx]);
                }
            }
            if (m_restore_latest_prefix_block_only && m_linear_attention_live_states.count(sequence_id) == 0) {
                BlocksPerLayer live_rows;
                live_rows.reserve(m_num_layers);
                for (size_t layer_idx = 0; layer_idx < m_num_layers; ++layer_idx) {
                    live_rows.push_back(m_block_table[sequence_id][layer_idx].front());
                }
                set_empty_linear_attention_live_state(sequence_id, live_rows);
            }
        }
    }
};


}
