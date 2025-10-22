// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <list>
#include <map>
#include <algorithm>
#include <fstream>
#include <chrono>

#include "sequence_group.hpp"

namespace ov::genai {

class KVCacheBlock {
    int m_ref_count;
    int m_index;
    size_t m_hash;
    std::chrono::time_point<std::chrono::steady_clock> m_timestamp;
public:
    using Ptr = std::shared_ptr<KVCacheBlock>;
    using CPtr = std::shared_ptr<const KVCacheBlock>;

    explicit KVCacheBlock(int index)
        : m_ref_count(0),
          m_index(index),
          m_hash(0),
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

    bool copy_on_write() const {
        return m_ref_count > 1;
    }

    int get_references_count() const {
        return m_ref_count;
    }

    size_t get_hash() const {
        return m_hash;
    }

    void set_hash(size_t hash) {
        m_hash = hash;
    }

    void set_timestamp(const std::chrono::time_point<std::chrono::steady_clock>& timestamp) {
        m_timestamp = timestamp;
    }

    std::chrono::time_point<std::chrono::steady_clock> get_timestamp() {
        return m_timestamp;
    }
};

using BlocksPerLayer = std::vector<KVCacheBlock::Ptr>;

/**
 * @brief Allows to store and retrieve KV-cache blocks based on their content- and position-based hash.
 * Blocks with the same prefix in the generated sequence will have the same hash. Blocks within this store
 * are not owned by any sequence (but had been once) and may be either selected for overwriting, if the allocator
 * runs out of fresh blocks, or reused if their contents match to the prefix-based requested hash.
 */
class OverwritableBlocksHashStore {
    std::map<size_t, BlocksPerLayer> m_blocks;
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
        bool is_all_free = std::all_of(blocks_for_all_layers.begin(), blocks_for_all_layers.end(), [](const KVCacheBlock::Ptr& block_ptr) { return block_ptr->is_free(); });
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
        auto timestamp = std::chrono::steady_clock::now();
        for (auto& block_ptr : blocks_for_all_layers) {
            block_ptr->set_timestamp(timestamp);
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
 * @brief Maintains a pool of KV cache block descriptors (layered as configured at initialization), freeing or allocating
 * them as requested.
 */
class BlockAllocator {
    std::vector<std::list<KVCacheBlock::Ptr>> m_free_blocks;
    // We keep m_free_blocks_num instead of m_free_blocks[X].size() to WA old CXX library implementation issue for std::list::size()
    // see https://stackoverflow.com/questions/13157164/why-isnt-stdlist-size-constant-time
    std::vector<size_t> m_free_blocks_num;
    size_t m_total_num_blocks;
    friend class CacheStateDumper;
    size_t m_num_layers;
    bool m_enable_prefix_caching;
    ov::genai::OverwritableBlocksHashStore m_overwriteable_blocks;

public:
    /**
     * Constructs the BlockAllocator.
     * @param num_blocks Number of KV cache blocks in the free block pool to be owned by this allocator.
     * @param enable_prefix_caching Whether prefix caching should be enabled for this allocator.
     * See also the equivalent parameter in ov::genai::ContinuousBatchingPipeline
     * @param num_layers The number of separate attention layers with KV caches in the LLM associated with the pipeline.
     * Blocks returned will be vectors with this size, each vector entry to be associated with a separate layer's KV cache.
     */
    BlockAllocator(size_t num_blocks, bool enable_prefix_caching, size_t num_layers = 1) :
            m_total_num_blocks(num_blocks), m_num_layers(num_layers), m_enable_prefix_caching(enable_prefix_caching), m_overwriteable_blocks(num_layers) {
        OPENVINO_ASSERT(num_layers != 0, "num_layers must be non-zero");
        m_free_blocks.resize(m_num_layers);
        if (num_blocks > 0) {
            m_free_blocks_num = std::vector<size_t>(num_layers, num_blocks);
            for (auto& per_layer_block_list : m_free_blocks) {
                for (int block_id = 0; block_id < m_total_num_blocks; ++block_id) {
                    per_layer_block_list.push_back(std::make_shared<KVCacheBlock>(block_id));
                }
            }
        } else {
            m_free_blocks_num = std::vector<size_t>(m_num_layers, 0);
        }
    }

    ~BlockAllocator() {
        // sanity check to validate that all blocks are freed
        for (auto& free_block : m_free_blocks_num) {
            size_t free_and_overwritable_block_cnt = free_block + num_overwriteable_blocks();
            OPENVINO_ASSERT(m_total_num_blocks == free_and_overwritable_block_cnt, "Expected num free blocks: ", m_total_num_blocks, ", actual: ", free_and_overwritable_block_cnt);
        }
    }

    void increase_kv_blocks_number(size_t new_kv_blocks_count) {
        OPENVINO_ASSERT(new_kv_blocks_count > m_total_num_blocks, "New blocks number should be more than previous blocks number.");
        size_t added_blocks = new_kv_blocks_count - m_total_num_blocks;
        for (auto idx = 0; idx < m_free_blocks_num.size(); idx++) {
            m_free_blocks_num[idx] += added_blocks;
        }
        for (auto& per_layer_block_list : m_free_blocks) {
            for (int block_id = m_total_num_blocks; block_id < new_kv_blocks_count; ++block_id) {
                per_layer_block_list.push_back(std::make_shared<KVCacheBlock>(block_id));
            }
        }
        m_total_num_blocks = new_kv_blocks_count;
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

    /**
     * Frees a given block for a given layer. If no sequence is associated with the block after freeing, the block
     * is returned to the "free" pool.
     * @param block_ptr The block to be freed.
     * @param layer_idx The index of the layer with which the block is associated.
     */
    void free(KVCacheBlock::Ptr& block_ptr, size_t layer_idx) {
        OPENVINO_ASSERT(!m_enable_prefix_caching);
        OPENVINO_ASSERT(layer_idx < m_num_layers);
        block_ptr->release();
        if (block_ptr->is_free()) {
            m_free_blocks[layer_idx].push_back(block_ptr);
            ++m_free_blocks_num[layer_idx];
        }
    }

    /**
     * Frees a block for each layer. If no sequence is associated with the blocks after freeing, the blocks
     * are either returned to the "free" pool, or, if prefix caching is enabled, stored internally for its contents
     * to be potentially reused if a prefix of a new sequence matches to the prefix with which the currently freed blocks
     * were computed.
     * @param blocks_for_all_layers The blocks to be freed (one for each layer).
     */
    void free(const BlocksPerLayer& blocks_for_all_layers) {
        OPENVINO_ASSERT(blocks_for_all_layers.size() == m_num_layers);
        for (size_t i = 0; i < m_num_layers; i++) {
            auto& block_ptr = blocks_for_all_layers[i];
            block_ptr->release();
        }

        auto free_predicate = [](const KVCacheBlock::Ptr& block_ptr) { return block_ptr->is_free(); };
        bool is_any_free = std::any_of(blocks_for_all_layers.begin(), blocks_for_all_layers.end(), free_predicate);
        bool is_all_free = false;
        if (is_any_free && m_num_layers > 1) {
            is_all_free = std::all_of(blocks_for_all_layers.begin(), blocks_for_all_layers.end(), free_predicate);
            OPENVINO_ASSERT(is_all_free, "blocks across layers must be freed simultaneously");
        }

        if (is_any_free) {
            // is_all_free == true due to assert above
            if (m_enable_prefix_caching)
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
    KVCacheBlock::Ptr allocate_block(size_t layer_idx) {
        OPENVINO_ASSERT(layer_idx < m_free_blocks.size());
        OPENVINO_ASSERT(!m_enable_prefix_caching);
        OPENVINO_ASSERT(can_allocate_blocks(1, layer_idx));
        KVCacheBlock::Ptr allocated_block = m_free_blocks[layer_idx].front();
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
    BlocksPerLayer allocate_block(size_t hash, std::map<uint64_t, BlocksPerLayer>& cached_blocks) {
        OPENVINO_ASSERT(m_enable_prefix_caching);
        OPENVINO_ASSERT(can_allocate_blocks(1));

        if (m_free_blocks_num[0] > 0) {
            // allocate new empty block
            BlocksPerLayer allocated_blocks;
            allocated_blocks.reserve(m_num_layers);
            for (size_t i = 0; i < m_num_layers; i++) {
                KVCacheBlock::Ptr allocated_block = m_free_blocks[i].front();
                allocated_block->increment();
                allocated_block->set_hash(hash);
                allocated_blocks.push_back(allocated_block);
                m_free_blocks[i].pop_front();
                --m_free_blocks_num[i];
            }
            cached_blocks[hash] = allocated_blocks;
            return allocated_blocks;
        }
        if (m_overwriteable_blocks.num_blocks() > 0) {
            // get least recently used block from store and reuse it
            BlocksPerLayer blocks_for_all_layers = m_overwriteable_blocks.get_lru_block_to_overwrite();
            cached_blocks.erase(blocks_for_all_layers[0]->get_hash());

            // update block with new hash
            for (auto& block : blocks_for_all_layers) {
                block->set_hash(hash);
            }
            cached_blocks[hash] = blocks_for_all_layers;
            return blocks_for_all_layers;
        }
        // should not be reachable due to the can_allocate_blocks assert in the beginning
        return {};
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

    /**
     * @return The percentage of the allocator's free block pool utilization.
     */
    float get_used_percentage() const {
        size_t sum = 0;
        for (size_t layer_idx = 0; layer_idx < m_num_layers; layer_idx++) sum += num_free_blocks(layer_idx);
        return static_cast<float>(m_num_layers * m_total_num_blocks - sum) / (m_num_layers * m_total_num_blocks) * 100;
    }

    /**
     * @return The total number of KV blocks .
     */
    size_t get_total_number_of_kv_blocks() const {
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
 * @brief Works with `ov::genai::SequenceGroup`s and individual `ov::genai::Sequence`s to assign KV cache blocks to these
 * at each pipeline generation step. A block table is kept for each sequence, storing the indices of "physical"
 * KV cache blocks currently allocated to a given sequence. Each block table defines a linear "logical" block space, with positions of
 * blocks within the block table being associated with "logical" block indices.
 */
class BlockManager {
    friend class CacheStateDumper;
    BlockAllocator m_allocator;
    bool m_enable_prefix_caching;
    size_t m_block_size;
    size_t m_num_layers;
    // TODO: caching time can probably be improved if we use the prefix tree
    std::map<uint64_t, BlocksPerLayer> m_prefix_hash_to_occupied_block_map;

    // stores blocks for each sequence (not sequence group)
    // the same block can be seen in multiple block_tables for different sequences
    std::map<uint64_t, std::vector<BlocksPerLayer>> m_block_table;

    std::mutex m_cached_blocks_map_mutex;
public:
    /**
     * Constructs the BlockManager.
     * @param num_blocks Number of KV cache blocks available for assignment to the sequences.
     * @param enable_prefix_caching Whether prefix caching should be enabled for this allocator.
     * See also the equivalent parameter in ov::genai::ContinuousBatchingPipeline
     * @param block_size The size of an individual KV cache block in tokens.
     * @param num_layers The number of separate attention layers with KV caches in the LLM associated with the pipeline.
     * In current implementation each layer must have the same number of logical blocks allocated at all times.
     */
    BlockManager(int num_blocks, bool enable_prefix_caching, size_t block_size, size_t num_layers = 1)
        : m_allocator(num_blocks, enable_prefix_caching, num_layers), m_enable_prefix_caching(enable_prefix_caching), m_block_size(block_size),
        m_num_layers(num_layers) {
        OPENVINO_ASSERT(num_layers != 0, "num_layers must be non-zero");
    }

    ~BlockManager() {
        // sanity check that all sequences are freed
        OPENVINO_ASSERT(m_block_table.empty());
    }

    /**
     * Gets the block table for a given sequence.
     * @param seq_id The identifier of an ov::genai::Sequence.
     * @return A vector of per-layer blocks occupied by this sequence.
     * Per-layer blocks are themselves each a vector with one KV cache block per layer.
     */
    const std::vector<BlocksPerLayer>& get_block_tables(uint64_t seq_id) const {
        return m_block_table.at(seq_id);
    }

    /**
     * Gets the block table for a given sequence and given layer.
     * @param seq_id The identifier of an ov::genai::Sequence.
     * @param layer_idx The index of a layer.
     * @return A vector of blocks (one for each layer) occupied by this sequence for this layer.
     */
    const std::vector<KVCacheBlock::Ptr>& get_block_table(uint64_t seq_id, size_t layer_idx) {
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
     * @return Number of blocks freed in each sequence in the group.
     */
    const size_t free_group_partially(SequenceGroup::Ptr sequence_group, size_t num_required_blocks) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        size_t blocks_num = std::ceil(num_required_blocks / sequence_group->get_not_finished_sequences().size());
        auto not_finished_sequences = sequence_group->get_not_finished_sequences();
        for (size_t idx = 0; idx < not_finished_sequences.size(); ++idx) {
            auto seq_id = not_finished_sequences[idx]->get_id();
            OPENVINO_ASSERT(m_block_table.count(seq_id) > 0, "Invalid sequence group.");
            free_sequence_partially(seq_id, blocks_num);
        }
        return blocks_num;
    }

    const size_t free_last_block_from_each_sequence(SequenceGroup::Ptr sequence_group) {
        size_t blocks_released = 0;
        auto not_finished_sequences = sequence_group->get_not_finished_sequences();
        for (size_t idx = 0; idx < not_finished_sequences.size(); ++idx) {
            auto seq_id = not_finished_sequences[idx]->get_id();
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
        m_allocator.free(blocks_to_free);
        for (size_t layer_idx = 0; layer_idx < m_num_layers; layer_idx++) {
            block_table[layer_idx].resize(block_table[layer_idx].size() - 1);
        }

        if (block_table[0].size() == 0) {
            OPENVINO_ASSERT(m_block_table.erase(seq_id) == 1);
         }
        return blocks_to_free[0]->is_free();
    }

    const size_t free_partially_beam_search_group(SequenceGroup::Ptr sequence_group, size_t num_required_blocks) {
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
        return logical_blocks_released;
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
     * @param seq_id The identifier of an ov::genai::Sequence
     * @return Whether or not this BlockManager is managing this sequence group.
     */
    const bool has_block_table(uint64_t seq_id) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        return m_block_table.count(seq_id) > 0;
    }

    /**
     * @return The number of KV cache blocks available to be assigned to new sequences.
     */
    size_t num_free_blocks() const {
        return m_allocator.num_free_blocks(0); // relying on the invariant that all layers have identical number of blocks
    }

    /**
     * @param num_blocks A number of KV cache blocks
     * @return Whether this number of KV cache blocks may be assigned to new sequences.
     */
    bool can_allocate_blocks(size_t num_blocks) const {
        for (size_t layer_idx = 0; layer_idx < m_num_layers; layer_idx++) {
            if (!m_allocator.can_allocate_blocks(num_blocks, layer_idx)) return false;
        }
        return true;
    }

    /**
     * Allocates a given number of KV cache blocks to a given sequence.
     * @param sequence The sequence for the blocks to be allocated to.
     * @param num_blocks The number of KV cache blocks to be allocated.
     * @param prompt_size Prompt size for this sequence.
     */
    void allocate(ov::genai::Sequence::Ptr sequence, size_t num_blocks, size_t prompt_size = 0) {
        OPENVINO_ASSERT(num_blocks > 0 && can_allocate_blocks(num_blocks));

        auto sequence_id = sequence->get_id();
        if (m_block_table.find(sequence_id) == m_block_table.end()) {
            m_block_table[sequence_id].resize(m_num_layers);
        }

        auto& block_table = m_block_table[sequence_id][0];
        auto content_length = sequence->get_generated_len() + prompt_size;
        size_t allocated_blocks = block_table.size(); // assuming all layers have the same number of allocated blocks
        size_t num_hashed_tokens = allocated_blocks * m_block_size;


        if (!m_enable_prefix_caching) {
            for (size_t layer_idx = 0; layer_idx < m_block_table[sequence_id].size(); layer_idx++) {
                auto& block_table = m_block_table[sequence_id][layer_idx];
                for (size_t i = 0; i < num_blocks; ++i) {
                    ov::genai::KVCacheBlock::Ptr block = m_allocator.allocate_block(layer_idx);
                    OPENVINO_ASSERT(block != nullptr);
                    m_block_table[sequence_id][layer_idx].push_back(block);
                }
            }
        } else {
            // If last block was restored from cache by using of a partially filled block,
            // its hash would correspond to partially filled block.
            // In this case hash needs to be updated to the hash of fully filled block.
            if (block_table.size() > 0) {
                KVCacheBlock::Ptr last_block = block_table.back();
                auto hash = sequence->get_hash(block_table.size() * m_block_size);
                auto prev_hash = last_block->get_hash();
                if (prev_hash != hash) {
                    BlocksPerLayer last_blocks_vec;
                    last_blocks_vec.reserve(m_num_layers);
                    for (size_t layer_idx = 0; layer_idx < m_num_layers; layer_idx++) {
                        auto& lst_blk = m_block_table[sequence_id][layer_idx].back();
                        lst_blk->set_hash(hash);
                        m_prefix_hash_to_occupied_block_map.erase(prev_hash);
                        last_blocks_vec.push_back(lst_blk);
                    }
                    m_prefix_hash_to_occupied_block_map[hash] = last_blocks_vec;
                }
            }
            for (size_t i = 0; i < num_blocks; ++i) {
                num_hashed_tokens += m_block_size;
                if (num_hashed_tokens > content_length) {
                    num_hashed_tokens = content_length;
                }
                auto hash = sequence->get_hash(num_hashed_tokens);
                auto blocks_for_all_layers = m_allocator.allocate_block(hash, m_prefix_hash_to_occupied_block_map);
                for (size_t layer_idx = 0; layer_idx < blocks_for_all_layers.size(); layer_idx++) {
                    m_block_table[sequence_id][layer_idx].push_back(blocks_for_all_layers[layer_idx]);
                }
            }
        }
    }

    /**
     * @return Percentage of KV cache used by all sequences.
     */
    float get_used_percentage() const {
        return m_allocator.get_used_percentage();
    }

    /**
     * Increases the number of KV blocks.
     * @param num_blocks The new number of KV-blocks.
     */
    void increase_kv_blocks_number(size_t num_blocks) {
        m_allocator.increase_kv_blocks_number(num_blocks);
    }

    /**
     * @return The total number of KV blocks .
     */
    size_t get_total_number_of_kv_blocks() const {
        return m_allocator.get_total_number_of_kv_blocks();
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
        OPENVINO_ASSERT(m_block_table.count(child_id) == 0);
        m_block_table[child_id].resize(m_num_layers);
        for (size_t layer_idx = 0; layer_idx < m_num_layers; layer_idx++) {
            m_block_table[child_id][layer_idx].reserve(m_block_table[parent_id][layer_idx].size());
            for (KVCacheBlock::Ptr &block: m_block_table[parent_id][layer_idx]) {
                block->increment();
                m_block_table[child_id][layer_idx].push_back(block);
            }
        }
    }

    /**
     * @brief Frees all blocks for a given sequence.
     * @param seq_id Identifier of the sequence to free.
     */
    void free_sequence(size_t seq_id) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        OPENVINO_ASSERT(m_block_table.find(seq_id) != m_block_table.end(), "sequence with id ", seq_id,
                        " not found in BlockManager, but requested to free");
        auto& block_table = m_block_table[seq_id];
        size_t effective_num_layers = block_table.size();
        size_t num_allocated_blocks = block_table[0].size();
        for (size_t i = 0; i < num_allocated_blocks; i++) {
            BlocksPerLayer blocks_to_free;
            blocks_to_free.reserve(effective_num_layers);
            for (size_t layer_idx = 0; layer_idx < effective_num_layers; layer_idx++) {
               blocks_to_free.push_back(block_table[layer_idx][i]);
            }
            m_allocator.free(blocks_to_free);
        }

        OPENVINO_ASSERT(m_block_table.erase(seq_id) == 1);
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
            m_allocator.free(blocks_to_free);
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
                    OPENVINO_ASSERT(logical_block_idx <= block_table_size,
                                    "cannot free logical block ", logical_block_idx,
                                    "from sequence ", seq_id, " since it only has ", block_table_size, "logical blocks");
                auto block = per_layer_block_table[logical_block_idx];
                per_layer_cache_blocks_to_free.push_back(block);
            }
            m_allocator.free(per_layer_cache_blocks_to_free);
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
     * @return Whether enough KV cache blocks are available to host the sequences in the group.
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
        std::vector<Sequence::CPtr> running_sequences = seq_group->get_running_sequences();
        size_t blocks_count = 0; // total number of needed blocks for sequence group
        std::set<size_t> last_block_ids; // unique last block indices

        for (auto seq: running_sequences) {
            auto seq_id = seq->get_id();
            if (m_block_table.find(seq_id) == m_block_table.end()) {
                // the block table is empty, so we need to allocate the number of blocks equal to number of logical blocks
                blocks_count += seq_group->get_num_logical_blocks();
                continue;
            }
            auto& block_table = m_block_table[seq_id][0];
            size_t num_physical_blocks = block_table.size();
            OPENVINO_ASSERT(num_physical_blocks > 0);

            if (num_physical_blocks > seq_group->get_num_logical_blocks())
                // new blocks are not required
                // Case when num_physical_blocks == seq_group->get_num_logical_blocks() may still need block allocation
                // (such as when a sequence with an incomplete last block was forked) and is handled further in the
                // iteration
                continue;

            size_t last_block_id = block_table.back()->get_index();

            if (last_block_ids.find(last_block_id) != last_block_ids.end())
                // this block was already processed
                continue;
            last_block_ids.insert(last_block_id);

            size_t needed_blocks_per_sequence = seq_group->get_num_logical_blocks() - num_physical_blocks;

            KVCacheBlock::Ptr last_block = block_table.back();
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
        return blocks_count;
    }

    /**
     * Clean up not busy physical KV cache blocks in a sequence group.
     * @param seq_group Pointer to a sequence group.
     */
    void free_empty_physical_blocks(SequenceGroup::Ptr seq_group) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        size_t num_logical_blocks = seq_group->get_num_logical_blocks();
        if (num_logical_blocks == 0) {
            return;
        }
        for (const auto& sequence : seq_group->get_running_sequences()) {
            auto seq_id = sequence->get_id();
            auto& block_table = m_block_table[seq_id];
            size_t num_physical_blocks = block_table[0].size();
            if (num_physical_blocks > num_logical_blocks) {
                free_sequence_partially(seq_id, num_physical_blocks - num_logical_blocks);
            }
        }
    }


    /**
     * Allocates just enough physical KV cache blocks to a sequence group to be enough for the sequences in it. If the sequences
     * in the group were forked before and their last block is a copy-on-write, then the block contents will have to be copied separately
     * into the freshly allocated block copies as reported in the returned map.
     * @param seq_group Pointer to a sequence group.
     * @return A map where each key is an index of a source *physical* block, and the corresponding value is a list of newly allocated *physical* block
     * indices into which the source block contents should be copied into separately.
     */
    std::map<size_t, std::list<size_t>> append_slots(SequenceGroup::Ptr seq_group) {
        std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        // Will always allocate the identical number of new blocks (if any) to each of the "layers" to keep the
        // number of blocks occupied by each "layer" identical at all times.
        size_t num_logical_blocks = seq_group->get_num_logical_blocks();
        std::vector<Sequence::Ptr> running_sequences = seq_group->get_running_sequences();

        std::map<size_t, std::list<size_t>> copy_blocks_map;
        for (size_t i = 0; i < running_sequences.size(); ++i) {
            Sequence::Ptr sequence = running_sequences[i];
            auto seq_id = sequence->get_id();
            size_t num_physical_blocks = 0;

            if (m_block_table.find(seq_id) != m_block_table.end())
            {
                num_physical_blocks = m_block_table[seq_id][0].size();
            }

            if (num_logical_blocks > num_physical_blocks) {
                OPENVINO_ASSERT(can_allocate_blocks(num_logical_blocks - num_physical_blocks));
                allocate(sequence, num_logical_blocks - num_physical_blocks, seq_group->get_prompt_len());
            } else {
                OPENVINO_ASSERT(num_logical_blocks == num_physical_blocks, "A number of physical and logic blocks must be the same in this code path");

                size_t effective_num_layers = m_block_table[seq_id].size();
                BlocksPerLayer last_blocks;
                last_blocks.reserve(m_block_table[seq_id].size());
                for (size_t i = 0; i < effective_num_layers; i++) {
                    last_blocks.push_back(m_block_table[seq_id][i].back());
                }

                bool is_copy_on_write = last_blocks[0]->copy_on_write();

                if (is_copy_on_write) {
                    BlocksPerLayer new_blocks_for_all_layers;
                    new_blocks_for_all_layers.reserve(effective_num_layers);
                    if (m_enable_prefix_caching) {
                        auto hash = sequence->get_hash();
                        new_blocks_for_all_layers = m_allocator.allocate_block(hash, m_prefix_hash_to_occupied_block_map);
                    } else {
                        for (size_t i = 0; i < effective_num_layers; i++) {
                            new_blocks_for_all_layers.push_back(m_allocator.allocate_block(i));
                        }
                    }

                    for (size_t i = 0; i < effective_num_layers; i++) {
                        auto& new_block = new_blocks_for_all_layers[i];
                        auto& block_table = m_block_table[seq_id][i];
                        block_table[num_physical_blocks - 1] = new_blocks_for_all_layers[i];
                        auto& last_block = last_blocks[i];
                        copy_blocks_map[last_block->get_index()].push_back(new_block->get_index());
                    }
                    m_allocator.free(last_blocks);
                } else {
                    // we are the only users of this block
                    if (m_enable_prefix_caching) {
                        // update hash of block
                        auto prev_hash = last_blocks[0]->get_hash();
                        auto hash = sequence->get_hash();
                        for (size_t i = 0; i < effective_num_layers; i++) {
                            auto& last_block = last_blocks[i];
                            last_block->set_hash(hash);
                        }
                        m_prefix_hash_to_occupied_block_map.erase(prev_hash);
                        m_prefix_hash_to_occupied_block_map[hash] = last_blocks;
                    }
                }
            }
        }

        // it returns information which blocks should be forked by CacheManager
        return copy_blocks_map;
    }

    void restore_cached_blocks(SequenceGroup::Ptr group) {
        // When add_request() is executed in multiple threads accessing to cached_blocks causes segfault.
        // The mutex is needed to prevent such segfaults.
        const std::lock_guard<std::mutex> lock(m_cached_blocks_map_mutex);
        auto prompt_len = group->get_prompt_len();
        auto sequences = group->get_not_finished_sequences();
        OPENVINO_ASSERT(sequences.size() == 1);
        auto sequence = sequences[0];
        auto seq_id = sequence->get_id();

        if (m_block_table.find(seq_id) == m_block_table.end()) {
            m_block_table[seq_id].resize(m_num_layers);
        }
        auto& block_table = m_block_table[seq_id];

        size_t content_len = 0;
        while (content_len < prompt_len) {
            size_t prev_iteration_content_len = content_len;
            content_len += m_block_size;
            if (content_len > prompt_len) {
                content_len = prompt_len;
            }
            // restore fully filled blocks
            auto full_block_hash = sequence->get_hash(content_len);
            auto blocks = m_allocator.get_cached_block(full_block_hash, m_prefix_hash_to_occupied_block_map);
            auto timestamp = std::chrono::steady_clock::now();
            if (!blocks.empty()) {
                for (size_t layer_idx = 0; layer_idx < block_table.size(); layer_idx++) {
                    auto& block = blocks[layer_idx];
                    block->set_timestamp(timestamp);
                    block_table[layer_idx].push_back(block);
                }
                group->update_processed_tokens_num(content_len == prompt_len ? content_len - 1 : content_len);
            } else {
            // restore partially filled block
                for (size_t i = 1; i < m_block_size; i++) {
                    if (prev_iteration_content_len + i > prompt_len) {
                        break;
                    }
                    auto hash = sequence->get_hash(prev_iteration_content_len + i);
                    auto blocks = m_allocator.get_cached_block(hash, m_prefix_hash_to_occupied_block_map);
                    if (!blocks.empty()) {
                        auto timestamp = std::chrono::steady_clock::now();

                        for (size_t layer_idx = 0; layer_idx < block_table.size(); layer_idx++) {
                            auto& block = blocks[layer_idx];
                            block->set_timestamp(timestamp);
                            block_table[layer_idx].push_back(block);
                        }
                        group->update_processed_tokens_num(prev_iteration_content_len + i == prompt_len ? prev_iteration_content_len + i - 1 : prev_iteration_content_len + i);

                        break;
                    }
                }
                break;
            }
        }
    }

    void clear() {
        // KV-cache should not be cleared if prefix caching is enabled
        OPENVINO_ASSERT(m_enable_prefix_caching == false);

        m_allocator.clear();
        m_prefix_hash_to_occupied_block_map.clear();

        // Block tables should be cleared when generation is finished
        OPENVINO_ASSERT(m_block_table.empty());
    }
};


}
