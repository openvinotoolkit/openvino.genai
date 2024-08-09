// Copyright (C) 2023-2024 Intel Corporation
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
    size_t m_num_hashed_tokens;
    std::chrono::time_point<std::chrono::system_clock> m_timestamp;
public:
    using Ptr = std::shared_ptr<KVCacheBlock>;
    using CPtr = std::shared_ptr<const KVCacheBlock>;

    explicit KVCacheBlock(int index)
        : m_ref_count(0),
          m_index(index),
          m_timestamp(std::chrono::system_clock::now()) { }

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

    size_t get_num_hashed_tokens() const {
        return m_num_hashed_tokens;
    }

    void set_hash(size_t hash, size_t num_hashed_tokens) {
        m_hash = hash;
        m_num_hashed_tokens = num_hashed_tokens;
    }

    void set_timestamp(const std::chrono::time_point<std::chrono::system_clock>& timestamp) {
        m_timestamp = timestamp;
    }

    std::chrono::time_point<std::chrono::system_clock> get_timestamp() {
        return m_timestamp;
    }
};


class Evictor {
    std::map<size_t, std::vector<KVCacheBlock::Ptr>> blocks;
    size_t m_num_layers;
public:
    Evictor(size_t num_layers) : m_num_layers(num_layers) {}

    void add(size_t hash, KVCacheBlock::Ptr block) {
        OPENVINO_ASSERT(m_num_layers == 0, "this overload may only be used if num_layers == 0");
        blocks[hash] = { block };
    }

    void add(size_t hash, const std::vector<KVCacheBlock::Ptr>& blocks_for_all_layers) {
        blocks[hash] = blocks_for_all_layers;
    }

    static bool block_is_less(const std::pair<size_t, std::vector<KVCacheBlock::Ptr>>& lhs, const std::pair<size_t, std::vector<KVCacheBlock::Ptr>>& rhs) {
        // assuming in the num_layers > 1 that all registered blocks with the same hash across layers have the same timestamp
        return lhs.second[0]->get_timestamp() < rhs.second[0]->get_timestamp();
    }

    std::vector<KVCacheBlock::Ptr> get_block(size_t hash) {
        if (blocks.find(hash)== blocks.end())
        {
            return {};
        }
        std::vector<KVCacheBlock::Ptr> blocks_for_all_layers = blocks[hash];
        for (auto& block_ptr : blocks_for_all_layers) {
            block_ptr->set_timestamp(std::chrono::system_clock::now());
            block_ptr->increment();
        }
        blocks.erase(hash);
        return blocks_for_all_layers;
    }

    std::vector<KVCacheBlock::Ptr> get_lru_block() {
        if (blocks.empty()) {
            return {};
        }
        auto hash_and_blocks_for_all_layers = std::min_element(std::begin(blocks), std::end(blocks), block_is_less);
        auto blocks_for_all_layers = hash_and_blocks_for_all_layers->second;
        auto timestamp = std::chrono::system_clock::now();
        for (auto& block_ptr : blocks_for_all_layers) {
            block_ptr->set_timestamp(timestamp);
            block_ptr->increment();
        }
        blocks.erase(hash_and_blocks_for_all_layers->first);
        return blocks_for_all_layers;
    }

    size_t num_blocks() const {
        return blocks.size();
    }
};

class CacheStateDumper;

class BlockAllocator {
    std::vector<std::list<KVCacheBlock::Ptr>> m_free_blocks;
    int m_total_num_blocks;
    friend class CacheStateDumper;
    size_t m_num_layers;
    bool m_enable_prefix_caching;
    ov::genai::Evictor m_evictor;
public:
    BlockAllocator(int num_blocks, bool enable_prefix_caching, size_t num_layers = 0) :
        m_total_num_blocks(num_blocks), m_num_layers(num_layers), m_enable_prefix_caching(enable_prefix_caching), m_evictor(num_layers) {
        if (m_num_layers == 0) {
             m_free_blocks.resize(1);
        }
        else {
            m_free_blocks.resize(m_num_layers);
        }
        for (auto& per_layer_block_list : m_free_blocks) {
            for (int block_id = 0; block_id < m_total_num_blocks; ++block_id) {
                per_layer_block_list.push_back(std::make_shared<KVCacheBlock>(block_id));
            }
        }
    }

    ~BlockAllocator() {
        // sanity check to validate that all blocks are freed
        // OPENVINO_ASSERT(m_total_num_blocks == m_free_blocks.size());
    }

    size_t num_free_blocks() const {
        OPENVINO_ASSERT(m_num_layers == 0, "this overload may only be used when num_layers == 0");
        return num_free_blocks(0);
    }

    size_t num_free_blocks(size_t layer_idx) const {
        return m_free_blocks[layer_idx].size() + m_evictor.num_blocks();
    }

    bool can_allocate_blocks(size_t num_blocks) const {
        OPENVINO_ASSERT(m_num_layers == 0, "this overload may only be used when num_layers == 0");
        return can_allocate_blocks(num_blocks, 0);
    }

    bool can_allocate_blocks(size_t num_blocks, size_t layer_idx) const {
        return num_blocks <= num_free_blocks(layer_idx);
    }

    void free(KVCacheBlock::Ptr block) {
        OPENVINO_ASSERT(m_num_layers == 0, "this overload may only be used when num_layers == 0");
        return free(std::vector<KVCacheBlock::Ptr>{block});
    }

    void free(const std::vector<KVCacheBlock::Ptr>& blocks_for_all_layers) {
        size_t effective_num_layers = m_num_layers != 0 ? m_num_layers : 1;
        OPENVINO_ASSERT(blocks_for_all_layers.size() == effective_num_layers);
        for (size_t i = 0; i < effective_num_layers; i++) {
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
                bool is_all_have_same_hash = std::equal(blocks_for_all_layers.begin(), blocks_for_all_layers.begin() + 1, blocks_for_all_layers.end(),
                        [](const KVCacheBlock::Ptr& lhs, const KVCacheBlock::Ptr& rhs) { return lhs->get_hash() == rhs->get_hash(); });
                if (is_all_have_same_hash) {
                    m_evictor.add(blocks_for_all_layers[0]->get_hash(), blocks_for_all_layers);
                } else {
                    // This set of blocks to be freed corresponds to blocks from different time steps, and thus not eligible for caching
                    for (size_t layer_idx = 0; layer_idx < blocks_for_all_layers.size(); layer_idx++) {
                        m_free_blocks[layer_idx].push_back(blocks_for_all_layers[layer_idx]);
                    }
                }
            }
            else {
                for (size_t layer_idx = 0; layer_idx < blocks_for_all_layers.size(); layer_idx++) {
                    m_free_blocks[layer_idx].push_back(blocks_for_all_layers[layer_idx]);
                }
            }
        }
    }


    KVCacheBlock::Ptr allocate_block() {
       OPENVINO_ASSERT(m_num_layers == 0, "this overload may only be used when num_layers == 0");
       return allocate_block(0);
    }

    KVCacheBlock::Ptr allocate_block(size_t layer_idx) {
        OPENVINO_ASSERT(!m_enable_prefix_caching);
        OPENVINO_ASSERT(can_allocate_blocks(1, layer_idx));
        KVCacheBlock::Ptr allocated_block = m_free_blocks[layer_idx].front();
        allocated_block->increment();
        m_free_blocks[layer_idx].pop_front();
        return allocated_block;
    }

    KVCacheBlock::Ptr allocate_block(size_t hash, size_t num_hashed_tokens, std::map<uint64_t, KVCacheBlock::Ptr>& cached_blocks) {
        OPENVINO_ASSERT(m_num_layers == 0, "this overload may only be used when num_layers == 0");
        std::map<uint64_t, std::vector<KVCacheBlock::Ptr>> expanded_map;
        for (auto& entry : cached_blocks) {
            expanded_map[entry.first] = { entry.second };
        }
        return allocate_block(hash, num_hashed_tokens, expanded_map)[0];
    }

    std::vector<KVCacheBlock::Ptr> allocate_block(size_t hash, size_t num_hashed_tokens, std::map<uint64_t, std::vector<KVCacheBlock::Ptr>>& cached_blocks) {
        OPENVINO_ASSERT(m_enable_prefix_caching);
        OPENVINO_ASSERT(can_allocate_blocks(1));
        auto blocks_for_all_layers = m_evictor.get_block(hash);
        if (!blocks_for_all_layers.empty()) {
            // use cached block from evictor
            cached_blocks[hash] = blocks_for_all_layers;
            return blocks_for_all_layers;
        }
        // TODO: Currently we cache all allocated blocks which might be redundant for beam search,
        // where blocks of non-used candidates are not needed in cache.
        // This part can be improved if we cache only blocks for prompt.
        if (cached_blocks.find(hash) != cached_blocks.end()) {
            // use cashed block from cached_blocks
            blocks_for_all_layers = cached_blocks[hash];
            auto& cached_blocks_for_all_layers = cached_blocks[hash];
            for (auto& block_ptr : cached_blocks_for_all_layers) {
                block_ptr->increment();
            }
            return blocks_for_all_layers;
        }

        size_t effective_num_layers = m_num_layers ? m_num_layers : 1;
        if (m_free_blocks[0].size() > 0) {
            // allocate new empty block
            std::vector<KVCacheBlock::Ptr> allocated_blocks;
            allocated_blocks.reserve(effective_num_layers);
            for (size_t i = 0; i < effective_num_layers; i++) {
                KVCacheBlock::Ptr allocated_block = m_free_blocks[i].front();
                allocated_block->increment();
                allocated_block->set_hash(hash, num_hashed_tokens);
                m_free_blocks[i].pop_front();
            }
            cached_blocks[hash] = allocated_blocks;
            return allocated_blocks;
        }
        if (m_evictor.num_blocks() > 0) {
            // get least resently used block from evictor and reuse it
            std::vector<KVCacheBlock::Ptr> blocks_for_all_layers = m_evictor.get_lru_block();
            cached_blocks.erase(blocks_for_all_layers[0]->get_hash());

            // update block with new hash
            for (auto& block : blocks_for_all_layers) {
                block->set_hash(hash, num_hashed_tokens);
            }
            cached_blocks[hash] = blocks_for_all_layers;
            return blocks_for_all_layers;
        }
        // out of memory
        return {};
    }

    KVCacheBlock::Ptr get_cached_block(size_t hash, std::map<uint64_t, KVCacheBlock::Ptr>& cached_blocks) {
        OPENVINO_ASSERT(m_num_layers == 0, "this overload may only be used when num_layers == 0");
        std::map<uint64_t, std::vector<KVCacheBlock::Ptr>> expanded_map;
        for (auto& entry : cached_blocks) {
            expanded_map[entry.first] = { entry.second };
        }
        return get_cached_block(hash, expanded_map)[0];
    }

    std::vector<KVCacheBlock::Ptr> get_cached_block(size_t hash, std::map<uint64_t, std::vector<KVCacheBlock::Ptr>>& cached_blocks) {
        auto blocks_for_all_layers = m_evictor.get_block(hash);
        if (!blocks_for_all_layers.empty()) {
            // use cached block from evictor
            cached_blocks[hash] = blocks_for_all_layers;
            return blocks_for_all_layers;
        }
        if (cached_blocks.find(hash) != cached_blocks.end()) {
            // use cached block from cached_blocks
            // TODO: add tokens validation in case of hash collision
            blocks_for_all_layers = cached_blocks[hash];
            for (auto& block_ptr : cached_blocks[hash]) {
                block_ptr->increment();
            }
            return blocks_for_all_layers;
        }
        return {};
    }

    float get_used_percentage() const {
        std::cout << "VSHAMPOR: m_total_num_blocks " << m_total_num_blocks << std::endl;
        if (m_num_layers == 0) {
            return static_cast<float>(m_total_num_blocks - num_free_blocks()) / m_total_num_blocks;
        }
        else {
            size_t sum = 0;
            for (size_t layer_idx = 0; layer_idx < m_num_layers; layer_idx++) sum += num_free_blocks(layer_idx);
            return static_cast<float>(m_num_layers * m_total_num_blocks - sum) / m_total_num_blocks;
        }
    }
};

class BlockManager {
    friend class CacheStateDumper;
    BlockAllocator m_allocator;
    bool m_enable_prefix_caching;
    size_t m_block_size;
    size_t m_num_layers;
    // TODO: caching time can probably be improved if we use the prefix tree
    std::map<uint64_t, std::vector<KVCacheBlock::Ptr>> cached_blocks;

    // stores blocks for each sequence (not sequence group)
    // the same block can be seen in multiple block_tables for different sequences
    std::map<uint64_t, std::vector<std::vector<KVCacheBlock::Ptr>>> m_block_table;
public:
    BlockManager(int num_blocks, bool enable_prefix_caching, size_t block_size, size_t num_layers = 0)
        : m_allocator(num_blocks, enable_prefix_caching, num_layers), m_enable_prefix_caching(enable_prefix_caching), m_block_size(block_size),
        m_num_layers(num_layers) { }

    ~BlockManager() {
        // sanity check that all sequences are freed
        // OPENVINO_ASSERT(m_block_table.empty());
    }

    const std::vector<KVCacheBlock::Ptr>& get_block_table(uint64_t seq_id) {
        OPENVINO_ASSERT(m_num_layers == 0, "this overload may only be used when num_layers == 0");
        return get_block_table(seq_id, 0);
    }

    const std::vector<std::vector<KVCacheBlock::Ptr>>& get_block_tables(uint64_t seq_id) {
        return m_block_table[seq_id];
    }

    const std::vector<KVCacheBlock::Ptr>& get_block_table(uint64_t seq_id, size_t layer_idx) {
        OPENVINO_ASSERT(m_block_table.count(seq_id) == 1);
        return m_block_table[seq_id][layer_idx];
    }

    const size_t free_group_partially(SequenceGroup::Ptr sequence_group, size_t num_required_blocks) {
        size_t blocks_num = std::ceil(num_required_blocks / sequence_group->get_not_finished_sequences().size());
        auto running_sequences = sequence_group->get_not_finished_sequences();
        std::set<size_t> blocks_released_indices;
        for (size_t idx = 0; idx < running_sequences.size(); ++idx) {
            auto seq_id = running_sequences[idx]->get_id();
            OPENVINO_ASSERT(m_block_table.count(seq_id) > 0, "Invalid sequence group.");
            free_sequence_partially(seq_id, blocks_num);
        }
        return blocks_num;
    }

    const size_t get_number_of_blocks_occupied_by_sequence(SequenceGroup::Ptr sequence_group) {
        auto running_sequences = sequence_group->get_not_finished_sequences();
        size_t num_blocks = 0;
        std::set<size_t> indices;
        for (size_t idx = 0; idx < running_sequences.size(); ++idx) {
            auto seq_id = running_sequences[idx]->get_id();
            if (m_block_table.count(seq_id) == 0) {
                continue;
            }
           // OPENVINO_ASSERT(m_block_table.count(seq_id) > 0, "Invalid sequence group.");
            auto block_table = m_block_table[seq_id][0];  // assuming all layers always have equal number of blocks
            size_t last_idx = block_table.back()->get_index();
            if (indices.find(last_idx) != indices.end()) {
                continue;
            }
            else {
                indices.insert(last_idx);
                num_blocks += block_table.size();
            }
        }
        return num_blocks;
    }

    const bool has_block_table(uint64_t seq_id) {
        return m_block_table.count(seq_id) > 0;
    }

    size_t num_free_blocks() const {
        return m_allocator.num_free_blocks(0); // relying on the invariant that all layers have identical number of blocks
    }

    bool can_allocate_blocks(size_t num_blocks) const {
        if (m_num_layers == 0) {
            return m_allocator.can_allocate_blocks(num_blocks);
        }
        else {
            for (size_t layer_idx = 0; layer_idx < m_num_layers; layer_idx++) {
                if (!m_allocator.can_allocate_blocks(num_blocks, layer_idx)) return false;
            }
            return true;
        }
    }


    void allocate(ov::genai::Sequence::CPtr sequence, size_t num_blocks, const ov::genai::TokenIds& prompt_ids = {}) {
        OPENVINO_ASSERT(num_blocks > 0 && can_allocate_blocks(num_blocks));
        if (m_enable_prefix_caching) {
            OPENVINO_ASSERT(prompt_ids.size() > 0, "prompt_ids should be set for hash calculation.");
        }
        auto sequence_id = sequence->get_id();
        size_t effective_num_layers = m_num_layers != 0 ? m_num_layers : 1;
        if (m_block_table.find(sequence_id) == m_block_table.end()) {
            m_block_table[sequence_id].resize(m_num_layers);
        }

        auto content_length = sequence->get_generated_len() + prompt_ids.size();
        size_t allocated_blocks = m_block_table[sequence_id][0].size(); // assuming all layers have the same number of allocated blocks
        size_t num_hashed_tokens = allocated_blocks * m_block_size;

        if (!m_enable_prefix_caching) {
            for (size_t layer_idx = 0; layer_idx < m_block_table[sequence_id].size(); layer_idx++) {
                auto block_table = m_block_table[sequence_id][layer_idx];
                for (size_t i = 0; i < num_blocks; ++i) {
                    ov::genai::KVCacheBlock::Ptr block = m_allocator.allocate_block(layer_idx);
                    OPENVINO_ASSERT(block != nullptr);
                    m_block_table[sequence_id][layer_idx].push_back(block);
                }
            }
        } else {
            num_hashed_tokens += m_block_size;
            if (num_hashed_tokens > content_length) {
                num_hashed_tokens = content_length;
            }
            auto hash = sequence->get_hash(num_hashed_tokens, prompt_ids);
            auto blocks_for_all_layers = m_allocator.allocate_block(hash, num_hashed_tokens, cached_blocks);
            for (size_t i = 0; i < blocks_for_all_layers.size(); i++) {
                m_block_table[sequence_id][i].push_back(blocks_for_all_layers[i]);
            }
        }
    }

    float get_used_percentage() const {
        return m_allocator.get_used_percentage();
    }

    void fork_sequence(uint64_t parent_id, uint64_t child_id) {
        OPENVINO_ASSERT(m_block_table.count(child_id) == 0);
        size_t effective_num_layers = m_num_layers != 0 ? m_num_layers : 1;
        m_block_table[child_id].resize(effective_num_layers);
        for (size_t layer_idx = 0; layer_idx < effective_num_layers; layer_idx++) {
            m_block_table[child_id][layer_idx].reserve(m_block_table[parent_id].size());
            for (KVCacheBlock::Ptr &block: m_block_table[parent_id][layer_idx]) {
                block->increment();
                m_block_table[child_id][layer_idx].push_back(block);
            }
        }
    }

    void free_sequence(size_t seq_id) {
        if (m_block_table.find(seq_id) == m_block_table.end()) {
            return;
        }
        auto& block_table = m_block_table[seq_id];
        size_t effective_num_layers = block_table.size();
        size_t num_allocated_blocks = block_table[0].size();
        for (size_t i = 0; i < num_allocated_blocks; i++) {
            std::vector<KVCacheBlock::Ptr> blocks_to_free;
            blocks_to_free.reserve(effective_num_layers);
            for (size_t layer_idx = 0; layer_idx < effective_num_layers; layer_idx++) {
               blocks_to_free.push_back(block_table[layer_idx][i]);
            }
            m_allocator.free(blocks_to_free);
        }

        OPENVINO_ASSERT(m_block_table.erase(seq_id) == 1);
    }

    void free_sequence_partially(size_t seq_id, size_t block_num) {
        size_t effective_num_layers = m_block_table[seq_id].size();
        for (size_t layer_idx = 0; layer_idx < effective_num_layers; layer_idx++) {
            auto& layer_block_table = m_block_table[seq_id][layer_idx];
            OPENVINO_ASSERT(layer_block_table.size() >= block_num);
        }

        for (size_t idx = 0; idx < block_num; idx++) {
            std::vector<KVCacheBlock::Ptr> blocks_to_free;
            blocks_to_free.reserve(effective_num_layers);
            for (size_t layer_idx = 0; layer_idx < effective_num_layers; layer_idx++) {
                auto& layer_block_table = m_block_table[seq_id][layer_idx];
                size_t block_idx = layer_block_table.size() - idx - 1;
                blocks_to_free.push_back(layer_block_table[block_idx]);
            }
            m_allocator.free(blocks_to_free);
            for (size_t layer_idx = 0; layer_idx < effective_num_layers; layer_idx++) {
                auto& layer_block_table = m_block_table[seq_id][layer_idx];
                layer_block_table.resize(layer_block_table.size() - block_num);
            }

        }

        bool sequence_freed_completely = false;
        auto empty_predicate = [](const std::vector<KVCacheBlock::Ptr>& v) { return v.empty(); };
        bool any_freed_completely = std::any_of(m_block_table[seq_id].begin(), m_block_table[seq_id].end(), empty_predicate);
        if (any_freed_completely) {
            bool all_freed_completely = std::all_of(m_block_table[seq_id].begin(), m_block_table[seq_id].end(), empty_predicate);
            // The invariant must hold at BlockManager level that all per-layer block tables
            // must have the same size
            OPENVINO_ASSERT(all_freed_completely, "block tables across layers should only be empty all at once");
            OPENVINO_ASSERT(m_block_table.erase(seq_id) == 1);
        }
    }


    void free_blocks_from_sequence(size_t seq_id, const std::set<size_t>& logical_block_indices_to_free) {
        OPENVINO_ASSERT(m_num_layers == 0, "this overload may only be called if num_layers == 0");
        std::vector<std::set<size_t>> expanded_input = { logical_block_indices_to_free };
        return free_blocks_from_sequence(seq_id, expanded_input);
    }


    void free_blocks_from_sequence(size_t seq_id, const std::vector<std::set<size_t>>& logical_block_index_sets_to_free) {
        std::vector<std::vector<size_t>> logical_block_indices_to_free(logical_block_index_sets_to_free.size());
        for (size_t i = 0; i < logical_block_index_sets_to_free.size(); i++) {
            const auto& index_set = logical_block_index_sets_to_free[i];
            auto& index_vector = logical_block_indices_to_free[i];
            index_vector.resize(index_set.size());
            std::copy(index_set.begin(), index_set.end(), index_vector.begin());
        }

        size_t presumed_num_layers = logical_block_indices_to_free.size();
        OPENVINO_ASSERT(m_num_layers == presumed_num_layers || (m_num_layers == 0 && presumed_num_layers == 1));
        for (size_t i = 0; i < presumed_num_layers; i++) {
            OPENVINO_ASSERT(logical_block_indices_to_free[i].size() == logical_block_indices_to_free[0].size(), "must free the same amount of blocks per each layer at once");
        }

        if (logical_block_indices_to_free[0].empty()) {
            return;
        }

        size_t num_blocks_to_free = logical_block_indices_to_free[0].size();

        // free blocks at the allocator level
        for (size_t block_idx = 0; block_idx < num_blocks_to_free; block_idx++) {
            std::vector<KVCacheBlock::Ptr> per_layer_cache_blocks_to_free;
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
            std::vector<KVCacheBlock::Ptr> new_sequence_blocks;
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

    bool can_append_slots(SequenceGroup::CPtr seq_group) {
        return required_blocks_count(std::move(seq_group)) <= m_allocator.num_free_blocks(0);
    }

    size_t required_blocks_count(SequenceGroup::CPtr seq_group) {
        std::vector<Sequence::CPtr> running_sequences = seq_group->get_running_sequences();
        size_t blocks_count= 0; // total number of needed blocks for sequence group
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

            if (num_physical_blocks >= seq_group->get_num_logical_blocks(seq_id))
                // new blocks are not required
                continue;

            size_t last_block_id = block_table.back()->get_index();

            if (last_block_ids.find(last_block_id) != last_block_ids.end()) 
                // this block was already processed
                continue;
            last_block_ids.insert(last_block_id);

            size_t needed_blocks_per_sequence = seq_group->get_num_logical_blocks(seq_id) - num_physical_blocks;

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

    std::map<size_t, std::list<size_t>> append_slots(SequenceGroup::CPtr seq_group) {
        // Will always allocate the identical number of new blocks (if any) to each of the "layers" to keep the
        // number of blocks occupied by each "layer" identical at all times.
        size_t num_logical_blocks = seq_group->get_num_logical_blocks();
        std::vector<Sequence::CPtr> running_sequences = seq_group->get_running_sequences();

        std::map<size_t, std::list<size_t>> copy_blocks_map;
        for (auto& sequence : running_sequences) {
            auto seq_id = sequence->get_id();
            size_t effective_num_layers = m_block_table[seq_id].size();
            size_t num_physical_blocks = m_block_table[seq_id][0].size();

            if (num_logical_blocks > num_physical_blocks) {
                OPENVINO_ASSERT(can_allocate_blocks(num_logical_blocks - num_physical_blocks));
                allocate(sequence, num_logical_blocks - num_physical_blocks, seq_group->get_prompt_ids());
            } else {
                OPENVINO_ASSERT(num_logical_blocks == num_physical_blocks, "A number of physical and logic blocks must be the same in this code path");
                std::vector<KVCacheBlock::Ptr> last_blocks;
                last_blocks.reserve(m_block_table[seq_id].size());
                for (size_t i = 0; i < effective_num_layers; i++) {
                    last_blocks.push_back(m_block_table[seq_id][i].back());
                }

                bool is_copy_on_write = last_blocks[0]->copy_on_write();

                if (is_copy_on_write) {
                    std::vector<KVCacheBlock::Ptr> new_blocks_for_all_layers;
                    new_blocks_for_all_layers.reserve(effective_num_layers);
                    if (m_enable_prefix_caching) {
                        auto hash = sequence->get_hash(seq_group->get_context_len(), seq_group->get_prompt_ids());
                        new_blocks_for_all_layers = m_allocator.allocate_block(hash, seq_group->get_context_len(), cached_blocks);
                        cached_blocks[hash] = new_blocks_for_all_layers;
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
                        auto hash = sequence->get_hash(seq_group->get_context_len(), seq_group->get_prompt_ids());
                        for (size_t i = 0; i < effective_num_layers; i++) {
                            auto& last_block = last_blocks[i];
                            last_block->set_hash(hash, seq_group->get_context_len());
                        }
                        cached_blocks.erase(prev_hash);
                        cached_blocks[hash] = last_blocks;
                    }
                }
            }
        }

        // it returns information which blocks should be forked by CacheManager
        return copy_blocks_map;
    }


    void _restore_cached_blocks(SequenceGroup::Ptr group, size_t block_size) {
        auto prompt_ids = group->get_prompt_ids();
        auto sequences = group->get_not_finished_sequences();
        OPENVINO_ASSERT(sequences.size() == 1);
        auto sequence = sequences[0];
        auto seq_id = sequence->get_id();
        auto& block_table = m_block_table[seq_id];

        size_t content_len = 0;
        while (content_len < prompt_ids.size()) {
            size_t prev_iteration_content_len = content_len;
            content_len += block_size;
            if (content_len > prompt_ids.size()) {
                content_len = prompt_ids.size();
            }
            // restore fully filled blocks
            auto hash = sequence->get_hash(content_len, prompt_ids);
            auto blocks = m_allocator.get_cached_block(hash, cached_blocks);
            auto timestamp = std::chrono::system_clock::now();
            if (!blocks.empty()) {
                for (size_t layer_idx = 0; layer_idx < m_block_table[seq_id].size(); layer_idx++) {
                    auto& block = blocks[layer_idx];
                    block->set_timestamp(timestamp);
                    m_block_table[seq_id][layer_idx].push_back(block);
                }
                group->update_processed_tokens_num(content_len);
            } else {
                // restore partially filled block
                for (size_t i = 1; i < block_size; i++) {
                    if (prev_iteration_content_len + i > prompt_ids.size()) {
                        break;
                    }
                    auto hash = sequence->get_hash(prev_iteration_content_len + i, prompt_ids);
                    auto blocks = m_allocator.get_cached_block(hash, cached_blocks);
                    if (!blocks.empty()) {
                        for (size_t layer_idx = 0; layer_idx < m_block_table[seq_id].size(); layer_idx++) {
                            auto& block = blocks[layer_idx];
                            block->set_timestamp(std::chrono::system_clock::now());
                            m_block_table[seq_id][layer_idx].push_back(block);
                        }

                        group->update_processed_tokens_num(prev_iteration_content_len + i);

                        size_t new_tokens_count_in_block = std::min(content_len,
                                                                    prev_iteration_content_len + block_size);
                        if (new_tokens_count_in_block > prev_iteration_content_len + i) {
                            cached_blocks.erase(hash);
                            auto new_hash = sequence->get_hash(new_tokens_count_in_block, prompt_ids);
                            cached_blocks[new_hash] = blocks;
                        }

                        break;
                    }
                }
                break;
            }
        }
    }
};


}
