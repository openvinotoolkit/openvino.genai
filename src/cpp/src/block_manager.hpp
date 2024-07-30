// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <list>
#include <map>
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
    std::map<size_t, KVCacheBlock::Ptr> blocks;
public:
    void add(size_t hash, KVCacheBlock::Ptr block) {
        blocks[hash] = block;
    }

    static bool block_is_less(const std::pair<size_t, KVCacheBlock::Ptr>& lhs, const std::pair<size_t, KVCacheBlock::Ptr>& rhs) {
        return lhs.second->get_timestamp() < rhs.second->get_timestamp();
    }

    KVCacheBlock::Ptr get_block(size_t hash) {
        if (blocks.find(hash)== blocks.end())
        {
            return nullptr;
        }
        KVCacheBlock::Ptr block = blocks[hash];
        block->set_timestamp(std::chrono::system_clock::now());
        block->increment();
        blocks.erase(hash);
        return block;
    }

    KVCacheBlock::Ptr get_lru_block() {
        if (!blocks.size()) {
            return nullptr;
        }
        auto hash_block = std::min_element(std::begin(blocks), std::end(blocks), block_is_less);
        auto block = hash_block->second;
        block->set_timestamp(std::chrono::system_clock::now());
        block->increment();
        blocks.erase(hash_block->first);
        return block;
    }

    size_t num_blocks() const {
        return blocks.size();
    }
};


class BlockAllocator {
    std::list<KVCacheBlock::Ptr> m_free_blocks;
    ov::genai::Evictor m_evictor;
    int m_total_num_blocks;
    bool m_enable_prefix_caching;
public:
    BlockAllocator(int num_blocks, bool enable_prefix_caching) :
        m_total_num_blocks(num_blocks), m_enable_prefix_caching(enable_prefix_caching) {
        for (int block_id = 0; block_id < m_total_num_blocks; ++block_id) {
            m_free_blocks.push_back(std::make_shared<KVCacheBlock>(block_id));
        }
    }

    ~BlockAllocator() {
        // sanity check to validate that all blocks are freed
        // OPENVINO_ASSERT(m_total_num_blocks == m_free_blocks.size());
    }

    size_t num_free_blocks() const {
        return m_free_blocks.size() + m_evictor.num_blocks();
    }

    bool can_allocate_blocks(size_t num_blocks) const {
        return num_blocks <= num_free_blocks();
    }

    void free(KVCacheBlock::Ptr block) {
        block->release();
        if (block->is_free()) {
            if (m_enable_prefix_caching)
            {
                m_evictor.add(block->get_hash(), block);
            }
            else {
                m_free_blocks.push_back(block);
            }
        }
    }

    KVCacheBlock::Ptr allocate_block() {
        OPENVINO_ASSERT(!m_enable_prefix_caching);
        OPENVINO_ASSERT(can_allocate_blocks(1));
        KVCacheBlock::Ptr allocated_block = m_free_blocks.front();
        allocated_block->increment();
        m_free_blocks.pop_front();
        return allocated_block;
    }

    KVCacheBlock::Ptr allocate_block(size_t hash, size_t num_hashed_tokens, std::map<uint64_t, KVCacheBlock::Ptr>& cached_blocks) {
        OPENVINO_ASSERT(m_enable_prefix_caching);
        OPENVINO_ASSERT(can_allocate_blocks(1));
        auto block = m_evictor.get_block(hash);
        if (block != nullptr) {
            // use cached block from evictor
            cached_blocks[hash] = block;
            return block;
        }
        // TODO: Currently we cache all allocated blocks which might be redundant for beam search,
        // where blocks of non-used candidates are not needed in cache.
        // This part can be improved if we cache only blocks for prompt.
        if (cached_blocks.find(hash) != cached_blocks.end()) {
            // use cashed block from cached_blocks
            block = cached_blocks[hash];
            cached_blocks[hash]->increment();
            return block;
        }
        if (m_free_blocks.size() > 0) {
            // allocate new empty block
            KVCacheBlock::Ptr allocated_block = m_free_blocks.front();
            allocated_block->increment();
            allocated_block->set_hash(hash, num_hashed_tokens);
            cached_blocks[hash] = allocated_block;

            m_free_blocks.pop_front();
            return allocated_block;
        }
        if (m_evictor.num_blocks() > 0) {
            // get least resently used block from evictor and reuse it
            KVCacheBlock::Ptr block = m_evictor.get_lru_block();
            cached_blocks.erase(block->get_hash());

            // update block with new hash
            block->set_hash(hash, num_hashed_tokens);
            cached_blocks[hash] = block;
            return block;
        }
        // out of memory
        return nullptr;
    }

    KVCacheBlock::Ptr get_cached_block(size_t hash, std::map<uint64_t, KVCacheBlock::Ptr>& cached_blocks) {
        auto block = m_evictor.get_block(hash);
        if (block != nullptr) {
            // use cashed block from evictor
            cached_blocks[hash] = block;
            return block;
        }
        if (cached_blocks.find(hash) != cached_blocks.end()) {
            // use cashed block from cached_blocks
            // TODO: add tokens validation in case of hash collision
            block = cached_blocks[hash];
            cached_blocks[hash]->increment();
            return block;
        }
        return nullptr;
    }

    float get_used_percentage() const {
        return static_cast<float>(m_total_num_blocks - num_free_blocks()) / m_total_num_blocks;
    }
};

class BlockManager {
    BlockAllocator m_allocator;
    bool m_enable_prefix_caching;
    size_t m_block_size;
    // TODO: caching time can probably be improved if we use the prefix tree
    std::map<uint64_t, KVCacheBlock::Ptr> cached_blocks;

    // stores blocks for each sequence (not sequence group)
    // the same block can be seen in multiple block_tables for different sequences
    std::map<uint64_t, std::vector<KVCacheBlock::Ptr>> m_block_table;
public:
    BlockManager(int num_blocks, bool enable_prefix_caching, size_t block_size)
        : m_allocator(num_blocks, enable_prefix_caching), m_enable_prefix_caching(enable_prefix_caching), m_block_size(block_size) { }

    ~BlockManager() {
        // sanity check that all sequences are freed
        // OPENVINO_ASSERT(m_block_table.empty());
    }

    const std::vector<KVCacheBlock::Ptr>& get_block_table(uint64_t seq_id) {
        OPENVINO_ASSERT(m_block_table.count(seq_id) == 1);
        return m_block_table[seq_id];
    }

    const size_t free_rightest_blocks(SequenceGroup::Ptr sequence_group) {
        size_t blocks_released = 0;
        auto running_sequences = sequence_group->get_not_finished_sequences();
        std::set<size_t> blocks_released_indices;
        for (size_t idx = 0; idx < running_sequences.size(); ++idx) {
            auto seq_id = running_sequences[idx]->get_id();
            OPENVINO_ASSERT(m_block_table.count(seq_id) > 0, "Invalid sequence group.");
            auto block_table = m_block_table[seq_id];
            if (free_last_block(seq_id)) {
                blocks_released++;
            }
        }
        return blocks_released;
    }

    const bool free_group_partially_multiple_runnning_sequence(SequenceGroup::Ptr sequence_group, size_t num_required_blocks, size_t& phisical_blocks_released, size_t& logical_blocks_released) {
        phisical_blocks_released = 0;
        logical_blocks_released = 0;
        while (num_required_blocks > phisical_blocks_released) {
            size_t released_count = free_rightest_blocks(sequence_group);
            logical_blocks_released += 1;
            if (get_number_of_blocks_occupied_by_sequence(sequence_group) == 0) {
                break;
            }
            phisical_blocks_released += released_count;
        }
        phisical_blocks_released = phisical_blocks_released;
        return num_required_blocks <= phisical_blocks_released;
    }

    const bool free_group_partially_single_runnning_sequence(SequenceGroup::Ptr sequence_group, size_t num_required_blocks, size_t& phisical_blocks_released) {
        auto sequences = sequence_group->get_not_finished_sequences();
        OPENVINO_ASSERT(sequences.size() == 1);
        auto running_sequence = sequences[0];
        auto seq_id = running_sequence->get_id();
        if (!has_block_table(seq_id)) {
            // no blocks are allocated for this sequence, so it can't be preempted
            return false;
        }
        auto block_table = get_block_table(seq_id);
        auto prev_blocks_count = num_free_blocks();
        free_sequence_partially_single_runnning_sequence(seq_id, num_required_blocks);

        // calculate the number of released blocks
        phisical_blocks_released = num_free_blocks() - prev_blocks_count;

        return num_required_blocks <= phisical_blocks_released;
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
            auto block_table = m_block_table[seq_id];
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
        return m_allocator.num_free_blocks();
    }

    bool can_allocate_blocks(size_t num_blocks) const {
        return m_allocator.can_allocate_blocks(num_blocks);
    }

    void allocate(ov::genai::Sequence::CPtr sequence, size_t num_blocks, const ov::genai::TokenIds& prompt_ids = {}) {
        OPENVINO_ASSERT(num_blocks > 0 && can_allocate_blocks(num_blocks));
        if (m_enable_prefix_caching) {
            OPENVINO_ASSERT(prompt_ids.size() > 0, "prompt_ids should be set for hash calculation.");
        }
        auto sequence_id = sequence->get_id();
        auto block_table = m_block_table[sequence_id];
        auto content_length = sequence->get_generated_len() + prompt_ids.size();
        size_t num_hashed_tokens = block_table.size() * m_block_size;

        for (size_t i = 0; i < num_blocks; ++i) {

            ov::genai::KVCacheBlock::Ptr block = nullptr; 
            if (m_enable_prefix_caching) {
                num_hashed_tokens += m_block_size;
                if (num_hashed_tokens > content_length) {
                    num_hashed_tokens = content_length;
                }
                auto hash = sequence->get_hash(num_hashed_tokens, prompt_ids);
                block = m_allocator.allocate_block(hash, num_hashed_tokens, cached_blocks);
            }
            else {
                block = m_allocator.allocate_block();
            }
            OPENVINO_ASSERT(block != nullptr);
            m_block_table[sequence_id].push_back(block);
        }
    }

    float get_used_percentage() const {
        return m_allocator.get_used_percentage();
    }

    void fork_sequence(uint64_t parent_id, uint64_t child_id) {
        OPENVINO_ASSERT(m_block_table.count(child_id) == 0);
        m_block_table[child_id].reserve(m_block_table[parent_id].size());
        for (KVCacheBlock::Ptr & block : m_block_table[parent_id]) {
            block->increment();
            m_block_table[child_id].push_back(block);
        }
    }

    void free_sequence(size_t seq_id) {
        auto block_table = m_block_table[seq_id];

        for (KVCacheBlock::Ptr& block : block_table) {
            m_allocator.free(block);
        }

        OPENVINO_ASSERT(m_block_table.erase(seq_id) == 1);
    }

    bool free_last_block(size_t seq_id) {
        auto block_table = m_block_table[seq_id];
        OPENVINO_ASSERT(block_table.size() >= 1);
        size_t block_idx = m_block_table[seq_id].size() - 1;
        m_allocator.free(block_table[block_idx]);
        m_block_table[seq_id].resize(m_block_table[seq_id].size() - 1);

        if (m_block_table[seq_id].size() == 0) {
            OPENVINO_ASSERT(m_block_table.erase(seq_id) == 1);
        }
        return block_table[block_idx]->is_free();
    }

    void free_sequence_partially_single_runnning_sequence(size_t seq_id, size_t block_num) {
        // this method is applicable only for groups with single sequences

        auto block_table = m_block_table[seq_id];
        OPENVINO_ASSERT(block_table.size() >= block_num);
        for (size_t idx = 0; idx < block_num; idx++) {
            size_t block_idx = m_block_table[seq_id].size() - idx - 1;
            m_allocator.free(block_table[block_idx]);
            OPENVINO_ASSERT(block_table[block_idx]->is_free());
        } 
        m_block_table[seq_id].resize(m_block_table[seq_id].size() - block_num);

        if (m_block_table[seq_id].size() == 0) {
            OPENVINO_ASSERT(m_block_table.erase(seq_id) == 1);
        }
    }

    bool can_append_slots(SequenceGroup::CPtr seq_group) {
        return required_blocks_count(seq_group) <= m_allocator.num_free_blocks();
    }

    size_t required_blocks_count(SequenceGroup::CPtr seq_group) {
        std::vector<Sequence::CPtr> running_sequences = seq_group->get_running_sequences();
        size_t blocks_count= 0; // totat number of needed blocks for sequence group
        std::set<size_t> last_block_ids; // unique last block indices

        for (auto seq: running_sequences) {
            auto seq_id = seq->get_id();
            if (m_block_table.find(seq_id) == m_block_table.end()) {
                // the block table is empty, so we need to allocate the number of blocks equal to number of logical blocks
                blocks_count += seq_group->get_num_logical_blocks();
                continue;
            }
            auto& block_table = m_block_table[seq_id];
            size_t num_physical_blocks = block_table.size();
            OPENVINO_ASSERT(num_physical_blocks > 0);

            if (num_physical_blocks > seq_group->get_num_logical_blocks())
                // new blocks are not required
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

    std::map<size_t, std::list<size_t>> append_slots(SequenceGroup::CPtr seq_group) {

        size_t num_logical_blocks = seq_group->get_num_logical_blocks();
        std::vector<Sequence::CPtr> running_sequences = seq_group->get_running_sequences();

        std::map<size_t, std::list<size_t>> copy_blocks_map;
        for (size_t i = 0; i < running_sequences.size(); ++i) {
            Sequence::CPtr sequence = running_sequences[i];
            auto seq_id = sequence->get_id();
            auto& block_table = m_block_table[seq_id];
            size_t num_physical_blocks = block_table.size();

            if (num_logical_blocks > num_physical_blocks) {
                OPENVINO_ASSERT(can_allocate_blocks(num_logical_blocks - num_physical_blocks));
                allocate(sequence, num_logical_blocks - num_physical_blocks, seq_group->get_prompt_ids());
            } else {
                OPENVINO_ASSERT(num_logical_blocks == num_physical_blocks, "A number of physical and logic blocks must be the same in this code path");
                KVCacheBlock::Ptr last_block = block_table.back();
                if (last_block->copy_on_write()) {
                    // we need to fork current block, because reference counter is more than 1
                    KVCacheBlock::Ptr new_block = nullptr;
                    if (m_enable_prefix_caching) {
                        auto hash = sequence->get_hash(seq_group->get_context_len(), seq_group->get_prompt_ids());
                        new_block = m_allocator.allocate_block(hash, seq_group->get_context_len(), cached_blocks);
                        cached_blocks[hash] = new_block;
                    }
                    else {
                        new_block = m_allocator.allocate_block();
                    }
                    block_table[num_physical_blocks - 1] = new_block;
                    // write information about block forking for later usage in CacheManager
                    copy_blocks_map[last_block->get_index()].push_back(new_block->get_index());
                    // release `last_block` usage
                    m_allocator.free(last_block);
                } else {
                    // we are the only users of this block
                    if (m_enable_prefix_caching) {
                        // update hash of block
                        auto prev_hash = last_block->get_hash();
                        auto hash = sequence->get_hash(seq_group->get_context_len(), seq_group->get_prompt_ids());
                        last_block->set_hash(hash, seq_group->get_context_len());
                        cached_blocks.erase(prev_hash);
                        cached_blocks[hash] = last_block;
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
            auto block = m_allocator.get_cached_block(hash, cached_blocks);
            if (block != nullptr) {
                block->set_timestamp(std::chrono::system_clock::now());
                m_block_table[seq_id].push_back(block);
                group->update_processed_tokens_num(content_len);
            }
            else {
                // restore partially filled block
                for (size_t i = 1; i < block_size; i++) {
                    if (prev_iteration_content_len + i > prompt_ids.size()) {
                        break;
                    }
                    auto hash = sequence->get_hash(prev_iteration_content_len + i, prompt_ids);
                    auto block = m_allocator.get_cached_block(hash, cached_blocks);
                    if (block != nullptr) {
                        block->set_timestamp(std::chrono::system_clock::now());
                        m_block_table[seq_id].push_back(block);
                        group->update_processed_tokens_num(prev_iteration_content_len + i);
                        
                        size_t new_tokens_count_in_block = std::min(content_len, prev_iteration_content_len + block_size);
                        if (new_tokens_count_in_block > prev_iteration_content_len + i) {
                            cached_blocks.erase(hash);
                            auto new_hash = sequence->get_hash(new_tokens_count_in_block, prompt_ids);
                            cached_blocks[new_hash] = block;
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
