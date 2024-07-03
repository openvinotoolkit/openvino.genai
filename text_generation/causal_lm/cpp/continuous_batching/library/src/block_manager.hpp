// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <list>
#include <map>

#include "sequence_group.hpp"

class KVCacheBlock {
    int m_ref_count;
    int m_index;
public:
    using Ptr = std::shared_ptr<KVCacheBlock>;
    using CPtr = std::shared_ptr<const KVCacheBlock>;

    explicit KVCacheBlock(int index)
        : m_ref_count(0),
          m_index(index) { }

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
        --m_ref_count;
    }

    bool copy_on_write() const {
        return m_ref_count > 1;
    }

    int get_references_count() const {
        return m_ref_count;
    }
};


class BlockAllocator {
    std::list<KVCacheBlock::Ptr> m_free_blocks;
    int m_total_num_blocks;
public:
    BlockAllocator(int num_blocks) :
        m_total_num_blocks(num_blocks) {
        for (int block_id = 0; block_id < m_total_num_blocks; ++block_id) {
            m_free_blocks.push_back(std::make_shared<KVCacheBlock>(block_id));
        }
    }

    ~BlockAllocator() {
        // sanity check to validate that all blocks are freed
        // OPENVINO_ASSERT(m_total_num_blocks == m_free_blocks.size());
    }

    size_t num_free_blocks() const {
        return m_free_blocks.size();
    }

    bool can_allocate_blocks(size_t num_blocks) const {
        return num_blocks <= m_free_blocks.size();
    }

    void free(KVCacheBlock::Ptr block) {
        block->release();
        if (block->is_free()) {
            m_free_blocks.push_back(block);
        }
    }

    KVCacheBlock::Ptr allocate_block() {
        OPENVINO_ASSERT(can_allocate_blocks(1));
        KVCacheBlock::Ptr allocated_block = m_free_blocks.front();
        allocated_block->increment();
        m_free_blocks.pop_front();
        return allocated_block;
    }

    float get_used_percentage() const {
        return static_cast<float>(m_total_num_blocks - m_free_blocks.size()) / m_total_num_blocks;
    }
};

class BlockManager {
    BlockAllocator m_allocator;

    // stores blocks for each sequence (not sequence group)
    // the same block can be seen in multiple block_tables for different sequences
    std::map<uint64_t, std::vector<KVCacheBlock::Ptr>> m_block_table;
public:
    BlockManager(int num_blocks)
        : m_allocator(num_blocks) { }

    ~BlockManager() {
        // sanity check that all sequences are freed
        // OPENVINO_ASSERT(m_block_table.empty());
    }

    const std::vector<KVCacheBlock::Ptr>& get_block_table(uint64_t seq_id) {
        OPENVINO_ASSERT(m_block_table.count(seq_id) == 1);
        return m_block_table[seq_id];
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

    void allocate(uint64_t sequence_id, size_t num_blocks) {
        OPENVINO_ASSERT(num_blocks > 0 && can_allocate_blocks(num_blocks));

        for (size_t i = 0; i < num_blocks; ++i) {
            m_block_table[sequence_id].push_back(m_allocator.allocate_block());
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

    void free_sequence_partially(size_t seq_id, size_t block_num) {
        // currently this method is applicable only for groups with single sequences
        // TODO: support for groups with multiple sequences
        auto block_table = m_block_table[seq_id];

        OPENVINO_ASSERT(block_table.size() >= block_num);
        for (size_t idx = 0; idx < block_num; idx++) {
            size_t block_idx = m_block_table[seq_id].size() - idx - 1;
            m_allocator.free(block_table[block_idx]);
            OPENVINO_ASSERT(block_table[block_idx]->is_free());
        } 
        m_block_table[seq_id].resize(m_block_table[seq_id].size() - block_num);

        if (m_block_table.size() == 0) {
            OPENVINO_ASSERT(m_block_table.erase(seq_id) == 1);
        }
    }

    bool can_append_slots(SequenceGroup::CPtr seq_group) {
        return required_blocks_count(std::move(seq_group)) <= m_allocator.num_free_blocks();
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
                allocate(seq_id, num_logical_blocks - num_physical_blocks);
            } else {
                OPENVINO_ASSERT(num_logical_blocks == num_physical_blocks, "A number of physical and logic blocks must be the same in this code path");
                KVCacheBlock::Ptr last_block = block_table.back();

                if (last_block->copy_on_write()) {
                    // we need to fork current block, because reference counter is more than 1
                    KVCacheBlock::Ptr new_block = m_allocator.allocate_block();
                    block_table[num_physical_blocks - 1] = new_block;
                    // write information about block forking for later usage in CacheManager
                    copy_blocks_map[last_block->get_index()].push_back(new_block->get_index());
                    // release `last_block` usage
                    m_allocator.free(std::move(last_block));
                } else {
                    // nothing to do, because we are the only users of this block
                }
            }
        }

        // it returns information which blocks should be forked by CacheManager
        return copy_blocks_map;
    }
};
