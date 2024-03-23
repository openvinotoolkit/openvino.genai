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
        OPENVINO_ASSERT(m_total_num_blocks == m_free_blocks.size());
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
        KVCacheBlock::Ptr allocated_block = m_free_blocks.back();
        allocated_block->increment();
        m_free_blocks.pop_back();
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
        OPENVINO_ASSERT(m_block_table.empty());
    }

    const std::vector<KVCacheBlock::Ptr>& get_block_table(uint64_t seq_id) {
        OPENVINO_ASSERT(m_block_table.count(seq_id) == 1);
        return m_block_table[seq_id];
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

    bool can_append_slot(const SequenceGroup& seq_group) {
        // TODO: optimize this heuristic
        // it assumes that all sequences require new block, but maybe some of them
        // don't share the same block
        // let's count actual number of sequences, where last_block_id is the same
        return seq_group.num_running_seqs() <= m_allocator.num_free_blocks();
    }

    std::map<size_t, size_t> append_slot(const SequenceGroup& seq_group) {
        OPENVINO_ASSERT(can_append_slot(seq_group));
        size_t num_logical_blocks = seq_group.get_num_logical_blocks();
        std::vector<Sequence::CPtr> unfinished_sequences = seq_group.get_running_sequences();

        std::map<size_t, size_t> copy_blocks_map;
        for (size_t i = 0; i < unfinished_sequences.size(); ++i) {
            const Sequence& sequence = *unfinished_sequences[i];
            auto seq_id = sequence.get_id();
            auto& block_table = m_block_table[seq_id];
            size_t num_physical_blocks = block_table.size();

            if (num_logical_blocks > num_physical_blocks) {
                // we require to allocate a new physical block
                block_table.push_back(m_allocator.allocate_block());
            } else {
                KVCacheBlock::Ptr last_block = block_table[num_physical_blocks - 1];

                if (last_block->copy_on_write()) {
                    // we need to fork current block, because reference counter is more than 1
                    KVCacheBlock::Ptr new_block = m_allocator.allocate_block();
                    block_table[num_physical_blocks - 1] = new_block;
                    // write information about block forking for later usage in CacheManager
                    copy_blocks_map[last_block->get_index()] = new_block->get_index();
                } else {
                    // nothing to do, because we are the only users of this block
                }
            }
        }

        // it returns information which blocks should be forked by CacheManager
        return copy_blocks_map;
    }
};
