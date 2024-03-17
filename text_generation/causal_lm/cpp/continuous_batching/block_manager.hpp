// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>
#include <map>

class KVCacheBlock {
    std::shared_ptr<int> m_ref_count = std::make_shared<int>(0);
    int m_index;
public:
    explicit KVCacheBlock(int index)
        : m_index(index) { }

    int get_index() const {
        return m_index;
    }

    bool is_free() const {
        return (*m_ref_count) == 0;
    }

    bool copy_on_write() const {
        return (*m_ref_count) > 1;
    }
};


class BlockAllocator {
    std::vector<KVCacheBlock> m_free_blocks;
public:
    BlockAllocator(int num_blocks) {
        m_free_blocks.reserve(num_blocks);
        for (int block_id = 0; block_id < num_blocks; ++block_id) {
            m_free_blocks.push_back(KVCacheBlock(block_id));
        }
    }

    size_t num_free_blocks() const {
        return m_free_blocks.size();
    }

    bool can_allocate_blocks(size_t num_blocks) const {
        return num_blocks <= m_free_blocks.size();
    }

    void free(KVCacheBlock block) {
        if (block.is_free()) {
            m_free_blocks.push_back(block);
        }
    }

    KVCacheBlock allocate_block() {
        OPENVINO_ASSERT(can_allocate_blocks(1));
        auto allocated_block = m_free_blocks.back();
        m_free_blocks.pop_back();
        return allocated_block;
    }
};

class BlockManager {
    BlockAllocator m_allocator;

    // stores blocks for each sequence (not sequence group)
    // the same block can be seen in multiple block_tables for different sequences
    std::map<uint64_t, std::vector<KVCacheBlock>> m_block_table;
public:
    BlockManager(int num_blocks)
        : m_allocator(num_blocks) { }

    const std::vector<KVCacheBlock>& get_block_table(uint64_t seq_id) {
        OPENVINO_ASSERT(m_block_table.count(seq_id) == 1);
        return m_block_table[seq_id];
    }

    size_t num_free_blocks() const {
        return m_allocator.num_free_blocks();
    }

    bool can_allocate_blocks(size_t num_blocks) const {
        return m_allocator.can_allocate_blocks(num_blocks);
    }

    void allocate(const Sequence& sequence, size_t num_blocks) {
        OPENVINO_ASSERT(can_allocate_blocks(num_blocks));

        for (size_t i = 0; i < num_blocks; ++i) {
            m_block_table[sequence.get_id()].push_back(m_allocator.allocate_block());
        }
    }

    void fork_sequence(const Sequence& parent, const Sequence& child) {
        // note, that reference counters are automatically incremented
        m_block_table[child.get_id()] = m_block_table[parent.get_id()];
    }

    void free_sequence(size_t seq_id) {
        auto block_table = m_block_table[seq_id];

        for (KVCacheBlock& block : block_table) {
            m_allocator.free(block);
        }

        m_block_table.erase(seq_id);
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
        std::vector<Sequence> unifinished_sequences = seq_group.get_running_sequences();

        std::map<size_t, size_t> copy_blocks_map;
        for (size_t i = 0; i < unifinished_sequences.size(); ++i) {
            const Sequence& sequence = seq_group[i];
            auto seq_id = sequence.get_id();
            auto& block_table = m_block_table[seq_id];
            size_t num_physical_blocks = block_table.size();

            if (num_logical_blocks > num_physical_blocks) {
                // we require to allocate a new physical block
                block_table.push_back(m_allocator.allocate_block());
            } else {
                KVCacheBlock last_block = block_table[num_physical_blocks - 1];

                if (last_block.copy_on_write()) {
                    // we need to fork current block, because reference counter is more than 1
                    KVCacheBlock new_block = m_allocator.allocate_block();
                    block_table[num_physical_blocks - 1] = new_block;
                    // write information about block forking for later usage in CacheManager
                    copy_blocks_map[last_block.get_index()] = new_block.get_index();
                } else {
                    // nothing to do, because we are the only users of this block
                }
            }
        }

        // it returns information which blocks should be forked by CacheManager
        return copy_blocks_map;
    }
};
