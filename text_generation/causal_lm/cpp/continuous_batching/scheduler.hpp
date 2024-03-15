
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <vector>

#include "model_config.hpp"
#include "block_manager.hpp"
#include "sequence_group.hpp"
#include "block_manager.hpp"

struct SchedulerConfig {
    // a maximum number of tokens to batch
    // (in constrast to max_batch_size which combines independent sequences, we consider total amount of tokens in a batch)
    // TODO: benchmark this value and understand a required value to ensure inference is not memory bound
    const size_t max_tokens_to_batch = 16;

    // total number of KV blocks available to scheduler logic
    const size_t num_kv_blocks = NUM_BLOCKS;
};

class Scheduler {
    SchedulerConfig m_config;
    BlockManager m_block_manager;
public:
    struct Output {
        std::vector<uint64_t> m_scheduled_sequence_groups_ids;
        // a number of scheduled tokens per sequence ID
        std::map<uint64_t, size_t> m_num_scheduled_tokens;
        // map of src -> dst blocks copies, which need to be performed by CacheManager
        std::map<size_t, size_t> m_block_copy_map;
    };

    Scheduler(const SchedulerConfig & config = {}) :
        m_config(config), m_block_manager(m_config.num_kv_blocks) { }

    void schedule(std::vector<SequenceGroup>& sequence_groups) {
        for (size_t sequence_group_id = 0, current_num_of_scheduled_tokens = 0;
            sequence_group_id < sequence_groups.size() && current_num_of_scheduled_tokens < m_config.max_tokens_to_batch; ++sequence_group_id) {
            SequenceGroup & sequence_group = sequence_groups[sequence_group_id];
            OPENVINO_ASSERT(!sequence_group.has_finished());

            // TODO: implement the logic of whether current sequence can be processed or we don't have memory for its execution
            // Handle cases, like:
            // 1. sequence does not require new blocks (e.g. generation phase, where we still have some free physical slots)
            // 2. only part of prompt can be allocated by BlockManager, because not all prompt tokens can fit into remainging KV cache
            // 3. generation sequences should always be processed before prompt sequences
            // 4. equally split remaining number of tokens in batch between prompt sequences
            //    (align chunk size of mini-prompt-batch by BLOCK_SIZE)
            // 5. we need to implement cache eviction (by BLOCK_SIZE) in order to continue generation of sequences with high priority
            //    (sequences with lower priority will lose blocks in KV cache in this case)
            //    Note: that we need to evict low-priority sequences while we have generation sequence groups (it should be either evicted or scheduled)
            bool can_allocate_current_sequence = true;

            if (!can_allocate_current_sequence) {
                continue;
            }

            size_t num_batch_available_tokens = m_config.max_tokens_to_batch - current_num_of_scheduled_tokens;
            size_t num_seq_available_tokens = sequence_group.get_num_available_tokens_for_batching();

            // schedule all bare minimum of tokens from current sequence to fill up a batch!
            size_t num_scheduled_tokens = std::min(num_batch_available_tokens, num_seq_available_tokens);
            // TODO: remove this limitation
            OPENVINO_ASSERT(num_scheduled_tokens == num_seq_available_tokens);
            sequence_group.schedule_tokens(num_scheduled_tokens);

            // iteratively allocate new blocks for sequence group
            // TODO: optimize:
            // 1. allocate required amount of blocks for prompt in a single shot
            size_t num_blocks_to_allocate = (num_scheduled_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
            m_block_manager.allocate(sequence_group, num_blocks_to_allocate);

            current_num_of_scheduled_tokens += num_scheduled_tokens;
        }
    }

    const std::vector<KVCacheBlock>& get_block_table(const Sequence& seq) {
        return m_block_manager.get_block_table(seq);
    }
};
