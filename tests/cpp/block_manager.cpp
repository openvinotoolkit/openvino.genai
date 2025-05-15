// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "openvino/genai/generation_config.hpp"
#include "sequence_group.hpp"
#include "continuous_batching/scheduler.hpp"

TEST(TestBlockManager, general_test) {
    ov::genai::BlockManager bm = ov::genai::BlockManager(6, false, 4);
    ov::genai::TokenIds prompt_ids = {10, 0};

    ov::genai::SequenceGroup::Ptr sequence_group = std::make_shared<ov::genai::SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {
        prompt_ids.size()}, prompt_ids.data()),
        ov::genai::beam_search(),
        4);
    auto sequence = sequence_group->get_not_finished_sequences()[0];
    bm.allocate(sequence, 6);
    auto seq_id = sequence->get_id();
    EXPECT_TRUE(bm.has_block_table(seq_id));
    EXPECT_EQ(bm.get_block_table(seq_id, 0).size(), 6);
    EXPECT_EQ(bm.num_free_blocks(), 0);

    bm.free_sequence_partially(seq_id, 4);
    EXPECT_EQ(bm.get_block_table(seq_id, 0).size(), 2);
    EXPECT_EQ(bm.num_free_blocks(), 4);

    bm.free_sequence(seq_id);
    EXPECT_FALSE(bm.has_block_table(seq_id));
    EXPECT_EQ(bm.num_free_blocks(), 6);

    bm.allocate(sequence, 2);
    bm.fork_sequence(seq_id, 1);
    EXPECT_TRUE(bm.has_block_table(1));
    EXPECT_EQ(bm.get_block_table(1, 0).back()->get_references_count(), 2);
    bm.free_sequence(0);
    bm.free_sequence(1);
}

TEST(TestBlockManager, required_blocks_count) {
    ov::genai::BlockManager bm = ov::genai::BlockManager(8, false, 4, 3);

    std::vector<uint64_t> tokens = {0,1,2,3,4};
    ov::genai::SequenceGroup::Ptr sequence_group = std::make_shared<ov::genai::SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {
        tokens.size()}, tokens.data()),
        ov::genai::beam_search(),
        4);
    sequence_group->schedule_tokens(5);
    auto required_blocks = bm.required_blocks_count(sequence_group);
    EXPECT_EQ(required_blocks, 2);
    EXPECT_TRUE(bm.can_append_slots(sequence_group));
    bm.append_slots(sequence_group);
    EXPECT_EQ(bm.num_free_blocks(), 6);
    EXPECT_EQ(bm.get_number_of_blocks_occupied_by_sequence(sequence_group), 2);

    sequence_group->finish_iteration();
    auto sequence_to_fork = sequence_group->get_running_sequences()[0];
    for (size_t i = 0; i < 4; ++i) {
        const auto forked_sequence = sequence_group->fork_sequence(sequence_to_fork);
        bm.fork_sequence(sequence_to_fork->get_id(), forked_sequence->get_id());
    }
    EXPECT_EQ(bm.get_number_of_blocks_occupied_by_sequence(sequence_group), 2);
    sequence_group->schedule_tokens(1);
    required_blocks = bm.required_blocks_count(sequence_group);
    // The last block was incomplete before forking, therefore need to allocate an extra block for each new forked
    // sequence (excluding the original)
    EXPECT_EQ(required_blocks, 4);
    EXPECT_TRUE(bm.can_append_slots(sequence_group));
    bm.append_slots(sequence_group);
    EXPECT_EQ(bm.get_number_of_blocks_occupied_by_sequence(sequence_group), 6);
    EXPECT_EQ(bm.num_free_blocks(), 2);
    sequence_group->finish_iteration();

    sequence_group->schedule_tokens(3);
    required_blocks = bm.required_blocks_count(sequence_group);
    // Each sequence in group had 3 tokens scheduled in addition to 6 already processed, e.g. with block size 4 we
    // require 1 extra block for each sequence in group
    EXPECT_EQ(required_blocks, 5);
    EXPECT_FALSE(bm.can_append_slots(sequence_group));

    for (auto& sequence : sequence_group->get_sequences()) {
        bm.free_sequence(sequence->get_id());
    }
}


TEST(TestBlockManager, CanFreeBlocksFromSequence) {
    const size_t BLOCK_SIZE = 2;
    ov::genai::BlockManager bm = ov::genai::BlockManager(8, false, BLOCK_SIZE, 3);

    std::vector<uint64_t> tokens = {0,1,2,3,4};
    ov::genai::SequenceGroup::Ptr sequence_group = std::make_shared<ov::genai::SequenceGroup>(
            0,
            ov::Tensor(ov::element::i64, {
                    tokens.size()}, tokens.data()),
            ov::genai::beam_search(),
            BLOCK_SIZE);
    sequence_group->schedule_tokens(5);
    bm.append_slots(sequence_group);
    ASSERT_EQ(bm.num_free_blocks(), 5);

    size_t seq_id = sequence_group->get_sequences()[0]->get_id();
    bm.free_blocks_from_sequence(seq_id, { {0}, {1}, {2} });
    EXPECT_EQ(bm.num_free_blocks(), 6);

    for (auto& sequence : sequence_group->get_sequences()) {
        bm.free_sequence(sequence->get_id());
    }
}