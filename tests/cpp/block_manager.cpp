// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"
#include "sequence_group.hpp"
#include "scheduler.hpp"

TEST(TestBlockManager, general_test) {
    ov::genai::BlockManager bm = ov::genai::BlockManager(6);

    bm.allocate(0, 6);
    EXPECT_TRUE(bm.has_block_table(0));
    EXPECT_EQ(bm.get_block_table(0).size(), 6);
    EXPECT_EQ(bm.num_free_blocks(), 0);

    bm.free_sequence_partially_single_runnning_sequence(0, 4);
    EXPECT_EQ(bm.get_block_table(0).size(), 2);
    EXPECT_EQ(bm.num_free_blocks(), 4);

    bm.free_sequence(0);
    EXPECT_FALSE(bm.has_block_table(0));
    EXPECT_EQ(bm.num_free_blocks(), 6);

    bm.allocate(0, 2);
    bm.fork_sequence(0, 1);
    EXPECT_TRUE(bm.has_block_table(1));
    EXPECT_EQ(bm.get_block_table(1).back()->get_references_count(), 2);

}

TEST(TestBlockManager, required_blocks_count) {
    ov::genai::BlockManager bm = ov::genai::BlockManager(8);

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

    sequence_group->finish_iteration();
    auto sequence_to_fork = sequence_group->get_running_sequences()[0];    
    for (size_t i = 0; i < 4; ++i) {
        const auto forked_sequence = sequence_group->fork_sequence(sequence_to_fork);
        bm.fork_sequence(sequence_to_fork->get_id(), forked_sequence->get_id());
    }
    sequence_group->schedule_tokens(1);
    required_blocks = bm.required_blocks_count(sequence_group);
    EXPECT_EQ(required_blocks, 4);
    EXPECT_TRUE(bm.can_append_slots(sequence_group));
    bm.append_slots(sequence_group);
    EXPECT_EQ(bm.num_free_blocks(), 2);
    sequence_group->finish_iteration();

    sequence_group->schedule_tokens(3);
    required_blocks = bm.required_blocks_count(sequence_group);
    EXPECT_EQ(required_blocks, 5);
    EXPECT_FALSE(bm.can_append_slots(sequence_group));
}