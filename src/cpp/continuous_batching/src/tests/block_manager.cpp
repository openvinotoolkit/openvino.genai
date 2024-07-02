// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "continuous_batching_pipeline.hpp"
#include "sequence_group.hpp"
#include "scheduler.hpp"
#include "openvino/genai/generation_config.hpp"

TEST(TestBlockManager, general_test) {
    BlockManager bm = BlockManager(6);

    bm.allocate(0, 6);
    EXPECT_TRUE(bm.has_block_table(0));
    EXPECT_EQ(bm.get_block_table(0).size(), 6);
    EXPECT_EQ(bm.num_free_blocks(), 0);

    bm.free_sequence_partially(0, 4);
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
