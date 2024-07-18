// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "scheduler.hpp"

TEST(TestEvictor, general_test) {
    ov::genai::Evicor evictor;
    auto block2 = std::make_shared<ov::genai::KVCacheBlock>(2);
    evictor.add(77, std::make_shared<ov::genai::KVCacheBlock>(0));
    evictor.add(56, std::make_shared<ov::genai::KVCacheBlock>(1));
    evictor.add(23, block2);
    EXPECT_EQ(evictor.num_blocks(), 3);

    auto block = evictor.get_block(56);
    EXPECT_EQ(block->get_index(), 1);
    EXPECT_EQ(block->get_hash(), 56);
    EXPECT_EQ(block->get_references_count(), 1);
    EXPECT_EQ(evictor.num_blocks(), 2);

    EXPECT_EQ(evictor.get_block(44), nullptr);
    EXPECT_EQ(evictor.num_blocks(), 2);

    EXPECT_EQ(evictor.get_lru_block()->get_index(), 0);
    EXPECT_EQ(evictor.num_blocks(), 1);

    evictor.add(12, std::make_shared<ov::genai::KVCacheBlock>(7));
    evictor.add(99, std::make_shared<ov::genai::KVCacheBlock>(10));
    block2->set_timestamp(time(NULL));
    
    EXPECT_EQ(evictor.get_lru_block()->get_index(), 7);
    EXPECT_EQ(evictor.get_lru_block()->get_index(), 10);
    EXPECT_EQ(evictor.get_lru_block()->get_index(), 2);
    EXPECT_EQ(evictor.get_lru_block(), nullptr);
    EXPECT_EQ(evictor.num_blocks(), 0);
}