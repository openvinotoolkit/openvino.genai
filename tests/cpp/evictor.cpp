// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "scheduler.hpp"
#include <chrono>

TEST(TestEvictor, general_test) {
    ov::genai::Evicor evictor;
    auto block0 = std::make_shared<ov::genai::KVCacheBlock>(0);
    block0->set_hash(77, 1);
    auto block1 = std::make_shared<ov::genai::KVCacheBlock>(1);
    block1->set_hash(56, 2);
    auto block2 = std::make_shared<ov::genai::KVCacheBlock>(2);
    block2->set_hash(23, 3);
    evictor.add(block0->get_hash(), block0);
    evictor.add(block1->get_hash(), block1);
    evictor.add(block2->get_hash(), block2);
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

    auto block3 = std::make_shared<ov::genai::KVCacheBlock>(7);
    block3->set_hash(12, 4);
    auto block4 = std::make_shared<ov::genai::KVCacheBlock>(10);
    block4->set_hash(99, 5);
    evictor.add(block3->get_hash(), block3);
    evictor.add(block4->get_hash(), block4);
    block2->set_timestamp(std::chrono::system_clock::now());

    EXPECT_EQ(evictor.get_lru_block()->get_index(), 7);
    EXPECT_EQ(evictor.get_lru_block()->get_index(), 10);
    EXPECT_EQ(evictor.get_lru_block()->get_index(), 2);
    EXPECT_EQ(evictor.get_lru_block(), nullptr);
    EXPECT_EQ(evictor.num_blocks(), 0);
}