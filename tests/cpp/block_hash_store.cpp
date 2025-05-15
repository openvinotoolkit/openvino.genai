// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "continuous_batching/scheduler.hpp"
#include <chrono>
#include <thread>

TEST(TestBlockHashStore, general_test) {
    ov::genai::OverwritableBlocksHashStore block_hash_store(1);
    auto block0 = std::make_shared<ov::genai::KVCacheBlock>(0);
    block0->set_hash(77);
    std::this_thread::sleep_until(std::chrono::steady_clock::now() + std::chrono::seconds(1));
    auto block1 = std::make_shared<ov::genai::KVCacheBlock>(1);
    block1->set_hash(56);
    std::this_thread::sleep_until(std::chrono::steady_clock::now() + std::chrono::seconds(1));
    auto block2 = std::make_shared<ov::genai::KVCacheBlock>(2);
    block2->set_hash(23);
    std::this_thread::sleep_until(std::chrono::steady_clock::now() + std::chrono::seconds(1));
    block_hash_store.add(ov::genai::BlocksPerLayer{block0});
    block_hash_store.add(ov::genai::BlocksPerLayer{block1});
    block_hash_store.add(ov::genai::BlocksPerLayer{block2});
    EXPECT_EQ(block_hash_store.num_blocks(), 3);

    auto block = block_hash_store.get_block_to_restore(56)[0];
    EXPECT_EQ(block->get_index(), 1);
    EXPECT_EQ(block->get_hash(), 56);
    EXPECT_EQ(block->get_references_count(), 1);
    EXPECT_EQ(block_hash_store.num_blocks(), 2);

    EXPECT_TRUE(block_hash_store.get_block_to_restore(44).empty());
    EXPECT_EQ(block_hash_store.num_blocks(), 2);

    EXPECT_EQ(block_hash_store.get_lru_block_to_overwrite()[0]->get_index(), 0);
    EXPECT_EQ(block_hash_store.num_blocks(), 1);

    auto block3 = std::make_shared<ov::genai::KVCacheBlock>(7);
    block3->set_hash(12);
    std::this_thread::sleep_until(std::chrono::steady_clock::now() + std::chrono::seconds(1));
    auto block4 = std::make_shared<ov::genai::KVCacheBlock>(10);
    block4->set_hash(99);
    std::this_thread::sleep_until(std::chrono::steady_clock::now() + std::chrono::seconds(1));
    block_hash_store.add(ov::genai::BlocksPerLayer{block3});
    block_hash_store.add(ov::genai::BlocksPerLayer{block4});
    block2->set_timestamp(std::chrono::steady_clock::now());

    EXPECT_EQ(block_hash_store.get_lru_block_to_overwrite()[0]->get_index(), 7);
    EXPECT_EQ(block_hash_store.get_lru_block_to_overwrite()[0]->get_index(), 10);
    EXPECT_EQ(block_hash_store.get_lru_block_to_overwrite()[0]->get_index(), 2);
    EXPECT_TRUE(block_hash_store.get_lru_block_to_overwrite().empty());
    EXPECT_EQ(block_hash_store.num_blocks(), 0);
}
