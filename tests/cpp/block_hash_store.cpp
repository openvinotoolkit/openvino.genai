// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "continuous_batching/scheduler.hpp"
#include <chrono>
#include <thread>

TEST(TestBlockHashStore, general_test) {
    ov::genai::OverwritableBlocksHashStore block_hash_store(1);
    auto block0 = std::make_shared<ov::genai::CacheBlock>(0);
    block0->set_hash(77);
    std::this_thread::sleep_until(std::chrono::steady_clock::now() + std::chrono::seconds(1));
    auto block1 = std::make_shared<ov::genai::CacheBlock>(1);
    block1->set_hash(56);
    std::this_thread::sleep_until(std::chrono::steady_clock::now() + std::chrono::seconds(1));
    auto block2 = std::make_shared<ov::genai::CacheBlock>(2);
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

    auto block3 = std::make_shared<ov::genai::CacheBlock>(7);
    block3->set_hash(12);
    std::this_thread::sleep_until(std::chrono::steady_clock::now() + std::chrono::seconds(1));
    auto block4 = std::make_shared<ov::genai::CacheBlock>(10);
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

namespace {

ov::genai::BlocksPerLayer make_stored_block(size_t index,
                                            size_t hash,
                                            std::chrono::steady_clock::time_point base,
                                            int age_seconds) {
    auto block = std::make_shared<ov::genai::CacheBlock>(index);
    block->set_hash(hash);
    block->set_timestamp(base + std::chrono::seconds(age_seconds));
    return ov::genai::BlocksPerLayer{block};
}

}  // namespace

TEST(TestBlockHashStore, chunk_boundaries_make_eviction_prefer_highest_chunk) {
    ov::genai::OverwritableBlocksHashStore store(1);
    const auto base = std::chrono::steady_clock::now();
    // chunk 0 = [0, 4), chunk 1 = [4, 8), chunk 2 = [8, ...)
    store.set_chunk_boundaries({0, 4, 8});
    store.add(make_stored_block(1, 11, base, 0));   // chunk 0, oldest
    store.add(make_stored_block(5, 22, base, 1));   // chunk 1
    store.add(make_stored_block(10, 33, base, 3));  // chunk 2, newest
    store.add(make_stored_block(9, 44, base, 2));   // chunk 2

    EXPECT_EQ(store.get_lru_block_to_overwrite()[0]->get_index(), 9);  // LRU within the highest chunk
    EXPECT_EQ(store.get_lru_block_to_overwrite()[0]->get_index(), 10);
    EXPECT_EQ(store.get_lru_block_to_overwrite()[0]->get_index(), 5);
    EXPECT_EQ(store.get_lru_block_to_overwrite()[0]->get_index(), 1);
    EXPECT_TRUE(store.get_lru_block_to_overwrite().empty());
}

TEST(TestBlockHashStore, blocks_beyond_last_boundary_belong_to_last_chunk) {
    ov::genai::OverwritableBlocksHashStore store(1);
    const auto base = std::chrono::steady_clock::now();
    store.set_chunk_boundaries({0, 4});
    store.add(make_stored_block(2, 11, base, 0));    // chunk 0, oldest
    store.add(make_stored_block(5, 22, base, 3));    // chunk 1
    store.add(make_stored_block(100, 33, base, 1));  // beyond the last boundary -> chunk 1

    EXPECT_EQ(store.get_lru_block_to_overwrite()[0]->get_index(), 100);
    EXPECT_EQ(store.get_lru_block_to_overwrite()[0]->get_index(), 5);
    EXPECT_EQ(store.get_lru_block_to_overwrite()[0]->get_index(), 2);
}

TEST(TestBlockHashStore, empty_chunk_boundaries_keep_plain_lru) {
    ov::genai::OverwritableBlocksHashStore store(1);
    const auto base = std::chrono::steady_clock::now();
    store.set_chunk_boundaries({0, 4, 8});
    store.set_chunk_boundaries({});
    store.add(make_stored_block(10, 11, base, 0));
    store.add(make_stored_block(1, 22, base, 1));
    store.add(make_stored_block(5, 33, base, 2));

    EXPECT_EQ(store.get_lru_block_to_overwrite()[0]->get_index(), 10);
    EXPECT_EQ(store.get_lru_block_to_overwrite()[0]->get_index(), 1);
    EXPECT_EQ(store.get_lru_block_to_overwrite()[0]->get_index(), 5);
}
