// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <vector>

#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/kv_cache_offload_cache.hpp"
#include "continuous_batching/cache/kv_cache_offload_manager.hpp"
#include "helper.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/runtime/core.hpp"
#include "sequence_group.hpp"
#include "utils.hpp"

using namespace ov::genai;

namespace {

constexpr size_t NUM_DECODER_LAYERS = 2;
constexpr size_t NUM_PHYSICAL_BLOCKS = 4;

/// Owns the OpenVINO objects a KVCacheManager needs so that tests can allocate a real CPU KV cache.
struct CacheFixture {
    ov::Core core;
    ov::InferRequest request;
    std::shared_ptr<KVCacheManager> cache_manager;

    explicit CacheFixture(size_t num_blocks = NUM_PHYSICAL_BLOCKS) {
        request = core.compile_model(get_dummy_model(core, NUM_DECODER_LAYERS)).create_infer_request();
        cache_manager = std::make_shared<KVCacheManager>(request);
        cache_manager->allocate_cache_if_needed(num_blocks);
    }

    /// Fills a physical block with a byte pattern derived from `seed` and returns the expected bytes.
    std::vector<uint8_t> fill_block(size_t block_id, uint8_t seed) {
        std::vector<uint8_t> data(cache_manager->get_block_layout().get_slot_size());
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<uint8_t>(seed + i);
        }
        cache_manager->write_block(block_id, data);
        return data;
    }
};

CacheOffloadConfig make_offload_config(const CacheFixture& fixture, size_t num_slots) {
    CacheOffloadConfig config;
    config.capacity_bytes = fixture.cache_manager->get_block_layout().get_slot_size() * num_slots;
    return config;
}

std::unique_ptr<KVCacheOffloadCache> make_offload_cache(CacheFixture& fixture, size_t num_slots) {
    auto backend = std::make_unique<KVCacheOffloadManager>(fixture.cache_manager->get_block_layout(),
                                                           make_offload_config(fixture, num_slots),
                                                           "CPU");
    return std::make_unique<KVCacheOffloadCache>(*fixture.cache_manager, std::move(backend));
}

BlocksPerLayer make_block_set(int physical_block_id) {
    return BlocksPerLayer{std::make_shared<CacheBlock>(physical_block_id)};
}

SequenceGroup::Ptr make_group(const std::vector<int64_t>& tokens, uint64_t request_id) {
    return std::make_shared<SequenceGroup>(request_id,
                                           ov::Tensor(ov::element::i64, {tokens.size()}, const_cast<int64_t*>(tokens.data())),
                                           utils::get_greedy_config());
}

}  // namespace

TEST(TestKVCacheOffloadCache, StoresBlockContentsOnOverwrite) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto expected = fixture.fill_block(1, /*seed=*/7);

    offload_cache->on_blocks_overwritten(/*hash=*/42, make_block_set(1));

    EXPECT_TRUE(offload_cache->contains(42));
    EXPECT_EQ(offload_cache->get_num_entries(), 1);
    EXPECT_EQ(offload_cache->get_statistics().num_stored, 1);

    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(42, restored));
    EXPECT_EQ(restored, expected);
}

TEST(TestKVCacheOffloadCache, KeepsFirstCopyOfKnownHash) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto expected = fixture.fill_block(0, /*seed=*/3);
    offload_cache->on_blocks_overwritten(/*hash=*/5, make_block_set(0));

    fixture.fill_block(1, /*seed=*/200);
    offload_cache->on_blocks_overwritten(/*hash=*/5, make_block_set(1));

    EXPECT_EQ(offload_cache->get_num_entries(), 1);
    EXPECT_EQ(offload_cache->get_statistics().num_stored, 1);
    EXPECT_EQ(offload_cache->get_statistics().num_already_present, 1);

    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(5, restored));
    EXPECT_EQ(restored, expected);
}

TEST(TestKVCacheOffloadCache, KeepsDistinctHashesInSeparateSlots) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto first = fixture.fill_block(0, /*seed=*/11);
    const auto second = fixture.fill_block(1, /*seed=*/77);

    offload_cache->on_blocks_overwritten(/*hash=*/1, make_block_set(0));
    offload_cache->on_blocks_overwritten(/*hash=*/2, make_block_set(1));

    ASSERT_EQ(offload_cache->get_num_entries(), 2);
    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(1, restored));
    EXPECT_EQ(restored, first);
    ASSERT_TRUE(offload_cache->read(2, restored));
    EXPECT_EQ(restored, second);
}

TEST(TestKVCacheOffloadCache, ReplacesOldestEntryWhenFileIsFull) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);
    fixture.fill_block(0, /*seed=*/1);
    offload_cache->on_blocks_overwritten(/*hash=*/100, make_block_set(0));
    ASSERT_EQ(offload_cache->get_num_free_slots(), 0);

    const auto newer = fixture.fill_block(1, /*seed=*/60);
    offload_cache->on_blocks_overwritten(/*hash=*/200, make_block_set(1));

    EXPECT_EQ(offload_cache->get_num_entries(), 1);
    EXPECT_FALSE(offload_cache->contains(100));
    EXPECT_EQ(offload_cache->get_statistics().num_replaced, 1);

    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(200, restored));
    EXPECT_EQ(restored, newer);
}

TEST(TestKVCacheOffloadCache, ReadReportsMissForUnknownHash) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);

    std::vector<uint8_t> restored;
    EXPECT_FALSE(offload_cache->read(/*hash=*/999, restored));
    EXPECT_TRUE(restored.empty());
}

TEST(TestKVCacheOffloadCache, RejectsPerLayerBlockTables) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);
    BlocksPerLayer per_layer_blocks{std::make_shared<CacheBlock>(0), std::make_shared<CacheBlock>(1)};

    EXPECT_THROW(offload_cache->on_blocks_overwritten(/*hash=*/1, per_layer_blocks), ov::Exception);
}

TEST(TestKVCacheOffloadCache, RemovesOffloadFileOnDestruction) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);
    fixture.fill_block(0, /*seed=*/5);
    offload_cache->on_blocks_overwritten(/*hash=*/1, make_block_set(0));

    offload_cache.reset();
    // The backend owns a run-specific file that must not outlive the cache.
    EXPECT_EQ(fixture.cache_manager->get_num_allocated_blocks(), NUM_PHYSICAL_BLOCKS);
}

TEST(TestKVCacheOffloadCache, BlockManagerStoresEvictedPrefixBlock) {
    constexpr size_t block_size = 4;
    CacheFixture fixture(/*num_blocks=*/2);
    auto offload_cache = make_offload_cache(fixture, 2);

    BlockManager block_manager(/*num_blocks=*/2, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);
    block_manager.set_overwritten_block_observer(offload_cache.get());

    const std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto producer = make_group(tokens, /*request_id=*/1);
    producer->schedule_tokens(tokens.size());
    block_manager.append_slots(producer);
    producer->finish_iteration();

    const auto producer_seq_id = producer->get_running_sequences().at(0)->get_id();
    ASSERT_EQ(block_manager.get_block_table(producer_seq_id, 0).size(), 2);
    block_manager.free_sequence(producer_seq_id);
    ASSERT_EQ(offload_cache->get_num_entries(), 0);

    // Allocating for an unrelated prefix exhausts the free pool and overwrites a cached block.
    const std::vector<int64_t> pressure_tokens = {10, 11, 12, 13, 14, 15, 16, 17};
    auto pressure = make_group(pressure_tokens, /*request_id=*/2);
    pressure->schedule_tokens(pressure_tokens.size());
    block_manager.append_slots(pressure);

    EXPECT_GT(offload_cache->get_num_entries(), 0);
    EXPECT_EQ(offload_cache->get_statistics().num_stored, offload_cache->get_num_entries());

    block_manager.set_overwritten_block_observer(nullptr);
    block_manager.free_sequence(pressure->get_running_sequences().at(0)->get_id());
}

TEST(TestKVCacheOffloadCache, DetachedObserverStopsStoring) {
    constexpr size_t block_size = 4;
    CacheFixture fixture(/*num_blocks=*/2);
    auto offload_cache = make_offload_cache(fixture, 2);

    BlockManager block_manager(/*num_blocks=*/2, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);
    block_manager.set_overwritten_block_observer(offload_cache.get());
    block_manager.set_overwritten_block_observer(nullptr);

    const std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto producer = make_group(tokens, /*request_id=*/3);
    producer->schedule_tokens(tokens.size());
    block_manager.append_slots(producer);
    producer->finish_iteration();
    block_manager.free_sequence(producer->get_running_sequences().at(0)->get_id());

    const std::vector<int64_t> pressure_tokens = {20, 21, 22, 23, 24, 25, 26, 27};
    auto pressure = make_group(pressure_tokens, /*request_id=*/4);
    pressure->schedule_tokens(pressure_tokens.size());
    block_manager.append_slots(pressure);

    EXPECT_EQ(offload_cache->get_num_entries(), 0);

    block_manager.free_sequence(pressure->get_running_sequences().at(0)->get_id());
}
