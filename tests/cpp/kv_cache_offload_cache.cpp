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

TEST(TestKVCacheOffloadCache, LoadsStoredContentsIntoAnotherBlock) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto expected = fixture.fill_block(0, /*seed=*/23);
    offload_cache->on_blocks_overwritten(/*hash=*/77, make_block_set(0));
    fixture.fill_block(2, /*seed=*/0);

    ASSERT_TRUE(offload_cache->load_into(/*hash=*/77, /*block_index=*/2));

    EXPECT_EQ(fixture.cache_manager->read_block(2), expected);
    EXPECT_EQ(offload_cache->get_statistics().num_loaded, 1);
}

TEST(TestKVCacheOffloadCache, LoadReportsMissForUnknownHash) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);
    const auto untouched = fixture.fill_block(1, /*seed=*/9);

    EXPECT_FALSE(offload_cache->load_into(/*hash=*/12345, /*block_index=*/1));
    EXPECT_EQ(fixture.cache_manager->read_block(1), untouched);
}

TEST(TestKVCacheOffloadCache, WarmPrefixCacheRestoresBlocksFromDisk) {
    constexpr size_t block_size = 4;
    CacheFixture fixture(/*num_blocks=*/2);
    auto offload_cache = make_offload_cache(fixture, 2);

    BlockManager block_manager(/*num_blocks=*/2, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);
    block_manager.set_overwritten_block_observer(offload_cache.get());

    const std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto producer = make_group(tokens, /*request_id=*/5);
    producer->schedule_tokens(tokens.size());
    block_manager.append_slots(producer);
    producer->finish_iteration();
    block_manager.free_sequence(producer->get_running_sequences().at(0)->get_id());

    // Evict the cached prefix from memory, which pushes its contents to disk.
    const std::vector<int64_t> pressure_tokens = {30, 31, 32, 33, 34, 35, 36, 37};
    auto pressure = make_group(pressure_tokens, /*request_id=*/6);
    pressure->schedule_tokens(pressure_tokens.size());
    block_manager.append_slots(pressure);
    const auto pressure_seq_id = pressure->get_running_sequences().at(0)->get_id();
    ASSERT_GT(offload_cache->get_num_entries(), 0);
    block_manager.free_sequence(pressure_seq_id);

    auto consumer = make_group(tokens, /*request_id=*/7);
    size_t num_warmed = 0;
    {
        KVCacheOffloadCache::ScopedReclamationPause keep_entries(*offload_cache);
        num_warmed = block_manager.warm_prefix_cache(consumer, *offload_cache);
    }

    EXPECT_GT(num_warmed, 0);
    EXPECT_EQ(offload_cache->get_statistics().num_loaded, num_warmed);
    // The warmed blocks are unowned, so the regular restore path can now claim them.
    EXPECT_TRUE(block_manager.restore_cached_blocks(consumer));
    EXPECT_GT(consumer->get_num_processed_tokens(), 0);

    block_manager.set_overwritten_block_observer(nullptr);
    block_manager.free_sequence(consumer->get_running_sequences().at(0)->get_id());
}

TEST(TestKVCacheOffloadCache, WarmPrefixCacheIsNoOpWithoutEntries) {
    constexpr size_t block_size = 4;
    CacheFixture fixture(/*num_blocks=*/2);
    auto offload_cache = make_offload_cache(fixture, 2);

    BlockManager block_manager(/*num_blocks=*/2, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);
    const std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto consumer = make_group(tokens, /*request_id=*/8);

    EXPECT_EQ(block_manager.warm_prefix_cache(consumer, *offload_cache), 0);
    EXPECT_EQ(block_manager.num_free_blocks(), 2);
}

TEST(TestKVCacheOffloadCache, WarmPrefixCacheKeepsChainWhenNoBlockIsUnused) {
    constexpr size_t block_size = 4;
    CacheFixture fixture(/*num_blocks=*/2);
    auto offload_cache = make_offload_cache(fixture, 4);

    BlockManager block_manager(/*num_blocks=*/2, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);
    block_manager.set_overwritten_block_observer(offload_cache.get());

    const std::vector<int64_t> wanted_tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto producer = make_group(wanted_tokens, /*request_id=*/9);
    producer->schedule_tokens(wanted_tokens.size());
    block_manager.append_slots(producer);
    producer->finish_iteration();
    block_manager.free_sequence(producer->get_running_sequences().at(0)->get_id());

    // Push both blocks of the wanted prefix to disk and leave the cache full of unrelated blocks.
    const std::vector<int64_t> other_tokens = {40, 41, 42, 43, 44, 45, 46, 47};
    auto other = make_group(other_tokens, /*request_id=*/10);
    other->schedule_tokens(other_tokens.size());
    block_manager.append_slots(other);
    other->finish_iteration();
    block_manager.free_sequence(other->get_running_sequences().at(0)->get_id());
    ASSERT_EQ(offload_cache->get_num_entries(), 2);

    auto consumer = make_group(wanted_tokens, /*request_id=*/11);
    // Both chain blocks have to survive warming even though every block has to be taken from the cache.
    {
        KVCacheOffloadCache::ScopedReclamationPause keep_entries(*offload_cache);
        EXPECT_EQ(block_manager.warm_prefix_cache(consumer, *offload_cache), 2);
    }
    ASSERT_TRUE(block_manager.restore_cached_blocks(consumer));

    const auto consumer_seq_id = consumer->get_running_sequences().at(0)->get_id();
    EXPECT_EQ(block_manager.get_block_table(consumer_seq_id, 0).size(), 2);

    block_manager.set_overwritten_block_observer(nullptr);
    block_manager.free_sequence(consumer_seq_id);
}

TEST(TestKVCacheOffloadCache, StoresBlocksFreedByPartialRelease) {
    constexpr size_t block_size = 4;
    CacheFixture fixture(/*num_blocks=*/4);
    auto offload_cache = make_offload_cache(fixture, 4);

    BlockManager block_manager(/*num_blocks=*/4, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);
    block_manager.set_overwritten_block_observer(offload_cache.get());

    const std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto preempted = make_group(tokens, /*request_id=*/12);
    preempted->schedule_tokens(tokens.size());
    block_manager.append_slots(preempted);
    preempted->finish_iteration();

    const auto seq_id = preempted->get_running_sequences().at(0)->get_id();
    ASSERT_EQ(block_manager.get_block_table(seq_id, 0).size(), 2);

    // Preemption releases blocks from the tail of the sequence.
    block_manager.free_sequence_partially(seq_id, 1);
    EXPECT_EQ(block_manager.get_block_table(seq_id, 0).size(), 1);

    // Fill the cache so the released block is overwritten and therefore offloaded.
    const std::vector<int64_t> other_tokens = {50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61};
    auto other = make_group(other_tokens, /*request_id=*/13);
    other->schedule_tokens(other_tokens.size());
    block_manager.append_slots(other);

    EXPECT_GT(offload_cache->get_num_entries(), 0);

    block_manager.set_overwritten_block_observer(nullptr);
    block_manager.free_sequence(seq_id);
    block_manager.free_sequence(other->get_running_sequences().at(0)->get_id());
}

TEST(TestKVCacheOffloadCache, HandlesCopyOnWriteOfSharedBlock) {
    constexpr size_t block_size = 4;
    CacheFixture fixture(/*num_blocks=*/8);
    auto offload_cache = make_offload_cache(fixture, 8);

    BlockManager block_manager(/*num_blocks=*/8, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);
    block_manager.set_overwritten_block_observer(offload_cache.get());

    // The trailing block stays incomplete, so consumers sharing it have to split it on the next token.
    const std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5};
    auto producer = make_group(tokens, /*request_id=*/14);
    producer->schedule_tokens(tokens.size());
    block_manager.append_slots(producer);
    producer->finish_iteration();
    const auto producer_seq_id = producer->get_running_sequences().at(0)->get_id();
    const auto shared_block_idx = block_manager.get_block_table(producer_seq_id, 0).at(1)->get_index();
    block_manager.free_sequence(producer_seq_id);

    auto first = make_group(tokens, /*request_id=*/15);
    auto second = make_group(tokens, /*request_id=*/16);
    ASSERT_TRUE(block_manager.restore_cached_blocks(first));
    ASSERT_TRUE(block_manager.restore_cached_blocks(second));

    first->schedule_tokens(1);
    second->schedule_tokens(1);
    const auto first_copy_map = block_manager.append_slots(first);
    const auto second_copy_map = block_manager.append_slots(second);

    EXPECT_TRUE(first_copy_map.count(shared_block_idx));
    EXPECT_TRUE(second_copy_map.empty());
    // Copy-on-write must not publish anything to disk, since no cached contents were overwritten.
    EXPECT_EQ(offload_cache->get_num_entries(), 0);

    block_manager.set_overwritten_block_observer(nullptr);
    block_manager.free_sequence(first->get_running_sequences().at(0)->get_id());
    block_manager.free_sequence(second->get_running_sequences().at(0)->get_id());
}
