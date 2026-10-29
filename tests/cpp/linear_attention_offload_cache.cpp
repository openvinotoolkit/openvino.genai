// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <thread>
#include <vector>

#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/cache/file_kv_cache_store.hpp"
#include "continuous_batching/cache/linear_attention_cache_manager.hpp"
#include "continuous_batching/cache/linear_attention_offload_cache.hpp"
#include "continuous_batching/cache/kv_cache_offload_manager.hpp"
#include "helper.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/runtime/core.hpp"
#include "sequence_group.hpp"
#include "utils.hpp"

using namespace ov::genai;

namespace {

constexpr size_t NUM_LA_LAYERS = 2;
constexpr size_t NUM_PHYSICAL_BLOCKS = 4;

/// Owns the OpenVINO objects a LinearAttentionCacheManager needs so that tests can allocate a real CPU cache.
struct LaCacheFixture {
    ov::Core core;
    ov::InferRequest request;
    std::shared_ptr<LinearAttentionCacheManager> cache_manager;

    explicit LaCacheFixture(size_t num_blocks = NUM_PHYSICAL_BLOCKS) {
        request = core.compile_model(get_dummy_hybrid_model(core, /*kv_num_layers=*/0, NUM_LA_LAYERS))
                      .create_infer_request();
        cache_manager = std::make_shared<LinearAttentionCacheManager>(request);
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

CacheOffloadConfig make_offload_config(const LaCacheFixture& fixture, size_t num_slots) {
    CacheOffloadConfig config;
    config.capacity_bytes = fixture.cache_manager->get_block_layout().get_slot_size() * num_slots;
    return config;
}

std::unique_ptr<LinearAttentionOffloadCache> make_offload_cache(LaCacheFixture& fixture, size_t num_slots) {
    auto backend = std::make_unique<KVCacheOffloadManager>(fixture.cache_manager->get_block_layout(),
                                                           make_offload_config(fixture, num_slots),
                                                           "CPU");
    auto store = std::make_unique<FileKVCacheStore>(std::move(backend));
    return std::make_unique<LinearAttentionOffloadCache>(*fixture.cache_manager, std::move(store), /*max_queued_stores=*/8);
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

TEST(TestLinearAttentionOffloadCache, StoresBlockContentsOnOverwrite) {
    LaCacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto expected = fixture.fill_block(1, /*seed=*/7);

    offload_cache->on_blocks_overwritten(/*hash=*/42, make_block_set(1));
    offload_cache->flush();

    EXPECT_TRUE(offload_cache->contains(42));
    EXPECT_EQ(offload_cache->get_num_entries(), 1);
    EXPECT_EQ(offload_cache->get_statistics().num_stored, 1);

    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(42, restored));
    EXPECT_EQ(restored, expected);
}

TEST(TestLinearAttentionOffloadCache, KeepsFirstCopyOfKnownHash) {
    LaCacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto expected = fixture.fill_block(0, /*seed=*/3);
    offload_cache->on_blocks_overwritten(/*hash=*/5, make_block_set(0));

    fixture.fill_block(1, /*seed=*/200);
    offload_cache->on_blocks_overwritten(/*hash=*/5, make_block_set(1));
    offload_cache->flush();

    EXPECT_EQ(offload_cache->get_num_entries(), 1);
    EXPECT_EQ(offload_cache->get_statistics().num_stored, 1);
    EXPECT_EQ(offload_cache->get_statistics().num_already_present, 1);

    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(5, restored));
    EXPECT_EQ(restored, expected);
}

TEST(TestLinearAttentionOffloadCache, ReplacesOldestEntryWhenFileIsFull) {
    LaCacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);
    fixture.fill_block(0, /*seed=*/1);
    offload_cache->on_blocks_overwritten(/*hash=*/100, make_block_set(0));
    offload_cache->flush();
    ASSERT_EQ(offload_cache->get_num_free_slots(), 0);

    const auto newer = fixture.fill_block(1, /*seed=*/60);
    offload_cache->on_blocks_overwritten(/*hash=*/200, make_block_set(1));
    offload_cache->flush();

    EXPECT_EQ(offload_cache->get_num_entries(), 1);
    EXPECT_FALSE(offload_cache->contains(100));
    EXPECT_EQ(offload_cache->get_statistics().num_replaced, 1);

    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(200, restored));
    EXPECT_EQ(restored, newer);
}

TEST(TestLinearAttentionOffloadCache, ReadReportsMissForUnknownHash) {
    LaCacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);

    std::vector<uint8_t> restored;
    EXPECT_FALSE(offload_cache->read(/*hash=*/999, restored));
    EXPECT_TRUE(restored.empty());
}

TEST(TestLinearAttentionOffloadCache, RejectsPerLayerBlockTables) {
    LaCacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);
    BlocksPerLayer per_layer_blocks{std::make_shared<CacheBlock>(0), std::make_shared<CacheBlock>(1)};

    EXPECT_THROW(offload_cache->on_blocks_overwritten(/*hash=*/1, per_layer_blocks), ov::Exception);
}

TEST(TestLinearAttentionOffloadCache, LoadsStoredContentsIntoAnotherBlock) {
    LaCacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto expected = fixture.fill_block(0, /*seed=*/23);
    offload_cache->on_blocks_overwritten(/*hash=*/77, make_block_set(0));
    offload_cache->flush();
    fixture.fill_block(2, /*seed=*/0);

    ASSERT_TRUE(offload_cache->load_into(/*hash=*/77, /*block_index=*/2));

    EXPECT_EQ(fixture.cache_manager->read_block(2), expected);
    EXPECT_EQ(offload_cache->get_statistics().num_loaded, 1);
}

TEST(TestLinearAttentionOffloadCache, LoadReportsMissForUnknownHash) {
    LaCacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);
    const auto untouched = fixture.fill_block(1, /*seed=*/9);

    EXPECT_FALSE(offload_cache->load_into(/*hash=*/12345, /*block_index=*/1));
    EXPECT_EQ(fixture.cache_manager->read_block(1), untouched);
}

TEST(TestLinearAttentionOffloadCache, BlockManagerWarmsPrefixBlockFromOffloadCache) {
    constexpr size_t cache_interval = 4;
    LaCacheFixture fixture(/*num_blocks=*/2);
    auto offload_cache = make_offload_cache(fixture, 2);

    BlockManager block_manager(/*num_blocks=*/2, /*enable_prefix_caching=*/true, cache_interval, /*num_layers=*/1);
    block_manager.set_overwritten_block_observer(offload_cache.get());

    const std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto consumer = make_group(tokens, /*request_id=*/3);
    const size_t prefix_hash = consumer->get_running_sequences().at(0)->get_hash(cache_interval, cache_interval);
    fixture.fill_block(0, /*seed=*/31);
    offload_cache->on_blocks_overwritten(prefix_hash, make_block_set(0));
    offload_cache->flush();
    ASSERT_GT(offload_cache->get_num_entries(), 0);

    EXPECT_GT(block_manager.warm_prefix_cache(consumer, *offload_cache), 0);
    EXPECT_TRUE(block_manager.restore_cached_blocks(consumer));

    block_manager.set_overwritten_block_observer(nullptr);
    block_manager.free_sequence(consumer->get_running_sequences().at(0)->get_id());
}

TEST(TestLinearAttentionOffloadCache, PersistStoresNewHashSynchronously) {
    LaCacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto expected = fixture.fill_block(1, /*seed=*/7);

    ASSERT_TRUE(offload_cache->persist(/*hash=*/42, /*block_index=*/1));

    EXPECT_EQ(offload_cache->get_num_entries(), 1);
    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(42, restored));
    EXPECT_EQ(restored, expected);
}

TEST(TestLinearAttentionOffloadCache, PersistOfKnownHashKeepsTheStoredCopy) {
    LaCacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto expected = fixture.fill_block(0, /*seed=*/3);
    offload_cache->on_blocks_overwritten(/*hash=*/5, make_block_set(0));
    offload_cache->flush();
    fixture.fill_block(1, /*seed=*/200);

    ASSERT_TRUE(offload_cache->persist(/*hash=*/5, /*block_index=*/1));

    EXPECT_EQ(offload_cache->get_statistics().num_stored, 1);
    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(5, restored));
    EXPECT_EQ(restored, expected);
}

TEST(TestLinearAttentionOffloadCache, PersistWaitsForAQueuedStoreOfTheSameHash) {
    LaCacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto expected = fixture.fill_block(0, /*seed=*/9);
    offload_cache->on_blocks_overwritten(/*hash=*/7, make_block_set(0));

    ASSERT_TRUE(offload_cache->persist(/*hash=*/7, /*block_index=*/0));

    EXPECT_EQ(offload_cache->get_num_entries(), 1);
    EXPECT_EQ(offload_cache->get_statistics().num_stored, 1);
    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(7, restored));
    EXPECT_EQ(restored, expected);
}

TEST(TestLinearAttentionOffloadCache, PersistReclaimsTheOldestUnpinnedEntryWhenFull) {
    LaCacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    fixture.fill_block(0, /*seed=*/1);
    fixture.fill_block(1, /*seed=*/2);
    offload_cache->on_blocks_overwritten(/*hash=*/1, make_block_set(0));
    offload_cache->on_blocks_overwritten(/*hash=*/2, make_block_set(1));
    offload_cache->flush();
    ASSERT_TRUE(offload_cache->pin(1));
    const auto newest = fixture.fill_block(2, /*seed=*/50);

    ASSERT_TRUE(offload_cache->persist(/*hash=*/3, /*block_index=*/2));

    EXPECT_TRUE(offload_cache->contains(1));
    EXPECT_FALSE(offload_cache->contains(2));
    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(3, restored));
    EXPECT_EQ(restored, newest);
    EXPECT_EQ(offload_cache->get_statistics().num_replaced, 1);
}

TEST(TestLinearAttentionOffloadCache, PersistFailsWithoutSideEffectsWhenEveryEntryIsPinned) {
    LaCacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);
    fixture.fill_block(0, /*seed=*/1);
    fixture.fill_block(1, /*seed=*/2);
    ASSERT_TRUE(offload_cache->persist(/*hash=*/1, /*block_index=*/0));

    EXPECT_FALSE(offload_cache->persist(/*hash=*/2, /*block_index=*/1));

    EXPECT_TRUE(offload_cache->contains(1));
    EXPECT_FALSE(offload_cache->contains(2));
    EXPECT_EQ(offload_cache->get_num_entries(), 1);
}

TEST(TestLinearAttentionOffloadCache, PinnedEntrySurvivesOverwriteReclamationUntilUnpinned) {
    LaCacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);
    fixture.fill_block(0, /*seed=*/1);
    fixture.fill_block(1, /*seed=*/2);
    ASSERT_TRUE(offload_cache->persist(/*hash=*/1, /*block_index=*/0));

    offload_cache->on_blocks_overwritten(/*hash=*/2, make_block_set(1));
    offload_cache->flush();
    EXPECT_TRUE(offload_cache->contains(1));
    EXPECT_FALSE(offload_cache->contains(2));
    EXPECT_EQ(offload_cache->get_statistics().num_failed, 1);

    offload_cache->unpin(1);
    offload_cache->on_blocks_overwritten(/*hash=*/3, make_block_set(1));
    offload_cache->flush();
    EXPECT_FALSE(offload_cache->contains(1));
    EXPECT_TRUE(offload_cache->contains(3));
}

TEST(TestLinearAttentionOffloadCache, PinsAreReferenceCounted) {
    LaCacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);
    fixture.fill_block(0, /*seed=*/1);
    fixture.fill_block(1, /*seed=*/2);

    EXPECT_FALSE(offload_cache->pin(/*hash=*/999));
    EXPECT_NO_THROW(offload_cache->unpin(/*hash=*/999));

    ASSERT_TRUE(offload_cache->persist(/*hash=*/1, /*block_index=*/0));
    ASSERT_TRUE(offload_cache->persist(/*hash=*/1, /*block_index=*/0));
    offload_cache->unpin(1);
    offload_cache->on_blocks_overwritten(/*hash=*/2, make_block_set(1));
    offload_cache->flush();
    EXPECT_TRUE(offload_cache->contains(1));

    offload_cache->unpin(1);
    offload_cache->on_blocks_overwritten(/*hash=*/2, make_block_set(1));
    offload_cache->flush();
    EXPECT_FALSE(offload_cache->contains(1));
    EXPECT_TRUE(offload_cache->contains(2));
}

TEST(TestLinearAttentionOffloadCache, PersistWithQueuedStoresInFlightDoesNotDeadlockOrLoseEntries) {
    LaCacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 4);
    for (size_t i = 0; i < 3; ++i) {
        fixture.fill_block(i, /*seed=*/static_cast<uint8_t>(10 * (i + 1)));
        offload_cache->on_blocks_overwritten(/*hash=*/100 + i, make_block_set(static_cast<int>(i)));
    }
    const auto last = fixture.fill_block(3, /*seed=*/99);

    ASSERT_TRUE(offload_cache->persist(/*hash=*/200, /*block_index=*/3));

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(offload_cache->contains(100 + i));
    }
    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(200, restored));
    EXPECT_EQ(restored, last);
    EXPECT_EQ(offload_cache->get_num_entries(), 4);
}

namespace {

/// A snapshot keeps one checkpoint, at the last computed position (num_tokens - 1), in the store. A latest-only
/// block manager must restore from it even though none of the earlier checkpoints exist.
void expect_snapshot_checkpoint_is_restored(size_t num_tokens) {
    constexpr size_t cache_interval = 4;
    LaCacheFixture fixture(/*num_blocks=*/2);
    auto offload_cache = make_offload_cache(fixture, 2);
    BlockManager block_manager(/*num_blocks=*/2,
                               /*enable_prefix_caching=*/true,
                               cache_interval,
                               /*num_layers=*/1,
                               /*fixed_blocks_per_sequence=*/0,
                               /*restore_latest_prefix_block_only=*/true);

    std::vector<int64_t> tokens(num_tokens);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = static_cast<int64_t>(i);
    }
    auto consumer = make_group(tokens, /*request_id=*/30);
    const auto sequence = consumer->get_running_sequences().at(0);
    const auto expected = fixture.fill_block(0, /*seed=*/41);
    ASSERT_TRUE(offload_cache->persist(sequence->get_hash(num_tokens - 1, cache_interval), /*block_index=*/0));

    EXPECT_EQ(block_manager.warm_prefix_cache(consumer, *offload_cache), 1);
    EXPECT_TRUE(block_manager.restore_cached_blocks(consumer));
    EXPECT_EQ(consumer->get_num_processed_tokens(), num_tokens - 1);

    block_manager.free_sequence(sequence->get_id());
}

}  // namespace

TEST(TestLinearAttentionOffloadCache, WarmRestoresASnapshotCheckpointThatIsNotIntervalAligned) {
    expect_snapshot_checkpoint_is_restored(/*num_tokens=*/10);
}

TEST(TestLinearAttentionOffloadCache, WarmRestoresASnapshotCheckpointOnAnIntervalBoundary) {
    expect_snapshot_checkpoint_is_restored(/*num_tokens=*/9);
}

TEST(TestLinearAttentionOffloadCache, WarmRestoresASnapshotCheckpointWhenThePromptEndsOnAnIntervalBoundary) {
    expect_snapshot_checkpoint_is_restored(/*num_tokens=*/8);
}

TEST(TestLinearAttentionOffloadCache, ConcurrentOverwritesNeverEvictAPinnedEntry) {
    LaCacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    fixture.fill_block(0, /*seed=*/1);
    fixture.fill_block(1, /*seed=*/2);
    ASSERT_TRUE(offload_cache->persist(/*hash=*/1000, /*block_index=*/0));

    std::thread overwriter([&] {
        for (size_t i = 0; i < 200; ++i) {
            offload_cache->on_blocks_overwritten(/*hash=*/i, make_block_set(1));
        }
    });
    for (size_t i = 0; i < 50; ++i) {
        offload_cache->persist(/*hash=*/2000 + i, /*block_index=*/1);
        offload_cache->unpin(2000 + i);
    }
    overwriter.join();
    offload_cache->flush();

    EXPECT_TRUE(offload_cache->contains(1000));
    EXPECT_LE(offload_cache->get_num_entries(), 2);
}
