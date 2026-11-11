// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/cache/file_kv_cache_store.hpp"
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
    auto store = std::make_unique<FileKVCacheStore>(std::move(backend));
    return std::make_unique<KVCacheOffloadCache>(*fixture.cache_manager, std::move(store), /*max_queued_stores=*/8);
}

BlocksPerLayer make_block_set(int physical_block_id) {
    return BlocksPerLayer{std::make_shared<CacheBlock>(physical_block_id)};
}

SequenceGroup::Ptr make_group(const std::vector<int64_t>& tokens, uint64_t request_id) {
    return std::make_shared<SequenceGroup>(request_id,
                                           ov::Tensor(ov::element::i64, {tokens.size()}, const_cast<int64_t*>(tokens.data())),
                                           utils::get_greedy_config());
}

/// Stores the contents of a physical block under `hash`, the way the block manager does when the block is overwritten.
template <typename OffloadCacheT>
void store_block(OffloadCacheT& cache, size_t hash, size_t block_index) {
    cache.on_blocks_overwritten(hash, make_block_set(static_cast<int>(block_index)));
    cache.flush();
}

}  // namespace

TEST(TestKVCacheOffloadCache, StoresBlockContentsOnOverwrite) {
    CacheFixture fixture;
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

TEST(TestKVCacheOffloadCache, KeepsFirstCopyOfKnownHash) {
    CacheFixture fixture;
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

TEST(TestKVCacheOffloadCache, KeepsDistinctHashesInSeparateSlots) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto first = fixture.fill_block(0, /*seed=*/11);
    const auto second = fixture.fill_block(1, /*seed=*/77);

    offload_cache->on_blocks_overwritten(/*hash=*/1, make_block_set(0));
    offload_cache->on_blocks_overwritten(/*hash=*/2, make_block_set(1));
    offload_cache->flush();

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
    offload_cache->flush();

    EXPECT_GT(offload_cache->get_num_entries(), 0);
    EXPECT_EQ(offload_cache->get_statistics().num_stored, offload_cache->get_num_entries());

    block_manager.set_overwritten_block_observer(nullptr);
    block_manager.free_sequence(pressure->get_running_sequences().at(0)->get_id());
}

TEST(TestKVCacheOffloadCache, PublishedBlockIsOnlyStoredWhenStoringCompletedBlocksIsEnabled) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    fixture.fill_block(0, /*seed=*/6);

    offload_cache->on_blocks_published(/*hash=*/1, make_block_set(0));
    offload_cache->flush();
    EXPECT_FALSE(offload_cache->contains(1));

    offload_cache->set_store_completed_blocks(true);
    offload_cache->on_blocks_published(/*hash=*/1, make_block_set(0));
    offload_cache->flush();
    EXPECT_TRUE(offload_cache->contains(1));
    EXPECT_EQ(offload_cache->get_statistics().num_stored, 1);

    // A block stored when it was completed is not stored a second time when it is overwritten.
    offload_cache->on_blocks_overwritten(/*hash=*/1, make_block_set(0));
    offload_cache->flush();
    EXPECT_EQ(offload_cache->get_statistics().num_stored, 1);
    EXPECT_EQ(offload_cache->get_statistics().num_already_present, 1);
}

TEST(TestKVCacheOffloadCache, BlockManagerStoresCompletedBlocksWithoutAnyEviction) {
    constexpr size_t block_size = 4;
    CacheFixture fixture(/*num_blocks=*/4);
    auto offload_cache = make_offload_cache(fixture, 4);
    offload_cache->set_store_completed_blocks(true);

    BlockManager block_manager(/*num_blocks=*/4, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);
    block_manager.set_overwritten_block_observer(offload_cache.get());

    const std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto group = make_group(tokens, /*request_id=*/1);
    const auto sequence = group->get_running_sequences().at(0);
    group->schedule_tokens(tokens.size());
    block_manager.append_slots(group);
    group->finish_iteration();
    block_manager.publish_completed_blocks(sequence, 0, tokens.size());
    offload_cache->flush();

    EXPECT_EQ(offload_cache->get_num_entries(), 2u) << "both full blocks are stored although nothing was overwritten";

    // Publishing a range that is already published stores nothing more.
    block_manager.publish_completed_blocks(sequence, 0, tokens.size());
    offload_cache->flush();
    EXPECT_EQ(offload_cache->get_statistics().num_stored, 2u);

    block_manager.set_overwritten_block_observer(nullptr);
    block_manager.free_sequence(sequence->get_id());
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
    offload_cache->flush();

    EXPECT_EQ(offload_cache->get_num_entries(), 0);

    block_manager.free_sequence(pressure->get_running_sequences().at(0)->get_id());
}

TEST(TestKVCacheOffloadCache, LoadsStoredContentsIntoAnotherBlock) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto expected = fixture.fill_block(0, /*seed=*/23);
    offload_cache->on_blocks_overwritten(/*hash=*/77, make_block_set(0));
    offload_cache->flush();
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
    offload_cache->flush();
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
    offload_cache->flush();
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
    offload_cache->flush();

    EXPECT_GT(offload_cache->get_num_entries(), 0);

    block_manager.set_overwritten_block_observer(nullptr);
    block_manager.free_sequence(seq_id);
    block_manager.free_sequence(other->get_running_sequences().at(0)->get_id());
}

TEST(TestKVCacheOffloadCache, ServesStoredBlockWhileTheWriteIsStillQueued) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    const auto expected = fixture.fill_block(0, /*seed=*/31);

    // The store hands off to a background writer, yet the contents must be readable right away.
    offload_cache->on_blocks_overwritten(/*hash=*/64, make_block_set(0));
    EXPECT_TRUE(offload_cache->contains(64));

    fixture.fill_block(3, /*seed=*/0);
    ASSERT_TRUE(offload_cache->load_into(/*hash=*/64, /*block_index=*/3));
    EXPECT_EQ(fixture.cache_manager->read_block(3), expected);

    offload_cache->flush();
    EXPECT_EQ(offload_cache->get_num_entries(), 1);
    std::vector<uint8_t> restored;
    ASSERT_TRUE(offload_cache->read(64, restored));
    EXPECT_EQ(restored, expected);
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
    offload_cache->flush();
    EXPECT_EQ(offload_cache->get_num_entries(), 0);

    block_manager.set_overwritten_block_observer(nullptr);
    block_manager.free_sequence(first->get_running_sequences().at(0)->get_id());
    block_manager.free_sequence(second->get_running_sequences().at(0)->get_id());
}

TEST(TestKVCacheOffloadCache, ConcurrentOverwritesNeverExceedTheStoreCapacity) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 2);
    fixture.fill_block(0, /*seed=*/1);
    fixture.fill_block(1, /*seed=*/2);

    std::thread overwriter([&] {
        for (size_t i = 0; i < 200; ++i) {
            offload_cache->on_blocks_overwritten(/*hash=*/i, make_block_set(1));
        }
    });
    for (size_t i = 0; i < 200; ++i) {
        offload_cache->on_blocks_overwritten(/*hash=*/1000 + i, make_block_set(0));
    }
    overwriter.join();
    offload_cache->flush();

    EXPECT_LE(offload_cache->get_num_entries(), 2);
}

namespace {

/// Lets several offload caches see one store, the way pipelines do through a persistent plugin.
class SharedStore : public IKVCacheStore {
public:
    explicit SharedStore(std::shared_ptr<IKVCacheStore> inner) : m_inner(std::move(inner)) {}

    bool has_block(size_t hash) const override {
        return m_inner->has_block(hash);
    }
    bool write_block(size_t hash, const std::vector<uint8_t>& data) override {
        return m_inner->write_block(hash, data);
    }
    bool read_block(size_t hash, std::vector<uint8_t>& data) const override {
        return m_inner->read_block(hash, data);
    }
    void drop_block(size_t hash) override {
        m_inner->drop_block(hash);
    }
    size_t get_block_size() const override {
        return m_inner->get_block_size();
    }
    size_t get_capacity_blocks() const override {
        return m_inner->get_capacity_blocks();
    }
    size_t get_num_free_blocks() const override {
        return m_inner->get_num_free_blocks();
    }
    void flush() override {
        m_inner->flush();
    }

private:
    std::shared_ptr<IKVCacheStore> m_inner;
};

std::shared_ptr<IKVCacheStore> make_shared_file_store(CacheFixture& fixture, size_t num_slots) {
    auto backend = std::make_unique<KVCacheOffloadManager>(fixture.cache_manager->get_block_layout(),
                                                           make_offload_config(fixture, num_slots),
                                                           "CPU");
    return std::make_shared<FileKVCacheStore>(std::move(backend));
}

std::unique_ptr<KVCacheOffloadCache> make_offload_cache_on(CacheFixture& fixture, std::shared_ptr<IKVCacheStore> store) {
    return std::make_unique<KVCacheOffloadCache>(*fixture.cache_manager,
                                                 std::make_unique<SharedStore>(std::move(store)),
                                                 /*max_queued_stores=*/8);
}

}  // namespace

TEST(TestKVCacheOffloadCache, AnotherInstanceServesEntriesItNeverWrote) {
    CacheFixture fixture;
    auto store = make_shared_file_store(fixture, 1);
    const auto expected = fixture.fill_block(0, /*seed=*/17);
    {
        auto first = make_offload_cache_on(fixture, store);
        first->on_blocks_overwritten(/*hash=*/42, make_block_set(0));
        first->flush();
    }

    auto second = make_offload_cache_on(fixture, store);
    EXPECT_EQ(second->get_num_entries(), 0);
    // The store is only asked through probe(); contains() answers from what the probes found.
    EXPECT_EQ(second->probe({42, 43}), (std::vector<bool>{true, false}));
    EXPECT_TRUE(second->contains(42));
    EXPECT_FALSE(second->contains(43));

    std::vector<uint8_t> restored;
    ASSERT_TRUE(second->read(42, restored));
    EXPECT_EQ(restored, expected);
    fixture.fill_block(1, /*seed=*/0);
    ASSERT_TRUE(second->load_into(/*hash=*/42, /*block_index=*/1));
    EXPECT_EQ(fixture.cache_manager->read_block(1), expected);
}

TEST(TestKVCacheOffloadCache, ReportsWhetherItsStoreIsAPlugin) {
    CacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);
    EXPECT_FALSE(offload_cache->is_plugin_backed());
}

namespace {

/// Records every question and write the host puts to the store, in order.
class RecordingStore : public IKVCacheStore {
public:
    RecordingStore(std::shared_ptr<IKVCacheStore> inner, bool plugin_backed, std::vector<std::string>& events)
        : m_inner(std::move(inner)),
          m_plugin_backed(plugin_backed),
          m_events(events) {}

    bool has_block(size_t hash) const override {
        m_events.push_back("has_block");
        return m_inner->has_block(hash);
    }
    std::vector<bool> has_blocks(const std::vector<size_t>& chain) const override {
        std::string event = "has_blocks";
        for (const size_t hash : chain) {
            event += " " + std::to_string(hash);
        }
        m_events.push_back(event);
        return m_inner->has_blocks(chain);
    }
    bool write_block(size_t hash, const std::vector<uint8_t>& data) override {
        m_events.push_back("write_block " + std::to_string(hash));
        return m_inner->write_block(hash, data);
    }
    bool read_block(size_t hash, std::vector<uint8_t>& data) const override {
        m_events.push_back("read_block " + std::to_string(hash));
        return m_inner->read_block(hash, data);
    }
    std::vector<bool> read_blocks(const std::vector<size_t>& hashes, std::vector<std::vector<uint8_t>>& data) const override {
        m_events.push_back("read_blocks " + std::to_string(hashes.size()));
        return m_inner->read_blocks(hashes, data);
    }
    void drop_block(size_t hash) override {
        m_inner->drop_block(hash);
    }
    size_t get_block_size() const override {
        return m_inner->get_block_size();
    }
    size_t get_capacity_blocks() const override {
        return m_inner->get_capacity_blocks();
    }
    size_t get_num_free_blocks() const override {
        return m_inner->get_num_free_blocks();
    }
    void flush() override {
        m_events.push_back("flush");
        m_inner->flush();
    }
    bool is_plugin_backed() const override {
        return m_plugin_backed;
    }

private:
    std::shared_ptr<IKVCacheStore> m_inner;
    bool m_plugin_backed;
    std::vector<std::string>& m_events;
};

std::unique_ptr<KVCacheOffloadCache> make_recording_cache(CacheFixture& fixture,
                                                          bool plugin_backed,
                                                          std::vector<std::string>& events,
                                                          size_t num_slots = 4) {
    return std::make_unique<KVCacheOffloadCache>(
        *fixture.cache_manager,
        std::make_unique<RecordingStore>(make_shared_file_store(fixture, num_slots), plugin_backed, events),
        /*max_queued_stores=*/8);
}

}  // namespace

TEST(TestKVCacheOffloadCache, PluginStoreWriteIsPrecededByAProbeOfItsChainFromBlockZero) {
    CacheFixture fixture;
    std::vector<std::string> events;
    auto offload_cache = make_recording_cache(fixture, /*plugin_backed=*/true, events);
    fixture.fill_block(0, /*seed=*/3);

    offload_cache->on_prefix_link(/*hash=*/11, /*parent_hash=*/0, /*has_parent=*/false);
    offload_cache->on_prefix_link(/*hash=*/22, /*parent_hash=*/11, /*has_parent=*/true);
    offload_cache->on_prefix_link(/*hash=*/33, /*parent_hash=*/22, /*has_parent=*/true);
    offload_cache->on_blocks_overwritten(/*hash=*/33, make_block_set(0));
    offload_cache->flush();

    const std::vector<std::string> expected{"has_blocks 11 22 33", "write_block 33"};
    EXPECT_EQ(events, expected);
}

TEST(TestKVCacheOffloadCache, FileStoreIsNotProbedBeforeWrites) {
    CacheFixture fixture;
    std::vector<std::string> events;
    auto offload_cache = make_recording_cache(fixture, /*plugin_backed=*/false, events);
    fixture.fill_block(0, /*seed=*/3);

    offload_cache->on_prefix_link(/*hash=*/11, /*parent_hash=*/0, /*has_parent=*/false);
    offload_cache->on_blocks_overwritten(/*hash=*/11, make_block_set(0));
    offload_cache->flush();

    const std::vector<std::string> expected{"write_block 11"};
    EXPECT_EQ(events, expected);
}

TEST(TestKVCacheOffloadCache, ProbeReportsEveryBlockOfTheChainInOrderWithOneStoreCall) {
    CacheFixture fixture;
    std::vector<std::string> events;
    auto offload_cache = make_recording_cache(fixture, /*plugin_backed=*/true, events);
    fixture.fill_block(0, /*seed=*/3);
    store_block(*offload_cache, /*hash=*/22, /*block_index=*/0);
    events.clear();

    EXPECT_EQ(offload_cache->probe({11, 22, 33}), (std::vector<bool>{false, true, false}));
    EXPECT_EQ(events, (std::vector<std::string>{"has_blocks 11 22 33"}));
}

namespace {

/// Keeps at most `max_blocks` blocks and drops the oldest by itself, the way a plugin with OV_KVCACHE_CAP_SELF_EVICT
/// does. It reports no free blocks at all, so a host that still counted capacity would refuse every write.
class SelfEvictingStore : public IKVCacheStore {
public:
    SelfEvictingStore(size_t block_size, size_t max_blocks) : m_block_size(block_size), m_max_blocks(max_blocks) {}

    bool has_block(size_t hash) const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_blocks.count(hash) != 0;
    }
    bool write_block(size_t hash, const std::vector<uint8_t>& data) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_num_writes;
        if (m_blocks.erase(hash) != 0) {
            m_order.remove(hash);
        }
        while (m_blocks.size() >= m_max_blocks) {
            m_blocks.erase(m_order.front());
            m_order.pop_front();
        }
        m_blocks[hash] = data;
        m_order.push_back(hash);
        return true;
    }
    bool read_block(size_t hash, std::vector<uint8_t>& data) const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto it = m_blocks.find(hash);
        if (it == m_blocks.end()) {
            return false;
        }
        data = it->second;
        return true;
    }
    void drop_block(size_t) override {
        ++m_num_host_drops;
    }
    size_t get_block_size() const override {
        return m_block_size;
    }
    size_t get_capacity_blocks() const override {
        return m_max_blocks;
    }
    size_t get_num_free_blocks() const override {
        return 0;
    }
    bool is_plugin_backed() const override {
        return true;
    }
    bool evicts_itself() const override {
        return true;
    }

    size_t num_writes() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_num_writes;
    }
    size_t num_host_drops() const {
        return m_num_host_drops;
    }

private:
    size_t m_block_size;
    size_t m_max_blocks;
    mutable std::mutex m_mutex;
    std::map<size_t, std::vector<uint8_t>> m_blocks;
    std::list<size_t> m_order;
    size_t m_num_writes = 0;
    std::atomic<size_t> m_num_host_drops{0};
};

struct SelfEvictingCache {
    SelfEvictingCache(CacheFixture& fixture, size_t max_blocks) {
        auto owned = std::make_unique<SelfEvictingStore>(fixture.cache_manager->get_block_layout().get_slot_size(),
                                                         max_blocks);
        store = owned.get();
        cache = std::make_unique<KVCacheOffloadCache>(*fixture.cache_manager, std::move(owned), /*max_queued_stores=*/8);
    }

    SelfEvictingStore* store = nullptr;
    std::unique_ptr<KVCacheOffloadCache> cache;
};

}  // namespace

TEST(TestKVCacheOffloadCache, SelfEvictingStoreIsNeverToldToDropBlocksAndKeepsAcceptingWrites) {
    CacheFixture fixture;
    SelfEvictingCache offload(fixture, /*max_blocks=*/2);

    for (size_t hash = 1; hash <= 4; ++hash) {
        fixture.fill_block(0, static_cast<uint8_t>(hash));
        offload.cache->on_blocks_overwritten(hash, make_block_set(0));
        offload.cache->flush();
    }

    EXPECT_EQ(offload.store->num_host_drops(), 0u);
    EXPECT_EQ(offload.cache->get_statistics().num_failed, 0u);
    EXPECT_EQ(offload.cache->get_statistics().num_stored, 4u);
    // The store's own policy decided that the two oldest blocks go, and contains() follows it.
    EXPECT_FALSE(offload.cache->contains(1));
    EXPECT_FALSE(offload.cache->contains(2));
    EXPECT_TRUE(offload.cache->contains(3));
    EXPECT_TRUE(offload.cache->contains(4));
}

TEST(TestKVCacheOffloadCache, LoadingABlockTheSelfEvictingStoreDroppedIsAMiss) {
    CacheFixture fixture;
    SelfEvictingCache offload(fixture, /*max_blocks=*/2);
    std::vector<uint8_t> newest;
    for (size_t hash = 1; hash <= 3; ++hash) {
        newest = fixture.fill_block(0, static_cast<uint8_t>(hash * 10));
        offload.cache->on_blocks_overwritten(hash, make_block_set(0));
        offload.cache->flush();
    }

    fixture.fill_block(1, /*seed=*/0);
    EXPECT_FALSE(offload.cache->load_into(/*hash=*/1, /*block_index=*/1));
    ASSERT_TRUE(offload.cache->load_into(/*hash=*/3, /*block_index=*/1));
    EXPECT_EQ(fixture.cache_manager->read_block(1), newest);
}

TEST(TestKVCacheOffloadCache, SelfEvictingStoreIsNotWrittenTwiceForTheSameBlock) {
    CacheFixture fixture;
    SelfEvictingCache offload(fixture, /*max_blocks=*/4);
    fixture.fill_block(0, /*seed=*/5);

    offload.cache->on_blocks_overwritten(/*hash=*/9, make_block_set(0));
    offload.cache->flush();
    offload.cache->on_blocks_overwritten(/*hash=*/9, make_block_set(0));
    offload.cache->flush();

    EXPECT_EQ(offload.store->num_writes(), 1u);
    EXPECT_EQ(offload.cache->get_statistics().num_already_present, 1u);
}

TEST(TestKVCacheOffloadCache, WarmPrefixCacheStillFindsBlocksThatReachTheSourceAfterTheProbe) {
    constexpr size_t block_size = 4;
    constexpr size_t num_blocks = 3;
    CacheFixture fixture(num_blocks);
    auto offload_cache = make_offload_cache(fixture, num_blocks);
    BlockManager block_manager(num_blocks, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);

    // Reports nothing at probe time, the way blocks evicted by the walk's own allocations are missed by it.
    struct BlindProbeSource : IExternalPrefixSource {
        explicit BlindProbeSource(KVCacheOffloadCache& inner) : m_inner(inner) {}
        bool contains(size_t hash) const override { return m_inner.contains(hash); }
        std::vector<bool> probe(const std::vector<size_t>& chain) override { return std::vector<bool>(chain.size(), false); }
        bool load_into(size_t hash, size_t block_index) override { return m_inner.load_into(hash, block_index); }
        KVCacheOffloadCache& m_inner;
    } source(*offload_cache);

    std::vector<int64_t> tokens(num_blocks * block_size + 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = static_cast<int64_t>(i);
    }
    auto consumer = make_group(tokens, /*request_id=*/31);
    const auto sequence = consumer->get_running_sequences().at(0);
    for (size_t block = 0; block < num_blocks; ++block) {
        fixture.fill_block(block, static_cast<uint8_t>(block + 1));
        store_block(*offload_cache, sequence->get_hash((block + 1) * block_size, block_size), block);
    }

    EXPECT_EQ(block_manager.warm_prefix_cache(consumer, source), num_blocks);
}

TEST(TestKVCacheOffloadCache, WarmPrefixCacheProbesTheWholeChainOnceBeforeReading) {
    constexpr size_t block_size = 4;
    constexpr size_t num_blocks = 3;
    CacheFixture fixture(num_blocks);
    std::vector<std::string> events;
    auto offload_cache = make_recording_cache(fixture, /*plugin_backed=*/true, events, num_blocks);
    BlockManager block_manager(num_blocks, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);

    std::vector<int64_t> tokens(num_blocks * block_size + 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = static_cast<int64_t>(i);
    }
    auto consumer = make_group(tokens, /*request_id=*/30);
    const auto sequence = consumer->get_running_sequences().at(0);
    std::string expected_probe = "has_blocks";
    for (size_t block = 0; block < num_blocks; ++block) {
        const size_t hash = sequence->get_hash((block + 1) * block_size, block_size);
        fixture.fill_block(block, static_cast<uint8_t>(block + 1));
        store_block(*offload_cache, hash, block);
        expected_probe += " " + std::to_string(hash);
    }
    events.clear();

    EXPECT_EQ(block_manager.warm_prefix_cache(consumer, *offload_cache), num_blocks);

    // One probe of the whole chain, then one batched read of the blocks to load; nothing is asked block by block.
    EXPECT_EQ(events, (std::vector<std::string>{expected_probe, "read_blocks " + std::to_string(num_blocks)}));
}

TEST(TestKVCacheOffloadCache, WarmPrefixCacheFallsBackToReadingBlocksOneByOneWhenTheBatchFails) {
    constexpr size_t block_size = 4;
    constexpr size_t num_blocks = 3;
    CacheFixture fixture(num_blocks);

    struct BatchFailsStore : SharedStore {
        using SharedStore::SharedStore;
        std::vector<bool> read_blocks(const std::vector<size_t>&, std::vector<std::vector<uint8_t>>&) const override {
            throw std::runtime_error("batch read is down");
        }
    };
    KVCacheOffloadCache offload_cache(*fixture.cache_manager,
                                      std::make_unique<BatchFailsStore>(make_shared_file_store(fixture, num_blocks)),
                                      /*max_queued_stores=*/8);
    BlockManager block_manager(num_blocks, /*enable_prefix_caching=*/true, block_size, /*num_layers=*/1);

    std::vector<int64_t> tokens(num_blocks * block_size + 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = static_cast<int64_t>(i);
    }
    auto consumer = make_group(tokens, /*request_id=*/32);
    const auto sequence = consumer->get_running_sequences().at(0);
    for (size_t block = 0; block < num_blocks; ++block) {
        fixture.fill_block(block, static_cast<uint8_t>(block + 1));
        store_block(offload_cache, sequence->get_hash((block + 1) * block_size, block_size), block);
    }

    EXPECT_EQ(block_manager.warm_prefix_cache(consumer, offload_cache), num_blocks);
}
