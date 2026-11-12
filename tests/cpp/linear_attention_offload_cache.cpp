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

TEST(TestLinearAttentionOffloadCache, WarmsNewestCheckpointAcrossMissingEarlierOnes) {
    constexpr size_t cache_interval = 4;
    LaCacheFixture fixture(/*num_blocks=*/4);
    auto offload_cache = make_offload_cache(fixture, 2);

    BlockManager block_manager(/*num_blocks=*/4, /*enable_prefix_caching=*/true, cache_interval, /*num_layers=*/1,
                               /*fixed_blocks_per_sequence=*/0, /*restore_latest_prefix_block_only=*/true);

    const std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    auto consumer = make_group(tokens, /*request_id=*/4);
    const auto sequence = consumer->get_running_sequences().at(0);
    // The checkpoint at token 8 is missing, as when a pass crossed it without keeping its state.
    fixture.fill_block(0, /*seed=*/5);
    offload_cache->on_blocks_overwritten(sequence->get_hash(4, cache_interval), make_block_set(0));
    // A store that arrives while another one is in flight is dropped, so wait for each one.
    offload_cache->flush();
    fixture.fill_block(1, /*seed=*/9);
    offload_cache->on_blocks_overwritten(sequence->get_hash(12, cache_interval), make_block_set(1));
    offload_cache->flush();

    EXPECT_EQ(block_manager.warm_prefix_cache(consumer, *offload_cache), 1);
    EXPECT_TRUE(block_manager.restore_cached_blocks(consumer));
    EXPECT_EQ(consumer->get_num_processed_tokens(), 12);

    block_manager.free_sequence(sequence->get_id());
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

std::shared_ptr<IKVCacheStore> make_shared_file_store(LaCacheFixture& fixture, size_t num_slots) {
    auto backend = std::make_unique<KVCacheOffloadManager>(fixture.cache_manager->get_block_layout(),
                                                           make_offload_config(fixture, num_slots),
                                                           "CPU");
    return std::make_shared<FileKVCacheStore>(std::move(backend));
}

std::unique_ptr<LinearAttentionOffloadCache> make_offload_cache_on(LaCacheFixture& fixture,
                                                                   std::shared_ptr<IKVCacheStore> store) {
    return std::make_unique<LinearAttentionOffloadCache>(*fixture.cache_manager,
                                                         std::make_unique<SharedStore>(std::move(store)),
                                                         /*max_queued_stores=*/8);
}

}  // namespace

TEST(TestLinearAttentionOffloadCache, AnotherInstanceServesEntriesItNeverWrote) {
    LaCacheFixture fixture;
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

TEST(TestLinearAttentionOffloadCache, ReportsWhetherItsStoreIsAPlugin) {
    LaCacheFixture fixture;
    auto offload_cache = make_offload_cache(fixture, 1);
    EXPECT_FALSE(offload_cache->is_plugin_backed());
}

TEST(TestLinearAttentionOffloadCache, ConcurrentOverwritesNeverExceedTheStoreCapacity) {
    LaCacheFixture fixture;
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
