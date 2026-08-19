// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "continuous_batching/scheduler.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/kv_cache_disk_layout.hpp"
#include "helper.hpp"

using namespace ov::genai;

size_t get_total_allocated_bytes(std::shared_ptr<KVCacheManager> cache_manager) {
    size_t allocated_bytes = 0;
    for (size_t i = 0; i < cache_manager->get_num_layers(); i++) {
        auto key_cache = cache_manager->get_key_cache(i);
        auto value_cache = cache_manager->get_value_cache(i);
        allocated_bytes += key_cache.get_byte_size() + value_cache.get_byte_size();
    }
    return allocated_bytes;
}

size_t get_num_kv_blocks(size_t cache_size, size_t block_size_bytes) {
    size_t cache_size_in_bytes = cache_size * 1024 * 1024 * 1024; // convert GBs to bytes
    return cache_size_in_bytes / block_size_bytes;
}

TEST(TestCacheManager, test_cache_size_param) {
    ov::Core core;
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 0;
    scheduler_config.cache_size = 2;
    scheduler_config.max_num_seqs = 2;

    const std::string device = "CPU";
    const size_t num_decoder_layers = 12;
    ov::InferRequest request = core.compile_model(get_dummy_model(core, num_decoder_layers)).create_infer_request();

    auto cache_manager = std::make_shared<KVCacheManager>(request);
    ASSERT_EQ(num_decoder_layers, cache_manager->get_num_layers());
    const size_t num_kv_blocks = get_num_kv_blocks(scheduler_config.cache_size, cache_manager->get_block_size_in_bytes());

    auto block_manager = BlockManager(num_kv_blocks, false, cache_manager->get_block_size(), cache_manager->get_num_layers());
    cache_manager->allocate_cache_if_needed(block_manager.get_total_block_count());

    const size_t kv_cache_total_size = scheduler_config.cache_size * 1024 * 1024 * 1024;
    const size_t cpu_block_size_total = cache_manager->get_block_size_in_bytes();
    size_t expected_size = kv_cache_total_size / cpu_block_size_total * cpu_block_size_total;
    ASSERT_EQ(get_total_allocated_bytes(cache_manager), expected_size);
}


TEST(TestCacheManager, test_kv_blocks_param) {
    ov::Core core;
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 150;
    scheduler_config.cache_size = 0;
    scheduler_config.max_num_seqs = 2;

    const size_t cpu_block_size = 32;
    const size_t num_decoder_layers = 12;

    auto block_manager = BlockManager(scheduler_config.num_kv_blocks, false, cpu_block_size, num_decoder_layers);
    ASSERT_EQ(block_manager.get_total_block_count(), scheduler_config.num_kv_blocks);
}


TEST(TestCacheManager, test_dynamic_cache_increase) {
    ov::Core core;
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 0;
    scheduler_config.cache_size = 0;
    scheduler_config.max_num_seqs = 2;

    const std::string device = "CPU";
    const size_t num_decoder_layers = 12;

    ov::InferRequest request = core.compile_model(get_dummy_model(core, num_decoder_layers)).create_infer_request();
    auto cache_manager = std::make_shared<KVCacheManager>(request);
    size_t block_size_in_bytes = cache_manager->get_block_size_in_bytes();
    const size_t num_kv_blocks = get_num_kv_blocks(scheduler_config.cache_size, block_size_in_bytes);

    auto block_manager = BlockManager(num_kv_blocks, false, cache_manager->get_block_size(), cache_manager->get_num_layers());
    ASSERT_EQ(num_decoder_layers, cache_manager->get_num_layers());

    // check initial cache allocation
    block_manager.increase_block_count(100);
    ASSERT_EQ(block_manager.get_total_block_count(), 100);

    cache_manager->allocate_cache_if_needed(block_manager.get_total_block_count());
    ASSERT_EQ(get_total_allocated_bytes(cache_manager), 100 * block_size_in_bytes);


    // check cache increase
    block_manager.increase_block_count(200);
    ASSERT_EQ(block_manager.get_total_block_count(), 200);

    cache_manager->allocate_cache_if_needed(block_manager.get_total_block_count());
    ASSERT_EQ(get_total_allocated_bytes(cache_manager), 200 * block_size_in_bytes);


    // check that cache does not increase if new blocks were not allocated
    cache_manager->allocate_cache_if_needed(block_manager.get_total_block_count());
    ASSERT_EQ(get_total_allocated_bytes(cache_manager), 200 * block_size_in_bytes);
}

TEST(TestCacheManager, test_cpu_block_round_trip) {
    ov::Core core;
    constexpr size_t num_layers = 2;
    constexpr size_t num_blocks = 3;
    ov::InferRequest request = core.compile_model(get_dummy_model(core, num_layers)).create_infer_request();
    auto cache_manager = std::make_shared<KVCacheManager>(request);
    cache_manager->allocate_cache_if_needed(num_blocks);

    std::vector<uint8_t> expected(cache_manager->get_block_size_in_bytes());
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<uint8_t>(i);
    }

    cache_manager->write_block(1, expected);
    EXPECT_EQ(cache_manager->read_block(1), expected);
    EXPECT_NE(cache_manager->read_block(0), expected);
}

TEST(TestCacheManager, test_disk_layout_is_layer_key_value_ordered) {
    KVCacheDiskLayout layout({10, 20}, {3, 4});

    EXPECT_EQ(layout.get_num_layers(), 2);
    EXPECT_EQ(layout.get_slot_size(), 37);
    EXPECT_EQ(layout.get_key_segment(0).offset, 0);
    EXPECT_EQ(layout.get_key_segment(0).size, 10);
    EXPECT_EQ(layout.get_value_segment(0).offset, 10);
    EXPECT_EQ(layout.get_value_segment(0).size, 3);
    EXPECT_EQ(layout.get_key_segment(1).offset, 13);
    EXPECT_EQ(layout.get_value_segment(1).offset, 33);
    EXPECT_EQ(layout.get_slot_offset(2), 74);
}
