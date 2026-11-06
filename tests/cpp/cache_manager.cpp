// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include "openvino/runtime/core.hpp"
#include "continuous_batching/scheduler.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
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

// Exercises KVCacheManager's legacy chunked allocation path (chunk_base_ptrs inputs) directly. GPU-only: chunked
// mode requires a RemoteContext (see KVCacheManager's constructor assert).
TEST(TestCacheManager, test_gpu_chunked_allocation_rounds_up_to_chunk_boundary) {
    ov::Core core;
    auto devices = core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), "GPU") == devices.end()) {
        GTEST_SKIP() << "No GPU device available on this machine.";
    }

    constexpr size_t num_layers = 2;
    constexpr size_t blocks_per_chunk = 4;
    ov::InferRequest request =
        core.compile_model(get_dummy_model_with_chunk_base_ptrs(core, num_layers), "GPU").create_infer_request();
    auto cache_manager = std::make_shared<KVCacheManager>(request, /*use_chunked_kv_cache=*/true, blocks_per_chunk);

    // 6 blocks requested -> rounds up to 2 chunks (8 blocks), not exactly 6: chunks are always fixed-size.
    cache_manager->allocate_cache_if_needed(6);
    EXPECT_EQ(cache_manager->get_num_allocated_blocks(), 2 * blocks_per_chunk);

    // Already-satisfied request is a no-op (matches the non-chunked contract).
    cache_manager->allocate_cache_if_needed(6);
    EXPECT_EQ(cache_manager->get_num_allocated_blocks(), 2 * blocks_per_chunk);

    // Growing past the current capacity adds exactly one more chunk (9 blocks needs a 3rd chunk).
    cache_manager->allocate_cache_if_needed(9);
    EXPECT_EQ(cache_manager->get_num_allocated_blocks(), 3 * blocks_per_chunk);
}

TEST(TestCacheManager, test_gpu_chunked_growth_preserves_existing_chunk_data) {
    ov::Core core;
    auto devices = core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), "GPU") == devices.end()) {
        GTEST_SKIP() << "No GPU device available on this machine.";
    }

    constexpr size_t num_layers = 2;
    constexpr size_t blocks_per_chunk = 2;
    ov::InferRequest request =
        core.compile_model(get_dummy_model_with_chunk_base_ptrs(core, num_layers), "GPU").create_infer_request();
    auto cache_manager = std::make_shared<KVCacheManager>(request, /*use_chunked_kv_cache=*/true, blocks_per_chunk);

    // First chunk only (blocks 0-1).
    cache_manager->allocate_cache_if_needed(2);
    std::vector<uint8_t> written_data(cache_manager->get_block_size_in_bytes());
    std::iota(written_data.begin(), written_data.end(), static_cast<uint8_t>(1));
    cache_manager->write_block(/*block_id=*/1, written_data);

    // Growing to 5 blocks allocates 2 brand-new chunks (chunk 1: blocks 2-3, chunk 2: blocks 4-5);
    // this must NOT touch the first chunk's already-written block 1 (the whole point of chunking is
    // that growth never copies/reallocates existing chunks).
    cache_manager->allocate_cache_if_needed(5);
    EXPECT_EQ(cache_manager->get_num_allocated_blocks(), 3 * blocks_per_chunk);
    EXPECT_EQ(cache_manager->read_block(1), written_data);
}

TEST(TestCacheManager, test_gpu_chunked_copy_blocks_across_chunk_boundary) {
    ov::Core core;
    auto devices = core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), "GPU") == devices.end()) {
        GTEST_SKIP() << "No GPU device available on this machine.";
    }

    constexpr size_t num_layers = 2;
    constexpr size_t blocks_per_chunk = 2;
    ov::InferRequest request =
        core.compile_model(get_dummy_model_with_chunk_base_ptrs(core, num_layers), "GPU").create_infer_request();
    auto cache_manager = std::make_shared<KVCacheManager>(request, /*use_chunked_kv_cache=*/true, blocks_per_chunk);

    // block 0 -> chunk 0, block 3 -> chunk 1: copy_blocks() must locate each side in its own chunk tensor.
    cache_manager->allocate_cache_if_needed(4);
    std::vector<uint8_t> written_data(cache_manager->get_block_size_in_bytes());
    std::iota(written_data.begin(), written_data.end(), static_cast<uint8_t>(7));
    cache_manager->write_block(/*block_id=*/0, written_data);

    cache_manager->copy_blocks({{0, {3}}});
    EXPECT_EQ(cache_manager->read_block(3), written_data);
}

namespace {
// Cache inputs only (no chunk_base_ptrs.N): the chunk-table mode binds pointer tables to key_cache.N / value_cache.N,
// which the GPU plugin recognises by the `kv_cache_chunk_table` flag on the Parameter's output tensor.
std::shared_ptr<ov::Model> get_dummy_model_with_chunk_tables(ov::Core core, size_t num_layers) {
    auto model = get_dummy_model(core, num_layers);
    for (const auto& param : model->get_parameters()) {
        param->get_output_tensor(0).get_rt_info()["kv_cache_chunk_table"] = true;
    }
    return model;
}
}  // namespace

// Table mode adds one chunk per growth step, sized from the shortfall (rounded up to the granule) and at least 1/16 of
// the blocks already allocated, so chunk sizes differ between steps.
TEST(TestCacheManager, test_gpu_chunk_tables_grow_by_variable_sized_chunks) {
    ov::Core core;
    auto devices = core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), "GPU") == devices.end()) {
        GTEST_SKIP() << "No GPU device available on this machine.";
    }

    ov::InferRequest request =
        core.compile_model(get_dummy_model_with_chunk_tables(core, /*num_layers=*/2), "GPU").create_infer_request();
    auto cache_manager = std::make_shared<KVCacheManager>(request,
                                                          /*use_chunked_kv_cache=*/true,
                                                          /*blocks_per_chunk=*/0,
                                                          /*granule_blocks=*/2);

    cache_manager->allocate_cache_if_needed(6);   // shortfall 6, already a multiple of 2
    EXPECT_EQ(cache_manager->get_num_allocated_blocks(), 6);
    cache_manager->allocate_cache_if_needed(6);   // already satisfied
    EXPECT_EQ(cache_manager->get_num_allocated_blocks(), 6);
    cache_manager->allocate_cache_if_needed(7);   // shortfall 1, rounded up to the granule: 2
    EXPECT_EQ(cache_manager->get_num_allocated_blocks(), 8);
    cache_manager->allocate_cache_if_needed(64);  // shortfall 56
    EXPECT_EQ(cache_manager->get_num_allocated_blocks(), 64);
    cache_manager->allocate_cache_if_needed(65);  // shortfall 1, but the floor is 64 / 16 = 4
    EXPECT_EQ(cache_manager->get_num_allocated_blocks(), 68);
}

TEST(TestCacheManager, test_gpu_chunk_tables_preserve_and_copy_data_across_unequal_chunks) {
    ov::Core core;
    auto devices = core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), "GPU") == devices.end()) {
        GTEST_SKIP() << "No GPU device available on this machine.";
    }

    ov::InferRequest request =
        core.compile_model(get_dummy_model_with_chunk_tables(core, /*num_layers=*/2), "GPU").create_infer_request();
    auto cache_manager = std::make_shared<KVCacheManager>(request,
                                                          /*use_chunked_kv_cache=*/true,
                                                          /*blocks_per_chunk=*/2,
                                                          /*granule_blocks=*/2);

    // Chunks of 4, 2 and 6 blocks: blocks [0,4), [4,6) and [6,12).
    cache_manager->allocate_cache_if_needed(3);
    cache_manager->allocate_cache_if_needed(5);
    std::vector<uint8_t> first(cache_manager->get_block_size_in_bytes()), second(cache_manager->get_block_size_in_bytes());
    std::iota(first.begin(), first.end(), static_cast<uint8_t>(1));
    std::iota(second.begin(), second.end(), static_cast<uint8_t>(101));
    cache_manager->write_block(/*block_id=*/3, first);    // last block of chunk 0
    cache_manager->write_block(/*block_id=*/4, second);   // first block of chunk 1

    cache_manager->allocate_cache_if_needed(11);
    EXPECT_EQ(cache_manager->get_num_allocated_blocks(), 12);
    EXPECT_EQ(cache_manager->read_block(3), first);
    EXPECT_EQ(cache_manager->read_block(4), second);

    cache_manager->copy_blocks({{3, {9}}, {4, {11}}});    // into the middle and last block of chunk 2
    EXPECT_EQ(cache_manager->read_block(9), first);
    EXPECT_EQ(cache_manager->read_block(11), second);
}
