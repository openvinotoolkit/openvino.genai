// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "scheduler.hpp"
#include "device_config.hpp"
#include "cache_manager.hpp"
#include "helper.hpp"

using namespace ov::genai;

size_t get_total_allocated_bytes(std::shared_ptr<ov::genai::CacheManager> cache_manager, size_t num_decoder_layers) {
    size_t allocated_bytes = 0;
    for (size_t i = 0; i < num_decoder_layers; i++) {
        auto key_cache = cache_manager->get_key_cache(i);
        auto value_cache = cache_manager->get_value_cache(i);
        allocated_bytes += key_cache.get_byte_size() + value_cache.get_byte_size();
    }
    return allocated_bytes;
}


TEST(TestCacheManager, test_cache_size_param) {
    ov::Core core;
    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 0;
    scheduler_config.cache_size = 2;
    scheduler_config.max_num_seqs = 2;

    const std::string device = "CPU";
    ov::genai::DeviceConfig device_config(core, scheduler_config, "CPU");
    size_t num_decoder_layers = 12;
    std::vector<size_t> num_kv_heads(12, 12);
    size_t head_size = 64;
    device_config.set_model_params(num_kv_heads, head_size, num_decoder_layers);

    ov::InferRequest request = core.compile_model(get_dummy_model(core, num_decoder_layers)).create_infer_request();
    auto cache_manager = std::make_shared<ov::genai::CacheManager>(device_config, request, core);
    auto block_manager = BlockManager(device_config.get_num_kv_blocks(), false, device_config.get_block_size(), device_config.get_num_layers());
    cache_manager->allocate_cache_if_needed(block_manager.get_total_number_of_kv_blocks());

    const size_t kv_cache_total_size = scheduler_config.cache_size * 1024 * 1024 * 1024;
    const size_t cpu_block_size = 32;
    // For u8 kvcahce, its scale, zero point and quantized data will be stored together.
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized data(u8,idx_1)|quantized data(u8,idx_2)|...|quantized data(u8,idx_head_size)|
    // so, we have to extend head_size by 2 * sizeof(float)
    const size_t cpu_block_size_total = num_decoder_layers * (num_kv_heads[0] + num_kv_heads[1]) * cpu_block_size * (head_size + 2 * sizeof(float)) * sizeof(uint8_t);
    size_t expected_size = kv_cache_total_size / cpu_block_size_total * cpu_block_size_total;
    ASSERT_EQ(get_total_allocated_bytes(cache_manager, num_decoder_layers), expected_size);
}


TEST(TestCacheManager, test_kv_blocks_param) {
    ov::Core core;
    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 150;
    scheduler_config.cache_size = 0;
    scheduler_config.max_num_seqs = 2;

    const std::string device = "CPU";
    ov::genai::DeviceConfig device_config(core, scheduler_config, "CPU");
    size_t num_decoder_layers = 12;
    std::vector<size_t> num_kv_heads(12, 12);
    device_config.set_model_params(num_kv_heads, 64, num_decoder_layers);

    ov::InferRequest request = core.compile_model(get_dummy_model(core, num_decoder_layers)).create_infer_request();
    auto cache_manager = std::make_shared<ov::genai::CacheManager>(device_config, request, core);
    auto block_manager = BlockManager(device_config.get_num_kv_blocks(), false, device_config.get_block_size(), device_config.get_num_layers());
    OPENVINO_ASSERT(block_manager.get_total_number_of_kv_blocks(), scheduler_config.num_kv_blocks);
}


TEST(TestCacheManager, test_dynamic_cache_increase) {
    ov::Core core;
    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 0;
    scheduler_config.cache_size = 0;
    scheduler_config.max_num_seqs = 2;

    const std::string device = "CPU";
    ov::genai::DeviceConfig device_config(core, scheduler_config, "CPU");
    size_t num_decoder_layers = 12;
    size_t head_size = 64;
    std::vector<size_t> num_kv_heads(12, 12);
    device_config.set_model_params(num_kv_heads, head_size, num_decoder_layers);
    size_t block_size_in_bytes = 0;
    for (size_t layer_id = 0; layer_id < num_decoder_layers; layer_id++) {
        block_size_in_bytes += 2 * num_kv_heads[layer_id] * device_config.get_block_size() * head_size * device_config.get_cache_precision().size();
    }


    ov::InferRequest request = core.compile_model(get_dummy_model(core, num_decoder_layers)).create_infer_request();
    auto cache_manager = std::make_shared<ov::genai::CacheManager>(device_config, request, core);
    auto block_manager = BlockManager(device_config.get_num_kv_blocks(), false, device_config.get_block_size(), device_config.get_num_layers());

    // check initial cache allocation
    block_manager.increase_kv_blocks_number(100);
    OPENVINO_ASSERT(block_manager.get_total_number_of_kv_blocks(), 100);

    cache_manager->allocate_cache_if_needed(block_manager.get_total_number_of_kv_blocks());
    OPENVINO_ASSERT(get_total_allocated_bytes(cache_manager, num_decoder_layers), 100 * block_size_in_bytes);


    // check cache increase
    block_manager.increase_kv_blocks_number(200);
    OPENVINO_ASSERT(block_manager.get_total_number_of_kv_blocks(), 200);

    cache_manager->allocate_cache_if_needed(block_manager.get_total_number_of_kv_blocks());
    OPENVINO_ASSERT(get_total_allocated_bytes(cache_manager, num_decoder_layers), 200 * block_size_in_bytes);


    // check that cache does not increase if new blocks were not allocated
    cache_manager->allocate_cache_if_needed(block_manager.get_total_number_of_kv_blocks());
    OPENVINO_ASSERT(get_total_allocated_bytes(cache_manager, num_decoder_layers), 200 * block_size_in_bytes);
}