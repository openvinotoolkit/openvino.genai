// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "scheduler.hpp"
#include "device_config.hpp"
#include "cache_manager.hpp"
#include "openvino/op/concat.hpp"

using namespace ov::genai;

std::shared_ptr<ov::Model> get_dummy_model(ov::Core core, size_t num_layers) {
    ov::NodeVector keys;
    ov::NodeVector values;
    ov::ParameterVector params;
    ov::element::Type inference_precision = core.get_property("CPU", ov::hint::inference_precision);
    ov::element::Type kv_cache_type = inference_precision == ov::element::bf16 ? ov::element::bf16 : ov::element::f16;

    auto shape = ov::PartialShape({ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()});
    for (size_t i = 0; i < num_layers; i++) {
        auto key = std::make_shared<ov::op::v0::Parameter>(kv_cache_type, shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(kv_cache_type, shape);
        key->get_output_tensor(0).set_names({"key_cache." + std::to_string(i)});
        value->get_output_tensor(0).set_names({"value_cache." + std::to_string(i)});
        keys.push_back(key);
        values.push_back(value);
        params.push_back(key);
        params.push_back(value);
    }
    const auto& concat1 = std::make_shared<ov::op::v0::Concat>(keys, 1);
    const auto& concat2 = std::make_shared<ov::op::v0::Concat>(values, 1);
    auto model = std::make_shared<ov::Model>(ov::NodeVector{concat1, concat2}, params);
    return std::make_shared<ov::Model>(ov::NodeVector{concat1, concat2}, params);
}

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
    const size_t num_decoder_layers = 12;
    const std::vector<KVHeadConfig> kv_heads_config(num_decoder_layers, KVHeadConfig { 12, 12, 64, 64 });
    device_config.set_kv_head_configs(kv_heads_config);

    ov::InferRequest request = core.compile_model(get_dummy_model(core, num_decoder_layers)).create_infer_request();
    auto cache_manager = std::make_shared<ov::genai::CacheManager>(device_config, request, core);
    auto block_manager = BlockManager(device_config.get_num_kv_blocks(), false, device_config.get_block_size(), device_config.get_num_layers());
    cache_manager->allocate_cache_if_needed(block_manager.get_total_number_of_kv_blocks());

    ASSERT_EQ(get_total_allocated_bytes(cache_manager, num_decoder_layers), 2146959360);
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
    const size_t num_decoder_layers = 12;
    const std::vector<KVHeadConfig> kv_heads_config(num_decoder_layers, KVHeadConfig { 12, 12, 64, 64 });
    device_config.set_kv_head_configs(kv_heads_config);

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
    const size_t num_decoder_layers = 12;
    const std::vector<KVHeadConfig> kv_heads_config(num_decoder_layers, KVHeadConfig { 12, 12, 64, 64 });
    device_config.set_kv_head_configs(kv_heads_config);

    size_t block_size_in_bytes = 0;
    for (size_t layer_id = 0; layer_id < num_decoder_layers; layer_id++) {
        KVHeadConfig config = kv_heads_config[layer_id];
        block_size_in_bytes += config.k_head_size * config.num_k_heads + config.v_head_size * config.num_v_heads;
    }
    block_size_in_bytes *= device_config.get_block_size() * device_config.get_cache_precision().size();

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
