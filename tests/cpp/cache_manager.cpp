// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "scheduler.hpp"
#include "device_config.hpp"
#include "cache_manager.hpp"

TEST(TestCacheManager, general_test) {
    ov::Core core;
    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 0;
    scheduler_config.cache_size = 2;
    scheduler_config.max_num_seqs = 2;

    const std::string device = "CPU";
    ov::genai::DeviceConfig device_config(core, scheduler_config, "CPU");
    size_t num_decoder_layers = 12;
    device_config.set_model_params(12, 64, num_decoder_layers);

    // auto cache_manager = std::make_shared<ov::genai::CacheManager>(device_config, core);

    // size_t allocated_bytes = 0;
    // for (size_t i = 0; i < num_decoder_layers; i++) {
    //     auto key_cache = cache_manager->get_key_cache(i);
    //     auto value_cache = cache_manager->get_value_cache(i);
    //     allocated_bytes += key_cache.get_byte_size() + value_cache.get_byte_size();
    // }
    
    // ASSERT_EQ(allocated_bytes, 2146959360);
}
