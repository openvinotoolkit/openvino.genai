// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "scheduler.hpp"
#include "device_config.hpp"

TEST(TestDeviceConfig, kv_cache_precision_u8) {
    ov::Core core;
    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 0;
    scheduler_config.cache_size = 2;
    scheduler_config.max_num_seqs = 2;

    const std::string device = "CPU";
    size_t num_decoder_layers = 12;
    size_t head_size = 64, head_size_u8 = head_size + 8;

    ov::genai::KVHeadConfig kv_head_config { 12, 12, head_size_u8, head_size_u8 };
    ov::genai::KVHeadConfig kv_head_config_u8 { 12, 12, head_size, head_size };

    ov::genai::DeviceConfig device_config_default(core, scheduler_config, "CPU");
    ov::genai::DeviceConfig device_config_u8(core, scheduler_config, "CPU", { ov::hint::kv_cache_precision(ov::element::u8) });

    device_config_default.set_kv_head_configs(std::vector<ov::genai::KVHeadConfig>(num_decoder_layers, kv_head_config));
    device_config_u8.set_kv_head_configs(std::vector<ov::genai::KVHeadConfig>(num_decoder_layers, kv_head_config_u8));

    const auto ratio = ov::element::f16.size() / ov::element::u8.size();
    ASSERT_EQ(device_config_default.get_num_kv_blocks() * ratio, device_config_u8.get_num_kv_blocks());
}
