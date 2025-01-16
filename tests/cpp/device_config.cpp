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
    std::vector<size_t> num_kv_heads(12, 12);

    ov::genai::DeviceConfig device_config_default(core, scheduler_config, "CPU");
    device_config_default.set_model_params(num_kv_heads, head_size_u8, num_decoder_layers);

    ov::genai::DeviceConfig device_config_u8(core, scheduler_config, "CPU", { ov::hint::kv_cache_precision(ov::element::u8) });
    device_config_u8.set_model_params(num_kv_heads, head_size, num_decoder_layers);

    const auto ratio = ov::element::f16.size() / ov::element::u8.size();
    ASSERT_EQ(device_config_default.get_num_kv_blocks() * ratio, device_config_u8.get_num_kv_blocks());
}
