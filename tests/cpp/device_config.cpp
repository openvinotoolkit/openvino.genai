// Copyright (C) 2018-2024 Intel Corporation
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
    scheduler_config.num_kv_blocks = 1;
    scheduler_config.cache_size = 2;
    scheduler_config.max_num_seqs = 2;

    const std::string device = "CPU";
    size_t num_decoder_layers = 12;
    size_t head_size = 64;
    size_t num_kv_heads = 12;

    ov::genai::DeviceConfig device_config_acc(core, scheduler_config, "CPU", { ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY) });
    device_config_acc.set_model_params(num_kv_heads, head_size, num_decoder_layers);
    ASSERT_EQ(device_config_acc.get_key_cache_precision(), ov::element::f32);
    ASSERT_EQ(device_config_acc.get_value_cache_precision(), ov::element::f32);

    ov::genai::DeviceConfig device_config_f16(core, scheduler_config, "CPU", { ov::hint::kv_cache_precision(ov::element::f16) });
    device_config_f16.set_model_params(num_kv_heads, head_size, num_decoder_layers);
    ASSERT_EQ(device_config_f16.get_key_cache_precision(), ov::element::f16);
    ASSERT_EQ(device_config_f16.get_value_cache_precision(), ov::element::f16);

    // ov::key/value_cache_precision has higher priority
    ov::genai::DeviceConfig device_config_mix(core, scheduler_config, "CPU", { ov::hint::kv_cache_precision(ov::element::f16), ov::key_cache_precision(ov::element::u8) });
    device_config_mix.set_model_params(num_kv_heads, head_size, num_decoder_layers);
    ASSERT_EQ(device_config_mix.get_key_cache_precision(), ov::element::u8);
    ASSERT_EQ(device_config_mix.get_value_cache_precision(), ov::element::f16);

    const size_t group_size = 16;
    ov::genai::DeviceConfig device_config_u8(core, scheduler_config, "CPU", {ov::key_cache_group_size(group_size), ov::value_cache_group_size(group_size)});
    device_config_u8.set_model_params(num_kv_heads, head_size, num_decoder_layers);
    ASSERT_EQ(device_config_u8.get_key_cache_precision(), ov::element::u8);
    ASSERT_EQ(device_config_u8.get_value_cache_precision(), ov::element::u8);
    ov::Shape expected_shape{scheduler_config.num_kv_blocks, num_kv_heads, size_t{32}, head_size + head_size / group_size * 2 * sizeof(float)};
    ASSERT_EQ(device_config_u8.get_key_cache_shape(), expected_shape);
    ASSERT_EQ(device_config_u8.get_value_cache_shape(), expected_shape);
}
