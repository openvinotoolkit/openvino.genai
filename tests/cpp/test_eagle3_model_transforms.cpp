// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>

#include "openvino/op/parameter.hpp"
#include "speculative_decoding/eagle3_model_transforms.hpp"

TEST(Eagle3ModelTransforms, PreservesKvCacheParameterRtInfo) {
    const auto cache_shape = ov::PartialShape{2, 4};
    auto key_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
    key_cache->set_friendly_name("key_cache.0");
    key_cache->output(0).set_names({"key_cache.0"});
    key_cache->get_rt_info()["enable_keep_const_precision"] = true;

    auto value_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
    value_cache->set_friendly_name("value_cache.0");
    value_cache->output(0).set_names({"value_cache.0"});
    value_cache->get_rt_info()["enable_keep_const_precision"] = true;

    auto main_model = std::make_shared<ov::Model>(ov::OutputVector{key_cache, value_cache},
                                                   ov::ParameterVector{key_cache, value_cache});
    const auto kv_update_model = ov::genai::utils::eagle3::create_eagle3_kv_update_model(main_model);

    for (const auto& input_name : {"key_cache.0", "value_cache.0"}) {
        const auto parameter = kv_update_model->input(input_name).get_node_shared_ptr();
        ASSERT_TRUE(parameter->get_rt_info().count("enable_keep_const_precision"));
        EXPECT_TRUE(parameter->get_rt_info().at("enable_keep_const_precision").as<bool>());
    }
}
