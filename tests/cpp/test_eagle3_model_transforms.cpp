// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>

#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
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

TEST(Eagle3ModelTransforms, AddsTreeMaskToEveryAttentionMaskSource) {
    // Two attention types with their own masks (e.g. full and sliding-window layers): the sliding mask is derived
    // from the full one, the full mask feeds two SDPA nodes.
    const auto qkv_shape = ov::PartialShape{1, 2, -1, 8};
    const auto mask_shape = ov::PartialShape{1, 1, -1, -1};
    auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, qkv_shape);
    auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, qkv_shape);
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, qkv_shape);
    auto full_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, mask_shape);
    auto window = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, mask_shape);
    auto sliding_mask = std::make_shared<ov::op::v1::Add>(full_mask, window);
    auto full_0 = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, full_mask, false);
    auto sliding = std::make_shared<ov::op::v13::ScaledDotProductAttention>(full_0, key, value, sliding_mask, false);
    auto full_1 = std::make_shared<ov::op::v13::ScaledDotProductAttention>(sliding, key, value, full_mask, false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{full_1},
                                             ov::ParameterVector{query, key, value, full_mask, window});

    ov::genai::utils::eagle3::apply_eagle3_attention_mask_transform(model);

    ASSERT_EQ(model->get_parameters().size(), 6u);
    const auto tree_mask = model->input("eagle_tree_mask").get_node_shared_ptr();
    EXPECT_EQ(tree_mask->get_output_partial_shape(0), mask_shape);

    const auto full_add = full_0->get_input_node_shared_ptr(3);
    ASSERT_TRUE(ov::is_type<ov::op::v1::Add>(full_add));
    EXPECT_EQ(full_add, full_1->get_input_node_shared_ptr(3));
    EXPECT_EQ(full_add->get_input_node_shared_ptr(0), full_mask);
    EXPECT_EQ(full_add->get_input_node_shared_ptr(1), tree_mask);

    const auto sliding_add = sliding->get_input_node_shared_ptr(3);
    ASSERT_TRUE(ov::is_type<ov::op::v1::Add>(sliding_add));
    EXPECT_NE(sliding_add, full_add);
    EXPECT_EQ(sliding_add->get_input_node_shared_ptr(0), sliding_mask);
    EXPECT_EQ(sliding_add->get_input_node_shared_ptr(1), tree_mask);
}

TEST(Eagle3ModelTransforms, RejectsAttentionMasksOfDifferentTypes) {
    const auto qkv_shape = ov::PartialShape{1, 2, -1, 8};
    auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, qkv_shape);
    auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, qkv_shape);
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, qkv_shape);
    auto float_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, -1});
    auto bool_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::PartialShape{1, 1, -1, -1});
    auto first = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, float_mask, false);
    auto second = std::make_shared<ov::op::v13::ScaledDotProductAttention>(first, key, value, bool_mask, false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{second},
                                             ov::ParameterVector{query, key, value, float_mask, bool_mask});

    EXPECT_THROW(ov::genai::utils::eagle3::apply_eagle3_attention_mask_transform(model), ov::Exception);
}
