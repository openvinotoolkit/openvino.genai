// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "gguf_utils/gguf_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "utils.hpp"

TEST(GGUFProperties, FilterConversionPropertiesAndPreservePluginProperties) {
    for (const std::string reader : {"LEGACY", "FRONTEND"}) {
        const ov::AnyMap properties{ov::genai::gguf_reader(reader),
                                    ov::genai::enable_save_ov_model(true),
                                    {"INFERENCE_NUM_THREADS", 2}};
        const auto extracted = ov::genai::gguf_utils::extract_gguf_properties(properties);
        EXPECT_EQ(extracted.use_legacy_reader(), reader == "LEGACY");
        EXPECT_TRUE(extracted.enable_save_ov_model);
        EXPECT_TRUE(extracted.frontend_properties.empty());
        ASSERT_EQ(extracted.rest.size(), 1);
        EXPECT_EQ(extracted.rest.at("INFERENCE_NUM_THREADS").as<int>(), 2);
        EXPECT_EQ(properties.size(), 3);
    }
    const auto defaults = ov::genai::gguf_utils::extract_gguf_properties({});
    EXPECT_TRUE(defaults.use_legacy_reader());
    EXPECT_FALSE(defaults.legacy_reader.has_value());
    EXPECT_FALSE(defaults.enable_save_ov_model);
    EXPECT_TRUE(defaults.frontend_properties.empty());
}

TEST(GGUFProperties, ExtractNativeWeightsWithoutLeakingToPluginProperties) {
    for (bool enabled : {false, true}) {
        const ov::AnyMap properties{ov::genai::gguf_reader("FRONTEND"),
                                    ov::hint::gguf_native_weights(enabled),
                                    {"INFERENCE_NUM_THREADS", 2}};
        const auto extracted = ov::genai::gguf_utils::extract_gguf_properties(properties);
        ASSERT_EQ(extracted.frontend_properties.size(), 1);
        EXPECT_EQ(extracted.frontend_properties.at(ov::hint::gguf_native_weights.name()).as<bool>(), enabled);
        ASSERT_EQ(extracted.rest.size(), 1);
        EXPECT_EQ(extracted.rest.at("INFERENCE_NUM_THREADS").as<int>(), 2);
        EXPECT_EQ(properties.size(), 3);
        EXPECT_EQ(properties.at(ov::hint::gguf_native_weights.name()).as<bool>(), enabled);
    }
}

TEST(GGUFProperties, RejectNativeWeightsWithLegacyReader) {
    for (bool enabled : {false, true}) {
        EXPECT_THROW(ov::genai::gguf_utils::extract_gguf_properties({ov::hint::gguf_native_weights(enabled)}),
                     ov::Exception);
        EXPECT_THROW(ov::genai::gguf_utils::extract_gguf_properties(
                         {ov::genai::gguf_reader("LEGACY"), ov::hint::gguf_native_weights(enabled)}),
                     ov::Exception);
    }
}

TEST(GGUFProperties, RejectInvalidNativeWeightsValue) {
    EXPECT_THROW(ov::genai::gguf_utils::extract_gguf_properties(
                     {ov::genai::gguf_reader("FRONTEND"), {"GGUF_NATIVE_WEIGHTS", "invalid"}}),
                 ov::Exception);
}

TEST(GGUFProperties, RejectInvalidReader) {
    EXPECT_THROW(ov::genai::gguf_utils::extract_gguf_properties({ov::genai::gguf_reader("invalid")}),
                 ov::Exception);
}

TEST(CacheTypes, GGUFRecurrentMetadataRequiresLinearCache) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    ov::Model model(ov::OutputVector{input}, ov::ParameterVector{input});
    EXPECT_FALSE(ov::genai::utils::get_cache_types(model).has_linear());
    model.set_rt_info(std::vector<std::string>{}, "gguf_recurrent_states");
    EXPECT_FALSE(ov::genai::utils::get_cache_types(model).has_linear());
    model.set_rt_info(std::vector<std::string>{"recurrent_state"}, "gguf_recurrent_states");
    EXPECT_TRUE(ov::genai::utils::get_cache_types(model).has_linear());
}
