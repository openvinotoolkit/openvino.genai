// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "gguf_utils/gguf_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "utils.hpp"

TEST(GGUFProperties, FilterConversionPropertiesAndPreservePluginProperties) {
    const ov::AnyMap properties{ov::genai::gguf_reader("FRONTEND"),
                                ov::genai::enable_save_ov_model(true),
                                {"INFERENCE_NUM_THREADS", 2}};
    const auto extracted = ov::genai::gguf_utils::extract_gguf_properties(properties);
    EXPECT_TRUE(extracted.enable_save_ov_model);
    ASSERT_EQ(extracted.rest.size(), 1);
    EXPECT_EQ(extracted.rest.at("INFERENCE_NUM_THREADS").as<int>(), 2);
    EXPECT_EQ(properties.size(), 3);

    const auto defaults = ov::genai::gguf_utils::extract_gguf_properties({});
    EXPECT_FALSE(defaults.enable_save_ov_model);
}

TEST(GGUFProperties, LegacyReaderThrowsInformativeError) {
    try {
        ov::genai::gguf_utils::extract_gguf_properties({ov::genai::gguf_reader("LEGACY")});
        FAIL() << "GGUF_READER='LEGACY' must throw an informative exception";
    } catch (const ov::Exception& e) {
        EXPECT_NE(std::string(e.what()).find("The legacy GGUF reader has been removed"), std::string::npos);
    }
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
