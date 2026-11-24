// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "gguf_utils/gguf_utils.hpp"
#include "model_desc.hpp"
#include "openvino/genai/llm_pipeline.hpp"
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
        ASSERT_EQ(extracted.rest.size(), 1);
        EXPECT_EQ(extracted.rest.at("INFERENCE_NUM_THREADS").as<int>(), 2);
        EXPECT_EQ(properties.size(), 3);
    }
    const auto defaults = ov::genai::gguf_utils::extract_gguf_properties({});
    EXPECT_TRUE(defaults.use_legacy_reader());
    EXPECT_FALSE(defaults.legacy_reader.has_value());
    EXPECT_FALSE(defaults.enable_save_ov_model);
}

TEST(GGUFProperties, RejectInvalidReader) {
    EXPECT_THROW(ov::genai::gguf_utils::extract_gguf_properties({ov::genai::gguf_reader("invalid")}),
                 ov::Exception);
}

TEST(GGUFProperties, IrMtpDraftRequiresIrTarget) {
    ov::genai::ModelDesc draft;
    draft.properties["mtp_mode"] = true;
    const ov::AnyMap properties{{ov::genai::utils::DRAFT_MODEL_ARG_NAME, ov::Any::make<ov::genai::ModelDesc>(draft)}};

    EXPECT_NO_THROW(ov::genai::resolve_draft_model_request("converted-model", "CPU", properties));
    EXPECT_THROW(ov::genai::resolve_draft_model_request("target.gguf", "CPU", properties), ov::Exception);
}

TEST(GGUFProperties, GgufMtpRequestDetection) {
    const ov::AnyMap mtp{{ov::genai::speculative_type.name(), std::string("mtp")}};
    EXPECT_TRUE(ov::genai::is_gguf_mtp_request("target.gguf", mtp));
    EXPECT_FALSE(ov::genai::is_gguf_mtp_request("converted-model", mtp));
    EXPECT_FALSE(ov::genai::is_gguf_mtp_request("target.gguf", {}));

    ov::AnyMap unsupported{{ov::genai::speculative_type.name(), std::string("eagle3")}};
    EXPECT_THROW(ov::genai::strip_gguf_mtp_request("target.gguf", unsupported), ov::Exception);
    ov::AnyMap with_lookup{{ov::genai::speculative_type.name(), std::string("mtp")},
                           {ov::genai::prompt_lookup.name(), true}};
    EXPECT_THROW(ov::genai::strip_gguf_mtp_request("target.gguf", with_lookup), ov::Exception);
    ov::AnyMap valid = mtp;
    EXPECT_NO_THROW(ov::genai::strip_gguf_mtp_request("target.gguf", valid));
    EXPECT_TRUE(valid.empty());
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
