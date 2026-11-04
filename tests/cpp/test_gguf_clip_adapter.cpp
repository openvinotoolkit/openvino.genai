// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_utils/gguf_clip_adapter.hpp"
#include "gtest/gtest.h"
#include "model_desc.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset15.hpp"

namespace {
using namespace ov::opset15;

std::shared_ptr<ov::Model> native_encoder() {
    auto pixels = std::make_shared<Parameter>(ov::element::f32, ov::Shape{1, 3, 2, 2});
    pixels->output(0).set_names({"vision.pixel_values"});
    auto weighted = std::make_shared<Multiply>(pixels, Constant::create(ov::element::f32, {1, 3, 1, 1}, {1, 2, 4}));
    auto sum = std::make_shared<ReduceSum>(weighted, Constant::create(ov::element::i64, {1}, {1}), false);
    auto features = std::make_shared<Reshape>(sum, Constant::create(ov::element::i64, {4}, {1, 1, 2, 2}), false);
    features->output(0).set_names({"vision.embeddings"});
    return std::make_shared<ov::Model>(ov::OutputVector{features}, ov::ParameterVector{pixels});
}
}  // namespace

TEST(GGUFClipAdapter, RejectsDraftModelBeforeLoading) {
    for (const std::string backend : {"SDPA", "PA"}) {
        ov::AnyMap properties{{"draft_model", ov::genai::ModelDesc{}},
                              {"ATTENTION_BACKEND", backend},
                              {"mmproj_path", std::string("unused.gguf")}};
        try {
            ov::genai::VLMPipeline pipeline("unused.gguf", "CPU", properties);
            FAIL() << "Expected draft_model rejection";
        } catch (const ov::Exception& error) {
            EXPECT_NE(std::string(error.what()).find("draft_model is not supported for GGUF VLMPipeline"),
                      std::string::npos);
        }
    }
}

TEST(GGUFClipAdapter, MultipleCropsMatchNativeInferenceAndPreserveSource) {
    ov::Core core;
    auto native = native_encoder();
    auto reference =
        core.compile_model(native, "CPU", ov::hint::inference_precision(ov::element::f32)).create_infer_request();
    auto adapted = ov::genai::adapt_gguf_clip_encoder(native);
    EXPECT_EQ(native->input().get_any_name(), "vision.pixel_values");
    EXPECT_EQ(native->output().get_shape(), (ov::Shape{1, 1, 2, 2}));
    EXPECT_EQ(adapted->input().get_any_name(), "pixel_values");
    EXPECT_EQ(adapted->input().get_partial_shape(), (ov::PartialShape{-1, 3, 2, 2}));
    auto request =
        core.compile_model(adapted, "CPU", ov::hint::inference_precision(ov::element::f32)).create_infer_request();
    for (size_t count : {size_t(1), size_t(3)}) {
        ov::Tensor crops(ov::element::f32, {count, 3, 2, 2});
        for (size_t i = 0; i < crops.get_size(); ++i)
            crops.data<float>()[i] = float(i) * 0.7f - 5.f;
        request.set_input_tensor(crops);
        request.infer();
        const auto actual = request.get_output_tensor();
        ASSERT_EQ(actual.get_shape(), (ov::Shape{count, 2, 2}));
        for (size_t crop = 0; crop < count; ++crop) {
            ov::Tensor pixels(ov::element::f32, {1, 3, 2, 2});
            std::copy_n(crops.data<float>() + 12 * crop, 12, pixels.data<float>());
            reference.set_input_tensor(pixels);
            reference.infer();
            const auto expected = reference.get_output_tensor();
            for (size_t i = 0; i < 4; ++i)
                EXPECT_NEAR(actual.data<float>()[4 * crop + i], expected.data<float>()[i], 1e-5f);
        }
    }
}

TEST(GGUFClipAdapter, RejectsUnsupportedNativeInterfacesWithoutChangingSource) {
    auto native = native_encoder();
    native->input().get_tensor().set_names({"other_pixels"});
    const auto parameters = native->get_parameters();
    const auto results = native->get_results();
    EXPECT_THROW(ov::genai::adapt_gguf_clip_encoder(native), ov::Exception);
    EXPECT_EQ(native->get_parameters(), parameters);
    EXPECT_EQ(native->get_results(), results);
}

TEST(GGUFClipAdapter, ImageSlotsPreserveTextAndTruncatedImageRows) {
    const std::vector<bool> slots{false, true, true, true, true, false};
    ov::Tensor first(ov::element::f32, {1, 2, 2}), second(ov::element::f32, {1, 2, 2});
    for (size_t i = 0; i < 4; ++i) {
        first.data<float>()[i] = float(i + 1);
        second.data<float>()[i] = float(i + 5);
    }
    const std::vector<float> expected{.25f, .25f, 1, 2, 3, 4, 5, 6, 7, 8, .25f, .25f};
    for (size_t length : {size_t(6), size_t(4), size_t(2), size_t(1)}) {
        ov::Tensor text(ov::element::f32, {1, length, 2});
        std::fill_n(text.data<float>(), text.get_size(), .25f);
        auto merged = ov::genai::merge_gguf_llava_embeddings(slots, text, {first, second});
        for (size_t i = 0; i < text.get_size(); ++i) {
            EXPECT_FLOAT_EQ(merged.data<float>()[i], expected[expected.size() - text.get_size() + i]);
            EXPECT_FLOAT_EQ(text.data<float>()[i], .25f);
        }
    }
}
