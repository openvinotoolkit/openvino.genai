// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_utils/gguf_clip_adapter.hpp"
#include "gtest/gtest.h"
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

std::shared_ptr<ov::Model> native_packed_encoder() {
    auto first = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 3, -1, -1});
    auto second = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 3, -1, -1});
    auto positions = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{1, 1, 1, -1});
    first->output(0).set_names({"vision.pixel_values"});
    second->output(0).set_names({"vision.pixel_values_1"});
    positions->output(0).set_names({"vision.positions"});
    const auto axes = [](std::initializer_list<int64_t> values) {
        return Constant::create(ov::element::i64, {values.size()}, values);
    };
    auto combined =
        std::make_shared<Add>(first, std::make_shared<Multiply>(second, Constant::create(ov::element::f32, {}, {2})));
    auto sampled =
        std::make_shared<Slice>(combined, axes({0, 0}), axes({INT64_MAX, INT64_MAX}), axes({2, 2}), axes({2, 3}));
    auto flattened =
        std::make_shared<Reshape>(std::make_shared<Transpose>(sampled, axes({0, 2, 3, 1})), axes({1, 1, -1, 3}), false);
    auto coordinates =
        std::make_shared<Reshape>(std::make_shared<Convert>(positions, ov::element::f32), axes({4, -1}), false);
    auto rows = std::make_shared<Gather>(coordinates, axes({0}), axes({0}));
    auto cols = std::make_shared<Gather>(coordinates, axes({1}), axes({0}));
    auto offsets =
        std::make_shared<Add>(rows, std::make_shared<Multiply>(cols, Constant::create(ov::element::f32, {}, {10})));
    auto features = std::make_shared<Add>(flattened, std::make_shared<Reshape>(offsets, axes({1, 1, -1, 1}), false));
    features->output(0).set_names({"vision.embeddings"});
    return std::make_shared<ov::Model>(ov::OutputVector{features}, ov::ParameterVector{first, second, positions});
}
}  // namespace

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

TEST(GGUFClipAdapter, PackedTemporalPatchesPreserveChannelsAndSpatialCoordinates) {
    ov::Core core;
    const auto native = native_packed_encoder();
    auto adapted = ov::genai::adapt_gguf_packed_vision_encoder(native, 2, 2);
    EXPECT_EQ(native->get_parameters().size(), 3);
    EXPECT_EQ(native->input(0).get_partial_shape(), (ov::PartialShape{1, 3, -1, -1}));
    EXPECT_EQ(adapted->input().get_partial_shape(), (ov::PartialShape{-1, -1, -1, 24}));
    auto request =
        core.compile_model(adapted, "CPU", ov::hint::inference_precision(ov::element::f32)).create_infer_request();
    for (const size_t groups : {size_t(1), size_t(3)}) {
        for (const auto grid : {ov::Shape{2, 4}, ov::Shape{4, 2}}) {
            const auto height = grid[0], width = grid[1];
            ov::Tensor patches(ov::element::f32, {groups, height, width, 24});
            for (size_t i = 0; i < patches.get_size(); ++i)
                patches.data<float>()[i] = float(i) * .03f - 8.f;
            request.set_input_tensor(patches);
            request.infer();
            const auto output = request.get_output_tensor();
            ASSERT_EQ(output.get_shape(), (ov::Shape{groups, height * width, 3}));
            for (size_t group = 0; group < groups; ++group) {
                for (size_t row = 0; row < height; ++row) {
                    for (size_t col = 0; col < width; ++col) {
                        for (size_t channel = 0; channel < 3; ++channel) {
                            const auto token = (group * height + row) * width + col;
                            const auto base = token * 24 + channel * 8;
                            const float expected = patches.data<float>()[base] + 2 * patches.data<float>()[base + 4] +
                                                   float(row + 10 * col);
                            EXPECT_NEAR(output.data<float>()[token * 3 + channel], expected, 1e-5f);
                        }
                    }
                }
            }
        }
    }
}

TEST(GGUFClipAdapter, PackedVisionRejectsUnsupportedInputsWithoutChangingSource) {
    auto native = native_packed_encoder();
    native->input("vision.positions").get_tensor().set_names({"other_positions"});
    EXPECT_THROW(ov::genai::adapt_gguf_packed_vision_encoder(native, 2, 2), ov::Exception);
    EXPECT_EQ(native->get_parameters().size(), 3);
    EXPECT_THROW(ov::genai::adapt_gguf_packed_vision_encoder(native_packed_encoder(), 0, 2), ov::Exception);
}
