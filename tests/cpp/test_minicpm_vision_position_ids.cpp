// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <sstream>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/serialize.hpp"
#include "visual_language/minicpmv4_6/classes.hpp"

namespace {

class MiniCPMVisionPositionIdsTest : public testing::TestWithParam<bool> {};

TEST_P(MiniCPMVisionPositionIdsTest, MatchesExportedInputRank) {
    const bool batched_positions = GetParam();
    auto pixels = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::PartialShape{1, 3, 14, -1});
    pixels->output(0).set_names({"pixel_values"});
    const ov::PartialShape position_shape = batched_positions ? ov::PartialShape{1, -1} : ov::PartialShape{-1};
    auto positions = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, position_shape);
    positions->output(0).set_names({"position_ids"});
    auto merge = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    merge->output(0).set_names({"merge_index"});
    ov::ParameterVector parameters{pixels, positions, merge};
    if (batched_positions) {
        auto window = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
        window->output(0).set_names({"window_index"});
        parameters.push_back(window);
    }
    auto converted = std::make_shared<ov::op::v0::Convert>(positions, ov::element::f32);
    auto output = std::make_shared<ov::op::v1::Reshape>(
        converted, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {-1, 1}), false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{output}, parameters);
    std::ostringstream xml;
    std::ostringstream bin;
    ov::pass::Serialize(xml, bin).run_on_model(model);
    const std::string weights_data = bin.str();
    ov::Tensor weights(ov::element::u8, ov::Shape{weights_data.size()});
    std::memcpy(weights.data(), weights_data.data(), weights_data.size());
    const ov::genai::ModelsMap models{{"vision_embeddings", {xml.str(), weights}}};
    const auto config_dir = std::filesystem::temp_directory_path() /
                            ("minicpm_vision_no_config_" +
                             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ov::genai::VisionEncoderMiniCPMV4_6 encoder(models, config_dir, "CPU", {});
    ov::Tensor image(ov::element::u8, ov::Shape{1, 56, 56, 3});
    std::fill_n(image.data<uint8_t>(), image.get_size(), uint8_t{255});
    const auto encoded = encoder.encode(image, {});
    ASSERT_EQ(encoded.resized_source.get_shape(), (ov::Shape{1024, 1}));
    const float* values = encoded.resized_source.data<float>();
    EXPECT_EQ(values[0], 0.0f);
    EXPECT_EQ(values[1], 2.0f);
    EXPECT_EQ(values[32], 140.0f);
}

INSTANTIATE_TEST_SUITE_P(OcrAndV46, MiniCPMVisionPositionIdsTest, testing::Bool());

}  // namespace
