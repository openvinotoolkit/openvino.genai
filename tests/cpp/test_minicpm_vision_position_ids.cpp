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
#include "visual_language/vlm_utils.hpp"

namespace {

TEST(MiniCPMResizeTest, KeepsFractionalDimensionsUntilPatchAlignment) {
    for (const int merge_factor : {2, 4}) {
        const double width = merge_factor == 2 ? 64.0 : 154.0;
        const auto resized = ov::genai::vlm_utils::find_best_resize(50, width, 112, 14, merge_factor, true);
        EXPECT_EQ(resized.height, merge_factor == 2 ? 112 : 56);
        EXPECT_EQ(resized.width, merge_factor == 2 ? 140 : 224);
    }
}

TEST(MiniCPMResizeTest, PreservesUpscaleAndDownscaleBehavior) {
    const auto small = ov::genai::vlm_utils::find_best_resize(56, 112, 224, 14, 4, false);
    EXPECT_EQ(small.height, 56);
    EXPECT_EQ(small.width, 112);
    const auto large = ov::genai::vlm_utils::find_best_resize(448, 448, 224, 14, 4, false);
    EXPECT_EQ(large.height, 224);
    EXPECT_EQ(large.width, 224);
    const auto fractional = ov::genai::vlm_utils::find_best_resize(50.5, 64.5, 112, 14, 2, true);
    EXPECT_EQ(fractional.height, 112);
    EXPECT_EQ(fractional.width, 140);
    const auto narrow = ov::genai::vlm_utils::find_best_resize(1, 112, 224, 14, 4, false);
    EXPECT_EQ(narrow.height, 56);
    EXPECT_EQ(narrow.width, 112);
}

TEST(MiniCPMResizeTest, RefinesEachSliceWithItsMergeFactor) {
    for (const int merge_factor : {2, 4}) {
        const auto landscape = ov::genai::vlm_utils::get_refine_size(50, 128, {1, 2}, 112, 14, merge_factor);
        EXPECT_EQ(landscape.height, 112);
        EXPECT_EQ(landscape.width, merge_factor == 2 ? 280 : 224);
        const auto portrait = ov::genai::vlm_utils::get_refine_size(128, 50, {2, 1}, 112, 14, merge_factor);
        EXPECT_EQ(portrait.height, merge_factor == 2 ? 280 : 224);
        EXPECT_EQ(portrait.width, 112);
    }
}

TEST(MiniCPMResizeTest, AlignsSourceDimensionsBeforeRefiningSlices) {
    const auto refined = ov::genai::vlm_utils::get_refine_size(5, 7, {2, 3}, 112, 14, 2);
    EXPECT_EQ(refined.height, 280);
    EXPECT_EQ(refined.width, 252);
}

TEST(MiniCPMResizeTest, RejectsEmptyRefinementGrid) {
    EXPECT_THROW(ov::genai::vlm_utils::get_refine_size(112, 112, {}, 112, 14, 4), ov::Exception);
    EXPECT_THROW(ov::genai::vlm_utils::get_refine_size(112, 112, {1, 0}, 112, 14, 4), ov::Exception);
    EXPECT_THROW(ov::genai::vlm_utils::get_refine_size(112, 112, {0, 1}, 112, 14, 4), ov::Exception);
}

TEST(MiniCPMResizeTest, SelectsGridMatchingImageAspectRatio) {
    const auto landscape = ov::genai::vlm_utils::get_slices_grid(112, 224, 2, 112);
    EXPECT_EQ(landscape.rows, 1);
    EXPECT_EQ(landscape.cols, 2);
    const auto portrait = ov::genai::vlm_utils::get_slices_grid(224, 112, 2, 112);
    EXPECT_EQ(portrait.rows, 2);
    EXPECT_EQ(portrait.cols, 1);
    const auto square = ov::genai::vlm_utils::get_slices_grid(224, 224, 4, 112);
    EXPECT_EQ(square.rows, 2);
    EXPECT_EQ(square.cols, 2);
}

TEST(MiniCPMResizeTest, SkipsSlicingForSmallImagesOrDisabledSlicing) {
    EXPECT_FALSE(ov::genai::vlm_utils::get_slices_grid(56, 56, 9, 112).has_slices());
    EXPECT_FALSE(ov::genai::vlm_utils::get_slices_grid(112, 112, 9, 112).has_slices());
    for (const int max_slice_nums : {0, 1}) {
        EXPECT_FALSE(ov::genai::vlm_utils::get_slices_grid(224, 224, max_slice_nums, 112).has_slices());
    }
}

TEST(MiniCPMResizeTest, RespectsSliceLimitAndPrefersMoreRowsOnTies) {
    const auto tied = ov::genai::vlm_utils::get_slices_grid(224, 224, 2, 112);
    EXPECT_EQ(tied.rows, 2);
    EXPECT_EQ(tied.cols, 1);
    for (const int max_slice_nums : {2, 3, 4, 9}) {
        const auto grid = ov::genai::vlm_utils::get_slices_grid(448, 896, max_slice_nums, 112);
        EXPECT_TRUE(grid.has_slices());
        EXPECT_LE(grid.rows * grid.cols, static_cast<size_t>(max_slice_nums));
    }
}

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
    EXPECT_TRUE(encoded.slices_shape.empty());
    EXPECT_FALSE(encoded.slices_grid.has_slices());
    EXPECT_EQ(ov::genai::vlm_utils::get_image_slice_count(encoded), 1);

    const ov::AnyMap overrides{{"scale_resolution", size_t{112}}, {"max_slice_nums", size_t{0}}};
    const auto smaller = encoder.encode(image, overrides);
    EXPECT_EQ(smaller.resized_source.get_shape(), (ov::Shape{64, 1}));
    EXPECT_EQ(encoder.get_processor_config().scale_resolution, 448);

    ov::Tensor large_image(ov::element::u8, ov::Shape{1, 112, 224, 3});
    std::fill_n(large_image.data<uint8_t>(), large_image.get_size(), uint8_t{255});
    const ov::AnyMap sliced_config{{"scale_resolution", size_t{112}}, {"max_slice_nums", size_t{2}}};
    const auto sliced = encoder.encode(large_image, sliced_config);
    EXPECT_TRUE(sliced.slices_grid.has_slices());
    EXPECT_EQ(sliced.slices_grid.rows, 1);
    EXPECT_EQ(sliced.slices_grid.cols, 2);
    EXPECT_EQ(ov::genai::vlm_utils::get_image_slice_count(sliced), 2);
    ASSERT_EQ(sliced.slices_shape.size(), 2);
    EXPECT_EQ(sliced.num_image_tokens, sliced.slices_shape[0] + 2 * sliced.slices_shape[1]);
    ov::Tensor portrait_image(ov::element::u8, ov::Shape{1, 224, 112, 3});
    std::fill_n(portrait_image.data<uint8_t>(), portrait_image.get_size(), uint8_t{255});
    const auto portrait = encoder.encode(portrait_image, sliced_config);
    EXPECT_EQ(portrait.slices_grid.rows, 2);
    EXPECT_EQ(portrait.slices_grid.cols, 1);
    ASSERT_EQ(portrait.slices_shape.size(), 2);
    EXPECT_EQ(portrait.num_image_tokens, portrait.slices_shape[0] + 2 * portrait.slices_shape[1]);
    const auto unsliced = encoder.encode(large_image, overrides);
    EXPECT_FALSE(unsliced.slices_grid.has_slices());
    EXPECT_TRUE(unsliced.slices_shape.empty());
    EXPECT_EQ(unsliced.num_image_tokens, batched_positions ? 48 : 72);

    ov::genai::VLMConfig vision_config;
    vision_config.vision_config_image_size = 448;
    encoder.update_vision_config(vision_config);
    const auto updated = encoder.encode(image, {});
    const float* updated_positions = updated.resized_source.data<float>();
    EXPECT_EQ(updated_positions[1], 1.0f);
    EXPECT_EQ(updated_positions[32], 32.0f);
}

INSTANTIATE_TEST_SUITE_P(OcrAndV46, MiniCPMVisionPositionIdsTest, testing::Bool());

}  // namespace
