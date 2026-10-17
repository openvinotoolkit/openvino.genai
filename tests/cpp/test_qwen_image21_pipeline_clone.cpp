// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "openvino/core/except.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"
#include "openvino/genai/image_generation/image2image_pipeline.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {

std::filesystem::path get_qwen_image21_model_dir() {
    const char* base_dir = std::getenv("TEST_MODELS_BASE_DIR");
    if (!base_dir) {
        return {};
    }
    return std::filesystem::path(base_dir) / "tiny-random-qwen-image-2.1";
}

ov::Tensor make_image(const size_t height, const size_t width) {
    ov::Tensor image(ov::element::u8, {1, height, width, 3});
    std::fill_n(image.data<uint8_t>(), image.get_size(), uint8_t{128});
    return image;
}

TEST(QwenImage21PipelineCloneTest, PreservesReshapeValidation) {
    const std::filesystem::path model_dir = get_qwen_image21_model_dir();
    if (model_dir.empty() || !std::filesystem::exists(model_dir)) {
        GTEST_SKIP() << "Converted tiny-random-qwen-image-2.1 model directory not found: " << model_dir;
    }
    ASSERT_TRUE(std::filesystem::is_directory(model_dir));
    ASSERT_TRUE(std::filesystem::exists(model_dir / "model_index.json"))
        << "Incomplete tiny-random-qwen-image-2.1 fixture: missing model_index.json in " << model_dir;

    ov::genai::Image2ImagePipeline pipeline(model_dir);
    pipeline.reshape(1, 64, 64, pipeline.get_generation_config().guidance_scale);
    pipeline.compile("CPU");
    ov::genai::Image2ImagePipeline clone = pipeline.clone();

    const ov::AnyMap generation_properties = {ov::genai::strength(0.8f), ov::genai::num_inference_steps(2)};
    bool got_reshape_validation_error = false;
    try {
        clone.generate("test prompt", make_image(64, 32), generation_properties);
    } catch (const ov::Exception& error) {
        const std::string message = error.what();
        got_reshape_validation_error =
            message.find("while reshape() fixed the VAE encoder to") != std::string::npos;
        EXPECT_TRUE(got_reshape_validation_error) << "Unexpected exception: " << message;
    }
    EXPECT_TRUE(got_reshape_validation_error) << "Expected the clone to reject a mismatched image aspect ratio";

    const ov::Tensor output = clone.generate("test prompt", make_image(64, 64), generation_properties);
    EXPECT_TRUE(output);
}

}  // namespace
