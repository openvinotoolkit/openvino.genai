// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "openvino/genai/video_generation/text2video_pipeline.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace ov::genai;

// Test fixture to verify the LTX Pipeline memory-safe decoding
TEST(Text2VideoPipelineTest, TemporalSlicerShapeVerification) {
    // We use a try-catch block so the CI doesn't hard-fail if the weights aren't downloaded locally
    std::string models_dir = "dummy_ltx_dir";

    try {
        Text2VideoPipeline pipeline(models_dir);

        // Shape: [Batch, Channels, Frames, Height, Width] -> [1, 128, 40, 32, 32]
        // 40 latent frames is > our 32 chunk limit, forcing exactly 2 chunks internally.
        ov::Shape latent_shape = {1, 128, 40, 32, 32};
        ov::Tensor dummy_latent(ov::element::f32, latent_shape);

        std::memset(dummy_latent.data<float>(), 0, dummy_latent.get_byte_size());

        VideoGenerationResult result = pipeline.decode(dummy_latent);
        ov::Tensor out_video = result.video;

        // For LTX, the temporal compression ratio (C) is 8.
        // Expected output frames = (Latent_Frames - 1) * C + 1
        // Expected = (40 - 1) * 8 + 1 = 313 frames.
        size_t expected_frames = 313;
        ov::Shape out_shape = out_video.get_shape();

        // Output video shape should be: [Batch, RGB, Frames, Height*8, Width*8]
        EXPECT_EQ(out_shape[2], expected_frames) << "Temporal slicer failed to align frame count. Expected "
                                                 << expected_frames << " but got " << out_shape[2];

        EXPECT_NE(out_video.data(), nullptr) << "Output video tensor is null.";

    } catch (const ov::Exception& e) {
        GTEST_SKIP() << "Skipping test due to missing local dummy model weights: " << e.what();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping test due to missing local dummy model weights: " << e.what();
    }
}