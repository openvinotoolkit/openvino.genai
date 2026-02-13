// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "openvino/genai/c/text2video_pipeline.h"

// Verify that the API handles invalid model paths gracefully

TEST(Text2VideoCAPI, CreatePipelineFailsOnInvalidPath) {
    ov_genai_text2video_pipeline* pipe = nullptr;

    // Try to load a fake model path
    int status = ov_genai_text2video_pipeline_create("invalid_path_does_not_exist", "CPU", &pipe);

    // We expect a non-zero status (Error) because the path is wrong
    ASSERT_NE(status, 0);

    // Ensure the pipe pointer wasn't wrongly assigned
    if (pipe) {
        ov_genai_text2video_pipeline_destroy(pipe);
    }
}

// Verify memory cleanup logic
TEST(Text2VideoCAPI, FreeTensorHandlesNullSafe) {
    // Test freeing a NULL tensor (Should not crash)
    ov_genai_text2video_free_tensor(nullptr);

    // Test freeing a tensor with NULL data (Should not crash)
    text2video_custom_tensor tensor = {0};
    tensor.data = nullptr;
    ov_genai_text2video_free_tensor(&tensor);

    SUCCEED();
}