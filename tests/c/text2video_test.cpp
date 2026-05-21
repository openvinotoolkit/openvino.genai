// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "openvino/genai/c/text2video_pipeline.h"

TEST(Text2VideoCAPI, CreatePipelineFailsOnInvalidPath) {
    ov_genai_text2video_pipeline* pipe = nullptr;
    int status = ov_genai_text2video_pipeline_create("invalid_path_does_not_exist", "CPU", &pipe);
    
    ASSERT_NE(status, 0);
    
    if (pipe) {
        ov_genai_text2video_pipeline_destroy(pipe);
    }
}

TEST(Text2VideoCAPI, FreeTensorHandlesNullSafe) {
    ov_genai_text2video_free_tensor(nullptr);
    
    text2video_custom_tensor tensor = {0};
    tensor.data = nullptr;
    ov_genai_text2video_free_tensor(&tensor);
    
    SUCCEED();
}
