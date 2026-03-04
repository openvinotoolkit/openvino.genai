# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Add the Text-to-Video C API Test
ov_add_test(
    NAME text2video_c_api_test
    SOURCES text2video_test.cpp
    LIBRARIES openvino_genai   
    LABELS "genai"
)

target_include_directories(text2video_c_api_test PRIVATE 
    "${OpenVINOGenAI_SOURCE_DIR}/src/c/include"
)
