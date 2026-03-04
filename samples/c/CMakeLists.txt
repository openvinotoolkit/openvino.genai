# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ADD YOUR NEW VIDEO FOLDER
add_executable(text2video_c_sample video_generation/text2video_c_sample.c)

# Link the GenAI C API
target_link_libraries(text2video_c_sample PRIVATE openvino::genai::c)

# Include paths for the headers we created
target_include_directories(text2video_c_sample PRIVATE 
    "${OpenVINOGenAI_SOURCE_DIR}/src/c/include"
    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/include"
)

# Installation rule so the .exe shows up in samples_bin
install(TARGETS text2video_c_sample
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)