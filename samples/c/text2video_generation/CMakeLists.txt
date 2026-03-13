# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

add_executable(text2video_generation_c main.c)

# Link directly to the internal C API target alias we saw in src/c/CMakeLists.txt
target_link_libraries(text2video_generation_c PRIVATE openvino::genai::c)