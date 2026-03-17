# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

add_executable(text2video_c_sample text2video_c_sample.c)

target_link_libraries(text2video_c_sample PRIVATE openvino::genai::c)

install(TARGETS text2video_c_sample
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)