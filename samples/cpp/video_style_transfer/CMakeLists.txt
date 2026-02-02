# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.10)

find_package(OpenCV REQUIRED)

ov_add_sample(
    NAME video_style_transfer
    SOURCES main.cpp
    DEPENDENCIES openvino_genai ${OpenCV_LIBS}
)