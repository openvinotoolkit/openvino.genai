# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.10)
project(text2video_generation_c C)

# Find the OpenVINO and GenAI packages
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(openvino_genai REQUIRED)

# Create the executable
add_executable(${PROJECT_NAME} main.c)

# Link against the OpenVINO GenAI C API library
target_link_libraries(${PROJECT_NAME} PRIVATE openvino::genai_c)