# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.15)
project(openvino_genai)

add_subdirectory(./thirdparty/openvino_tokenizers/ "${CMAKE_CURRENT_BINARY_DIR}/openvino_tokenizers/")

add_subdirectory(src)
add_subdirectory(text_generation/causal_lm/cpp)
