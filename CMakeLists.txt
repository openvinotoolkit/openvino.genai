# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)  # The requirement comes from thirdparty/openvino_contrib/modules/custom_operations/CMakeLists.txt
project(GenAI)

list(APPEND CUSTOM_OPERATIONS tokenizer)
add_subdirectory(llm/mincpmv2_cpp)
add_subdirectory(thirdparty/openvino_tokenizers/ "${CMAKE_CURRENT_BINARY_DIR}/openvino_tokenizers/")
target_compile_definitions(minicpmv2 PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")

