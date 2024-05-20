# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.15)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake build type")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" "RelWithDebInfo" "MinSizeRel")

project(openvino_genai VERSION 2024.2.0.0)

add_subdirectory(./thirdparty/openvino_tokenizers/ "${CMAKE_CURRENT_BINARY_DIR}/openvino_tokenizers/")
add_subdirectory(src)
add_subdirectory(text_generation/causal_lm/cpp)

install(DIRECTORY text_generation/causal_lm/cpp/ DESTINATION samples/cpp/causal_lm COMPONENT cpp_samples_genai)
install(FILES LICENSE third-party-programs.txt DESTINATION licensing_genai COMPONENT licensing_genai)  # TODO: how to merge with OPenvino
set(CPACK_GENERATOR "ZIP")
include(CPack)
