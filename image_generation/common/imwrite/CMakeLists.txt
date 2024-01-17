# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

add_library(imwrite STATIC src/imwrite.cpp)
add_library(imwrite::imwrite ALIAS imwrite)

target_include_directories(imwrite PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(imwrite PRIVATE openvino::runtime)
