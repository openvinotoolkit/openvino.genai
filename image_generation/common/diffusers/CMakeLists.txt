# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

add_library(diffusers STATIC src/scheduler_lms_discrete.cpp)
add_library(diffusers::diffusers ALIAS diffusers)

target_include_directories(diffusers PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(diffusers PRIVATE openvino::runtime)
