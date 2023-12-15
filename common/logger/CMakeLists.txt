# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

add_library(logger STATIC src/logger.cpp)
add_library(logger::logger ALIAS logger)

target_include_directories(logger PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
