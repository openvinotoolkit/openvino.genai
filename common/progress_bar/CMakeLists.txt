# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

add_library(progress_bar STATIC src/progress_bar.cpp)
add_library(progress_bar::progress_bar ALIAS progress_bar)

target_include_directories(progress_bar PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
