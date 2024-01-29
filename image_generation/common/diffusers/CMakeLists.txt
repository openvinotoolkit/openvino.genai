# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

file(GLOB_RECURSE "diffusers_SRC" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(diffusers STATIC ${diffusers_SRC})
add_library(diffusers::diffusers ALIAS diffusers)

target_include_directories(diffusers
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(diffusers PUBLIC -march=native -Wall)
endif()

# dependencies

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(diffusers PUBLIC openvino::runtime)

find_package(Eigen3 REQUIRED)
target_link_libraries(diffusers PRIVATE Eigen3::Eigen)

include(FetchContent)

FetchContent_Declare(safetensors.h
    URL https://github.com/hsnyder/safetensors.h/archive/974a85d7dfd6e010558353226638bb26d6b9d756.tar.gz
    URL_HASH SHA256=9aaf5961609601cf9aaa96582a207bce7c6e5fbf57ed2cc669bb7bde6a937d4b)
FetchContent_MakeAvailable(safetensors.h)

target_include_directories(diffusers SYSTEM PRIVATE "${safetensors.h_SOURCE_DIR}")
