# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)

project(lcm_dreamshaper LANGUAGES CXX)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake build type")

# dependencies

if(WIN32)
    message(STATUS "CMake is running on Windows")
    set(EIGEN3_INCLUDE_DIR "C:/Eigen3/eigen-3.4.0")
endif()

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(Eigen3 REQUIRED)

include(FetchContent)

FetchContent_Declare(cxxopts
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.1.1.tar.gz
    URL_HASH SHA256=523175f792eb0ff04f9e653c90746c12655f10cb70f1d5e6d6d9491420298a08)

FetchContent_MakeAvailable(cxxopts)

add_subdirectory(../../common/imwrite _deps/imwrite)
add_subdirectory(../../common/diffusers _deps/diffusers)

set(CUSTOM_OPERATIONS tokenizer)
add_subdirectory(../../../thirdparty/openvino_contrib/modules/custom_operations/ _deps/tokenizers)

# create executable

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
    # TODO: remove after migration to string tensors
    $<TARGET_PROPERTY:user_ov_extensions,INTERFACE_INCLUDE_DIRECTORIES>
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE
    openvino::runtime
    Eigen3::Eigen
    cxxopts::cxxopts
    diffusers::diffusers
    imwrite::imwrite)

add_dependencies(${PROJECT_NAME} user_ov_extensions)

target_compile_definitions(${PROJECT_NAME} PRIVATE TOKENIZERS_LIBRARY_PATH=\"$<TARGET_FILE:user_ov_extensions>\")

if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${PROJECT_NAME} PRIVATE -march=native -Wall)
endif()
