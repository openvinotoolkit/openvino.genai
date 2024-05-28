# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)

file(GLOB_RECURSE "detectors_SRC" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(detectors STATIC ${detectors_SRC})
add_library(detectors::detectors ALIAS detectors)

target_include_directories(detectors
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(detectors PUBLIC -march=native -Wall)
endif()

# dependencies

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
add_subdirectory(../../common/imwrite _deps/imwrite)

# opencv is only for resize now
find_package(OpenCV 4.9.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


target_link_libraries(detectors PUBLIC openvino::runtime imwrite::imwrite ${OpenCV_LIBS})

include(FetchContent)

FetchContent_Declare(eigen3)
FetchContent_GetProperties(eigen3)
if(NOT eigen3_POPULATED)
    FetchContent_Populate(eigen3 QUIET
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
        URL_HASH SHA256=1ccaabbfe870f60af3d6a519c53e09f3dcf630207321dffa553564a8e75c4fc8)
endif()

target_include_directories(detectors SYSTEM PRIVATE "${eigen3_SOURCE_DIR}")

# gtest
FetchContent_Declare(
    googletest
    # Specify the commit you depend on and update it regularly.
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)

  # Now simply link against gtest or gtest_main as needed. Eg
  add_executable(detectors_test detectors_test.cpp)
  target_link_libraries(detectors_test PRIVATE detectors gtest_main)
  gtest_discover_tests(detectors_test)