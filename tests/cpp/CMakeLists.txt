# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT TARGET gtest)
  set(INSTALL_GTEST OFF CACHE BOOL "")

  FetchContent_Declare(
    googletest
    URL       https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    URL_HASH SHA256=edd885a1ab32b6999515a880f669efadb80b3f880215f315985fa3f6eca7c4d3
  )
  FetchContent_MakeAvailable(googletest)
endif()

file(GLOB tests_src "*.cpp")

set(TEST_TARGET_NAME "tests_continuous_batching")
add_executable(${TEST_TARGET_NAME} ${tests_src} $<TARGET_OBJECTS:openvino_genai_obj>)

target_link_libraries(${TEST_TARGET_NAME} PRIVATE $<TARGET_PROPERTY:openvino::genai,LINK_LIBRARIES> gtest_main gmock_main)
target_include_directories(${TEST_TARGET_NAME} PRIVATE "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src"
                                                       "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/include")
