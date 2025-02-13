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

  foreach(gtarget IN ITEMS gtest gmock gtest_main gmock_main)
    if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR (OV_COMPILER_IS_INTEL_LLVM AND UNIX))
      target_compile_options(${gtarget} PUBLIC -Wno-undef)
    endif()
  endforeach()
endif()

set(TEST_TARGET_NAME "tests_continuous_batching")
file(GLOB tests_src "*.cpp")
file(GLOB src_files "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/sequence_group.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/cache_eviction.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/sampler.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/speculative_decoding/*.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/prompt_lookup/*.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/paged_attention_transformations.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/utils.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/continuous_batching*.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/icontinuous_batching.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/lora_helper.cpp")

add_executable(${TEST_TARGET_NAME} ${tests_src})

target_link_libraries(${TEST_TARGET_NAME} PRIVATE openvino::genai gtest_main gmock_main)
target_include_directories(${TEST_TARGET_NAME} PRIVATE "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src")
target_sources(${TEST_TARGET_NAME} PRIVATE ${src_files})
