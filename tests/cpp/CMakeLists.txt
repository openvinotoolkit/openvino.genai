# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT TARGET gtest)
  set(INSTALL_GTEST OFF CACHE BOOL "")
  set(BUILD_SHARED_LIBS OFF)
  
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

file(GLOB tests_src "*.cpp")

set(TEST_TARGET_NAME "tests_continuous_batching")

add_executable(${TEST_TARGET_NAME} ${tests_src} $<TARGET_OBJECTS:openvino_genai_obj>)

target_link_libraries(${TEST_TARGET_NAME} PRIVATE $<TARGET_PROPERTY:openvino::genai,LINK_LIBRARIES> gtest_main gmock_main)
target_include_directories(${TEST_TARGET_NAME} PRIVATE "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src"
                                                       $<TARGET_PROPERTY:openvino::genai,INTERFACE_INCLUDE_DIRECTORIES>)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_link_options(${TEST_TARGET_NAME} PRIVATE /IGNORE:4207,4286)
endif()

install(TARGETS ${TEST_TARGET_NAME}
        RUNTIME DESTINATION tests/
        COMPONENT tests
        EXCLUDE_FROM_ALL)