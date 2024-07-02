# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)

# start of dependencies

include(FetchContent)

FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
    URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406)

FetchContent_MakeAvailable(nlohmann_json)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# check that SDPA to PA transformtion exists
get_target_property(ov_include_dirs openvino::runtime INTERFACE_INCLUDE_DIRECTORIES)
find_file(spda_to_pa_header sdpa_to_paged_attention.hpp
    PATHS ${ov_include_dirs}
    PATH_SUFFIXES openvino/pass
    DOC "Path to sdpa_to_paged_attention.hpp header"
    NO_CACHE REQUIRED NO_DEFAULT_PATH)

# end of dependencies

set(TARGET_NAME openvino_continuous_batching)

add_library(${TARGET_NAME} STATIC
    src/generation_config.cpp
    src/generation_handle.cpp
    src/continuous_batching_pipeline.cpp
    src/paged_attention_transformations.cpp)

add_library(openvino::continuous_batching ALIAS openvino_continuous_batching)

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
                                          PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_TOKENIZERS_PATH="${OPENVINO_TOKENIZERS_PATH}")
set_target_properties(${TARGET_NAME} PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime openvino::genai PRIVATE nlohmann_json::nlohmann_json)

#
# Installation
#

include(GNUInstallDirs)

install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT openvino_continuous_batching
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT openvino_continuous_batching
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT openvino_continuous_batching)

install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT openvino_continuous_batching
        FILES_MATCHING PATTERN "*.hpp")


# gtest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)


set(TEST_TARGET_NAME "tests_continuous_batching")
add_executable(${TEST_TARGET_NAME} "src/tests/scheduler.cpp" "src/tests/block_manager.cpp" "src/tests/logit_filtering.cpp" "src/tests/cache_manager.cpp" "src/tests/generate_config.cpp")
target_link_libraries(${TEST_TARGET_NAME} PUBLIC ${TARGET_NAME} openvino::runtime gtest_main)
target_include_directories(${TEST_TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/"
                                          PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(${TEST_TARGET_NAME} PRIVATE cxx_std_20)
