# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# start of dependencies

include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(cxxopts
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.1.1.tar.gz
    URL_HASH SHA256=523175f792eb0ff04f9e653c90746c12655f10cb70f1d5e6d6d9491420298a08)
FetchContent_MakeAvailable(cxxopts)

if(NOT TARGET nlohmann_json)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
        URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(Threads REQUIRED)

# end of dependencies

genai_add_tool_target(
        NAME continuous_batching_benchmark
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        SOURCES continuous_batching_benchmark.cpp
        LINK_LIBRARIES nlohmann_json::nlohmann_json cxxopts::cxxopts Threads::Threads)
