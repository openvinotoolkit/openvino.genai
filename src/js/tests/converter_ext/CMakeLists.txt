# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(converter_test_ext CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DNAPI_VERSION=8)

include(FetchContent)

FetchContent_Declare(
    node-api-headers
    URL      https://github.com/nodejs/node-api-headers/archive/refs/tags/v1.9.0.tar.gz
    URL_HASH SHA256=071f773bd10479f865f6828b86fe8232d63e255290d9bbb7a0ec487c57310231
)
FetchContent_MakeAvailable(node-api-headers)

FetchContent_Declare(
    node-addon-api
    URL      https://github.com/nodejs/node-addon-api/archive/refs/tags/v8.9.0.tar.gz
    URL_HASH SHA256=d3c35d47f57461a49826349928da4913cb467d81288549973b7f9335f5d4c001
)
FetchContent_MakeAvailable(node-addon-api)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

set(JS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

add_library(converter_test_ext SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/converter_test_ext.cpp"
    "${JS_ROOT}/src/any_converter.cpp"
)

target_include_directories(converter_test_ext PRIVATE
    "${node-api-headers_SOURCE_DIR}/include"
    "${node-addon-api_SOURCE_DIR}"
    "${JS_ROOT}"
)

target_link_libraries(converter_test_ext PRIVATE openvino::runtime)

set_target_properties(converter_test_ext PROPERTIES PREFIX "" SUFFIX ".node")

# Node resolves N-API symbols when loading the addon.
if(APPLE)
    target_link_options(converter_test_ext PRIVATE -Wl,-undefined,dynamic_lookup)
endif()
