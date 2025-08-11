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

set(TARGET_NAME continuous_batching_benchmark)
add_executable(${TARGET_NAME} ${TARGET_NAME}.cpp)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::genai nlohmann_json::nlohmann_json cxxopts::cxxopts Threads::Threads)

set_target_properties(${TARGET_NAME} PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS ${TARGET_NAME} 
        RUNTIME DESTINATION samples_bin/
        COMPONENT tools_bin
        EXCLUDE_FROM_ALL)