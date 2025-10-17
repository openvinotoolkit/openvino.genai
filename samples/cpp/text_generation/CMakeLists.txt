# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

function(add_sample_executable target_name)
    add_executable(${target_name} ${target_name}.cpp)
    target_link_libraries(${target_name} PRIVATE openvino::genai)
    set_target_properties(${target_name} PROPERTIES
        # Ensure out-of-box LC_RPATH on macOS with SIP
        INSTALL_RPATH_USE_LINK_PATH ON)
    install(TARGETS ${target_name}
            RUNTIME DESTINATION samples_bin/
            COMPONENT samples_bin
            EXCLUDE_FROM_ALL)
endfunction()

set (SAMPLE_LIST
    greedy_causal_lm
    encrypted_model_causal_lm
    beam_search_causal_lm
    chat_sample
    structured_output_generation
    lora_greedy_causal_lm
    multinomial_causal_lm
    prompt_lookup_decoding_lm
    speculative_decoding_lm)

foreach(sample IN LISTS SAMPLE_LIST)
    add_sample_executable(${sample})
endforeach()

# benchmark_genai
include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(cxxopts
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.1.1.tar.gz
    URL_HASH SHA256=523175f792eb0ff04f9e653c90746c12655f10cb70f1d5e6d6d9491420298a08)
FetchContent_MakeAvailable(cxxopts)

add_executable(benchmark_genai benchmark_genai.cpp read_prompt_from_file.cpp)
target_link_libraries(benchmark_genai PRIVATE openvino::genai cxxopts::cxxopts)
set_target_properties(benchmark_genai PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS benchmark_genai
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
