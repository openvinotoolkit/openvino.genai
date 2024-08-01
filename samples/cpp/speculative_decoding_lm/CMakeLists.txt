# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if(TARGET openvino_tokenizers)
    set(OPENVINO_TOKENIZERS_PATH $<TARGET_FILE:openvino_tokenizers>)
else()
    message(FATAL_ERROR "multinomial_causal_lm must be compiled as part of OpenVIINOGenAI to have the path to openvino_tokenizers hardcoded.")
endif()

include(FetchContent)

FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
    URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406)
FetchContent_MakeAvailable(nlohmann_json)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(TBB REQUIRED COMPONENTS tbb)

add_executable(speculative_decoding_lm speculative_decoding_lm.cpp)

target_link_libraries(speculative_decoding_lm PRIVATE openvino::runtime TBB::tbb nlohmann_json::nlohmann_json)

set_target_properties(speculative_decoding_lm PROPERTIES
    COMPILE_PDB_NAME speculative_decoding_lm
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

    target_compile_definitions(speculative_decoding_lm PRIVATE OPENVINO_TOKENIZERS_PATH="${OPENVINO_TOKENIZERS_PATH}")
target_compile_features(speculative_decoding_lm PRIVATE cxx_std_17)

install(TARGETS speculative_decoding_lm
    RUNTIME DESTINATION samples_bin/
    COMPONENT samples_bin
    EXCLUDE_FROM_ALL)
