# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if(TARGET openvino_tokenizers)
    set(OPENVINO_TOKENIZERS_PATH $<TARGET_FILE:openvino_tokenizers>)
else()
    message(FATAL_ERROR "multinomial_causal_lm must be compiled as part of OpenVIINOGenAI to have the path to openvino_tokenizers hardcoded.")
endif()
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(TBB REQUIRED COMPONENTS tbb)
add_executable(prompt_lookup_decoding_lm prompt_lookup_decoding_lm.cpp)
target_link_libraries(prompt_lookup_decoding_lm PRIVATE openvino::runtime TBB::tbb)
target_compile_definitions(prompt_lookup_decoding_lm PRIVATE OPENVINO_TOKENIZERS_PATH="${OPENVINO_TOKENIZERS_PATH}")
target_compile_features(prompt_lookup_decoding_lm PRIVATE cxx_std_17)
install(TARGETS prompt_lookup_decoding_lm
    RUNTIME DESTINATION samples_bin/
    COMPONENT samples_bin
    EXCLUDE_FROM_ALL)
