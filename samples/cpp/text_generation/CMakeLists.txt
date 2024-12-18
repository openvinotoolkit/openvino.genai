# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)


# greedy_causal_lm
add_executable(greedy_causal_lm greedy_causal_lm.cpp)
target_link_libraries(greedy_causal_lm PRIVATE openvino::genai)
set_target_properties(greedy_causal_lm PROPERTIES
    COMPILE_PDB_NAME greedy_causal_lm
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
target_compile_features(greedy_causal_lm PRIVATE cxx_std_11)
install(TARGETS greedy_causal_lm
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)


# encrypted_model_causal_lm
add_executable(encrypted_model_causal_lm encrypted_model_causal_lm.cpp)
target_link_libraries(encrypted_model_causal_lm PRIVATE openvino::genai)
set_target_properties(encrypted_model_causal_lm PROPERTIES
    COMPILE_PDB_NAME encrypted_model_causal_lm
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
target_compile_features(encrypted_model_causal_lm PRIVATE cxx_std_11)
install(TARGETS encrypted_model_causal_lm
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)


# beam_search_causal_lm
add_executable(beam_search_causal_lm beam_search_causal_lm.cpp)
target_link_libraries(beam_search_causal_lm PRIVATE openvino::genai)
set_target_properties(beam_search_causal_lm PROPERTIES
    COMPILE_PDB_NAME beam_search_causal_lm
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
target_compile_features(beam_search_causal_lm PRIVATE cxx_std_11)
install(TARGETS beam_search_causal_lm
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)


# chat_sample
add_executable(chat_sample chat_sample.cpp)
target_link_libraries(chat_sample PRIVATE openvino::genai)
set_target_properties(chat_sample PROPERTIES
    COMPILE_PDB_NAME chat_sample
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
target_compile_features(chat_sample PRIVATE cxx_std_11)
install(TARGETS chat_sample
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)


# lora_greedy_causal_lm
add_executable(lora_greedy_causal_lm lora_greedy_causal_lm.cpp)
target_link_libraries(lora_greedy_causal_lm PRIVATE openvino::genai)
set_target_properties(lora_greedy_causal_lm PROPERTIES
    COMPILE_PDB_NAME lora_greedy_causal_lm
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
target_compile_features(lora_greedy_causal_lm PRIVATE cxx_std_11)
install(TARGETS lora_greedy_causal_lm
    RUNTIME DESTINATION samples_bin/
    COMPONENT samples_bin
    EXCLUDE_FROM_ALL)


# multinomial_causal_lm
add_executable(multinomial_causal_lm multinomial_causal_lm.cpp)
target_link_libraries(multinomial_causal_lm PRIVATE openvino::genai)
set_target_properties(multinomial_causal_lm PROPERTIES
    COMPILE_PDB_NAME multinomial_causal_lm
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
target_compile_features(multinomial_causal_lm PRIVATE cxx_std_11)

install(TARGETS multinomial_causal_lm
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)


# prompt_lookup_decoding_lm
add_executable(prompt_lookup_decoding_lm prompt_lookup_decoding_lm.cpp)
target_link_libraries(prompt_lookup_decoding_lm PRIVATE openvino::genai)

set_target_properties(prompt_lookup_decoding_lm PROPERTIES
    COMPILE_PDB_NAME prompt_lookup_decoding_lm
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
# Don't install prompt_lookup_decoding_lm because it doesn't use openvino_genai library and is not verified yet.
# install(TARGETS prompt_lookup_decoding_lm
#         RUNTIME DESTINATION samples_bin/
#         COMPONENT samples_bin
#         EXCLUDE_FROM_ALL)


# speculative_decoding_lm
add_executable(speculative_decoding_lm speculative_decoding_lm.cpp)
target_link_libraries(speculative_decoding_lm PRIVATE openvino::genai)

set_target_properties(speculative_decoding_lm PROPERTIES
    COMPILE_PDB_NAME speculative_decoding_lm
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS speculative_decoding_lm
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
