# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

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

add_executable(generate_with_encrypted_model generate_with_encrypted_model.cpp)
target_link_libraries(generate_with_encrypted_model PRIVATE openvino::genai)
set_target_properties(generate_with_encrypted_model PROPERTIES
    COMPILE_PDB_NAME generate_with_encrypted_model
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
target_compile_features(generate_with_encrypted_model PRIVATE cxx_std_11)

install(TARGETS generate_with_encrypted_model
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
