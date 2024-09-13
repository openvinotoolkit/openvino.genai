# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

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
