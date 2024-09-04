# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    HINTS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

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
