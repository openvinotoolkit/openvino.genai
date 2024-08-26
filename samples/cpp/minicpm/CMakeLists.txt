# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED PATHS
    "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
    ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
)

add_executable(minicpm minicpm.cpp)
target_link_libraries(minicpm PRIVATE openvino::genai)
set_target_properties(minicpm PROPERTIES
    COMPILE_PDB_NAME chat_sample
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
target_compile_features(minicpm PRIVATE cxx_std_11)

install(TARGETS minicpm
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
