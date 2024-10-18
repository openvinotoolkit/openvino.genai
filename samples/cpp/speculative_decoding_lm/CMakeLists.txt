# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# start of dependencies

include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

# end of dependencies

set(TARGET_NAME speculative_decoding_lm)
add_executable(${TARGET_NAME} ${TARGET_NAME}.cpp)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::genai cxxopts::cxxopts)

set_target_properties(${TARGET_NAME}  PROPERTIES
    COMPILE_PDB_NAME ${TARGET_NAME} 
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS ${TARGET_NAME} 
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
