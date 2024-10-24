# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

file(DOWNLOAD https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image.h ${CMAKE_BINARY_DIR}/stb_image.h
     EXPECTED_HASH MD5=27932e6fb3a2f26aee2fc33f2cb4e696)

# create main sample executable

add_executable(stable_diffusion
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imwrite.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/load_image.cpp)

target_include_directories(stable_diffusion PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(stable_diffusion PRIVATE openvino::genai)

set_target_properties(stable_diffusion PROPERTIES
    COMPILE_PDB_NAME stable_diffusion
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS stable_diffusion
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

# create LoRA sample executable

add_executable(lora_stable_diffusion
    ${CMAKE_CURRENT_SOURCE_DIR}/lora.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imwrite.cpp)

target_include_directories(lora_stable_diffusion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lora_stable_diffusion PRIVATE openvino::genai)

set_target_properties(lora_stable_diffusion PROPERTIES
    COMPILE_PDB_NAME lora_stable_diffusion
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS lora_stable_diffusion
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
