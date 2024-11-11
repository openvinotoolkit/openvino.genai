# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

# create main sample executable

add_executable(stable_diffusion
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imwrite.cpp)

target_include_directories(stable_diffusion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
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

# create txt2image_from_subcomponent sample executable

add_executable(txt2image_from_subcomponent
    ${CMAKE_CURRENT_SOURCE_DIR}/txt2image_from_subcomponent.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imwrite.cpp)

target_include_directories(txt2image_from_subcomponent PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(txt2image_from_subcomponent PRIVATE openvino::genai)

set_target_properties(txt2image_from_subcomponent PROPERTIES
    COMPILE_PDB_NAME txt2image_from_subcomponent
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS txt2image_from_subcomponent
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
