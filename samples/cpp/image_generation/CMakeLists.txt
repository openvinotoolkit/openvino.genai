# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

file(DOWNLOAD https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image.h ${CMAKE_BINARY_DIR}/stb_image.h
     EXPECTED_HASH MD5=27932e6fb3a2f26aee2fc33f2cb4e696)

include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(indicators
    URL https://github.com/p-ranav/indicators/archive/refs/tags/v2.3.tar.gz
    URL_HASH SHA256=70da7a693ff7a6a283850ab6d62acf628eea17d386488af8918576d0760aef7b)
FetchContent_MakeAvailable(indicators)

# create main sample executable

add_executable(text2image text2image.cpp imwrite.cpp)

target_include_directories(text2image PRIVATE ${CMAKE_BINARY_DIR} "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(text2image PRIVATE openvino::genai indicators::indicators)

set_target_properties(text2image PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS text2image
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

# create text2image concurrent sample executable

add_executable(text2image_concurrency text2image_concurrency.cpp imwrite.cpp)

target_include_directories(text2image_concurrency PRIVATE ${CMAKE_BINARY_DIR} "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(text2image_concurrency PRIVATE openvino::genai indicators::indicators)

set_target_properties(text2image_concurrency PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS text2image_concurrency
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

# create LoRA sample executable

add_executable(lora_text2image lora_text2image.cpp imwrite.cpp)

target_include_directories(lora_text2image PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(lora_text2image PRIVATE openvino::genai indicators::indicators)

set_target_properties(lora_text2image PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS lora_text2image
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

# create heterogeneous_stable_diffusion sample executable

add_executable(heterogeneous_stable_diffusion
    heterogeneous_stable_diffusion.cpp
    imwrite.cpp)

target_include_directories(heterogeneous_stable_diffusion PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(heterogeneous_stable_diffusion PRIVATE openvino::genai indicators::indicators)

set_target_properties(heterogeneous_stable_diffusion PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS heterogeneous_stable_diffusion
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

# create image2image sample executable

add_executable(image2image image2image.cpp load_image.cpp imwrite.cpp)

target_include_directories(image2image PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_BINARY_DIR}")
target_link_libraries(image2image PRIVATE openvino::genai indicators::indicators)

set_target_properties(image2image PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS image2image
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

# create image2image concurrent sample executable

add_executable(image2image_concurrency image2image_concurrency.cpp load_image.cpp imwrite.cpp)

target_include_directories(image2image_concurrency PRIVATE ${CMAKE_BINARY_DIR} "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(image2image_concurrency PRIVATE openvino::genai indicators::indicators)

set_target_properties(image2image_concurrency PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS image2image_concurrency
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

# create stable_diffusion_export_import sample executable

add_executable(stable_diffusion_export_import stable_diffusion_export_import.cpp load_image.cpp imwrite.cpp)

target_include_directories(stable_diffusion_export_import PRIVATE ${CMAKE_BINARY_DIR} "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(stable_diffusion_export_import PRIVATE openvino::genai indicators::indicators)

set_target_properties(stable_diffusion_export_import PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS stable_diffusion_export_import
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

# create inpainting executable

add_executable(inpainting inpainting.cpp load_image.cpp imwrite.cpp)

target_include_directories(inpainting PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_BINARY_DIR}")
target_link_libraries(inpainting PRIVATE openvino::genai indicators::indicators)

set_target_properties(inpainting PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS inpainting
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

# benchmark_image_gen

include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(cxxopts
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.1.1.tar.gz
    URL_HASH SHA256=523175f792eb0ff04f9e653c90746c12655f10cb70f1d5e6d6d9491420298a08)
FetchContent_MakeAvailable(cxxopts)

add_executable(benchmark_image_gen benchmark_image_gen.cpp load_image.cpp imwrite.cpp)
target_include_directories(benchmark_image_gen PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_BINARY_DIR}")
target_link_libraries(benchmark_image_gen PRIVATE openvino::genai cxxopts::cxxopts indicators::indicators)
set_target_properties(benchmark_image_gen PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS benchmark_image_gen
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
