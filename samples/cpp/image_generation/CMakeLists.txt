# Copyright (C) 2023-2026 Intel Corporation
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
include(${CMAKE_CURRENT_SOURCE_DIR}/../fetch_opencv.cmake)

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

# create denoising_process sample executable

add_executable(denoising_process denoising_process.cpp ../video_generation/imwrite_video.cpp)

target_include_directories(denoising_process PRIVATE
    ${CMAKE_BINARY_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../video_generation")
ov_genai_link_opencv(denoising_process core imgproc videoio imgcodecs)
target_link_libraries(denoising_process PRIVATE openvino::genai indicators::indicators)

set_target_properties(denoising_process PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

# Windows: when OpenCV was built from source via FetchContent, deploy the
# component DLLs next to denoising_process.exe in samples_bin/. The Windows
# loader has no rpath, and install/samples_bin/ is not on PATH at runtime
# unless the user puts it there. Skipped when find_package found a system
# OpenCV, since those installs ship their own runtime path.
get_property(_ov_genai_opencv_fetched GLOBAL PROPERTY OV_GENAI_FETCHED_OPENCV)
if(WIN32 AND _ov_genai_opencv_fetched)
    install(FILES "$<TARGET_FILE:opencv_core>"
                  "$<TARGET_FILE:opencv_imgproc>"
                  "$<TARGET_FILE:opencv_imgcodecs>"
                  "$<TARGET_FILE:opencv_videoio>"
            DESTINATION samples_bin/
            COMPONENT samples_bin
            EXCLUDE_FROM_ALL)
endif()

install(TARGETS denoising_process
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

# create TaylorSeer text2image sample executable

add_executable(taylorseer_text2image taylorseer_text2image.cpp imwrite.cpp)

target_include_directories(taylorseer_text2image PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(taylorseer_text2image PRIVATE openvino::genai indicators::indicators)

set_target_properties(taylorseer_text2image PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS taylorseer_text2image
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

# create encrypted_stable_diffusion sample executable

add_executable(encrypted_stable_diffusion encrypted_stable_diffusion.cpp imwrite.cpp)

target_include_directories(encrypted_stable_diffusion PRIVATE ${CMAKE_BINARY_DIR} "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(encrypted_stable_diffusion PRIVATE openvino::genai indicators::indicators)

set_target_properties(encrypted_stable_diffusion PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS encrypted_stable_diffusion
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
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.3.1.tar.gz
    URL_HASH SHA256=3bfc70542c521d4b55a46429d808178916a579b28d048bd8c727ee76c39e2072)
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
