# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

file(DOWNLOAD
    https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image.h
    ${CMAKE_BINARY_DIR}/stb_image.h
    EXPECTED_HASH MD5=27932e6fb3a2f26aee2fc33f2cb4e696)

file(DOWNLOAD
    https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image_resize2.h
    ${CMAKE_BINARY_DIR}/stb_image_resize2.h
    EXPECTED_HASH MD5=9293a158ddf9e44dd186c446372784d8)

include(${CMAKE_CURRENT_SOURCE_DIR}/../fetch_opencv.cmake)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

include(FetchContent)

if(NOT TARGET dr_libs)
    FetchContent_Declare(dr_libs
        URL https://github.com/mackron/dr_libs/archive/da35f9d6c7374a95353fd1df1d394d44ab66cf01.tar.gz
        URL_HASH SHA256=2704d347f480ca1bc92233fb01747e4550cc8031735b6ea62ca9990ebb8851ae)
    FetchContent_MakeAvailable(dr_libs)
endif()

# create main sample executable

add_executable(visual_language_chat visual_language_chat.cpp load_image.cpp)
target_include_directories(visual_language_chat PRIVATE "${CMAKE_BINARY_DIR}")
target_link_libraries(visual_language_chat PRIVATE openvino::genai)

set_target_properties(visual_language_chat PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS visual_language_chat
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

# create audio language chat sample executable
add_executable(audio_to_text_chat audio_to_text_chat.cpp ../speech_generation/audio_utils.cpp)
target_include_directories(audio_to_text_chat PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.." "$<BUILD_INTERFACE:${dr_libs_SOURCE_DIR}>")
target_link_libraries(audio_to_text_chat PRIVATE openvino::genai)

set_target_properties(audio_to_text_chat PROPERTIES

    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS audio_to_text_chat
    RUNTIME DESTINATION samples_bin/
    COMPONENT samples_bin
    EXCLUDE_FROM_ALL)

# create lora sample executable

add_executable(visual_language_lora visual_language_lora.cpp load_image.cpp)
target_include_directories(visual_language_lora PRIVATE "${CMAKE_BINARY_DIR}")
target_link_libraries(visual_language_lora PRIVATE openvino::genai)

set_target_properties(visual_language_lora PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS visual_language_lora
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

# create encrypted model sample executable

add_executable(encrypted_model_vlm encrypted_model_vlm.cpp load_image.cpp)
target_include_directories(encrypted_model_vlm PRIVATE "${CMAKE_BINARY_DIR}")
target_link_libraries(encrypted_model_vlm PRIVATE openvino::genai)

set_target_properties(encrypted_model_vlm PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS encrypted_model_vlm
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

# create benchmark executable
add_executable(benchmark_vlm benchmark_vlm.cpp load_image.cpp ../text_generation/read_prompt_from_file.cpp)
target_include_directories(benchmark_vlm PRIVATE "${CMAKE_BINARY_DIR}")
target_link_libraries(benchmark_vlm PRIVATE openvino::genai cxxopts::cxxopts)
set_target_properties(benchmark_vlm PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS benchmark_vlm
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

add_executable(video_to_text_chat video_to_text_chat.cpp load_video.cpp)
target_include_directories(video_to_text_chat PRIVATE "${CMAKE_BINARY_DIR}")
ov_genai_link_opencv(video_to_text_chat core imgproc videoio imgcodecs)
target_link_libraries(video_to_text_chat PRIVATE openvino::genai cxxopts::cxxopts)

install(TARGETS video_to_text_chat
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
