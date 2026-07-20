# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

include(FetchContent)

if(NOT TARGET dr_libs)
    FetchContent_Declare(dr_libs
        URL https://github.com/mackron/dr_libs/archive/da35f9d6c7374a95353fd1df1d394d44ab66cf01.tar.gz
        URL_HASH SHA256=2704d347f480ca1bc92233fb01747e4550cc8031735b6ea62ca9990ebb8851ae)
    FetchContent_MakeAvailable(dr_libs)
endif()

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# create main sample executable

add_executable(text2speech text2speech.cpp audio_utils.cpp)
add_executable(kokoro_phonemize_fallback kokoro_phonemize_fallback.cpp audio_utils.cpp)

# Qwen3-TTS samples (one per model variant)
add_executable(qwen3_customvoice qwen3_customvoice.cpp audio_utils.cpp)
add_executable(qwen3_voice_design qwen3_voice_design.cpp audio_utils.cpp)
add_executable(qwen3_base qwen3_base.cpp audio_utils.cpp)

set(speech_generation_samples
    text2speech
    kokoro_phonemize_fallback
    qwen3_customvoice
    qwen3_voice_design
    qwen3_base)

foreach(sample_target IN LISTS speech_generation_samples)
    target_include_directories(${sample_target} PRIVATE "$<BUILD_INTERFACE:${dr_libs_SOURCE_DIR}>")
    target_link_libraries(${sample_target} PRIVATE openvino::genai)
    set_target_properties(${sample_target} PROPERTIES
        # Ensure out of box LC_RPATH on macOS with SIP
        INSTALL_RPATH_USE_LINK_PATH ON)
    target_compile_features(${sample_target} PRIVATE cxx_std_17)
endforeach()

install(TARGETS ${speech_generation_samples}
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
