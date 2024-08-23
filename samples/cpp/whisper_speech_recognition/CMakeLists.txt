# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED PATHS
    "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
    ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

include(FetchContent)

if(NOT TARGET dr_libs)
    FetchContent_Declare(dr_libs
        GIT_REPOSITORY https://github.com/mackron/dr_libs.git
        GIT_TAG da35f9d6c7374a95353fd1df1d394d44ab66cf01)
    FetchContent_MakeAvailable(dr_libs)
endif()

add_executable(whisper_speech_recognition whisper_speech_recognition.cpp audio_utils.cpp)
target_link_libraries(whisper_speech_recognition PRIVATE openvino::genai)
target_include_directories(whisper_speech_recognition PUBLIC "$<BUILD_INTERFACE:${dr_libs_SOURCE_DIR}>")
set_target_properties(whisper_speech_recognition PROPERTIES
    COMPILE_PDB_NAME whisper_speech_recognition
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
target_compile_features(whisper_speech_recognition PRIVATE cxx_std_11)

install(TARGETS whisper_speech_recognition
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
