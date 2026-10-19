# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

include(FetchContent)

if(NOT TARGET dr_libs)
    FetchContent_Declare(dr_libs
        URL https://github.com/mackron/dr_libs/archive/da35f9d6c7374a95353fd1df1d394d44ab66cf01.tar.gz
        URL_HASH SHA256=2704d347f480ca1bc92233fb01747e4550cc8031735b6ea62ca9990ebb8851ae)
    FetchContent_MakeAvailable(dr_libs)
endif()

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

# Whisper Speech Recognition Sample
add_executable(whisper_speech_recognition_c whisper_speech_recognition.c ../common/audio_utils.c)

# Specifies that the source file should be compiled as a C source file
set_source_files_properties(whisper_speech_recognition.c ../common/audio_utils.c PROPERTIES LANGUAGE C)
target_include_directories(whisper_speech_recognition_c PRIVATE ../common "$<BUILD_INTERFACE:${dr_libs_SOURCE_DIR}>")
target_link_libraries(whisper_speech_recognition_c PRIVATE openvino::genai::c)

set_target_properties(whisper_speech_recognition_c PROPERTIES
    # Ensure out-of-box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

# Install
install(TARGETS whisper_speech_recognition_c
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
