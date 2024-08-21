# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED PATHS
    "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
    ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
)

add_executable(whisper_speech_recognition whisper_speech_recognition.cpp)
target_link_libraries(whisper_speech_recognition PRIVATE openvino::genai)
set_target_properties(whisper_speech_recognition PROPERTIES
    COMPILE_PDB_NAME whisper_speech_recognition
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
target_compile_features(whisper_speech_recognition PRIVATE cxx_std_11)

install(TARGETS whisper_speech_recognition
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
