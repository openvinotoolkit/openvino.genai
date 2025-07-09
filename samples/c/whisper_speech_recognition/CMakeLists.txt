# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINO REQUIRED)

# Whisper Speech Recognition Sample
add_executable(whisper_speech_recognition whisper_speech_recognition.c)
target_link_libraries(whisper_speech_recognition openvino::genai::c)

# Install
install(TARGETS whisper_speech_recognition
        RUNTIME DESTINATION samples_bin/c/whisper_speech_recognition
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)