# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINO REQUIRED)

# Whisper Speech Recognition Sample
add_executable(whisper_speech_recognition_c whisper_speech_recognition.c)
target_link_libraries(whisper_speech_recognition_c openvino::genai::c m)

# Install
install(TARGETS whisper_speech_recognition_c
        RUNTIME DESTINATION samples_bin/c/whisper_speech_recognition
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)