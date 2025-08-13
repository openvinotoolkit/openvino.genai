# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

# Whisper Speech Recognition Sample
genai_add_c_sample_target(
        NAME whisper_speech_recognition_c
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        SOURCES whisper_speech_recognition.c whisper_utils.c
)
