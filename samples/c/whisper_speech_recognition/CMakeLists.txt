# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

# Whisper Speech Recognition Sample
ov_add_sample(
        NAME whisper_speech_recognition_c
        SOURCES whisper_speech_recognition.c whisper_utils.c
        DEPENDENCIES openvino::genai::c
)

set_source_files_properties(whisper_speech_recognition.c whisper_utils.c PROPERTIES LANGUAGE C)
