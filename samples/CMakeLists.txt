# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
if(ENABLE_SAMPLES)
    add_subdirectory(cpp/text_generation)
    add_subdirectory(cpp/image_generation)
    add_subdirectory(cpp/speech_generation)
    add_subdirectory(cpp/visual_language_chat)
    add_subdirectory(cpp/whisper_speech_recognition)
    add_subdirectory(cpp/rag)
    add_subdirectory(c/text_generation)
    add_subdirectory(c/whisper_speech_recognition)
    add_subdirectory(c/visual_language_chat)
endif()

install(FILES
        deployment-requirements.txt
        export-requirements.txt
        requirements.txt
    DESTINATION samples
    COMPONENT cpp_samples_genai)

install(DIRECTORY
            cpp/text_generation
            cpp/image_generation
            cpp/speech_generation
            cpp/visual_language_chat
            cpp/whisper_speech_recognition
            cpp/rag
        DESTINATION samples/cpp COMPONENT cpp_samples_genai)

install(DIRECTORY
            python/text_generation
            python/image_generation
            python/speech_generation
            python/visual_language_chat
            python/whisper_speech_recognition
            python/rag
        DESTINATION samples/python COMPONENT cpp_samples_genai
        USE_SOURCE_PERMISSIONS)

install(DIRECTORY
        c/text_generation
        c/whisper_speech_recognition
        c/visual_language_chat
        DESTINATION samples/c COMPONENT cpp_samples_genai)
