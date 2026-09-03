# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)

project(OpenVINOGenAICppSamples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(text_generation)
add_subdirectory(image_generation)
add_subdirectory(video_generation)
add_subdirectory(speech_generation)
add_subdirectory(visual_language_chat)
add_subdirectory(automatic_speech_recognition)
add_subdirectory(omni)
add_subdirectory(rag)
