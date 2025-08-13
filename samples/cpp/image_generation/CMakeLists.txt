# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

file(DOWNLOAD https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image.h ${CMAKE_BINARY_DIR}/stb_image.h
     EXPECTED_HASH MD5=27932e6fb3a2f26aee2fc33f2cb4e696)

include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(indicators
    URL https://github.com/p-ranav/indicators/archive/refs/tags/v2.3.tar.gz
    URL_HASH SHA256=70da7a693ff7a6a283850ab6d62acf628eea17d386488af8918576d0760aef7b)
FetchContent_MakeAvailable(indicators)

# create main sample executable

ov_add_sample(
        NAME text2image
        SOURCES text2image.cpp imwrite.cpp
        DEPENDENCIES indicators::indicators openvino::genai
        INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR})

# create text2image concurrent sample executable

ov_add_sample(
        NAME text2image_concurrency
        SOURCES text2image_concurrency.cpp imwrite.cpp
        DEPENDENCIES indicators::indicators openvino::genai
        INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR})

# create LoRA sample executable

ov_add_sample(
        NAME lora_text2image
        SOURCES lora_text2image.cpp imwrite.cpp
        DEPENDENCIES indicators::indicators openvino::genai)

# create heterogeneous_stable_diffusion sample executable

ov_add_sample(
        NAME heterogeneous_stable_diffusion
        SOURCES heterogeneous_stable_diffusion.cpp imwrite.cpp
        DEPENDENCIES indicators::indicators openvino::genai)

# create image2image sample executable

ov_add_sample(
        NAME image2image
        SOURCES image2image.cpp load_image.cpp imwrite.cpp
        DEPENDENCIES indicators::indicators openvino::genai
        INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR})

# create image2image concurrent sample executable

ov_add_sample(
        NAME image2image_concurrency
        SOURCES image2image_concurrency.cpp load_image.cpp imwrite.cpp
        DEPENDENCIES indicators::indicators openvino::genai
        INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR})

# create inpainting executable

ov_add_sample(
        NAME inpainting
        SOURCES inpainting.cpp load_image.cpp imwrite.cpp
        DEPENDENCIES indicators::indicators openvino::genai
        INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR})

# benchmark_image_gen

include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(cxxopts
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.1.1.tar.gz
    URL_HASH SHA256=523175f792eb0ff04f9e653c90746c12655f10cb70f1d5e6d6d9491420298a08)
FetchContent_MakeAvailable(cxxopts)

ov_add_sample(
        NAME benchmark_image_gen
        SOURCES benchmark_image_gen.cpp load_image.cpp imwrite.cpp
        DEPENDENCIES cxxopts::cxxopts indicators::indicators openvino::genai
        INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR})
