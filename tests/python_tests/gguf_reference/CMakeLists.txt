# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(genai_gguf_reference LANGUAGES C CXX)

include(FetchContent)

# Keep generator and oracle on the same reviewed revision, independent of installed llama.cpp wheels.
FetchContent_Declare(llama_cpp
    URL https://codeload.github.com/ggml-org/llama.cpp/tar.gz/03fa73cb27f5c251b9528489b18d303b1366aca4
    URL_HASH SHA256=a1ea18cad07f6051b771b80f4a68a34ab11905187e642124ca93d4f9bd7d7f4c
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
foreach(backend BLAS CUDA HIP METAL MUSA OPENCL OPENVINO RPC SYCL VULKAN WEBGPU)
    set(GGML_${backend} OFF CACHE BOOL "" FORCE)
endforeach()
FetchContent_MakeAvailable(llama_cpp)

add_executable(genai-gguf-generate generate.cpp)
target_compile_features(genai-gguf-generate PRIVATE cxx_std_17)
target_include_directories(genai-gguf-generate PRIVATE ${llama_cpp_SOURCE_DIR})
target_link_libraries(genai-gguf-generate PRIVATE llama-common)
set_target_properties(genai-gguf-generate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(genai-gguf-reference reference.cpp)
target_compile_features(genai-gguf-reference PRIVATE cxx_std_17)
target_include_directories(genai-gguf-reference PRIVATE ${llama_cpp_SOURCE_DIR}/vendor)
target_link_libraries(genai-gguf-reference PRIVATE llama)
set_target_properties(genai-gguf-reference PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
