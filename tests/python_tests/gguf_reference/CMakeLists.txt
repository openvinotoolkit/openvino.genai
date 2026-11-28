# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(genai_gguf_reference LANGUAGES C CXX)

include(FetchContent)

# Keep generator and oracle on the same reviewed revision, independent of installed llama.cpp wheels.
FetchContent_Declare(llama_cpp
    URL https://codeload.github.com/ggml-org/llama.cpp/tar.gz/19cce6919992fdffd92e0ac152caaf1e2b8f254e
    URL_HASH SHA256=e16c6d1076af7a2bdb351e22176dd4b302f244ac4c3822604e2ec64a8a783d44
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
