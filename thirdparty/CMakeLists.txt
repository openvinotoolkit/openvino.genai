# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

option(BUILD_TOKENIZERS "Build OpenVINO Tokenizers together with OpenVINO GenAI" ON)

option(BUILD_LLAMA_CPP "Build llama.cpp with OpenVINO backend for dynamic GGUF loading" ON)


if(BUILD_TOKENIZERS)
    add_subdirectory(./openvino_tokenizers/ "${CMAKE_BINARY_DIR}/openvino_tokenizers/")
    # Put binaries to a single dir to mimic package structure.
    set_target_properties(openvino_tokenizers PROPERTIES
        # Generator expressions to disable appending a per-configuration subdirectory (Release, Debug).
        # ARCHIVE_OUTPUT is irrelevant. It's here just to keep all the artifacts in one place.
        ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
        LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
        RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
    )
endif()

if(BUILD_LLAMA_CPP)
    set(GGML_OPENVINO ON CACHE BOOL "Enable OpenVINO backend in llama.cpp" FORCE)

    include(FetchContent)
    
    FetchContent_Declare(
        opencl_headers
        GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
        GIT_TAG        v2023.12.14
    )
    FetchContent_Declare(
        opencl_hpp
        GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git
        GIT_TAG        v2023.12.14
    )

    FetchContent_GetProperties(opencl_headers)
    if(NOT opencl_headers_POPULATED)
        FetchContent_Populate(opencl_headers)
    endif()

    FetchContent_GetProperties(opencl_hpp)
    if(NOT opencl_hpp_POPULATED)
        FetchContent_Populate(opencl_hpp)
    endif()

    # Inject these paths so ggml-openvino.cpp can find <CL/cl2.hpp>
    include_directories(
        ${opencl_headers_SOURCE_DIR}
        ${opencl_hpp_SOURCE_DIR}/include
    )

    # Save and restore to avoid polluting global scope
    set(_orig_build_shared_libs ${BUILD_SHARED_LIBS})
    
    # Force static build ---
    # We build llama.cpp statically so it gets absorbed into the openvino_genai 
    # binary, hiding it from downstream users.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build llama.cpp as static library" FORCE)

    # EXCLUDE_FROM_ALL prevents building llama.cpp's standalone executables 
    # (like the server or main binary) which saves massive compile time.
    add_subdirectory(llama.cpp "${CMAKE_BINARY_DIR}/llama_cpp/" EXCLUDE_FROM_ALL)

    # Restore global scope
    set(BUILD_SHARED_LIBS ${_orig_build_shared_libs} CACHE BOOL "" FORCE)
endif()
