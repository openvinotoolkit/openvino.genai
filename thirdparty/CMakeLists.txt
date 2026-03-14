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

option(BUILD_LLAMA_CPP "Build llama.cpp with OpenVINO backend for dynamic GGUF loading" OFF)

if(BUILD_LLAMA_CPP)
    set(GGML_OPENVINO ON CACHE BOOL "Enable OpenVINO backend in llama.cpp" FORCE)

    # Save and restore to avoid polluting global scope
    set(_orig_build_shared_libs ${BUILD_SHARED_LIBS})
    
    # --- GSOC INTEGRATION: Force static build ---
    # We build llama.cpp statically so it gets absorbed into the openvino_genai 
    # binary, hiding it from downstream users.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build llama.cpp as static library" FORCE)

    # EXCLUDE_FROM_ALL prevents building llama.cpp's standalone executables 
    # (like the server or main binary) which saves massive compile time.
    add_subdirectory(llama.cpp "${CMAKE_BINARY_DIR}/llama_cpp/" EXCLUDE_FROM_ALL)

    # Restore global scope
    set(BUILD_SHARED_LIBS ${_orig_build_shared_libs} CACHE BOOL "" FORCE)
endif()
