# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

option(BUILD_TOKENIZERS "Build OpenVINO Tokenizers together with OpenVINO GenAI" ON)

if(ENABLE_PROFILING_ITT)
    find_package(ittapi QUIET)

    if(ittapi_FOUND)
        if(TARGET ittapi::ittapi)
            set_target_properties(ittapi::ittapi PROPERTIES
                INTERFACE_COMPILE_DEFINITIONS ENABLE_PROFILING_ITT)
        elseif(TARGET ittapi::ittnotify)
            set_target_properties(ittapi::ittnotify PROPERTIES
                INTERFACE_COMPILE_DEFINITIONS ENABLE_PROFILING_ITT)
        endif()
    else()
        if(DEFINED INTEL_VTUNE_DIR OR DEFINED ENV{INTEL_VTUNE_DIR})
            find_package(ITT QUIET
                         PATHS "${OpenVINOGenAI_SOURCE_DIR}/cmake"
                         NO_DEFAULT_PATH)
            if(NOT ITT_FOUND)
                message(WARNING "ENABLE_PROFILING_ITT=ON, but no ITT library was found under INTEL_VTUNE_DIR")
            endif()
        else()
            set(_genai_ittapi_local_dir "${CMAKE_CURRENT_SOURCE_DIR}/ittapi")
            set(_genai_ittapi_openvino_dir "${OpenVINOGenAI_SOURCE_DIR}/../openvino/thirdparty/ittapi")
            set(_genai_ittapi_local_repo_dir "${_genai_ittapi_local_dir}/ittapi")
            set(_genai_ittapi_openvino_repo_dir "${_genai_ittapi_openvino_dir}/ittapi")

            if(EXISTS "${_genai_ittapi_local_repo_dir}/CMakeLists.txt")
                add_subdirectory("${_genai_ittapi_local_repo_dir}" "${CMAKE_BINARY_DIR}/ittapi" EXCLUDE_FROM_ALL)
            elseif(EXISTS "${_genai_ittapi_local_dir}/CMakeLists.txt")
                add_subdirectory("${_genai_ittapi_local_dir}" "${CMAKE_BINARY_DIR}/ittapi" EXCLUDE_FROM_ALL)
            elseif(EXISTS "${_genai_ittapi_openvino_repo_dir}/CMakeLists.txt")
                message(STATUS "ENABLE_PROFILING_ITT=ON and system ittapi was not found, using ITT from ${_genai_ittapi_openvino_dir}")
                # Use raw ittapi sources to avoid OpenVINO-specific wrapper macros.
                add_subdirectory("${_genai_ittapi_openvino_repo_dir}" "${CMAKE_BINARY_DIR}/ittapi" EXCLUDE_FROM_ALL)
            elseif(EXISTS "${_genai_ittapi_openvino_dir}/CMakeLists.txt")
                message(STATUS "ENABLE_PROFILING_ITT=ON and system ittapi was not found, using ITT from ${_genai_ittapi_openvino_dir}")
                add_subdirectory("${_genai_ittapi_openvino_dir}" "${CMAKE_BINARY_DIR}/ittapi" EXCLUDE_FROM_ALL)
            else()
                include(FetchContent)
                message(STATUS "ENABLE_PROFILING_ITT=ON and no local ITT sources were found, fetching ittapi from https://github.com/intel/ittapi.git")
                FetchContent_Declare(genai_ittapi
                    GIT_REPOSITORY https://github.com/intel/ittapi.git
                    GIT_TAG ca45fef1a12cef3316e6ff362a4d36571270e392
                    GIT_SHALLOW FALSE)
                FetchContent_GetProperties(genai_ittapi)
                if(NOT genai_ittapi_POPULATED)
                    FetchContent_Populate(genai_ittapi)
                endif()

                if(EXISTS "${genai_ittapi_SOURCE_DIR}/CMakeLists.txt")
                    add_subdirectory("${genai_ittapi_SOURCE_DIR}" "${genai_ittapi_BINARY_DIR}" EXCLUDE_FROM_ALL)
                else()
                    message(WARNING "ENABLE_PROFILING_ITT=ON, but fetched ittapi source directory is invalid")
                endif()
            endif()

            if(TARGET ittnotify)
                target_compile_definitions(ittnotify INTERFACE ENABLE_PROFILING_ITT)
                if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                    target_compile_options(ittnotify PRIVATE -Wno-undef)
                endif()
            endif()

            if(TARGET ittnotify AND NOT TARGET ittapi::ittnotify)
                add_library(ittapi::ittnotify ALIAS ittnotify)
            endif()

            unset(_genai_ittapi_local_dir)
            unset(_genai_ittapi_openvino_dir)
            unset(_genai_ittapi_local_repo_dir)
            unset(_genai_ittapi_openvino_repo_dir)
        endif()
    endif()
endif()

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
