# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


# - Windows: `<openvino_dir>\runtime\bin\intel64\Release\`
# - MacOS_x86: `<openvino_dir>/runtime/lib/intel64/Release`
# - MacOS_arm64: `<openvino_dir>/runtime/lib/arm64/Release/`
# - Linux_x86: `<openvino_dir>/runtime/lib/intel64/`
# - Linux_arm64: `<openvino_dir>/runtime/lib/aarch64/`
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARCH_DIR)
if(ARCH_DIR MATCHES "amd64.*|x86_64.*|AMD64.*")
    set(ARCH_DIR intel64)
elseif(ARCH_DIR MATCHES "^(arm64.*|aarch64.*|AARCH64.*|ARM64.*)")
    if(APPLE)
        set(ARCH_DIR "arm64")
    else()
        set(ARCH_DIR "aarch64")
    endif()
elseif(ARCH_DIR STREQUAL "x86_64" OR ARCH_DIR STREQUAL "amd64"  # Windows detects Intel's 64-bit CPU as AMD64
        OR CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
    set(ARCH_DIR intel64)
endif()

if(WIN32 OR APPLE)
    if(GENERATOR_IS_MULTI_CONFIG_VAR)
        set(ARCH_DIR ${ARCH_DIR}/$<CONFIG>)
    else()
        set(ARCH_DIR ${ARCH_DIR}/${CMAKE_BUILD_TYPE})
    endif()
endif()

# Extract two last digits from OpenVINOGenAI_VERSION_MAJOR because SOVERSION can only contain up to 4 symbols.
string(REGEX MATCH [=[[0-9][0-9]$]=] MAJOR_SUFFIX ${OpenVINOGenAI_VERSION_MAJOR})

# Put binaries at the top level for NPM package
if(CPACK_GENERATOR STREQUAL "NPM")
    set(LIBRARY_DESTINATION .)
    set(ARCHIVE_DESTINATION .)
    set(RUNTIME_DESTINATION .)

    # setting RPATH / LC_RPATH depending on platform
    if(LINUX)
        # to find libopenvino_genai.so in the same folder
        set(rpaths "$ORIGIN")
        # to find libopenvino.so when installing from npm
        list(APPEND rpaths "$ORIGIN/../../openvino-node/bin")
        # to find libopenvino.so when installing from source
        list(APPEND rpaths "$ORIGIN/../node_modules/openvino-node/bin")
    elseif(APPLE)
        # to find libopenvino_genai.dylib in the same folder
        set(rpaths "@loader_path")
        # to find libopenvino.dylib when installing from npm
        list(APPEND rpaths "@loader_path/../../openvino-node/bin")
        # to find libopenvino.dylib when installing from source
        list(APPEND rpaths "@loader_path/../node_modules/openvino-node/bin")
    endif()
else()
    set(LIBRARY_DESTINATION runtime/lib/${ARCH_DIR})
    set(ARCHIVE_DESTINATION runtime/lib/${ARCH_DIR})
    set(RUNTIME_DESTINATION runtime/bin/${ARCH_DIR})
endif()

if(NOT DEFINED ENABLE_LTO)
    set(ENABLE_LTO OFF)
endif()

#
# Include source dirs
#

add_subdirectory(cpp)

if(NOT DEFINED PY_BUILD_CMAKE_PACKAGE_NAME)
    add_subdirectory(c)
endif()

if(ENABLE_PYTHON)
    add_subdirectory(python)
endif()

if(ENABLE_JS)
    add_subdirectory(js)
endif()
