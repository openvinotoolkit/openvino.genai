# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Reference, standalone implementation of the KV cache store plugin ABI declared in
# openvino/genai/c/kv_cache_store_provider_abi.h - for third-party vendors to use as a starting
# point. Deliberately NOT linked against openvino::genai (only its include path, to find the ABI
# header), to demonstrate that a real plugin needs nothing else.

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alongside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

add_library(sample_kv_cache_store_plugin SHARED sample_kv_cache_store_plugin.c)
set_source_files_properties(sample_kv_cache_store_plugin.c PROPERTIES LANGUAGE C)
target_include_directories(sample_kv_cache_store_plugin PRIVATE
    $<TARGET_PROPERTY:openvino::genai::c,INTERFACE_INCLUDE_DIRECTORIES>)

if(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(sample_kv_cache_store_plugin PRIVATE Threads::Threads)
endif()

install(TARGETS sample_kv_cache_store_plugin
        LIBRARY DESTINATION samples_bin/
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
