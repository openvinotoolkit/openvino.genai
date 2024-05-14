# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.12.0
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

pybind11_add_module(py_generate_pipeline py_generate_pipeline.cpp)
target_link_libraries(py_generate_pipeline PRIVATE generate_pipeline_lib)

install(TARGETS py_generate_pipeline LIBRARY DESTINATION . COMPONENT genai_python)

# setting RPATH / LC_RPATH depending on platform
if(LINUX)
    # to find libgenerate_pipeline_lib.so in the same folder
    set(rpaths "$ORIGIN")
elseif(APPLE)
    # to find libgenerate_pipeline_lib.dylib in the same folder
    set(rpaths "@loader_path")
    if(DEFINED SKBUILD)
        # in case we build pip package, we need to refer to libopenvino.dylib from 'openvino' package
        list(APPEND rpaths "@loader_path/../openvino/libs")
    endif()
endif()

if(rpaths)
    set_target_properties(py_generate_pipeline PROPERTIES INSTALL_RPATH "${rpaths}")
endif()

# Populate python with the libraries to allow skipping wheel installation
add_custom_command(TARGET py_generate_pipeline POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy
        "$<TARGET_FILE:py_generate_pipeline>"
        "${CMAKE_CURRENT_SOURCE_DIR}/openvino_genai/$<TARGET_FILE_NAME:py_generate_pipeline>"
    COMMENT "Copy py_generate_pipeline to src/python/openvino_genai/")
