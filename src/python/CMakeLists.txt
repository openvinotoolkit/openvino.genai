# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(FetchContent)
FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/3e9dfa2866941655c56877882565e7577de6fc7b.tar.gz
    URL_HASH SHA256=9a7d245f405f470798b9d2a48912cc97230658024775299eac203f7c9c9ae37c
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

pybind11_add_module(py_generate_pipeline py_generate_pipeline.cpp)
target_link_libraries(py_generate_pipeline PRIVATE openvino::genai)

# setting RPATH / LC_RPATH depending on platform
if(LINUX)
    # to find libgenai.so in the same folder
    set(rpaths "$ORIGIN")
elseif(APPLE)
    # to find libgenai.dylib in the same folder
    set(rpaths "@loader_path")
    if(DEFINED SKBUILD)
        # in case we build pip package, we need to refer to libopenvino.dylib from 'openvino' package
        list(APPEND rpaths "@loader_path/../openvino/libs")
    endif()
endif()

if(rpaths)
    set_target_properties(py_generate_pipeline PROPERTIES INSTALL_RPATH "${rpaths}")
endif()

# Copy the library to python to allow skipping wheel installation
add_custom_command(TARGET py_generate_pipeline POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy
        "$<TARGET_FILE:py_generate_pipeline>"
        "${CMAKE_CURRENT_SOURCE_DIR}/openvino_genai/$<TARGET_FILE_NAME:py_generate_pipeline>"
    COMMENT "Copy py_generate_pipeline to src/python/openvino_genai/")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/openvino_genai/ DESTINATION python/openvino_genai/ COMPONENT pygenai_${Python_VERSION_MAJOR}_${Python_VERSION_MINOR})
install(TARGETS py_generate_pipeline LIBRARY DESTINATION python/openvino_genai/ COMPONENT pygenai_${Python_VERSION_MAJOR}_${Python_VERSION_MINOR})

# wheel_genai component is used for wheel generation in pyproject.toml.
# Exclude wheel_genai from normal packaging process.
install(TARGETS genai py_generate_pipeline LIBRARY DESTINATION . COMPONENT wheel_genai RUNTIME DESTINATION . COMPONENT wheel_genai EXCLUDE_FROM_ALL)
