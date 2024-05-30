# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(FetchContent)
FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.tar.gz
    URL_HASH SHA256=bf8f242abd1abcd375d516a7067490fb71abd79519a282d22b6e4d19282185a7
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

pybind11_add_module(py_generate_pipeline py_generate_pipeline.cpp)
target_link_libraries(py_generate_pipeline PRIVATE openvino::genai nlohmann_json::nlohmann_json)
set_target_properties(py_generate_pipeline PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
)
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/openvino_genai/__init__.py" DESTINATION "${CMAKE_BINARY_DIR}/openvino_genai/")
write_file("${CMAKE_BINARY_DIR}/openvino_genai/__version__.py" "__version__ = \"${CMAKE_PROJECT_VERSION}\"")

# setting RPATH / LC_RPATH depending on platform
if(LINUX)
    # to find libopenvino_genai.so in the same folder
    set(rpaths "$ORIGIN")
elseif(APPLE)
    # to find libopenvino_genai.dylib in the same folder
    set(rpaths "@loader_path")
    if(DEFINED SKBUILD)
        # in case we build pip package, we need to refer to libopenvino.dylib from 'openvino' package
        list(APPEND rpaths "@loader_path/../openvino/libs")
    endif()
endif()

if(rpaths)
    set_target_properties(py_generate_pipeline PROPERTIES INSTALL_RPATH "${rpaths}")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
install(FILES "${CMAKE_BINARY_DIR}/openvino_genai/__init__.py" "${CMAKE_BINARY_DIR}/openvino_genai/__version__.py" DESTINATION python/openvino_genai/ COMPONENT pygenai_${Python_VERSION_MAJOR}_${Python_VERSION_MINOR})
install(TARGETS genai py_generate_pipeline LIBRARY DESTINATION python/openvino_genai/ COMPONENT pygenai_${Python_VERSION_MAJOR}_${Python_VERSION_MINOR})

# wheel_genai component is used for wheel generation in pyproject.toml.
# Exclude wheel_genai from normal packaging because there's pygenai_X_Y component for that.
install(TARGETS genai py_generate_pipeline
    LIBRARY DESTINATION . COMPONENT wheel_genai
    RUNTIME DESTINATION . COMPONENT wheel_genai
    EXCLUDE_FROM_ALL)
