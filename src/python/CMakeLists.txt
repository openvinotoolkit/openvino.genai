# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(FetchContent)

FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.5.tar.gz
    URL_HASH SHA256=b1e209c42b3a9ed74da3e0b25a4f4cd478d89d5efbb48f04b277df427faf6252
)
FetchContent_GetProperties(pybind11)
# search for FindPython3.cmake instead of legacy modules
set(PYBIND11_FINDPYTHON ON)

if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

file(GLOB python_sources "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

pybind11_add_module(py_openvino_genai ${python_sources})
target_link_libraries(py_openvino_genai PRIVATE openvino::genai)
target_include_directories(py_openvino_genai PRIVATE "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src") # for tokenizers_path.hpp
set_target_properties(py_openvino_genai PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
    LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
)
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/openvino_genai/__init__.py" DESTINATION "${CMAKE_BINARY_DIR}/openvino_genai/")

configure_file("${OpenVINOGenAI_SOURCE_DIR}/cmake/templates/__version__.py.in"
               "${CMAKE_BINARY_DIR}/openvino_genai/__version__.py" @ONLY)

if(DEFINED PY_BUILD_CMAKE_PACKAGE_NAME)
    # RPATH for wheel is mandatory to find openvino_genai library. It
    # must be forced because GenAI may be built with OpenVINO targeting
    # archive. Such OpenVINO configurations sets
    # CMAKE_SKIP_INSTALL_RPATH to ON because it relies on setupvars.sh.
    set(CMAKE_SKIP_INSTALL_RPATH OFF)
endif()
# setting RPATH / LC_RPATH depending on platform
if(LINUX)
    # to find libopenvino_genai.so in the same folder
    set(rpaths "$ORIGIN")
elseif(APPLE)
    # to find libopenvino_genai.dylib in the same folder
    set(rpaths "@loader_path")
    if(DEFINED PY_BUILD_CMAKE_PACKAGE_NAME)
        # in case we build pip package, we need to refer to libopenvino.dylib from 'openvino' package
        list(APPEND rpaths "@loader_path/../openvino/libs")
    endif()
endif()

if(rpaths)
    set_target_properties(py_openvino_genai PROPERTIES INSTALL_RPATH "${rpaths}")
endif()

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/openvino_genai/__init__.py"
              "${CMAKE_BINARY_DIR}/openvino_genai/__version__.py"
        DESTINATION python/openvino_genai
        COMPONENT pygenai_${Python3_VERSION_MAJOR}_${Python3_VERSION_MINOR})
install(TARGETS py_openvino_genai
        LIBRARY DESTINATION python/openvino_genai
        COMPONENT pygenai_${Python3_VERSION_MAJOR}_${Python3_VERSION_MINOR})

install(FILES "${CMAKE_BINARY_DIR}/openvino_genai/__version__.py"
        DESTINATION openvino_genai
        COMPONENT wheel_genai
        EXCLUDE_FROM_ALL)

install(FILES "${OpenVINOGenAI_SOURCE_DIR}/LICENSE"
              "${OpenVINOGenAI_SOURCE_DIR}/third-party-programs.txt"
              "${OpenVINOGenAI_SOURCE_DIR}/SECURITY.md"
        DESTINATION "${PY_BUILD_CMAKE_PACKAGE_NAME}-${PY_BUILD_CMAKE_PACKAGE_VERSION}.dist-info"
        COMPONENT wheel_genai
        EXCLUDE_FROM_ALL)

# wheel_genai component is used for wheel generation in pyproject.toml.
# Exclude wheel_genai from normal packaging because there's pygenai_X_Y component for that.
install(TARGETS openvino_genai py_openvino_genai
        LIBRARY DESTINATION openvino_genai COMPONENT wheel_genai EXCLUDE_FROM_ALL
        RUNTIME DESTINATION openvino_genai COMPONENT wheel_genai EXCLUDE_FROM_ALL)
