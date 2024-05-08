# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.12.0
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

pybind11_add_module(openvino_genai_bindings openvino_genai_bindings.cpp)
target_link_libraries(openvino_genai_bindings PRIVATE generate_pipeline_lib)
# TODO: how to link with tokenizers and openvino
install(TARGETS openvino_genai_bindings LIBRARY DESTINATION . COMPONENT openvino_genai_bindings_install_target)
