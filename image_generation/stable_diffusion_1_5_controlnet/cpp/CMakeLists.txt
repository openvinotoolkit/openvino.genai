# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)

project(stable_diffusion_controlnet LANGUAGES CXX)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake build type")

# dependencies

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

include(FetchContent)

FetchContent_Declare(cxxopts
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.1.1.tar.gz
    URL_HASH SHA256=523175f792eb0ff04f9e653c90746c12655f10cb70f1d5e6d6d9491420298a08)

FetchContent_MakeAvailable(cxxopts)

add_subdirectory(../../common/detectors _deps/detectors)
add_subdirectory(../../common/diffusers _deps/diffusers)
add_subdirectory(../../../thirdparty/openvino_tokenizers/ _deps/tokenizers)

# create executable
add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/core/core.cpp ${PROJECT_SOURCE_DIR}/src/cli/cli.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(${PROJECT_NAME} PRIVATE
    openvino::runtime
    cxxopts::cxxopts
    diffusers::diffusers
    imwrite::imwrite
    detectors::detectors)

add_dependencies(${PROJECT_NAME} openvino_tokenizers)

target_compile_definitions(${PROJECT_NAME} PRIVATE TOKENIZERS_LIBRARY_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")

# gui
set(GUI_APP_NAME "stable_diffusion_controlnet_gui")

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.0
)
FetchContent_MakeAvailable(imgui)

if(NOT imgui_POPULATED)
    FetchContent_Populate(imgui)
    add_subdirectory(${imgui_SOURCE_DIR} ${imgui_BINARY_DIR})
endif()

message(STATUS "imgui_SOURCE_DIR: ${imgui_SOURCE_DIR}")
message(STATUS "imgui_BINARY_DIR: ${imgui_BINARY_DIR}")

set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

set(APP_DEPS_SOURCE
    ${PROJECT_SOURCE_DIR}/src/app/tinyfiledialogs.c
)

add_executable(${GUI_APP_NAME}
    ${PROJECT_SOURCE_DIR}/src/app/app.cpp
    ${PROJECT_SOURCE_DIR}/src/app/worker.cpp
    ${PROJECT_SOURCE_DIR}/src/core/core.cpp
    ${APP_DEPS_SOURCE}
    ${IMGUI_SOURCES}
)
target_include_directories(${GUI_APP_NAME} PRIVATE
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/include/app
    ${CMAKE_CURRENT_SOURCE_DIR}/include/core
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(${GUI_APP_NAME} PRIVATE
    openvino::runtime
    cxxopts::cxxopts
    diffusers::diffusers
    imwrite::imwrite
    detectors::detectors
    glfw)
add_dependencies(${GUI_APP_NAME} openvino_tokenizers)
target_compile_definitions(${GUI_APP_NAME} PRIVATE TOKENIZERS_LIBRARY_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
find_package(OpenGL REQUIRED)
target_link_libraries(${GUI_APP_NAME} PRIVATE OpenGL::GL)
