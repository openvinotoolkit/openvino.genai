cmake_minimum_required(VERSION 3.14)
project(OVGenAITestUtils LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET_PYWRAPPER_NAME "genai_test_utils")

find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Duplication but works
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

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)
if(NOT DEFINED OpenVINOGenAI_SOURCE_DIR)
    get_target_property(OpenVINOGenAI_LOCATION openvino::genai LOCATION)
    get_filename_component(OpenVINOGenAI_SOURCE_DIR "${OpenVINOGenAI_LOCATION}/../../../" ABSOLUTE)
    message(STATUS "OpenVINOGenAI_SOURCE_DIR was found at: ${OpenVINOGenAI_LOCATION}")
endif()

file(GLOB src_files "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/text_callback_streamer.cpp")

# Add source files
add_library(${TARGET_PYWRAPPER_NAME} MODULE py_genai_test_utils.cpp)
target_link_libraries(${TARGET_PYWRAPPER_NAME} PRIVATE pybind11::module PRIVATE openvino::genai)
target_sources(${TARGET_PYWRAPPER_NAME} PRIVATE ${src_files})
target_include_directories(${TARGET_PYWRAPPER_NAME} PRIVATE "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src")

# Remove 'lib' prefix for Python modules
set_target_properties(${TARGET_PYWRAPPER_NAME} PROPERTIES PREFIX "")

# Set rpath for find GenAI
set_target_properties(${TARGET_PYWRAPPER_NAME} PROPERTIES
    INSTALL_RPATH "\$ORIGIN/../openvino_genai"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Install the Python module
install(TARGETS ${TARGET_PYWRAPPER_NAME} LIBRARY DESTINATION genai_test_utils)
