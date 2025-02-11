cmake_minimum_required(VERSION 3.14)
project(TextCallbackStreamerPython LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET_PYWRAPPER_NAME "ov_genai_test_utils")

file(GLOB src_files "${OpenVINOGenAI_SOURCE_DIR}src/cpp/src/text_callback_streamer.cpp")

add_library(${TARGET_PYWRAPPER_NAME} MODULE py_ov_genai_test_utils.cpp)
target_link_libraries(${TARGET_PYWRAPPER_NAME} PRIVATE pybind11::module PRIVATE openvino::genai)
target_sources(${TARGET_PYWRAPPER_NAME} PRIVATE ${src_files})

target_include_directories(${TARGET_PYWRAPPER_NAME} PRIVATE "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src")

# Remove 'lib' prefix for Python modules
set_target_properties(${TARGET_PYWRAPPER_NAME} PROPERTIES PREFIX "")
