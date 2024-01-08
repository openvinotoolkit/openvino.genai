cmake_minimum_required(VERSION 3.12)
project(qwen.cpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

set(CMAKE_CXX_STANDARD 23)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif ()

# third-party libraries
set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(third_party/abseil-cpp)

add_subdirectory(third_party/re2)

file(GLOB CPP_SOURCES
  ${PROJECT_SOURCE_DIR}/*.h
  ${PROJECT_SOURCE_DIR}/*.cpp)

set_source_files_properties(${CPP_SOURCES} PROPERTIES COMPILE_FLAGS "-pedantic-errors")

add_library(qwen STATIC qwen.cpp)
target_link_libraries(qwen PUBLIC re2::re2)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE qwen)
target_link_libraries(main PRIVATE openvino::runtime)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall /wd4710 /wd4711")
    add_definitions("/wd4267 /wd4244 /wd4305 /Zc:strictStrings /utf-8")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")
else()
    target_compile_options(main PRIVATE -Wall)  # Display all warnings
endif ()

