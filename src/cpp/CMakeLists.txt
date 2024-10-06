# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Dependencies

include(FetchContent)

if(NOT TARGET nlohmann_json)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
        URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

function(ov_genai_build_jinja2cpp)
    FetchContent_Declare(jinja2cpp
        URL https://github.com/jinja2cpp/Jinja2Cpp/archive/b32fbde7d98d13c34784c332c4a24a6f92c76e38.tar.gz
        URL_HASH SHA256=7cc25ddbc438a5c874d404e100b4eccd8a331c195417f5487c48aebcf4b9e7fb)

    FetchContent_GetProperties(jinja2cpp)
    if(NOT jinja2cpp_POPULATED)
        FetchContent_Populate(jinja2cpp)

        set(BUILD_SHARED_LIBS OFF)
        set(JINJA2CPP_INSTALL OFF CACHE BOOL "")
        set(JINJA2CPP_CXX_STANDARD 17 CACHE STRING "")
        set(JINJA2CPP_BUILD_SHARED OFF CACHE BOOL "")
        set(JINJA2CPP_USE_REGEX "std" CACHE STRING "")
        set(JINJA2CPP_WITH_JSON_BINDINGS "none" CACHE STRING "")
        set(JINJA2CPP_STRICT_WARNINGS OFF CACHE BOOL "")
        set(JINJA2CPP_PIC ON CACHE BOOL "")

        # options for Jinja2Cpp dependencies
        option(RAPIDJSON_BUILD_DOC "Build rapidjson documentation." OFF)

        add_subdirectory("${jinja2cpp_SOURCE_DIR}" "${jinja2cpp_BINARY_DIR}" EXCLUDE_FROM_ALL)
    endif()
endfunction()

FetchContent_Declare(safetensors.h
    URL https://github.com/hsnyder/safetensors.h/archive/974a85d7dfd6e010558353226638bb26d6b9d756.tar.gz
    URL_HASH SHA256=9aaf5961609601cf9aaa96582a207bce7c6e5fbf57ed2cc669bb7bde6a937d4b)
FetchContent_MakeAvailable(safetensors.h)

ov_genai_build_jinja2cpp()

# Library

file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")

set(TARGET_NAME openvino_genai)
add_library(${TARGET_NAME} SHARED ${SOURCE_FILES})
add_dependencies(${TARGET_NAME} openvino_tokenizers)
add_library(openvino::genai ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>" "$<INSTALL_INTERFACE:runtime/include>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE "${safetensors.h_SOURCE_DIR}")

target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime PRIVATE nlohmann_json::nlohmann_json jinja2cpp)

target_compile_features(${TARGET_NAME} PUBLIC cxx_std_17)

set_target_properties(${TARGET_NAME} PROPERTIES
    EXPORT_NAME genai
    ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
    LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
    RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
)
# Extract two last digits from OpenVINOGenAI_VERSION_MAJOR because SOVERSION can only contain up to 4 symbols.
string(REGEX MATCH [=[[0-9][0-9]$]=] MAJOR_SUFFIX ${OpenVINOGenAI_VERSION_MAJOR})
if(DEFINED PY_BUILD_CMAKE_PACKAGE_NAME AND LINUX)
    # Don't pack symlinks but append version hash to the name for wheel
    set_target_properties(${TARGET_NAME} PROPERTIES
        SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX}.${MAJOR_SUFFIX}${OpenVINOGenAI_VERSION_MINOR}${OpenVINOGenAI_VERSION_PATCH})
elseif(DEFINED PY_BUILD_CMAKE_PACKAGE_NAME AND APPLE)
    set_target_properties(${TARGET_NAME} PROPERTIES
        SUFFIX .${MAJOR_SUFFIX}${OpenVINOGenAI_VERSION_MINOR}${OpenVINOGenAI_VERSION_PATCH}${CMAKE_SHARED_LIBRARY_SUFFIX})
else()
    set_target_properties(${TARGET_NAME} PROPERTIES
        VERSION ${OpenVINOGenAI_VERSION}
        SOVERSION ${MAJOR_SUFFIX}${OpenVINOGenAI_VERSION_MINOR}${OpenVINOGenAI_VERSION_PATCH})
endif()

# - Windows: `<openvino_dir>\runtime\bin\intel64\Release\`
# - MacOS_x86: `<openvino_dir>/runtime/lib/intel64/Release`
# - MacOS_arm64: `<openvino_dir>/runtime/lib/arm64/Release/`
# - Linux_x86: `<openvino_dir>/runtime/lib/intel64/`
# - Linux_arm64: `<openvino_dir>/runtime/lib/aarch64/`
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARCH_DIR)
if(ARCH_DIR MATCHES "amd64.*|x86_64.*|AMD64.*")
    set(ARCH_DIR intel64)
elseif(ARCH_DIR MATCHES "^(arm64.*|aarch64.*|AARCH64.*|ARM64.*)")
    if(APPLE)
        set(ARCH_DIR "arm64")
    else()
        set(ARCH_DIR "aarch64")
    endif()
elseif(ARCH_DIR STREQUAL "x86_64" OR ARCH_DIR STREQUAL "amd64"  # Windows detects Intel's 64-bit CPU as AMD64
        OR CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
    set(ARCH_DIR intel64)
endif()
if(MSVC OR APPLE)
    set(ARCH_DIR ${ARCH_DIR}/${CMAKE_BUILD_TYPE})
endif()

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOGenAITargets
        LIBRARY DESTINATION runtime/lib/${ARCH_DIR} COMPONENT core_genai
            NAMELINK_COMPONENT core_genai_dev
        ARCHIVE DESTINATION runtime/lib/${ARCH_DIR} COMPONENT core_genai_dev
        RUNTIME DESTINATION runtime/bin/${ARCH_DIR} COMPONENT core_genai
        INCLUDES DESTINATION runtime/include)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION runtime/include COMPONENT core_genai_dev)
install(EXPORT OpenVINOGenAITargets FILE OpenVINOGenAITargets.cmake
        NAMESPACE openvino:: DESTINATION runtime/cmake
        COMPONENT core_genai_dev)

include(CMakePackageConfigHelpers)
configure_package_config_file("${OpenVINOGenAI_SOURCE_DIR}/cmake/templates/OpenVINOGenAIConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/OpenVINOGenAIConfig.cmake" INSTALL_DESTINATION runtime/cmake)
write_basic_package_version_file("${CMAKE_BINARY_DIR}/OpenVINOGenAIConfigVersion.cmake"
                                 VERSION ${OpenVINOGenAI_VERSION} COMPATIBILITY AnyNewerVersion)
install(FILES "${CMAKE_BINARY_DIR}/OpenVINOGenAIConfig.cmake" "${CMAKE_BINARY_DIR}/OpenVINOGenAIConfigVersion.cmake"
        DESTINATION runtime/cmake COMPONENT core_genai_dev)
export(EXPORT OpenVINOGenAITargets FILE "${CMAKE_BINARY_DIR}/OpenVINOGenAITargets.cmake" NAMESPACE openvino::)
