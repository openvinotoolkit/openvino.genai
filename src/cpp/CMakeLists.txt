# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Dependencies

include(FetchContent)

FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
    URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406)
FetchContent_MakeAvailable(nlohmann_json)

function(ov_genai_build_jinja2cpp)
    FetchContent_Declare(jinja2cpp
        URL https://github.com/ilya-lavrenov/Jinja2Cpp/archive/5433af6b225cd35df700023cf60df4acdd6cbcf3.tar.gz
        URL_HASH SHA256=b90f6c44908beaacae8eeb2690d11a6ebb183b4560434698ac00017e7bc07d11)

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

        add_subdirectory("${jinja2cpp_SOURCE_DIR}" "${jinja2cpp_BINARY_DIR}" EXCLUDE_FROM_ALL)
        # openvino::runtime exports _GLIBCXX_USE_CXX11_ABI=0 on CentOS7.
        # It needs to be propagated to every library GenAI links with.
        # It's enough to propagate to fmt, because fmt propagates to
        # jinja2cpp.
        target_compile_definitions(fmt PUBLIC $<TARGET_PROPERTY:openvino::runtime,INTERFACE_COMPILE_DEFINITIONS>)
    endif()
endfunction()

ov_genai_build_jinja2cpp()

# Library

file(GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

set(TARGET_NAME genai)
add_library(${TARGET_NAME} SHARED ${SOURCE_FILES})
add_library(openvino::${TARGET_NAME} ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>" "$<INSTALL_INTERFACE:runtime/include>")

target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime PRIVATE nlohmann_json::nlohmann_json jinja2cpp)

target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")

target_compile_features(${TARGET_NAME} PUBLIC cxx_std_17)

set_target_properties(${TARGET_NAME} PROPERTIES
    VERSION ${CMAKE_PROJECT_VERSION}
    SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
)

# Copy the library to python to allow skipping wheel installation
add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy
        "$<TARGET_FILE:${TARGET_NAME}>"
        "${CMAKE_CURRENT_SOURCE_DIR}/../python/openvino_genai/$<TARGET_FILE_NAME:${TARGET_NAME}>"
    COMMENT "Copy ${TARGET_NAME} to src/python/openvino_genai")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
install(TARGETS ${TARGET_NAME}
    LIBRARY DESTINATION python/openvino_genai/ COMPONENT pygenai_${Python_VERSION_MAJOR}_${Python_VERSION_MINOR}
    RUNTIME DESTINATION python/openvino_genai/ COMPONENT pygenai_${Python_VERSION_MAJOR}_${Python_VERSION_MINOR})

# Copy libcore_tokenizers.so to build_dir/openvino_tokenizers/src/
if(NOT MSVC)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy
            "${CMAKE_BINARY_DIR}/_deps/fast_tokenizer-src/lib/libcore_tokenizers.so"
            "${CMAKE_BINARY_DIR}/openvino_tokenizers/src/"
        COMMENT "Copy libcore_tokenizers.so to build_dir/openvino_tokenizers/src/")
endif()

# - Windows: `<openvino_dir>\runtime\bin\intel64\Release\`
# - MacOS_x86: `<openvino_dir>/runtime/lib/intel64/Release`
# - MacOS_arm64: `<openvino_dir>/runtime/lib/arm64/Release/`
# - Linux_x86: `<openvino_dir>/runtime/lib/intel64/`
# - Linux_arm64: `<openvino_dir>/runtime/lib/aarch64/`
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARCH_DIR)
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
    set(ARCH_DIR intel64)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm64.*|aarch64.*|AARCH64.*|ARM64.*)")
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
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION runtime/include COMPONENT core_genai_dev)
install(EXPORT OpenVINOGenAITargets FILE OpenVINOGenAITargets.cmake NAMESPACE openvino:: DESTINATION runtime/cmake)
include(CMakePackageConfigHelpers)
configure_package_config_file(OpenVINOGenAIConfig.cmake.in "${CMAKE_BINARY_DIR}/OpenVINOGenAIConfig.cmake" INSTALL_DESTINATION runtime/cmake)
install(FILES "${CMAKE_BINARY_DIR}/OpenVINOGenAIConfig.cmake" "${CMAKE_BINARY_DIR}/OpenVINOGenAIConfig.cmake" DESTINATION runtime/cmake COMPONENT core_genai_dev)
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${CMAKE_BINARY_DIR}/OpenVINOGenAIConfigVersion.cmake" VERSION ${CMAKE_PROJECT_VERSION} COMPATIBILITY AnyNewerVersion)
export(EXPORT OpenVINOGenAITargets FILE "${CMAKE_BINARY_DIR}/OpenVINOGenAITargets.cmake" NAMESPACE openvino::)
