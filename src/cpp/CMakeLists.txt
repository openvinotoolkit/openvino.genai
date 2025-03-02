# Copyright (C) 2018-2025 Intel Corporation
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
        URL https://github.com/jinja2cpp/Jinja2Cpp/archive/86dfb939b5c2beb7fabddae2df386be4e7fb9507.tar.gz
        URL_HASH SHA256=c222e2407316a573561bd74dfd1cd7b34b359f5481ac59529b3de355162ef9f7)

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

        if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR (OV_COMPILER_IS_INTEL_LLVM AND UNIX))
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undef")
        endif()
        if(SUGGEST_OVERRIDE_SUPPORTED)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
        endif()

        add_subdirectory("${jinja2cpp_SOURCE_DIR}" "${jinja2cpp_BINARY_DIR}" EXCLUDE_FROM_ALL)

        target_compile_definitions(jinja2cpp PUBLIC JINJA2CPP_LINK_AS_SHARED=0)
    endif()
endfunction()

FetchContent_Declare(safetensors.h
    URL https://github.com/hsnyder/safetensors.h/archive/974a85d7dfd6e010558353226638bb26d6b9d756.tar.gz
    URL_HASH SHA256=9aaf5961609601cf9aaa96582a207bce7c6e5fbf57ed2cc669bb7bde6a937d4b)
FetchContent_MakeAvailable(safetensors.h)

ov_genai_build_jinja2cpp()

# generate version files

configure_file("${OpenVINOGenAI_SOURCE_DIR}/cmake/templates/version.hpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/openvino/genai/version.hpp" @ONLY)

configure_file("${OpenVINOGenAI_SOURCE_DIR}/cmake/templates/version.cpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/version.cpp" @ONLY)

# Object library
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c" "${CMAKE_SOURCE_DIR}/src/c/src/*.cpp")
list(APPEND SOURCE_FILES "${CMAKE_CURRENT_BINARY_DIR}/version.cpp")

set(TARGET_NAME openvino_genai)
set(TARGET_NAME_OBJ ${TARGET_NAME}_obj)

add_library(${TARGET_NAME_OBJ} OBJECT ${SOURCE_FILES})

if(TARGET openvino_tokenizers)
    add_dependencies(${TARGET_NAME_OBJ} openvino_tokenizers)
endif()

target_include_directories(${TARGET_NAME_OBJ}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
           "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/c/include>"
           "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_include_directories(${TARGET_NAME_OBJ} SYSTEM PRIVATE "${safetensors.h_SOURCE_DIR}")

target_link_libraries(${TARGET_NAME_OBJ} PRIVATE openvino::runtime openvino::threading nlohmann_json::nlohmann_json jinja2cpp)

target_compile_features(${TARGET_NAME_OBJ} PRIVATE cxx_std_17)

target_compile_definitions(${TARGET_NAME_OBJ} PRIVATE openvino_genai_EXPORTS)

set_target_properties(${TARGET_NAME_OBJ} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Shared library

add_library(${TARGET_NAME} SHARED $<TARGET_OBJECTS:${TARGET_NAME_OBJ}>)
add_library(openvino::genai ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME} INTERFACE "$<INSTALL_INTERFACE:runtime/include>"
                                                    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                                    "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/c/include>"
                                                    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")

target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime PRIVATE openvino::threading nlohmann_json::nlohmann_json jinja2cpp ${CMAKE_DL_LIBS})

target_compile_features(${TARGET_NAME} INTERFACE cxx_std_17)

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

if(OpenVINODeveloperPackage_FOUND)
    # must be called after all target_link_libraries
    ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

    ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                        SOURCE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")

    # TODO: override versions as currently they come from OpenVINO
    # ov_add_vs_version_file(NAME ${TARGET_NAME}
    #                        FILEDESCRIPTION "OpenVINO GenAI library")

    # TODO: commit changes separately
    # ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
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

# Put binaries at the top level for NPM package
if(CPACK_GENERATOR STREQUAL "NPM")
    set(LIBRARY_DESTINATION .)
    set(ARCHIVE_DESTINATION .)
    set(RUNTIME_DESTINATION .)

    # setting RPATH / LC_RPATH depending on platform
    if(LINUX)
        # to find libopenvino_genai.so in the same folder
        set(rpaths "$ORIGIN")
        # to find libopenvino.so when installing from npm
        list(APPEND rpaths "$ORIGIN/../../openvino-node/bin")
        # to find libopenvino.so when installing from source
        list(APPEND rpaths "$ORIGIN/../node_modules/openvino-node/bin")
    elseif(APPLE)
        # to find libopenvino.dylib in the same folder
        set(rpaths "@loader_path")
    endif()

    if(rpaths)
        set_target_properties(${TARGET_NAME} PROPERTIES INSTALL_RPATH "${rpaths}")
    endif()
else()
    set(LIBRARY_DESTINATION runtime/lib/${ARCH_DIR})
    set(ARCHIVE_DESTINATION runtime/lib/${ARCH_DIR})
    set(RUNTIME_DESTINATION runtime/bin/${ARCH_DIR})
endif()

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOGenAITargets
        LIBRARY DESTINATION ${LIBRARY_DESTINATION} COMPONENT core_genai
            NAMELINK_COMPONENT core_genai_dev
        ARCHIVE DESTINATION ${ARCHIVE_DESTINATION} COMPONENT core_genai_dev
        RUNTIME DESTINATION ${RUNTIME_DESTINATION} COMPONENT core_genai
        INCLUDES DESTINATION runtime/include)

# development files do not need to be built for NPM package
if(CPACK_GENERATOR STREQUAL "NPM")
    return()
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
                  ${CMAKE_SOURCE_DIR}/src/c/include/
        DESTINATION runtime/include COMPONENT core_genai_dev)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/openvino/genai/version.hpp
        DESTINATION runtime/include/openvino/genai COMPONENT core_genai_dev)

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
