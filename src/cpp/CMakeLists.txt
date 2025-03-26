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
file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
list(APPEND SOURCE_FILES "${CMAKE_CURRENT_BINARY_DIR}/version.cpp")

set(TARGET_NAME openvino_genai)
set(TARGET_NAME_OBJ ${TARGET_NAME}_obj)

add_library(${TARGET_NAME_OBJ} OBJECT ${SOURCE_FILES})

target_include_directories(${TARGET_NAME_OBJ}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
           "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

if(ENABLE_GGUF)
    message(STATUS "Downloading gguflib")
    FetchContent_Declare(
      gguflib
      URL https://github.com/antirez/gguf-tools/archive/af7d88d808a7608a33723fba067036202910acb3.zip
      URL_HASH SHA256=d613559c7a398eb4a0919982e6a370055f8466497f0f866d331dc92b735927e7)
    FetchContent_MakeAvailable(gguflib)
    target_include_directories(${TARGET_NAME_OBJ}
                               PRIVATE "${gguflib_SOURCE_DIR}")
    target_include_directories(${TARGET_NAME_OBJ}
                              PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/gguf_utils")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    add_library(gguflib STATIC ${gguflib_SOURCE_DIR}/fp16.c
                               ${gguflib_SOURCE_DIR}/gguflib.c)
    #target_compile_features(gguflib PRIVATE fPIC)
    target_link_libraries(${TARGET_NAME_OBJ} PRIVATE $<BUILD_INTERFACE:gguflib>) 
    target_sources(${TARGET_NAME_OBJ} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/gguf_utils/gguf.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/gguf_utils/gguf_quants.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/gguf_utils/gguf_modeling.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/src/gguf_utils/building_blocks.cpp)
endif()

target_include_directories(${TARGET_NAME_OBJ} SYSTEM PRIVATE "${safetensors.h_SOURCE_DIR}" "${gguflib_SOURCE_DIR}")

target_link_libraries(${TARGET_NAME_OBJ} PRIVATE openvino::runtime openvino::threading nlohmann_json::nlohmann_json jinja2cpp)

target_compile_features(${TARGET_NAME_OBJ} PRIVATE cxx_std_17)

target_compile_definitions(${TARGET_NAME_OBJ} PRIVATE openvino_genai_EXPORTS)

set_target_properties(${TARGET_NAME_OBJ} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Shared library

add_library(${TARGET_NAME} SHARED $<TARGET_OBJECTS:${TARGET_NAME_OBJ}>)
add_library(openvino::genai ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME} INTERFACE "$<INSTALL_INTERFACE:runtime/include>"
                                                    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                                    "$<BUILD_INTERFACE:${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/gguf_utils>"
                                                    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")

target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime PRIVATE openvino::threading nlohmann_json::nlohmann_json jinja2cpp ${CMAKE_DL_LIBS})

if(ENABLE_GGUF)
  target_link_libraries(${TARGET_NAME} PRIVATE gguflib ${CMAKE_DL_LIBS})
endif()

target_compile_features(${TARGET_NAME} INTERFACE cxx_std_17)

if(TARGET openvino_tokenizers)
    add_dependencies(${TARGET_NAME} openvino_tokenizers)
endif()

set_target_properties(${TARGET_NAME} PROPERTIES
    EXPORT_NAME genai
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO}
    ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
    LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
    RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
)

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

if(rpaths)
    set_target_properties(${TARGET_NAME} PROPERTIES INSTALL_RPATH "${rpaths}")
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
                  ${OpenVINOGenAI_SOURCE_DIR}/src/c/include/
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
