# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
list(APPEND SOURCE_FILES "${CMAKE_CURRENT_BINARY_DIR}/version.cpp")

# Dependencies

include(FetchContent)

if(NOT TARGET nlohmann_json)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
        URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

if(NOT TARGET minja)
    FetchContent_Declare(
        minja
        GIT_REPOSITORY https://github.com/google/minja.git
        GIT_TAG 3e4c61c616eda133cfb1e440fc7a14bf1729bbee
    )

    FetchContent_GetProperties(minja)
    if(NOT minja_POPULATED)
        FetchContent_Populate(minja)

        add_library(minja INTERFACE)
        target_include_directories(minja INTERFACE
            $<BUILD_INTERFACE:${minja_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
        target_link_libraries(minja INTERFACE nlohmann_json::nlohmann_json)
    endif()
endif()

FetchContent_Declare(safetensors.h
    URL https://github.com/hsnyder/safetensors.h/archive/974a85d7dfd6e010558353226638bb26d6b9d756.tar.gz
    URL_HASH SHA256=9aaf5961609601cf9aaa96582a207bce7c6e5fbf57ed2cc669bb7bde6a937d4b)
FetchContent_MakeAvailable(safetensors.h)

if(ENABLE_GGUF)
    FetchContent_Declare(
      gguflib
      URL https://github.com/Lourdle/gguf-tools/archive/bac796ada809ac293e685db59b075971181cb008.zip
      URL_HASH SHA256=4d6eab5055468d222833f3f83fe2f7909ccd06114278c2c0b468570ef002c22d)
    FetchContent_MakeAvailable(gguflib)
    set_target_properties(gguf-tools PROPERTIES EXCLUDE_FROM_ALL ON)

    add_library(gguflib STATIC ${gguflib_SOURCE_DIR}/fp16.c ${gguflib_SOURCE_DIR}/gguflib.c)
    set_target_properties(gguflib PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(gguflib PUBLIC "${gguflib_SOURCE_DIR}")
endif()

# Disable XGrammar for Android platform, causes issues in CI
if(ANDROID)
    set(ENABLE_XGRAMMAR OFF CACHE BOOL "Enable XGrammar" FORCE)
endif()

if(ENABLE_XGRAMMAR)
    set(XGRAMMAR_VERSION v0.1.25)
    set(XGRAMMAR_DIR ${CMAKE_BINARY_DIR}/xgrammar)

    FetchContent_Declare(
      xgrammar
      GIT_REPOSITORY https://github.com/mlc-ai/xgrammar.git
      GIT_TAG ${XGRAMMAR_VERSION}
      GIT_SUBMODULES_RECURSE ON
    )

    set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Build Python bindings" FORCE)
    FetchContent_GetProperties(xgrammar)
    if(NOT xgrammar_POPULATED)
        FetchContent_Populate(xgrammar)

        # Modify the config.cmake file directly
        set(CONFIG_FILE "${xgrammar_SOURCE_DIR}/cmake/config.cmake")
        if(EXISTS "${CONFIG_FILE}")
            file(READ "${CONFIG_FILE}" CONFIG_CONTENT)
            string(REGEX REPLACE "XGRAMMAR_BUILD_PYTHON_BINDINGS ON"
                                "XGRAMMAR_BUILD_PYTHON_BINDINGS OFF"
                                MODIFIED_CONFIG "${CONFIG_CONTENT}")

            string(REGEX REPLACE "^set\\(CMAKE_BUILD_TYPE .*$"
                                "set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})"
                                MODIFIED_CONFIG "${MODIFIED_CONFIG}")
            file(WRITE "${CONFIG_FILE}" "${MODIFIED_CONFIG}")
        endif()

        set(XGRAMMAR_BUILD_CXX_TESTS OFF CACHE BOOL "Build CXX tests" FORCE)
        set(XGRAMMAR_ENABLE_CPPTRACE OFF CACHE BOOL "Enable CPPTrace" FORCE)
        set(XGRAMMAR_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA" FORCE)

        add_subdirectory("${xgrammar_SOURCE_DIR}" "${xgrammar_BINARY_DIR}" EXCLUDE_FROM_ALL)
        if(MSVC)
            # Suppress C4146 warning for MSVC compiler
            target_compile_options(xgrammar PRIVATE "/wd4146")
        endif()
    endif()
else()
    set(XGRAMMAR_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/sampling/structured_output/xgrammar_backend.hpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/src/sampling/structured_output/xgrammar_backend.cpp)
    list(REMOVE_ITEM SOURCE_FILES ${XGRAMMAR_SOURCES})
endif()

# generate version files
configure_file("${OpenVINOGenAI_SOURCE_DIR}/cmake/templates/version.hpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/openvino/genai/version.hpp" @ONLY)

configure_file("${OpenVINOGenAI_SOURCE_DIR}/cmake/templates/version.cpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/version.cpp" @ONLY)

# Object library
if(NOT ENABLE_GGUF)
    set(GGUF_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/gguf_utils/gguf.cpp
                     ${CMAKE_CURRENT_SOURCE_DIR}/src/gguf_utils/gguf_quants.cpp
                     ${CMAKE_CURRENT_SOURCE_DIR}/src/gguf_utils/gguf_modeling.cpp
                     ${CMAKE_CURRENT_SOURCE_DIR}/src/gguf_utils/building_blocks.cpp)
    list(REMOVE_ITEM SOURCE_FILES ${GGUF_SOURCES})
endif()

set(TARGET_NAME openvino_genai)
set(TARGET_NAME_OBJ ${TARGET_NAME}_obj)

add_library(${TARGET_NAME_OBJ} OBJECT ${SOURCE_FILES})

target_include_directories(${TARGET_NAME_OBJ}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
           "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

if(ENABLE_XGRAMMAR)
    target_link_libraries(${TARGET_NAME_OBJ} PRIVATE xgrammar)
endif()

if(ENABLE_GGUF)
    target_link_libraries(${TARGET_NAME_OBJ} PRIVATE gguflib)
    target_compile_definitions(${TARGET_NAME_OBJ} PRIVATE ENABLE_GGUF)
endif()

target_include_directories(${TARGET_NAME_OBJ} SYSTEM PRIVATE "${safetensors.h_SOURCE_DIR}")

target_link_libraries(${TARGET_NAME_OBJ} PRIVATE openvino::runtime openvino::threading nlohmann_json::nlohmann_json minja)

target_compile_features(${TARGET_NAME_OBJ} PRIVATE cxx_std_17)

target_compile_definitions(${TARGET_NAME_OBJ} PRIVATE openvino_genai_EXPORTS)

# Add /bigobj flag for MSVC to handle large object files
if(MSVC)
    target_compile_options(${TARGET_NAME_OBJ} PRIVATE "/bigobj")
endif()

set_target_properties(${TARGET_NAME_OBJ} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Shared library

add_library(${TARGET_NAME} SHARED $<TARGET_OBJECTS:${TARGET_NAME_OBJ}>)
add_library(openvino::genai ALIAS ${TARGET_NAME})

if(WIN32)
  add_vs_version_resource(${TARGET_NAME})
endif()

target_include_directories(${TARGET_NAME} INTERFACE "$<INSTALL_INTERFACE:runtime/include>"
                                                    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                                    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")

target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime PRIVATE openvino::threading nlohmann_json::nlohmann_json minja ${CMAKE_DL_LIBS})

if(ENABLE_XGRAMMAR)
    target_link_libraries(${TARGET_NAME} PRIVATE xgrammar)
endif()

if(ENABLE_GGUF)
  target_link_libraries(${TARGET_NAME} PRIVATE gguflib)
endif()

target_compile_features(${TARGET_NAME} INTERFACE cxx_std_17)

# Add /bigobj flag for MSVC to handle large object files  
if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE "/bigobj")
endif()

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
