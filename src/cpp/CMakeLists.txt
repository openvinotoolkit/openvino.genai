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
        URL https://github.com/ilya-lavrenov/Jinja2Cpp/archive/a5d002cbf44469775556daea14ba3ccdba1e365a.tar.gz
        URL_HASH SHA256=5aa5378d9acf3c44dfb607fd7f16f48b17ffa6495c219957901e9191ffe28900)

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

set(TARGET_NAME generate_pipeline_lib)
add_library(${TARGET_NAME} SHARED ${SOURCE_FILES})

target_include_directories(${TARGET_NAME}
    # TODO: remove it, because beam_search algo should not be exposed to end users
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../text_generation/causal_lm/cpp/
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime PRIVATE nlohmann_json::nlohmann_json jinja2cpp)

target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")

target_compile_features(${TARGET_NAME} PUBLIC cxx_std_17)

install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION . COMPONENT genai RUNTIME DESTINATION . COMPONENT genai)

# Populate python with the libraries to allow skipping wheel installation
add_custom_command(TARGET generate_pipeline_lib POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy
        "$<TARGET_FILE:generate_pipeline_lib>"
        "${CMAKE_CURRENT_SOURCE_DIR}/../python/openvino_genai/$<TARGET_FILE_NAME:generate_pipeline_lib>"
    COMMENT "Copy generate_pipeline_lib to src/python/openvino_genai")
