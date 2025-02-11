# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

include(FetchContent)

if(NOT TARGET gtest)
  set(INSTALL_GTEST OFF CACHE BOOL "")

  FetchContent_Declare(
    googletest
    URL       https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    URL_HASH SHA256=edd885a1ab32b6999515a880f669efadb80b3f880215f315985fa3f6eca7c4d3
  )
  FetchContent_MakeAvailable(googletest)
endif()

if(NOT TARGET nlohmann_json)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
        URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

FetchContent_Declare(safetensors.h
    URL https://github.com/hsnyder/safetensors.h/archive/974a85d7dfd6e010558353226638bb26d6b9d756.tar.gz
    URL_HASH SHA256=9aaf5961609601cf9aaa96582a207bce7c6e5fbf57ed2cc669bb7bde6a937d4b)
FetchContent_MakeAvailable(safetensors.h)

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

        add_subdirectory("${jinja2cpp_SOURCE_DIR}" "${jinja2cpp_BINARY_DIR}" EXCLUDE_FROM_ALL)

        if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR (OV_COMPILER_IS_INTEL_LLVM AND UNIX))
            target_compile_options(jinja2cpp PRIVATE -Wno-undef)
        endif()
        if(SUGGEST_OVERRIDE_SUPPORTED)
            target_compile_options(jinja2cpp PRIVATE -Wno-suggest-override)
        endif()

        target_compile_definitions(jinja2cpp PUBLIC JINJA2CPP_LINK_AS_SHARED=0)
    endif()
endfunction()

ov_genai_build_jinja2cpp()

set(TEST_TARGET_NAME "tests_continuous_batching")
file(GLOB tests_src "*.cpp")
file(GLOB src_files "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/sequence_group.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/cache_eviction.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/sampler.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/speculative_decoding/*.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/prompt_lookup/*.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/paged_attention_transformations.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/utils.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/continuous_batching*.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/icontinuous_batching.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/lora_helper.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/text_callback_streamer.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/generation_config.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/tokenizer.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/generation_handle.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/lora_adapter.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/perf_metrics.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/continious_batching_impl.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/lora_names_mapping.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/make_tokenizer_stateful.cpp"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/safetensors.c"
                    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src/tokenizers_path.cpp")



add_executable(${TEST_TARGET_NAME} ${tests_src})

if(TARGET openvino_tokenizers)
    add_dependencies(${TEST_TARGET_NAME} openvino_tokenizers)
endif()

target_include_directories(${TEST_TARGET_NAME} SYSTEM PRIVATE "${safetensors.h_SOURCE_DIR}")

target_link_libraries(${TEST_TARGET_NAME} PRIVATE openvino::runtime nlohmann_json::nlohmann_json jinja2cpp gtest_main gmock_main)
target_include_directories(${TEST_TARGET_NAME} PRIVATE "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src" "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/include")
target_sources(${TEST_TARGET_NAME} PRIVATE ${src_files})

target_compile_definitions(${TEST_TARGET_NAME} PRIVATE openvino_genai_STATIC)
