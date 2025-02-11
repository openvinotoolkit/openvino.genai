# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET gtest)
  set(INSTALL_GTEST OFF CACHE BOOL "")

  FetchContent_Declare(
    googletest
    URL       https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    URL_HASH SHA256=edd885a1ab32b6999515a880f669efadb80b3f880215f315985fa3f6eca7c4d3
  )
  FetchContent_MakeAvailable(googletest)
endif()

FetchContent_Declare(safetensors.h
    URL https://github.com/hsnyder/safetensors.h/archive/974a85d7dfd6e010558353226638bb26d6b9d756.tar.gz
    URL_HASH SHA256=9aaf5961609601cf9aaa96582a207bce7c6e5fbf57ed2cc669bb7bde6a937d4b)
FetchContent_MakeAvailable(safetensors.h)

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

target_link_libraries(${TEST_TARGET_NAME} PRIVATE openvino::runtime nlohmann_json::nlohmann_json jinja2cpp gtest_main gmock_main ${CMAKE_DL_LIBS})
target_include_directories(${TEST_TARGET_NAME} PRIVATE "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src" "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/include")
target_sources(${TEST_TARGET_NAME} PRIVATE ${src_files})

target_compile_definitions(${TEST_TARGET_NAME} PRIVATE openvino_genai_STATIC)
