# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.15)

# Multi config generators such as Visual Studio ignore CMAKE_BUILD_TYPE. Multi config generators are configured with
# CMAKE_CONFIGURATION_TYPES, but limiting options in it completely removes such build options
get_property(GENERATOR_IS_MULTI_CONFIG_VAR GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT GENERATOR_IS_MULTI_CONFIG_VAR AND NOT DEFINED CMAKE_BUILD_TYPE)
    message(STATUS "CMAKE_BUILD_TYPE is not defined, 'Release' will be used")
    # Setting CMAKE_BUILD_TYPE as CACHE must go before project(). Otherwise project() sets its value and set() doesn't take an effect
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel ...")
endif()

project(OpenVINOGenAI VERSION 2024.2.0.0)

add_subdirectory(./thirdparty/openvino_tokenizers/ "${CMAKE_CURRENT_BINARY_DIR}/openvino_tokenizers/")
# Put binaries to a single dir to mimic package structure.
set_target_properties(openvino_tokenizers PROPERTIES
    # Generator expressions to disable appending a per-configuration subdirectory (Release, Debug).
    # ARCHIVE_OUTPUT is irrelevant. It's here just to keep all the artifacts in one place.
    ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
    LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
    RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
)
if(TARGET core_tokenizers)
    set_target_properties(core_tokenizers PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
        LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
        RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
    )
else()
    # Prebuilt dependencies
    if(WIN32)
        set(extra_libs "${CMAKE_BINARY_DIR}/_deps/fast_tokenizer-src/lib/core_tokenizers.dll"
                       "${CMAKE_BINARY_DIR}/_deps/fast_tokenizer-src/third_party/lib/icudt70.dll"
                       "${CMAKE_BINARY_DIR}/_deps/fast_tokenizer-src/third_party/lib/icuuc70.dll")
    elseif(LINUX)
        set(extra_libs "${CMAKE_BINARY_DIR}/_deps/fast_tokenizer-src/lib/libcore_tokenizers.so")
    elseif(APPLE)
        set(extra_libs "${CMAKE_BINARY_DIR}/_deps/fast_tokenizer-src/lib/libcore_tokenizers.dylib")
    endif()
    add_custom_command(OUTPUT "${extra_libs}"
        COMMAND "${CMAKE_COMMAND}" -E copy "${extra_libs}" "${CMAKE_BINARY_DIR}/openvino_genai/"
        DEPENDS openvino_tokenizers)
endif()
add_subdirectory(src)
add_subdirectory(text_generation/causal_lm/cpp)

install(DIRECTORY text_generation/causal_lm/cpp/ DESTINATION samples/cpp/causal_lm COMPONENT cpp_samples_genai)
install(FILES LICENSE DESTINATION licensing COMPONENT licensing_genai RENAME LICENSE-GENAI)
install(FILES third-party-programs.txt DESTINATION licensing COMPONENT licensing_genai RENAME third-party-programs-genai.txt)
set(CPACK_GENERATOR "ZIP")
include(CPack)
