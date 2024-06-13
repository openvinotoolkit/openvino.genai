# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.23.0)  # The requirement comes from Jinja2Cpp

# Multi config generators such as Visual Studio ignore CMAKE_BUILD_TYPE. Multi config generators are configured with
# CMAKE_CONFIGURATION_TYPES, but limiting options in it completely removes such build options
get_property(GENERATOR_IS_MULTI_CONFIG_VAR GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(CMAKE_GENERATOR STREQUAL "Ninja Multi-Config")
    # 'Ninja Multi-Config' specific, see:
    # https://cmake.org/cmake/help/latest/variable/CMAKE_DEFAULT_BUILD_TYPE.html
    set(CMAKE_DEFAULT_BUILD_TYPE "Release" CACHE STRING "CMake default build type")
elseif(NOT GENERATOR_IS_MULTI_CONFIG_VAR AND NOT DEFINED CMAKE_BUILD_TYPE)
    message(STATUS "CMAKE_BUILD_TYPE is not defined, 'Release' will be used")
    # Setting CMAKE_BUILD_TYPE as CACHE must go before project(). Otherwise project() sets its value and set() doesn't take an effect
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel ...")
endif()

project(OpenVINOGenAI VERSION 2024.2.0.0)

option(CPACK_INCLUDE_TOPLEVEL_DIRECTORY "Boolean toggle to include/exclude top level directory." OFF)
if(WIN32)
    set(CPACK_GENERATOR "ZIP" CACHE STRING "List of CPack generators to use.")
endif()

add_subdirectory(./thirdparty/)
add_subdirectory(src)
add_subdirectory(samples/cpp/beam_search_causal_lm/)
add_subdirectory(samples/cpp/chat_sample/)
add_subdirectory(samples/cpp/greedy_causal_lm/)
add_subdirectory(samples/cpp/multinomial_causal_lm/)
add_subdirectory(samples/cpp/prompt_lookup_decoding_lm/)
add_subdirectory(samples/cpp/speculative_decoding_lm/)

install(DIRECTORY
        ./samples/cpp/beam_search_causal_lm
        ./samples/cpp/chat_sample
        ./samples/cpp/greedy_causal_lm
        ./samples/cpp/multinomial_causal_lm
        # Don't install prompt_lookup_decoding_lm and speculative_decoding_lm because they don't use openvino_genai library and arent verifyed yet.
    DESTINATION samples/cpp/ COMPONENT cpp_samples_genai)
install(DIRECTORY
        ./samples/python/beam_search_causal_lm
        ./samples/python/chat_sample
        ./samples/python/greedy_causal_lm
        ./samples/python/multinomial_causal_lm
    DESTINATION samples/python/ COMPONENT cpp_samples_genai
    USE_SOURCE_PERMISSIONS)
install(FILES ./samples/requirements.txt DESTINATION ./samples/ COMPONENT cpp_samples_genai)
install(FILES LICENSE DESTINATION licensing COMPONENT licensing_genai RENAME LICENSE-GENAI)
install(FILES third-party-programs.txt DESTINATION licensing COMPONENT licensing_genai RENAME third-party-programs-genai.txt)
include(CPack)
