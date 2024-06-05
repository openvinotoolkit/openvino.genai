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

add_subdirectory(./thirdparty/)
add_subdirectory(src)
add_subdirectory(text_generation/causal_lm/cpp)

install(DIRECTORY text_generation/causal_lm/cpp/ DESTINATION samples/cpp/causal_lm COMPONENT cpp_samples_genai)
install(FILES LICENSE DESTINATION licensing COMPONENT licensing_genai RENAME LICENSE-GENAI)
install(FILES third-party-programs.txt DESTINATION licensing COMPONENT licensing_genai RENAME third-party-programs-genai.txt)
if(MSVC AND NOT DEFINED CPACK_GENERATOR)
    set(CPACK_GENERATOR "ZIP")
endif()
include(CPack)
