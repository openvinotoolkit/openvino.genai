# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)  # The requirement comes from thirdparty/openvino_contrib/modules/custom_operations/CMakeLists.txt
project(GenAI)

list(APPEND CUSTOM_OPERATIONS tokenizer)
add_subdirectory(thirdparty/openvino_contrib/modules/custom_operations/)  # Build user_ov_extensions
if(NOT MSVC)  # # Disable the warning from user_ov_extensions
    target_compile_options(sentencepiece-static PRIVATE -Wno-stringop-overflow)
endif()

add_subdirectory(llm/cpp)
target_compile_definitions(llm PRIVATE USER_OV_EXTENSIONS_PATH=\"$<TARGET_FILE:user_ov_extensions>\")
target_link_libraries(llm PRIVATE user_ov_extensions)

add_subdirectory(llm/chatglm_cpp)
target_compile_definitions(chatglm PRIVATE USER_OV_EXTENSIONS_PATH=\"$<TARGET_FILE:user_ov_extensions>\")
target_link_libraries(chatglm PRIVATE user_ov_extensions)
