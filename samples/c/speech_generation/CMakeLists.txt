# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

include(FetchContent)

if(NOT TARGET dr_libs)
	FetchContent_Declare(dr_libs
		URL https://github.com/mackron/dr_libs/archive/da35f9d6c7374a95353fd1df1d394d44ab66cf01.tar.gz
		URL_HASH SHA256=2704d347f480ca1bc92233fb01747e4550cc8031735b6ea62ca9990ebb8851ae)
	FetchContent_MakeAvailable(dr_libs)
endif()

find_package(OpenVINOGenAI REQUIRED PATHS "${CMAKE_BINARY_DIR}" ${OpenVINO_DIR} NO_CMAKE_FIND_ROOT_PATH)
add_executable(text2speech_c text2speech.c ../common/audio_utils.c)
target_include_directories(text2speech_c PRIVATE ../common "$<BUILD_INTERFACE:${dr_libs_SOURCE_DIR}>")
target_link_libraries(text2speech_c PRIVATE openvino::genai::c)
set_target_properties(text2speech_c PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
install(TARGETS text2speech_c RUNTIME DESTINATION samples_bin/ COMPONENT samples_bin EXCLUDE_FROM_ALL)

add_executable(qwen3_tts_c qwen3_tts.c ../common/audio_utils.c ../common/cli_utf8.c)
target_include_directories(qwen3_tts_c PRIVATE ../common "$<BUILD_INTERFACE:${dr_libs_SOURCE_DIR}>")
target_link_libraries(qwen3_tts_c PRIVATE openvino::genai::c)
if(WIN32)
	target_link_libraries(qwen3_tts_c PRIVATE shell32)
endif()
set_target_properties(qwen3_tts_c PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
install(TARGETS qwen3_tts_c RUNTIME DESTINATION samples_bin/ COMPONENT samples_bin EXCLUDE_FROM_ALL)
