# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED PATHS "${CMAKE_BINARY_DIR}" ${OpenVINO_DIR} NO_CMAKE_FIND_ROOT_PATH)
add_executable(text2speech_c text2speech.c)
target_link_libraries(text2speech_c PRIVATE openvino::genai::c)
set_target_properties(text2speech_c PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
install(TARGETS text2speech_c RUNTIME DESTINATION samples_bin/ COMPONENT samples_bin EXCLUDE_FROM_ALL)
