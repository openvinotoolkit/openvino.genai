# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINO REQUIRED COMPONENTS Runtime Threading)

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

add_executable(prompt_lookup_decoding_lm prompt_lookup_decoding_lm.cpp)
target_link_libraries(prompt_lookup_decoding_lm PRIVATE openvino::runtime openvino::threading)
set_target_properties(prompt_lookup_decoding_lm PROPERTIES
    COMPILE_PDB_NAME prompt_lookup_decoding_lm
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
target_compile_features(prompt_lookup_decoding_lm PRIVATE cxx_std_17)

get_target_property(genai_imported openvino::genai IMPORTED_LOCATION)
set(OPENVINO_TOKENIZERS_PATH $<IF:$<BOOL:${genai_imported}>,${genai_imported},$<TARGET_FILE_DIR:openvino::genai>>)
set(OPENVINO_TOKENIZERS_FILENAME "${CMAKE_SHARED_LIBRARY_PREFIX}openvino_tokenizers${CMAKE_SHARED_LIBRARY_SUFFIX}")
target_compile_definitions(prompt_lookup_decoding_lm PRIVATE
    OPENVINO_TOKENIZERS_PATH="${OPENVINO_TOKENIZERS_PATH}/${OPENVINO_TOKENIZERS_FILENAME}")

install(TARGETS prompt_lookup_decoding_lm
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
