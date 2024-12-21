# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

set(TARGET_NAME prompt_lookup_decoding_lm)
add_executable(${TARGET_NAME} ${TARGET_NAME}.cpp)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::genai)

set_target_properties(${TARGET_NAME} PROPERTIES
    COMPILE_PDB_NAME ${TARGET_NAME} 
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS ${TARGET_NAME} 
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)


# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINO REQUIRED COMPONENTS Runtime Threading)
find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

add_executable(prompt_lookup_decoding_lm_sdpa prompt_lookup_decoding_lm_sdpa.cpp)
target_link_libraries(prompt_lookup_decoding_lm_sdpa PRIVATE openvino::runtime openvino::threading)
set_target_properties(prompt_lookup_decoding_lm_sdpa PROPERTIES
    COMPILE_PDB_NAME prompt_lookup_decoding_lm_sdpa
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
target_compile_features(prompt_lookup_decoding_lm_sdpa PRIVATE cxx_std_17)
get_target_property(genai_imported openvino::genai IMPORTED_LOCATION)
set(OPENVINO_TOKENIZERS_PATH $<IF:$<BOOL:${genai_imported}>,${genai_imported},$<TARGET_FILE_DIR:openvino::genai>>)
set(OPENVINO_TOKENIZERS_FILENAME "${CMAKE_SHARED_LIBRARY_PREFIX}openvino_tokenizers${CMAKE_SHARED_LIBRARY_SUFFIX}")
target_compile_definitions(prompt_lookup_decoding_lm_sdpa PRIVATE
    OPENVINO_TOKENIZERS_PATH="${OPENVINO_TOKENIZERS_PATH}/${OPENVINO_TOKENIZERS_FILENAME}")

install(TARGETS prompt_lookup_decoding_lm_sdpa
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)