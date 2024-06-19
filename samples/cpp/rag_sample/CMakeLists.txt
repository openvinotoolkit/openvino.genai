# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")

find_package(OpenVINOGenAI REQUIRED PATHS
    "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
    ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
)

include_directories(include)

if(TARGET openvino_tokenizers)
    set(OPENVINO_TOKENIZERS_PATH $<TARGET_FILE:openvino_tokenizers>)
else()
    message(FATAL_ERROR "multinomial_causal_lm must be compiled as part of OpenVIINOGenAI to have the path to openvino_tokenizers hardcoded.")
endif()
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_library(embeddings SHARED src/embeddings.cpp)
target_link_libraries(embeddings PRIVATE openvino::genai)
target_include_directories(embeddings PUBLIC include)
target_compile_definitions(embeddings PRIVATE OPENVINO_TOKENIZERS_PATH="${OPENVINO_TOKENIZERS_PATH}")

add_library(handle_master SHARED src/handle_master.cpp)
target_link_libraries(handle_master PRIVATE openvino::genai)
target_link_libraries(handle_master PRIVATE embeddings)
target_include_directories(handle_master PUBLIC include)
target_compile_definitions(handle_master PRIVATE OPENVINO_TOKENIZERS_PATH="${OPENVINO_TOKENIZERS_PATH}")

add_executable(rag_sample_server rag_sample_server.cpp)
add_executable(rag_sample_client rag_sample_client.cpp)

target_link_libraries(rag_sample_server PRIVATE openvino::genai)
target_link_libraries(rag_sample_server PRIVATE handle_master)
target_link_libraries(rag_sample_server PRIVATE embeddings)


target_compile_definitions(rag_sample_server PRIVATE OPENVINO_TOKENIZERS_PATH="${OPENVINO_TOKENIZERS_PATH}")

target_compile_features(rag_sample_server PRIVATE cxx_std_17)
target_compile_features(rag_sample_client PRIVATE cxx_std_17)

install(TARGETS rag_sample_server rag_sample_client 
    RUNTIME DESTINATION samples_bin/
    COMPONENT samples_bin
    EXCLUDE_FROM_ALL)
