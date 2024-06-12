# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
find_package(OpenVINOGenAI REQUIRED PATHS
    "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
    ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
)
add_executable(rag_sample_server rag_sample_server.cpp)
add_executable(rag_sample_client rag_sample_client.cpp)

target_link_libraries(rag_sample_server PRIVATE openvino::genai)
target_compile_features(rag_sample_server PRIVATE cxx_std_17)
target_compile_features(rag_sample_client PRIVATE cxx_std_17)

install(TARGETS rag_sample_server rag_sample_client
    RUNTIME DESTINATION samples_bin/
    COMPONENT samples_bin
    EXCLUDE_FROM_ALL)
