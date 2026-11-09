# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

option(BUILD_TOKENIZERS "Build OpenVINO Tokenizers together with OpenVINO GenAI" ON)

if(BUILD_TOKENIZERS)
    # Define the CMAKE_INSTALL_<dir> cache variables before OpenVINO Tokenizers adds sentencepiece and PCRE2.
    # sentencepiece requires CMake 3.5 (policy CMP0102 OLD), so its mark_as_advanced() call creates empty
    # CMAKE_INSTALL_BINDIR / CMAKE_INSTALL_INCLUDEDIR cache entries; PCRE2 >= 10.48 then keeps them and fails with
    # "install FILES given no DESTINATION!" unless GNUInstallDirs was included earlier (e.g. by the OpenVINO
    # Developer Package, which an installed OpenVINO does not provide).
    include(GNUInstallDirs)
    add_subdirectory(./openvino_tokenizers/ "${CMAKE_BINARY_DIR}/openvino_tokenizers/")
    # Put binaries to a single dir to mimic package structure.
    set_target_properties(openvino_tokenizers PROPERTIES
        # Generator expressions to disable appending a per-configuration subdirectory (Release, Debug).
        # ARCHIVE_OUTPUT is irrelevant. It's here just to keep all the artifacts in one place.
        ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
        LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
        RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/openvino_genai/>"
    )
endif()
