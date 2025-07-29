# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(PROJECT_COMPANY_NAME "Intel Corporation")
set(PROJECT_PRODUCT_NAME "OpenVINO GenAI")
set(PROJECT_COPYRIGHT "Copyright (C) 2018-2025, Intel Corporation")
set(PROJECT_COMMENTS "https://docs.openvino.ai/")

# This function generates a version resource (.rc) file from a template and adds it to the given target.
function(add_vs_version_resource TARGET_NAME)
    set(VS_VERSION_TEMPLATE "${PROJECT_SOURCE_DIR}/cmake/templates/vs_version.rc.in")
    set(VS_VERSION_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/vs_version.rc")

    configure_file("${VS_VERSION_TEMPLATE}" "${VS_VERSION_OUTPUT}" @ONLY)

    target_sources(${TARGET_NAME} PRIVATE "${VS_VERSION_OUTPUT}")
endfunction()
