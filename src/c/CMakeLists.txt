# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_genai_c)

file(GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

# shared library

add_library(${TARGET_NAME} SHARED ${SOURCE_FILES})
add_library(openvino::genai::c ALIAS ${TARGET_NAME})
if(WIN32)
  add_vs_version_resource(${TARGET_NAME})
endif()

target_include_directories(${TARGET_NAME} PUBLIC "$<INSTALL_INTERFACE:runtime/include>"
                                                 "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                                 $<BUILD_INTERFACE:$<TARGET_PROPERTY:openvino::genai,INTERFACE_INCLUDE_DIRECTORIES>>)

target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime::c PRIVATE openvino::genai)

set_target_properties(${TARGET_NAME} PROPERTIES
    EXPORT_NAME genai::c
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO}
    VERSION ${OpenVINOGenAI_VERSION}
    SOVERSION ${MAJOR_SUFFIX}${OpenVINOGenAI_VERSION_MINOR}${OpenVINOGenAI_VERSION_PATCH})

if(OpenVINODeveloperPackage_FOUND)
    # must be called after all target_link_libraries
    ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

    ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                        SOURCE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif()

# install

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOGenAITargets
        LIBRARY DESTINATION ${LIBRARY_DESTINATION} COMPONENT core_c_genai
            NAMELINK_COMPONENT core_c_genai_dev
        ARCHIVE DESTINATION ${ARCHIVE_DESTINATION} COMPONENT core_c_genai_dev
        RUNTIME DESTINATION ${RUNTIME_DESTINATION} COMPONENT core_c_genai
        INCLUDES DESTINATION runtime/include)
