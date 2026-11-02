# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET openvino::runtime)
     # The OpenVINO installed from PyPI can be used to find OpenVINO_DIR
     if(NOT CMAKE_CROSSCOMPILING)
         find_package(Python3 QUIET COMPONENTS Interpreter)
         message(STATUS "Python3_Interpreter_FOUND=${Python3_Interpreter_FOUND}")
         if(Python3_Interpreter_FOUND)
             execute_process(
                 COMMAND ${Python3_EXECUTABLE} -c "from openvino.utils import get_cmake_path; print(get_cmake_path(), end='')"
                 OUTPUT_VARIABLE OpenVINO_DIR_PY
                 ERROR_QUIET)
         endif()
    endif()
    message(STATUS "OpenVINO_DIR_PY=${OpenVINO_DIR_PY}")
    find_package(OpenVINO REQUIRED PATHS "${OpenVINO_DIR_PY}")
endif()

function(add_test_extension TARGET_NAME)
    add_library(${TARGET_NAME} MODULE ${ARGN})

    if(DEFINED TEST_TARGET_NAME)
        set_target_properties(${TARGET_NAME} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:${TEST_TARGET_NAME}>"
            RUNTIME_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:${TEST_TARGET_NAME}>"
        )
    endif()

    target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)

    install(TARGETS ${TARGET_NAME}
            COMPONENT tests
            LIBRARY DESTINATION tests
            RUNTIME DESTINATION tests
            ARCHIVE DESTINATION tests)
endfunction()

add_test_extension(openvino_custom_add_extension custom_add.cpp ov_extension.cpp)
if(ENABLE_GGUF)
    add_test_extension(openvino_gguf_test_extension gguf_extension.cpp)
endif()
