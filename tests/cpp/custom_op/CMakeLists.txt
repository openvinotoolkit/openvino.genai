# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(TARGET_NAME "openvino_custom_add_extension")

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

set(SRC custom_add.cpp ov_extension.cpp)

add_library(${TARGET_NAME} MODULE ${SRC})

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
