# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

foreach(var IN ITEMS init_pyi_file)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "Variable ${var} is not defined")
    endif()
endforeach()

file(STRINGS ${init_pyi_file} file_lines)

foreach(file_line IN LISTS file_lines)
    if(file_line MATCHES "^from openvino_genai\\.py_openvino_genai\\..* import draft_model$")
        set(file_line "from openvino_genai.py_openvino_genai import draft_model")
    endif()
    if(file_line MATCHES "^from openvino_genai\\.py_openvino_genai\\..* import get_version$")
        set(file_line "from openvino_genai.py_openvino_genai import get_version")
    endif()

    set(file_content "${file_content}${file_line}\n")
endforeach()

file(WRITE ${init_pyi_file} ${file_content})
