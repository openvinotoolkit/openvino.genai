# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

foreach(var IN ITEMS init_pyi_file)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "Variable ${var} is not defined")
    endif()
endforeach()

file(STRINGS ${init_pyi_file} file_lines)

foreach(file_line IN LISTS file_lines)
    if(file_line MATCHES "^__version__.*")
        set(file_line "__version__: str")
    endif()

    set(file_content "${file_content}${file_line}\n")
endforeach()

file(WRITE ${init_pyi_file} ${file_content})
