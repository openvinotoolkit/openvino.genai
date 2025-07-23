# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

foreach(var IN ITEMS generated_pyi_files_location source_pyi_files_location)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "Variable ${var} is not defined")
    endif()
endforeach()

file(GLOB_RECURSE pyi_files ${generated_pyi_files_location}/*.pyi)

# perform comparison of generated files with committed ones
foreach(pyi_file IN LISTS pyi_files)
    string(REPLACE ${generated_pyi_files_location} ${source_pyi_files_location} commited_pyi_file "${pyi_file}")
    if(NOT EXISTS "${commited_pyi_file}")
        message(FATAL_ERROR "${commited_pyi_file} does not exists. Please, install pybind11-stubgen and generate .pyi files")
    else()
        execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files "${pyi_file}" "${commited_pyi_file}"
                        OUTPUT_VARIABLE output_message
                        ERROR_VARIABLE error_message
                        RESULT_VARIABLE exit_code
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT exit_code EQUAL 0)
            execute_process(COMMAND diff -u "${commited_pyi_file}" "${pyi_file}"
                            OUTPUT_VARIABLE diff_output
                            ERROR_VARIABLE diff_error
                            RESULT_VARIABLE diff_exit_code)
            message(FATAL_ERROR "File ${commited_pyi_file} is outdated and needs to be regenerated with pybind11-stubgen\nDifferences:\n${diff_output}")
        endif()
    endif()
endforeach()
