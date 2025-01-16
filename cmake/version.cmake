# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(Git QUIET)

function(ov_genai_branch_name VAR)
    if(GIT_FOUND)
        execute_process(
                COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
                WORKING_DIRECTORY ${OpenVINOGenAI_SOURCE_DIR}
                OUTPUT_VARIABLE GIT_BRANCH
                RESULT_VARIABLE EXIT_CODE
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(EXIT_CODE EQUAL 0)
            set(${VAR} ${GIT_BRANCH} PARENT_SCOPE)
        endif()
    endif()
endfunction()

function(ov_genai_commit_hash VAR)
    if(GIT_FOUND)
        execute_process(
                COMMAND ${GIT_EXECUTABLE} rev-parse --short=11 HEAD
                WORKING_DIRECTORY ${OpenVINOGenAI_SOURCE_DIR}
                OUTPUT_VARIABLE GIT_COMMIT_HASH
                RESULT_VARIABLE EXIT_CODE
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(EXIT_CODE EQUAL 0)
            set(${VAR} ${GIT_COMMIT_HASH} PARENT_SCOPE)
        endif()
    endif()
endfunction()

function(ov_genai_commit_number VAR)
    set(GIT_COMMIT_NUMBER_FOUND OFF)
    if(GIT_FOUND)
        execute_process(
                COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
                WORKING_DIRECTORY ${OpenVINOGenAI_SOURCE_DIR}
                OUTPUT_VARIABLE GIT_COMMIT_NUMBER
                RESULT_VARIABLE EXIT_CODE
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(EXIT_CODE EQUAL 0)
            set(GIT_COMMIT_NUMBER_FOUND ON)
            set(${VAR} ${GIT_COMMIT_NUMBER} PARENT_SCOPE)
        endif()
    endif()
    if(NOT GIT_COMMIT_NUMBER_FOUND)
        # set zeros since git is not available
        set(${VAR} "000" PARENT_SCOPE)
    endif()
endfunction()

function(ov_genai_full_version full_version)
    if(GIT_FOUND)
        ov_genai_branch_name(GIT_BRANCH)
        ov_genai_commit_hash(GIT_COMMIT_HASH)
        ov_genai_commit_number(GIT_COMMIT_NUMBER)

        if(NOT GIT_BRANCH MATCHES "^(master|HEAD)$")
            set(GIT_BRANCH_POSTFIX "-${GIT_BRANCH}")
        endif()

        set(${full_version} "${OpenVINOGenAI_VERSION}-${GIT_COMMIT_NUMBER}-${GIT_COMMIT_HASH}${GIT_BRANCH_POSTFIX}" PARENT_SCOPE)
    else()
        set(${full_version} "${OpenVINOGenAI_VERSION}" PARENT_SCOPE)
    endif()
endfunction()

ov_genai_full_version(OpenVINOGenAI_FULL_VERSION)
message(STATUS "OpenVINO GenAI full version: ${OpenVINOGenAI_FULL_VERSION}")
