# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT GENAI_SAMPLES_ARCHIVE_OUTPUT_DIRECTORY)
    if (CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
        set (GENAI_SAMPLES_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/samples/")
    else()
        set (GENAI_SAMPLES_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/samples/")
    endif()
endif()

if (NOT GENAI_SAMPLES_LIBRARY_OUTPUT_DIRECTORY)
    if (CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set (GENAI_SAMPLES_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/samples/")
    else()
        set (GENAI_SAMPLES_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/samples/")
    endif()
endif()

if (NOT GENAI_SAMPLES_RUNTIME_OUTPUT_DIRECTORY)
    if (CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        set (GENAI_SAMPLES_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/samples/")
    else()
        set (GENAI_SAMPLES_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/samples/")
    endif()
endif()

function(_add_sample)
    set(options
        ADD_CLANG_FORMAT              # Enables code style checks for the target
        )
    set(oneValueRequiredArgs
        NAME # name of target
        )
    set(multiValueArgs
        INCLUDE_DIRECTORIES                      # Extra include directories
        DEPENDENCIES                # Link libraries (in form of target name or file name)
        SOURCES                       # list of sources.
        )
    cmake_parse_arguments(ARG "${options}" "${oneValueRequiredArgs}" "${multiValueArgs}" ${ARGN} )

    # sanity checks
    foreach(argName IN LISTS oneValueRequiredArgs)
        if (NOT ARG_${argName})
            message(SEND_ERROR "Argument '${argName}' is required.")
        endif()
    endforeach()
    if (ARG_UNPARSED_ARGUMENTS)
        message(SEND_ERROR "Unexpected parameters have passed to function: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    add_executable(${ARG_NAME} ${ARG_SOURCES})

    if (ARG_INCLUDE_DIRECTORIES)
        target_include_directories(${ARG_NAME} PRIVATE ${ARG_INCLUDE_DIRECTORIES})
    endif()
    if (ARG_DEPENDENCIES)
        target_link_libraries(${ARG_NAME} PRIVATE ${ARG_DEPENDENCIES})
    endif()
    if (ARG_ADD_CLANG_FORMAT)
        # code style
        ov_add_clang_format_target(${ARG_NAME}_clang FOR_TARGETS ${ARG_NAME})
    endif()

    set(SAMPLES_ROOT "${PROJECT_SOURCE_DIR}/samples")

    set (RELATIVE_SAMPLE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
    cmake_path(RELATIVE_PATH RELATIVE_SAMPLE_PATH
               BASE_DIRECTORY "${SAMPLES_ROOT}")

    set_target_properties(${ARG_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${GENAI_SAMPLES_RUNTIME_OUTPUT_DIRECTORY}/${RELATIVE_SAMPLE_PATH}"
        LIBRARY_OUTPUT_DIRECTORY "${GENAI_SAMPLES_LIBRARY_OUTPUT_DIRECTORY}/${RELATIVE_SAMPLE_PATH}"
        ARCHIVE_OUTPUT_DIRECTORY "${GENAI_SAMPLES_ARCHIVE_OUTPUT_DIRECTORY}/${RELATIVE_SAMPLE_PATH}"
    )

    set_target_properties(${ARG_NAME} PROPERTIES
        # Ensure out of box LC_RPATH on macOS with SIP
        INSTALL_RPATH_USE_LINK_PATH ON)

    install(TARGETS ${ARG_NAME}
            RUNTIME DESTINATION samples_bin/
            COMPONENT samples_bin
            EXCLUDE_FROM_ALL)

endfunction()

function(ov_add_sample)
    set(options)
    set(oneValueRequiredArgs
        NAME # name of target
        )
    set(multiValueArgs
        )
    cmake_parse_arguments(ARG "${options}" "${oneValueRequiredArgs}" "${multiValueArgs}" ${ARGN} )

    _add_sample(${ARGV})
endfunction()

if(ENABLE_SAMPLES)
    add_subdirectory(cpp/text_generation)
    add_subdirectory(cpp/image_generation)
    add_subdirectory(cpp/speech_generation)
    add_subdirectory(cpp/visual_language_chat)
    add_subdirectory(cpp/whisper_speech_recognition)
    add_subdirectory(cpp/rag)
    add_subdirectory(c/text_generation)
    add_subdirectory(c/whisper_speech_recognition)
endif()

install(FILES
        deployment-requirements.txt
        export-requirements.txt
        requirements.txt
    DESTINATION samples
    COMPONENT cpp_samples_genai)

install(DIRECTORY
            cpp/text_generation
            cpp/image_generation
            cpp/speech_generation
            cpp/visual_language_chat
            cpp/whisper_speech_recognition
            cpp/rag
        DESTINATION samples/cpp COMPONENT cpp_samples_genai)

install(DIRECTORY
            python/text_generation
            python/image_generation
            python/speech_generation
            python/visual_language_chat
            python/whisper_speech_recognition
            python/rag
        DESTINATION samples/python COMPONENT cpp_samples_genai
        USE_SOURCE_PERMISSIONS)

install(DIRECTORY
        c/text_generation
        c/whisper_speech_recognition
        DESTINATION samples/c COMPONENT cpp_samples_genai)
