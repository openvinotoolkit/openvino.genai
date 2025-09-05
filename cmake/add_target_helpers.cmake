# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(_add_target)
    set(options
        ADD_CLANG_FORMAT              # Enables code style checks for the target
        )
    set(oneValueRequiredArgs
        TYPE # type of target, SHARED|STATIC|EXECUTABLE. SHARED and STATIC correspond to add_library, EXECUTABLE to add_executable
        NAME # name of target
        ROOT # root directory to be used for recursive search of source files
        )
    set(multiValueArgs
        INCLUDES                      # Extra include directories
        LINK_LIBRARIES                # Link libraries (in form of target name or file name)
        DEPENDENCIES                  # compile order dependencies (no link implied)
        DEFINES                       # extra preprocessor definitions
        ADDITIONAL_SOURCE_DIRS        # list of directories which will be used to recursive search of source files in addition to ROOT
        OBJECT_FILES                  # list of object files to be additionally built into the target
        EXCLUDED_SOURCE_PATHS         # list of paths excluded from the global recursive search of source files
        LINK_FLAGS                    # list of extra commands to linker
        SOURCES                       # list of sources. If not defined, GLOB searching in ROOT will be used
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

    # adding files to target
    set(includeSearch)
    set(sourceSearch)
    if (ARG_SOURCES)
        list(APPEND sourceSearch ${ARG_SOURCES})
        foreach(directory ${ARG_ADDITIONAL_SOURCE_DIRS})
            list(APPEND includeSearch ${directory}/*.h ${directory}/*.hpp)
            list(APPEND sourceSearch  ${directory}/*.cpp)
        endforeach()
    else()
        foreach(directory ${ARG_ROOT} ${ARG_ADDITIONAL_SOURCE_DIRS})
            list(APPEND includeSearch ${directory}/*.h ${directory}/*.hpp)
            list(APPEND sourceSearch  ${directory}/*.cpp)
        endforeach()
    endif()

    file(GLOB_RECURSE includes ${includeSearch})
    file(GLOB_RECURSE sources  ${sourceSearch})

    # remove unnecessary directories
    foreach(excludedDir IN LISTS ARG_EXCLUDED_SOURCE_PATHS)
        list(FILTER includes EXCLUDE REGEX "${excludedDir}.*")
        list(FILTER sources EXCLUDE REGEX "${excludedDir}.*")
    endforeach()

    source_group("include" FILES ${includes})
    source_group("src"     FILES ${sources})

    set(all_sources ${sources} ${includes} ${ARG_OBJECT_FILES})

    # defining a target
    if (ARG_TYPE STREQUAL EXECUTABLE)
        add_executable(${ARG_NAME} ${all_sources})
    elseif(ARG_TYPE STREQUAL STATIC OR ARG_TYPE STREQUAL SHARED OR ARG_TYPE STREQUAL OBJECT)
        add_library(${ARG_NAME} ${ARG_TYPE} ${all_sources})
    else()
        message(SEND_ERROR "Invalid target type ${ARG_TYPE} specified for target name ${ARG_NAME}")
    endif()

    if (ARG_DEFINES)
        target_compile_definitions(${ARG_NAME} PRIVATE ${ARG_DEFINES})
    endif()
    if (ARG_INCLUDES)
        target_include_directories(${ARG_NAME} PRIVATE ${ARG_INCLUDES})
    endif()
    if (ARG_LINK_LIBRARIES)
        target_link_libraries(${ARG_NAME} PRIVATE ${ARG_LINK_LIBRARIES})
    endif()
    if (ARG_DEPENDENCIES)
        add_dependencies(${ARG_NAME} ${ARG_DEPENDENCIES})
    endif()
    if (ARG_LINK_FLAGS)
        get_target_property(oldLinkFlags ${ARG_NAME} LINK_FLAGS)
        string(REPLACE ";" " " ARG_LINK_FLAGS "${ARG_LINK_FLAGS}")
        set_target_properties(${ARG_NAME} PROPERTIES LINK_FLAGS "${oldLinkFlags} ${ARG_LINK_FLAGS}")
    endif()
    if (ARG_ADD_CLANG_FORMAT)
        # code style
        ov_add_clang_format_target(${ARG_NAME}_clang FOR_TARGETS ${ARG_NAME})
    endif()
endfunction()

#[[
function to create CMake target and setup its options in a declarative style. The target is built into openvino_genai directory.
Example:
genai_add_target(
   NAME core_lib
   ADD_CLANG_FORMAT
   TYPE <SHARED / STATIC / EXECUTABLE>
   ROOT ${CMAKE_CURRENT_SOURCE_DIR}
   ADDITIONAL_SOURCE_DIRS
        /some/additional/sources
   SOURCES
        /some/specific/source.cpp
   EXCLUDED_SOURCE_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/unnecessary_sources/
   INCLUDES
        ${SDL_INCLUDES}
        /some/specific/path
   LINK_LIBRARIES
        link_dependencies
   DEPENDENCIES
        dependencies
        openvino::important_plugin
   OBJECT_FILES
        object libraries
   DEFINES
        DEF1 DEF2
   LINK_FLAGS
        flag1 flag2
)
#]]
function(genai_add_target)
    set(options)
    set(oneValueRequiredArgs
        NAME # name of target
        )
    set(multiValueArgs
        )
    cmake_parse_arguments(ARG "${options}" "${oneValueRequiredArgs}" "${multiValueArgs}" ${ARGN} )

    if (NOT ARG_NAME)
        message(SEND_ERROR "Target name was not specified")
    endif()

    _add_target(${ARGV})

    set_target_properties(${ARG_NAME} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${GENAI_ARCHIVE_OUTPUT_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY "${GENAI_LIBRARY_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY "${GENAI_RUNTIME_OUTPUT_DIRECTORY}"
    )
endfunction()

#[[
function to create CMake target and setup its options in a declarative style. The target is built into tools directory.
Example:
genai_add_tool_target(
   NAME core_lib
   ADD_CLANG_FORMAT
   TYPE <SHARED / STATIC / EXECUTABLE>
   ROOT ${CMAKE_CURRENT_SOURCE_DIR}
   ADDITIONAL_SOURCE_DIRS
        /some/additional/sources
   SOURCES
        /some/specific/source.cpp
   EXCLUDED_SOURCE_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/unnecessary_sources/
   INCLUDES
        ${SDL_INCLUDES}
        /some/specific/path
   LINK_LIBRARIES
        link_dependencies
   DEPENDENCIES
        dependencies
        openvino::important_plugin
   OBJECT_FILES
        object libraries
   DEFINES
        DEF1 DEF2
   LINK_FLAGS
        flag1 flag2
)
#]]
function(genai_add_tool_target)
    set(options)
    set(oneValueRequiredArgs
        NAME # name of target
        )
    set(multiValueArgs
        )
    cmake_parse_arguments(ARG "${options}" "${oneValueRequiredArgs}" "${multiValueArgs}" ${ARGN} )

    if (NOT ARG_NAME)
        message(SEND_ERROR "Target name was not specified")
    endif()

    _add_target(${ARGV} TYPE EXECUTABLE)

    target_link_libraries(${ARG_NAME} PRIVATE openvino::genai)

    set(TOOLS_ROOT "${PROJECT_SOURCE_DIR}/tools")

    set (RELATIVE_TOOL_PATH ${CMAKE_CURRENT_SOURCE_DIR})
    cmake_path(RELATIVE_PATH RELATIVE_TOOL_PATH
               BASE_DIRECTORY "${TOOLS_ROOT}")

    set_target_properties(${ARG_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${GENAI_TOOLS_RUNTIME_OUTPUT_DIRECTORY}/${RELATIVE_TOOL_PATH}"
        LIBRARY_OUTPUT_DIRECTORY "${GENAI_TOOLS_LIBRARY_OUTPUT_DIRECTORY}/${RELATIVE_TOOL_PATH}"
        ARCHIVE_OUTPUT_DIRECTORY "${GENAI_TOOLS_ARCHIVE_OUTPUT_DIRECTORY}/${RELATIVE_TOOL_PATH}"
    )

    install(TARGETS ${ARG_NAME}
        RUNTIME DESTINATION samples_bin/
        COMPONENT tools_bin
        EXCLUDE_FROM_ALL)
endfunction()
