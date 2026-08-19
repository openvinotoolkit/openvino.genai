# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

function(ov_genai_link_opencv target_name)
    set(required_components ${ARGN})
    if(NOT required_components)
        set(required_components core imgproc videoio imgcodecs)
    endif()

    # A static OpenCV is preferred. On Windows OpenCVConfig.cmake selects staticlib/ or
    # lib/ depending on OpenCV_STATIC, so probe the static layout first and fall back to
    # the shared one. On other platforms OpenCV_STATIC has no effect on the search.
    set(OpenCV_STATIC ON)
    find_package(OpenCV QUIET COMPONENTS ${required_components})

    if(NOT OpenCV_FOUND)
        # No OpenCV available: build a static one from source.
        include(FetchContent)

        if(POLICY CMP0135)
            cmake_policy(SET CMP0135 NEW)
        endif()

        set(BUILD_SHARED_LIBS OFF)
        set(BUILD_WITH_STATIC_CRT OFF CACHE BOOL "" FORCE)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        set(OPENCV_ENABLE_PLUGINS OFF CACHE BOOL "" FORCE)
        set(WITH_FFMPEG ON)
        set(WITH_PROTOBUF OFF CACHE BOOL "" FORCE)
        set(WITH_GSTREAMER OFF CACHE BOOL "" FORCE)
        set(WITH_OPENCLAMDBLAS OFF CACHE BOOL "" FORCE)
        set(WITH_OPENCLAMDFFT OFF CACHE BOOL "" FORCE)
        set(WITH_MATLAB OFF CACHE BOOL "" FORCE)
        set(HIGHGUI_ENABLE_PLUGINS OFF CACHE BOOL "" FORCE)
        set(BUILD_JAVA OFF CACHE BOOL "" FORCE)
        set(OPENCV_GAPI_GSTREAMER OFF CACHE BOOL "" FORCE)
        set(INSTALL_TESTS OFF CACHE BOOL "" FORCE)
        set(INSTALL_C_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(INSTALL_PYTHON_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
        set(BUILD_ANDROID_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(BUILD_ITT OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_java_bindings_generator OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_calib3d OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_dnn OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_features2d OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_flann OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_gapi OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_highgui OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_ml OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_objdetect OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_photo OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_python2 OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_python_tests OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_stitching OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_ts OFF CACHE BOOL "" FORCE)

        FetchContent_Declare(opencv
            GIT_REPOSITORY https://github.com/opencv/opencv.git
            GIT_TAG 4.14.0
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(opencv)

        set(opencv_targets)
        foreach(component IN LISTS required_components)
            list(APPEND opencv_targets opencv_${component})
        endforeach()

        target_include_directories(${target_name} PRIVATE ${OPENCV_CONFIG_FILE_INCLUDE_DIR})
        foreach(component IN LISTS required_components)
            target_include_directories(${target_name} PRIVATE
                ${OPENCV_MODULE_opencv_${component}_LOCATION}/include)
        endforeach()
    else()
        set(opencv_targets ${OpenCV_LIBS})
        if(OpenCV_INCLUDE_DIRS)
            target_include_directories(${target_name} PRIVATE ${OpenCV_INCLUDE_DIRS})
        endif()

        if(OpenCV_SHARED)
            message(STATUS "${target_name}: linking against a shared OpenCV from ${OpenCV_DIR}")
            # The OpenCV shared libraries must be locatable at runtime. Samples are
            # installed to samples_bin/ next to the lib/ folder of the package.
            if(LINUX)
                set_target_properties(${target_name} PROPERTIES
                    INSTALL_RPATH "$ORIGIN/../lib"
                    INSTALL_RPATH_USE_LINK_PATH ON)
            elseif(APPLE)
                set_target_properties(${target_name} PROPERTIES
                    INSTALL_RPATH "@loader_path/../lib"
                    INSTALL_RPATH_USE_LINK_PATH ON)
            endif()
        else()
            message(STATUS "${target_name}: linking against a static OpenCV from ${OpenCV_DIR}")
        endif()
    endif()

    target_link_libraries(${target_name} PRIVATE ${opencv_targets})
endfunction()
