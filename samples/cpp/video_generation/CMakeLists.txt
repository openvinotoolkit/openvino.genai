# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alongside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

file(DOWNLOAD https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image_write.h ${CMAKE_BINARY_DIR}/stb_image_write.h
     EXPECTED_HASH MD5=845b8b43d7d941890a57a477455558ad)

include(FetchContent)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(indicators
    URL https://github.com/p-ranav/indicators/archive/refs/tags/v2.3.tar.gz
    URL_HASH SHA256=70da7a693ff7a6a283850ab6d62acf628eea17d386488af8918576d0760aef7b)
FetchContent_MakeAvailable(indicators)

# OpenCV is required for this sample
find_package(OpenCV QUIET COMPONENTS core imgproc videoio imgcodecs)

# create main sample executable
add_executable(text2video text2video.cpp)
target_sources(text2video PRIVATE imwrite_video.cpp)

target_include_directories(text2video PRIVATE ${CMAKE_BINARY_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/../image_generation/" "${CMAKE_CURRENT_SOURCE_DIR}/../../../src/cpp/src/")

if(NOT OpenCV_FOUND)
    set(BUILD_SHARED_LIBS ON)
    set(BUILD_SHARED_LIBS ON)
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
    set(BUILD_opencv_python_tests OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_stitching OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_ts OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(opencv
        GIT_REPOSITORY https://github.com/opencv/opencv.git
        GIT_TAG 4.11.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(opencv)

    set(TEXT2VIDEO_OPENCV_TARGETS opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs)
else()
    set(TEXT2VIDEO_OPENCV_TARGETS ${OpenCV_LIBS})
endif()

if(OpenCV_FOUND)
    target_include_directories(text2video PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

target_link_libraries(text2video PRIVATE openvino::genai indicators::indicators ${TEXT2VIDEO_OPENCV_TARGETS})

if(UNIX AND NOT APPLE)
    set_target_properties(text2video PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib"
        INSTALL_RPATH_USE_LINK_PATH ON
    )
elseif(APPLE)
    set_target_properties(text2video PROPERTIES
        INSTALL_RPATH "@loader_path/../lib"
        INSTALL_RPATH_USE_LINK_PATH ON
    )
endif()

set_target_properties(text2video PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS text2video
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
