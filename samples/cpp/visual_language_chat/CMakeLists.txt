# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if (MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alogside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

file(DOWNLOAD
    https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image.h
    ${CMAKE_BINARY_DIR}/stb_image.h
    EXPECTED_HASH MD5=27932e6fb3a2f26aee2fc33f2cb4e696)

# create main sample executable

add_executable(visual_language_chat visual_language_chat.cpp load_image.cpp)
target_include_directories(visual_language_chat PRIVATE "${CMAKE_BINARY_DIR}")
target_link_libraries(visual_language_chat PRIVATE openvino::genai)

set_target_properties(visual_language_chat PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS visual_language_chat
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

# create encrypted model sample executable

add_executable(encrypted_model_vlm encrypted_model_vlm.cpp load_image.cpp)
target_include_directories(encrypted_model_vlm PRIVATE "${CMAKE_BINARY_DIR}")
target_link_libraries(encrypted_model_vlm PRIVATE openvino::genai)

set_target_properties(encrypted_model_vlm PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS encrypted_model_vlm
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

# create benchmark executable
add_executable(benchmark_vlm benchmark_vlm.cpp load_image.cpp ../text_generation/read_prompt_from_file.cpp)
target_include_directories(benchmark_vlm PRIVATE "${CMAKE_BINARY_DIR}")
target_link_libraries(benchmark_vlm PRIVATE openvino::genai cxxopts::cxxopts)
set_target_properties(benchmark_vlm PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS benchmark_vlm
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)

find_package(OpenCV QUIET)

add_executable(video_to_text_chat video_to_text_chat.cpp)

if(NOT OpenCV_FOUND)
    include(FetchContent)

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


    FetchContent_Declare(
            opencv
            GIT_REPOSITORY https://github.com/opencv/opencv.git
            GIT_TAG 4.11.0
            GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(opencv)

    target_include_directories(video_to_text_chat PRIVATE
            ${OPENCV_CONFIG_FILE_INCLUDE_DIR}
            ${OPENCV_MODULE_opencv_core_LOCATION}/include
            ${OPENCV_MODULE_opencv_videoio_LOCATION}/include
            )


    target_link_libraries(video_to_text_chat opencv_core opencv_imgcodecs opencv_imgproc opencv_videoio openvino::genai cxxopts::cxxopts)

    if(LINUX)
        set_target_properties(video_to_text_chat opencv_core opencv_imgcodecs opencv_imgproc opencv_videoio PROPERTIES
            INSTALL_RPATH "$ORIGIN/../lib"
            # Ensure out of box LC_RPATH on macOS with SIP
            INSTALL_RPATH_USE_LINK_PATH ON)
    elseif(APPLE)
        set_target_properties(video_to_text_chat opencv_core opencv_imgcodecs opencv_imgproc opencv_videoio PROPERTIES
            INSTALL_RPATH "@loader_path/../lib"
            # Ensure out of box LC_RPATH on macOS with SIP
            INSTALL_RPATH_USE_LINK_PATH ON)
    endif()
else()
    set(OPENCV_LIBS ${OpenCV_LIBS})

    target_include_directories(video_to_text_chat PRIVATE ${OpenCV_INCLUDE_DIRS} "${CMAKE_BINARY_DIR}")
    target_link_libraries(video_to_text_chat PRIVATE ${OpenCV_LIBS} openvino::genai cxxopts::cxxopts)
endif()

install(TARGETS video_to_text_chat
        RUNTIME DESTINATION samples_bin/
        COMPONENT samples_bin
        EXCLUDE_FROM_ALL)
