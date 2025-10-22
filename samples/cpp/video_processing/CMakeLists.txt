# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(OpenVINOGenAI REQUIRED
    PATHS
        "${CMAKE_BINARY_DIR}"  # Reuse the package from the build.
        ${OpenVINO_DIR}  # GenAI may be installed alongside OpenVINO.
    NO_CMAKE_FIND_ROOT_PATH
)

# Only build if FFmpeg and VPL support is enabled
if(ENABLE_FFMPEG_VPL)
    # Find FFmpeg
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
        libavcodec
        libavformat
        libavutil
    )
    
    # Find VPL
    pkg_check_modules(VPL REQUIRED IMPORTED_TARGET vpl)
    
    add_executable(video_processing_sample video_processing_sample.cpp)
    target_link_libraries(video_processing_sample PRIVATE 
        openvino::genai
        PkgConfig::LIBAV
        PkgConfig::VPL
    )
    
    set_target_properties(video_processing_sample PROPERTIES
        # Ensure out of box LC_RPATH on macOS with SIP
        INSTALL_RPATH_USE_LINK_PATH ON)
    target_compile_features(video_processing_sample PRIVATE cxx_std_17)
    
    install(TARGETS video_processing_sample
            RUNTIME DESTINATION samples_bin/
            COMPONENT samples_bin
            EXCLUDE_FROM_ALL)
else()
    message(STATUS "Skipping video_processing_sample - ENABLE_FFMPEG_VPL is not enabled")
endif()
