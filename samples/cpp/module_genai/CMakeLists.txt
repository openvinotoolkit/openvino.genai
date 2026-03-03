cmake_minimum_required(VERSION 3.20)

set(TARGET_NAME "module_genai_ut_app")

project($TARGET_NAME)

find_package(OpenVINO REQUIRED)
find_package(OpenVINOGenAI REQUIRED)
find_package(OpenCV REQUIRED)

# yaml-cpp dependency (use centralized cmake module or fetch if standalone)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/yaml-cpp.cmake)

if(OpenCV_FOUND)
    message(STATUS "OpenCV found. Version: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found! Check your installation and environment.")
endif()

function(add_sample_executable target_name)
    add_executable(${target_name}
        ${target_name}.cpp
        utils/utils.cpp
        utils/vision_utils.cpp
    )
    target_link_libraries(${target_name} PRIVATE openvino::genai ${OpenCV_LIBS} ${YAML_CPP_TARGET})
    set_target_properties(${target_name} PROPERTIES
        # Ensure out-of-box LC_RPATH on macOS with SIP
        INSTALL_RPATH_USE_LINK_PATH ON)
    install(TARGETS ${target_name}
            RUNTIME DESTINATION samples_bin/
            COMPONENT samples_bin
            EXCLUDE_FROM_ALL)
endfunction()

set (SAMPLE_LIST
    md_image_generation
    md_video_generation
    md_visual_language_chat)

foreach(sample IN LISTS SAMPLE_LIST)
    add_sample_executable(${sample})
endforeach()

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

add_subdirectory(comfyui)
