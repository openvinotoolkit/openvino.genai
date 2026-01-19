cmake_minimum_required(VERSION 3.20)

set(TARGET_NAME "module_genai_ut_app")

project($TARGET_NAME)

find_package(OpenVINO REQUIRED)
find_package(OpenVINOGenAI REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "OpenCV found. Version: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found! Check your installation and environment.")
endif()

file(GLOB SRC_FILES CONFIGURE_DEPENDS *.cpp ut_modules/*.cpp)
# Exclude module_pipeline_comfyui.cpp as it has its own main function
list(FILTER SRC_FILES EXCLUDE REGEX ".*module_pipeline_comfyui\\.cpp$")
add_executable(${TARGET_NAME} ${SRC_FILES})

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOUCE_DIR}" "${CMAKE_BINARY_DIR}" "${OpenCV_INCLUDE_DIRS}")

target_link_libraries(${TARGET_NAME} PRIVATE openvino::genai ${OpenCV_LIBS} ${YAML_CPP_LIBRARIES})

# ============================================================================
# MODULE PIPELINE COMFYUI - Standalone tool for running pipelines
# ============================================================================
# Supports both ComfyUI JSON and YAML config input
set(MODULE_PIPELINE_COMFYUI_TARGET "module_pipeline_comfyui")
add_executable(${MODULE_PIPELINE_COMFYUI_TARGET}
  ut_pipelines/comfyui/module_pipeline_comfyui.cpp
  utils_image.cpp
)
target_include_directories(${MODULE_PIPELINE_COMFYUI_TARGET} PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${OpenCV_INCLUDE_DIRS}"
)
target_link_libraries(${MODULE_PIPELINE_COMFYUI_TARGET} PRIVATE
  openvino::genai
  ${OpenCV_LIBS}
)

install(TARGETS ${MODULE_PIPELINE_COMFYUI_TARGET}
        RUNTIME DESTINATION bin/
        COMPONENT tools)
