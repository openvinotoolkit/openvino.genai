cmake_minimum_required(VERSION 3.20)

set(TARGET_NAME "md_comfyui")

project($TARGET_NAME)

find_package(OpenVINO REQUIRED)
find_package(OpenVINOGenAI REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(${TARGET_NAME}
  main.cpp
  ../utils/vision_utils.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${OpenCV_INCLUDE_DIRS}"
)

target_link_libraries(${TARGET_NAME} PRIVATE
  openvino::genai
  ${OpenCV_LIBS}
)

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION bin/
        COMPONENT tools)