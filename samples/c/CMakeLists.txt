cmake_minimum_required(VERSION 3.13)
project(openvino_genai_c_samples)

# Add the executable
add_executable(text2video_c_sample 
    text2video_c_sample.c
    ../../src/c/src/text2video_pipeline.cpp
)

# Link GenAI library
target_link_libraries(text2video_c_sample PRIVATE openvino_genai)

# Include Directories (This fixes the "file not found" red line!)
target_include_directories(text2video_c_sample PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/c/include"    # Finds text2video_pipeline.h (C)
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/cpp/include"  # Finds text2video_pipeline.hpp (C++)
)