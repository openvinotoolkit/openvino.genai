# Add the Text-to-Video C API Test
ov_add_test(
    NAME text2video_c_api_test
    SOURCES text2video_test.cpp
    LIBRARIES openvino_genai   
    LABELS "genai"
)