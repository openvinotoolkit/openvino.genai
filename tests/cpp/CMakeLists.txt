FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
set(TEST_TARGET_NAME "tests_continuous_batching")
file(GLOB tests_src
     "*.cpp"
)
add_executable(${TEST_TARGET_NAME} ${tests_src})
target_link_libraries(${TEST_TARGET_NAME} PUBLIC openvino::genai gtest_main)
target_include_directories(${TEST_TARGET_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/src/cpp/src")
