cmake_minimum_required(VERSION 3.20)
project(misaki_cpp_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MisakiCpp CONFIG REQUIRED)

add_executable(consumer_example main.cpp)
target_link_libraries(consumer_example PRIVATE MisakiCpp::misaki_cpp)
