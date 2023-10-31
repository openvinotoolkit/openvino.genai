cmake_minimum_required(VERSION 3.5)
project(OV_SD_CPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake build type")
add_compile_options(-O3 -march=native -Wall)

if (WIN32)
    message("CMake is running on Windows")
    set(BOOST_ROOT "C:/Boost")
    set(EIGEN3_INCLUDE_DIR "C:/Eigen3/eigen-3.4.0")
endif (WIN32)

find_package(OpenVINO REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(SD-generate ${PROJECT_SOURCE_DIR}/src/main.cpp)

target_include_directories(SD-generate PRIVATE 
                            ${PROJECT_BINARY_DIR}/src
                            ${OV_INCLUDE_DIR}
                            ${EIGEN3_INCLUDE_DIR})

target_link_libraries(SD-generate PRIVATE openvino::runtime Eigen3::Eigen ${OV_LIBRARY} ${Boost_LIBRARIES} )