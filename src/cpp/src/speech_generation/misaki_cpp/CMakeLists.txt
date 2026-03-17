cmake_minimum_required(VERSION 3.20)
project(misaki_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MISAKI_CPP_BUILD_PYTHON_BINDINGS "Build Python bindings for misaki_cpp" OFF)

set(_MISAKI_DEFAULT_LEXICON_DIR "")
if(EXISTS "${CMAKE_SOURCE_DIR}/data/us_gold.json" AND
   EXISTS "${CMAKE_SOURCE_DIR}/data/us_silver.json" AND
   EXISTS "${CMAKE_SOURCE_DIR}/data/gb_gold.json" AND
   EXISTS "${CMAKE_SOURCE_DIR}/data/gb_silver.json")
  set(_MISAKI_DEFAULT_LEXICON_DIR "${CMAKE_SOURCE_DIR}/data")
endif()

set(MISAKI_ENGLISH_LEXICON_DATA_DIR "${_MISAKI_DEFAULT_LEXICON_DIR}" CACHE PATH
    "Directory containing required Misaki English lexicon files (us/gb gold/silver JSON)")

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(misaki_cpp STATIC
  src/en_g2p.cpp
  src/espeak_fallback.cpp
  src/espeak_g2p.cpp
  src/english_lexicon.cpp
  src/factory_stub.cpp
)
target_include_directories(misaki_cpp
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(misaki_cpp PRIVATE
  $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
)
target_compile_definitions(misaki_cpp PRIVATE
  MISAKI_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
add_library(MisakiCpp::misaki_cpp ALIAS misaki_cpp)


add_executable(sample_basic_usage samples/basic_usage.cpp)
target_link_libraries(sample_basic_usage PRIVATE misaki_cpp)

add_executable(sample_fallback_and_unknown samples/fallback_and_unknown.cpp)
target_link_libraries(sample_fallback_and_unknown PRIVATE misaki_cpp)

if(MISAKI_CPP_BUILD_PYTHON_BINDINGS)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)

  pybind11_add_module(misaki_cpp_py python/misaki_cpp_py.cpp)
  target_link_libraries(misaki_cpp_py PRIVATE misaki_cpp)
endif()

install(
  TARGETS misaki_cpp
  EXPORT MisakiCppTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  FILES
    data/us_gold.json
    data/us_silver.json
    data/gb_gold.json
    data/gb_silver.json
  DESTINATION ${CMAKE_INSTALL_DATADIR}/misaki/data
)

install(
  EXPORT MisakiCppTargets
  NAMESPACE MisakiCpp::
  FILE MisakiCppTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MisakiCpp
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MisakiCppConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/MisakiCppConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MisakiCpp
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/MisakiCppConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MisakiCppConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MisakiCppConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MisakiCpp
)
