cmake_minimum_required(VERSION 3.20)
project(misaki_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.6.0
)
FetchContent_MakeAvailable(catch2)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(misaki_cpp STATIC
  src/en_g2p.cpp
  src/espeak_fallback.cpp
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

add_executable(test_english tests/test_english.cpp)
target_link_libraries(test_english PRIVATE
  misaki_cpp
  Catch2::Catch2WithMain
  nlohmann_json::nlohmann_json
)

add_executable(test_english_exactness tests/test_english_exactness.cpp)
target_link_libraries(test_english_exactness PRIVATE
  misaki_cpp
  Catch2::Catch2WithMain
  nlohmann_json::nlohmann_json
)
target_compile_definitions(test_english_exactness PRIVATE
  MISAKI_ENGLISH_GOLDEN_CASES_PATH="${CMAKE_SOURCE_DIR}/tests/english_golden_cases.jsonl"
)

add_executable(test_english_gb_exactness tests/test_english_gb_exactness.cpp)
target_link_libraries(test_english_gb_exactness PRIVATE
  misaki_cpp
  Catch2::Catch2WithMain
  nlohmann_json::nlohmann_json
)
target_compile_definitions(test_english_gb_exactness PRIVATE
  MISAKI_ENGLISH_GB_GOLDEN_CASES_PATH="${CMAKE_SOURCE_DIR}/tests/english_gb_golden_cases.jsonl"
)


add_executable(sample_basic_usage samples/basic_usage.cpp)
target_link_libraries(sample_basic_usage PRIVATE misaki_cpp)

add_executable(sample_fallback_and_unknown samples/fallback_and_unknown.cpp)
target_link_libraries(sample_fallback_and_unknown PRIVATE misaki_cpp)

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

include(CTest)
include(Catch)
catch_discover_tests(test_english)
catch_discover_tests(test_english_exactness)
catch_discover_tests(test_english_gb_exactness)
