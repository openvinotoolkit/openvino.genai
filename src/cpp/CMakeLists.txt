# Generate Pipeline library

set(JINJA2CPP_DEPS_MODE internal)

add_subdirectory(../../thirdparty/nlohmann_json/ "${CMAKE_CURRENT_BINARY_DIR}/nlohmann_json/")

# todo: remove hardcodes and make submodule work
# include_directories($ENV{HOME}/opt/jinja2cpp/include)
# add_subdirectory(../../../thirdparty/Jinja2Cpp/ "${CMAKE_CURRENT_BINARY_DIR}/Jinja2Cpp/")
# include_directories(../../../thirdparty/inja/include/Jinja2Cpp)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

set(TARGET_NAME generate_pipeline_lib)
file(GLOB SOURCE_FILES "src/*.cpp")
add_library(${TARGET_NAME} SHARED ${SOURCE_FILES})
target_include_directories(${TARGET_NAME} PRIVATE ../../text_generation/causal_lm/cpp/)
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime)
target_link_libraries(${TARGET_NAME} PUBLIC nlohmann_json::nlohmann_json)
target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
# target_link_libraries(${TARGET_NAME} PRIVATE $ENV{HOME}/opt/jinja2cpp/lib/static/libjinja2cpp.a)  # todo: remove hardcode
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)
