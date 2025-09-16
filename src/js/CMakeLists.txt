cmake_minimum_required(VERSION 3.18)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

set(dist_folder "${CMAKE_SOURCE_DIR}/bin/")

if(WIN32)
    set(CMAKE_SHARED_LINKER_FLAGS /DELAYLOAD:NODE.EXE)
    set(CMAKE_JS_LIB ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/node.lib)
    set(CMAKE_JS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/win_delay_load_hook.cc)

    set(CMAKE_JS_NODELIB_DEF ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/node-lib.def)
    set(CMAKE_JS_NODELIB_TARGET ${CMAKE_JS_LIB})
    set(DELAYIMP_LIB delayimp.lib)
endif()

project(genai_node_addon)

# Specify NAPI version 8
# supports v12.22.0+, v14.17.0+, v15.12.0+, 16.0.0 and all later Node.js versions
add_definitions(-DNAPI_VERSION=8)

include(FetchContent)

FetchContent_Declare(
    node-api-headers
    URL      https://github.com/nodejs/node-api-headers/archive/refs/tags/v1.1.0.tar.gz
    URL_HASH SHA256=70608bc1e6dddce280285f3462f18a106f687c0720a4b90893e1ecd86e5a8bbf
)
FetchContent_MakeAvailable(node-api-headers)

FetchContent_Declare(
    node-addon-api
    URL      https://github.com/nodejs/node-addon-api/archive/refs/tags/v8.0.0.tar.gz
    URL_HASH SHA256=42424c5206b9d67b41af4fcff5d6e3cb22074168035a03b8467852938a281d47
)
FetchContent_MakeAvailable(node-addon-api)

file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Create a library
add_library(${PROJECT_NAME} SHARED
    ${SOURCE_FILES}
    ${CMAKE_JS_SRC}
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
  "${node-api-headers_SOURCE_DIR}/include"
  "${node-addon-api_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

target_link_libraries(${PROJECT_NAME} PRIVATE openvino::genai ${DELAYIMP_LIB} ${CMAKE_JS_LIB})

if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET) # Generate node.lib
  execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()

if(APPLE)
    target_link_options(${PROJECT_NAME} PRIVATE -Wl,-undefined,suppress,-flat_namespace)
elseif(AARCH64 OR ARM)
    target_link_options(${PROJECT_NAME} PRIVATE -Wl,--unresolved-symbols=ignore-all)
endif()

# Set library properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".node"
)

# setting RPATH / LC_RPATH depending on platform
if(LINUX)
    # to find libopenvino_genai.so in the same folder
    set(rpaths "$ORIGIN")
    # to find libopenvino.so when installing from npm
    list(APPEND rpaths "$ORIGIN/../../openvino-node/bin")
    # to find libopenvino.so when installing from source
    list(APPEND rpaths "$ORIGIN/../node_modules/openvino-node/bin")
elseif(APPLE)
    # to find libopenvino_genai.dylib in the same folder
    set(rpaths "@loader_path")
    # to find libopenvino.dylib when installing from npm
    list(APPEND rpaths "@loader_path/../../openvino-node/bin")
    # to find libopenvino.dylib when installing from source
    list(APPEND rpaths "@loader_path/../node_modules/openvino-node/bin")
endif()

if(rpaths)
    set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "${rpaths}")
endif()

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION . COMPONENT ${PROJECT_NAME}
    RUNTIME DESTINATION . COMPONENT ${PROJECT_NAME}
)
