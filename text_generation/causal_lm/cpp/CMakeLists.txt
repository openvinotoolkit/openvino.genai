# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(causal_lm)

add_subdirectory(../../../thirdparty/openvino_tokenizers/ "${CMAKE_CURRENT_BINARY_DIR}/openvino_tokenizers/")

add_executable(greedy_causal_lm greedy_causal_lm.cpp)
target_compile_definitions(greedy_causal_lm PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(greedy_causal_lm PRIVATE openvino::runtime)
set_target_properties(greedy_causal_lm PROPERTIES CXX_STANDARD 17)
set_target_properties(greedy_causal_lm PROPERTIES CXX_STANDARD_REQUIRED ON)

add_executable(beam_search_causal_lm beam_search_causal_lm.cpp)
target_compile_definitions(beam_search_causal_lm PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_include_directories(beam_search_causal_lm PRIVATE ./)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(beam_search_causal_lm PRIVATE openvino::runtime)
set_target_properties(beam_search_causal_lm PROPERTIES CXX_STANDARD 17)
set_target_properties(beam_search_causal_lm PROPERTIES CXX_STANDARD_REQUIRED ON)


get_filename_component(PACKAGE_PREFIX_DIR "${INTEL_OPENVINO_DIR}" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()


# we have to use our own version of find_dependency because of support cmake 3.7
macro(_genai_find_dependency dep)
    set(cmake_fd_quiet_arg)
    if(${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
        set(cmake_fd_quiet_arg QUIET)
    endif()
    set(cmake_fd_required_arg)
    if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
        set(cmake_fd_required_arg REQUIRED)
    endif()

    get_property(cmake_fd_alreadyTransitive GLOBAL PROPERTY
        _CMAKE_${dep}_TRANSITIVE_DEPENDENCY)

    find_package(${dep} ${ARGN}
        ${cmake_fd_quiet_arg}
        ${cmake_fd_required_arg})

    if(NOT DEFINED cmake_fd_alreadyTransitive OR cmake_fd_alreadyTransitive)
        set_property(GLOBAL PROPERTY _CMAKE_${dep}_TRANSITIVE_DEPENDENCY TRUE)
    endif()

    if(NOT ${dep}_FOUND)
        set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE "${CMAKE_FIND_PACKAGE_NAME} could not be found because dependency ${dep} could not be found.")
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND False)
        return()
    endif()

    unset(cmake_fd_required_arg)
    unset(cmake_fd_quiet_arg)
endmacro()

macro(_genai_find_tbb)
    set(THREADING "TBB")
    if(THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO")
        set(enable_pkgconfig_tbb "")

        # try tbb.pc
        if(enable_pkgconfig_tbb AND NOT ANDROID)
            _genai_find_dependency(PkgConfig)
            if(PkgConfig_FOUND)
                if(${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
                    set(pkg_config_quiet_arg QUIET)
                endif()
                if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
                    set(pkg_config_required_arg REQUIRED)
                endif()

                pkg_search_module(tbb
                                  ${pkg_config_quiet_arg}
                                  ${pkg_config_required_arg}
                                  IMPORTED_TARGET
                                  tbb)
                unset(pkg_config_quiet_arg)
                unset(pkg_config_required_arg)

                if(tbb_FOUND)
                    if(TARGET PkgConfig::tbb)
                        set(TBB_VERSION ${tbb_VERSION})
                        set(TBB_FOUND ${tbb_FOUND})
                        unset(tbb_FOUND)
                        unset(tbb_VERSION)
                    elseif(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
                        message(FATAL_ERROR "cmake v${CMAKE_VERSION} contains bug in function 'pkg_search_module', need to update to at least v3.16.0 version")
                    endif()
                endif()
            endif()
        else()
            # try cmake TBB interface

            set(enable_system_tbb "ON")
            if(NOT enable_system_tbb)
                set_and_check(_tbb_dir "${PACKAGE_PREFIX_DIR}/")

                # see https://stackoverflow.com/questions/28070810/cmake-generate-error-on-windows-as-it-uses-as-escape-seq
                if(DEFINED ENV{TBBROOT})
                    file(TO_CMAKE_PATH $ENV{TBBROOT} ENV_TBBROOT)
                endif()
                if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
                    set(_no_cmake_install_prefix NO_CMAKE_INSTALL_PREFIX)
                endif()

                set(find_package_tbb_extra_args
                    CONFIG
                    PATHS
                        # oneTBB case exposed via export TBBROOT=<custom TBB root>
                        "${ENV_TBBROOT}/lib64/cmake/TBB"
                        "${ENV_TBBROOT}/lib/cmake/TBB"
                        "${ENV_TBBROOT}/lib/cmake/tbb"
                        # for custom TBB exposed via cmake -DTBBROOT=<custom TBB root>
                        "${TBBROOT}/cmake"
                        # _tbb_dir points to TBB_DIR (custom | temp | system) used to build OpenVINO
                        ${_tbb_dir}
                    CMAKE_FIND_ROOT_PATH_BOTH
                    NO_PACKAGE_ROOT_PATH
                    NO_SYSTEM_ENVIRONMENT_PATH
                    NO_CMAKE_PACKAGE_REGISTRY
                    NO_CMAKE_SYSTEM_PATH
                    ${_no_cmake_install_prefix}
                    NO_CMAKE_SYSTEM_PACKAGE_REGISTRY)
                unset(_tbb_dir)
                unset(_no_cmake_install_prefix)
            endif()
            unset(enable_system_tbb)

            _genai_find_dependency(TBB
                                COMPONENTS tbb tbbmalloc
                                ${find_package_tbb_extra_args})
            unset(find_package_tbb_extra_args)
        endif()
        unset(enable_pkgconfig_tbb)

        set(install_tbbbind "")
        if(install_tbbbind)
            set_and_check(_tbb_bind_dir "")
            _genai_find_dependency(TBBBIND_2_5
                                PATHS ${_tbb_bind_dir}
                                NO_CMAKE_FIND_ROOT_PATH
                                NO_DEFAULT_PATH)
            if(TARGET TBBbind::tbbbind_2_5)
                # To solve https://cmake.org/cmake/help/latest/policy/CMP0111.html warnings
                set_property(TARGET TBBbind::tbbbind_2_5 PROPERTY IMPORTED_CONFIGURATIONS RELEASE DEBUG)
            endif()
            unset(_tbb_bind_dir)
        endif()
        unset(install_tbbbind)
    endif()
endmacro()

add_executable(speculative_decoding_lm speculative_decoding_lm.cpp)
target_compile_definitions(speculative_decoding_lm PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_include_directories(speculative_decoding_lm PRIVATE ./)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(speculative_decoding_lm PRIVATE openvino::runtime)
set_target_properties(speculative_decoding_lm PROPERTIES CXX_STANDARD 17)
set_target_properties(speculative_decoding_lm PROPERTIES CXX_STANDARD_REQUIRED ON)
_genai_find_tbb()
target_link_libraries(speculative_decoding_lm PRIVATE TBB::tbb)

add_executable(prompt_lookup_decoding_lm prompt_lookup_decoding_lm.cpp)
target_compile_definitions(prompt_lookup_decoding_lm PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_include_directories(prompt_lookup_decoding_lm PRIVATE ./)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(prompt_lookup_decoding_lm PRIVATE openvino::runtime)
set_target_properties(prompt_lookup_decoding_lm PROPERTIES CXX_STANDARD 17)
set_target_properties(prompt_lookup_decoding_lm PROPERTIES CXX_STANDARD_REQUIRED ON)
_genai_find_tbb()
target_link_libraries(prompt_lookup_decoding_lm PRIVATE TBB::tbb)
