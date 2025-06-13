// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer/tokenizers_path.hpp"

#include <sstream>

#ifdef _WIN32
#    include <windows.h>
#    define MAX_ABS_PATH _MAX_PATH
#    define get_absolute_path(result, path) _fullpath(result, path.c_str(), MAX_ABS_PATH)
#else
#    include <dlfcn.h>
#    include <limits.h>
#    include <string.h>
#    define MAX_ABS_PATH PATH_MAX
#    define get_absolute_path(result, path) realpath(path.c_str(), result)
#endif

namespace {
#ifndef _WIN32
std::string get_absolute_file_path(const std::string& path) {
    std::string absolutePath;
    absolutePath.resize(MAX_ABS_PATH);
    std::ignore = get_absolute_path(&absolutePath[0], path);
    if (!absolutePath.empty()) {
        // on Linux if file does not exist or no access, function will return NULL, but
        // `absolutePath` will contain resolved path
        absolutePath.resize(absolutePath.find('\0'));
        return std::string(absolutePath);
    }
    std::stringstream ss;
    ss << "Can't get absolute file path for [" << path << "], err = " << strerror(errno);
    throw std::runtime_error(ss.str());
}
#endif

std::filesystem::path get_ov_genai_library_path() {
#ifdef _WIN32
    WCHAR genai_library_path_w[MAX_PATH];
    HMODULE hm = NULL;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(get_ov_genai_library_path),
                            &hm)) {
        std::stringstream ss;
        ss << "GetModuleHandleExW returned " << GetLastError();
        throw std::runtime_error(ss.str());
    }
    DWORD result = GetModuleFileNameW(hm, (LPWSTR)genai_library_path_w, sizeof(genai_library_path_w) / sizeof(genai_library_path_w[0]));
    if (result == 0) {
        std::stringstream ss;
        ss << "GetModuleFileNameW failed with error " << GetLastError();
        throw std::runtime_error(ss.str());
    }
    return std::filesystem::path(std::wstring(genai_library_path_w));
#elif defined(__APPLE__) || defined(__linux__) || defined(__EMSCRIPTEN__)
    Dl_info info;
    dladdr(reinterpret_cast<void*>(get_ov_genai_library_path), &info);
    return std::filesystem::path(get_absolute_file_path(info.dli_fname));
#else
#    error "Unsupported OS"
#endif  // _WIN32
}

std::filesystem::path with_openvino_tokenizers(const std::filesystem::path& path) {
#if !defined(NDEBUG) && (defined(__APPLE__) || defined(_WIN32))
# define LIB_POSTFIX "d"
#else
# define LIB_POSTFIX ""
#endif
#ifdef _WIN32
    constexpr char tokenizers[] = "openvino_tokenizers" LIB_POSTFIX ".dll";
#elif defined(__linux__)
    constexpr char tokenizers[] = "libopenvino_tokenizers" LIB_POSTFIX ".so";
#elif defined(__APPLE__)
    constexpr char tokenizers[] = "libopenvino_tokenizers" LIB_POSTFIX ".dylib";
#else
#    error "Unsupported OS"
#endif
    return path.parent_path() / tokenizers;
}
}

std::filesystem::path tokenizers_relative_to_genai() {
    return with_openvino_tokenizers(get_ov_genai_library_path());
}
