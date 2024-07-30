// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <sstream>
#ifdef _WIN32
#    include <windows.h>
#    define MAX_ABS_PATH _MAX_PATH
#    define get_absolute_path(result, path) _fullpath(result, path.c_str(), MAX_ABS_PATH)
#else
#    include <dlfcn.h>
#    include <limits.h>
#    define MAX_ABS_PATH PATH_MAX
#    define get_absolute_path(result, path) realpath(path.c_str(), result)

namespace {
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
}
#endif

// These utilites are used in openvino_genai library and python
// bindings. Put in anonymous namespace to avoid exposing the utilites
// but compile them twice.
namespace {
std::string get_ov_genai_library_path() {
#ifdef _WIN32
    CHAR genai_library_path[MAX_PATH];
    HMODULE hm = NULL;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPSTR>(get_ov_genai_library_path),
                            &hm)) {
        std::stringstream ss;
        ss << "GetModuleHandle returned " << GetLastError();
        throw std::runtime_error(ss.str());
    }
    GetModuleFileNameA(hm, (LPSTR)genai_library_path, sizeof(genai_library_path));
    return std::string(genai_library_path);
#elif defined(__APPLE__) || defined(__linux__) || defined(__EMSCRIPTEN__)
    Dl_info info;
    dladdr(reinterpret_cast<void*>(get_ov_genai_library_path), &info);
    return get_absolute_file_path(info.dli_fname).c_str();
#else
#    error "Unsupported OS"
#endif  // _WIN32
}

std::filesystem::path with_openvino_tokenizers(const std::filesystem::path& path) {
#ifdef _WIN32
    constexpr char tokenizers[] = "openvino_tokenizers.dll";
#elif __linux__
    constexpr char tokenizers[] = "libopenvino_tokenizers.so";
#elif __APPLE__
    constexpr char tokenizers[] = "libopenvino_tokenizers.dylib";
#endif
    return path.parent_path() / tokenizers;
}

// Returns an absolute path. The path is this library's directory
// concatenated with openvino_tokenizers OS specific
// * name (.so, .dll, .dylib, lib prefix). This is part of the interface
// because it's reused in Python bindings.
// tokenizers_relative_to_genai() and ScopedVar allow passing a path to
// openvino_tokenizers through env var removing one argument from
// Tokenizer's constructor.
std::filesystem::path tokenizers_relative_to_genai() {
    return with_openvino_tokenizers(get_ov_genai_library_path());
}

// Sets ENVIRONMENT_VARIABLE_NAME to environment_variable_value
// and unsets in destructor. Does nothing if ENVIRONMENT_VARIABLE_NAME
// was already defined.
class ScopedVar {
public:
    bool was_already_set{false};
    static constexpr char ENVIRONMENT_VARIABLE_NAME[] = "OPENVINO_TOKENIZERS_PATH_GENAI";
    explicit ScopedVar(const std::string& environment_variable_value) {
#ifdef _WIN32
        char* value = nullptr;
        size_t len = 0;
        _dupenv_s(&value, &len, ENVIRONMENT_VARIABLE_NAME);
        if (value == nullptr)
            _putenv_s(ENVIRONMENT_VARIABLE_NAME, environment_variable_value.c_str());
#else
        if (!getenv(ENVIRONMENT_VARIABLE_NAME))
            setenv(ENVIRONMENT_VARIABLE_NAME, environment_variable_value.c_str(), 1);
#endif
        else
            was_already_set = true;
    }
    ~ScopedVar() {
        if (!was_already_set) {
#ifdef _WIN32
            _putenv_s(ENVIRONMENT_VARIABLE_NAME, "");
#else
            unsetenv(ENVIRONMENT_VARIABLE_NAME);
#endif
        }
    }
};
}
