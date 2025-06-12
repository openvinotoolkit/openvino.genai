// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <iostream>

#include "openvino/genai/visibility.hpp"

// Returns an absolute path. The path is this library's directory
// concatenated with openvino_tokenizers OS specific
// * name (.so, .dll, .dylib, lib prefix). This is part of the interface
// because it's reused in Python bindings.
// tokenizers_relative_to_genai() and ScopedVar allow passing a path to
// openvino_tokenizers through env var removing one argument from
// Tokenizer's constructor.
OPENVINO_GENAI_EXPORTS
std::filesystem::path tokenizers_relative_to_genai();

namespace {
// Sets ENVIRONMENT_VARIABLE_NAME to environment_variable_value
// and unsets in destructor. Does nothing if ENVIRONMENT_VARIABLE_NAME
// was already defined.
class ScopedVar {
    bool was_already_set{false};
public:
    static constexpr char ENVIRONMENT_VARIABLE_NAME[] = "OPENVINO_TOKENIZERS_PATH_GENAI";

    explicit ScopedVar(const std::filesystem::path& environment_variable_value) {
        std::cout << "[DEBUG] ScopedVar constructor called" << std::endl;

#ifdef _WIN32
        std::cout << "[DEBUG] ScopedVar: Using Windows Unicode APIs" << std::endl;
        std::wstring env_var_name(std::string(ENVIRONMENT_VARIABLE_NAME).begin(), std::string(ENVIRONMENT_VARIABLE_NAME).end());

        wchar_t* value = nullptr;
        size_t len = 0;
        std::cout << "[DEBUG] ScopedVar: Checking if environment variable exists" << std::endl;
        _wdupenv_s(&value, &len, env_var_name.c_str());
        if (value == nullptr) {
            std::cout << "[DEBUG] ScopedVar: Environment variable not set, setting it now" << std::endl;
            std::wstring wide_path = environment_variable_value.wstring();
            int result = _wputenv_s(env_var_name.c_str(), wide_path.c_str());
            if (result == 0) {
                std::cout << "[DEBUG] ScopedVar: Environment variable set successfully" << std::endl;
            } else {
                std::cout << "[DEBUG] ScopedVar: Failed to set environment variable, error: " << result << std::endl;
            }
        } else {
            std::cout << "[DEBUG] ScopedVar: Environment variable already exists, not overriding" << std::endl;
            was_already_set = true;
            free(value);
        }
#else
        std::cout << "[DEBUG] ScopedVar: Using POSIX APIs" << std::endl;
        if (!getenv(ENVIRONMENT_VARIABLE_NAME)) {
            setenv(ENVIRONMENT_VARIABLE_NAME, environment_variable_value.string().c_str(), 1);
            std::cout << "[DEBUG] ScopedVar: Environment variable set on POSIX" << std::endl;
        } else {
            was_already_set = true;
            std::cout << "[DEBUG] ScopedVar: Environment variable already exists on POSIX" << std::endl;
        }
#endif
    }
    
    ~ScopedVar() {
        std::cout << "[DEBUG] ScopedVar destructor called, was_already_set: " << was_already_set << std::endl;

        if (!was_already_set) {
#ifdef _WIN32
            std::cout << "[DEBUG] ScopedVar destructor: Unsetting environment variable on Windows" << std::endl;
            std::wstring env_var_name(std::string(ENVIRONMENT_VARIABLE_NAME).begin(), std::string(ENVIRONMENT_VARIABLE_NAME).end());
            _wputenv_s(env_var_name.c_str(), L"");
#else
            std::cout << "[DEBUG] ScopedVar destructor: Unsetting environment variable on POSIX" << std::endl;
            unsetenv(ENVIRONMENT_VARIABLE_NAME);
#endif
        }
    }
};
}
