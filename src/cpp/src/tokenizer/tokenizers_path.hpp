// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

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
    static constexpr wchar_t ENVIRONMENT_VARIABLE_NAME_W[] = L"OPENVINO_TOKENIZERS_PATH_GENAI";

    explicit ScopedVar(const std::filesystem::path& environment_variable_value) {

#ifdef _WIN32
        wchar_t* value = nullptr;
        size_t len = 0;
        _wdupenv_s(&value, &len, ENVIRONMENT_VARIABLE_NAME_W);
        if (value == nullptr) {
            _wputenv_s(ENVIRONMENT_VARIABLE_NAME_W, environment_variable_value.wstring().c_str());
        } else {
            was_already_set = true;
            free(value);
        }
#else
        if (!getenv(ENVIRONMENT_VARIABLE_NAME)) {
            setenv(ENVIRONMENT_VARIABLE_NAME, environment_variable_value.string().c_str(), 1);
        } else {
            was_already_set = true;
        }
#endif
    }
    
    ~ScopedVar() {
        if (!was_already_set) {
#ifdef _WIN32
            _wputenv_s(ENVIRONMENT_VARIABLE_NAME_W, L"");
#else
            unsetenv(ENVIRONMENT_VARIABLE_NAME);
#endif
        }
    }
};
}
