// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>
#endif

// Small command-line helpers shared by the Qwen3-TTS C++ samples.
//
// Qwen3-TTS is multilingual (for example, `--instruct` and prompts may contain
// non-ASCII text). On Windows the ANSI `argv` cannot represent such characters,
// so `normalized_argv` re-reads the command line as UTF-8. On other platforms it
// simply forwards `argv` unchanged.
namespace qwen3_cli {

#ifdef _WIN32
inline std::vector<std::string> windows_utf8_argv(int argc, char* argv[]) {
    struct LocalFreeDeleter {
        void operator()(LPWSTR* ptr) const noexcept {
            if (ptr != nullptr) {
                LocalFree(ptr);
            }
        }
    };

    std::vector<std::string> args;
    int wide_argc = 0;
    std::unique_ptr<LPWSTR, LocalFreeDeleter> wide_argv(CommandLineToArgvW(GetCommandLineW(), &wide_argc));
    if (wide_argv == nullptr || wide_argc <= 0) {
        args.reserve(static_cast<size_t>(argc));
        for (int i = 0; i < argc; ++i) {
            args.emplace_back(argv[i]);
        }
        return args;
    }

    args.reserve(static_cast<size_t>(wide_argc));
    for (int i = 0; i < wide_argc; ++i) {
        const wchar_t* warg = wide_argv.get()[i];
        const int needed = WideCharToMultiByte(CP_UTF8, 0, warg, -1, nullptr, 0, nullptr, nullptr);
        OPENVINO_ASSERT(needed > 0, "Failed to convert command-line argument to UTF-8");
        std::string utf8(static_cast<size_t>(needed), '\0');
        const int written = WideCharToMultiByte(CP_UTF8, 0, warg, -1, utf8.data(), needed, nullptr, nullptr);
        OPENVINO_ASSERT(written == needed, "Failed to convert command-line argument to UTF-8");
        utf8.pop_back();
        args.push_back(std::move(utf8));
    }

    return args;
}
#endif

inline std::vector<std::string> normalized_argv(int argc, char* argv[]) {
#ifdef _WIN32
    return windows_utf8_argv(argc, argv);
#else
    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    return args;
#endif
}

}  // namespace qwen3_cli
