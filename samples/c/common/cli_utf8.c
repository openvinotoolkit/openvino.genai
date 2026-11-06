// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cli_utf8.h"

#include <stdlib.h>
#include <string.h>

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

static char* duplicate_string(const char* src) {
    size_t len = 0;
    char* out = NULL;
    if (!src)
        return NULL;
    len = strlen(src);
    out = (char*)malloc(len + 1);
    if (!out)
        return NULL;
    memcpy(out, src, len + 1);
    return out;
}

static void free_argv_strings(int argc, char** argv) {
    if (!argv)
        return;
    for (int i = 0; i < argc; ++i)
        free(argv[i]);
}

#ifdef _WIN32
static int build_windows_utf8_argv(int* out_argc, char*** out_argv) {
    LPWSTR* wide_argv = NULL;
    int wide_argc = 0;
    char** utf8_argv = NULL;

    wide_argv = CommandLineToArgvW(GetCommandLineW(), &wide_argc);
    if (!wide_argv || wide_argc <= 0)
        return 0;

    utf8_argv = (char**)calloc((size_t)wide_argc, sizeof(char*));
    if (!utf8_argv) {
        LocalFree(wide_argv);
        return 0;
    }

    for (int i = 0; i < wide_argc; ++i) {
        const wchar_t* warg = wide_argv[i];
        const int needed = WideCharToMultiByte(CP_UTF8, 0, warg, -1, NULL, 0, NULL, NULL);
        if (needed <= 0)
            goto fail;
        utf8_argv[i] = (char*)malloc((size_t)needed);
        if (!utf8_argv[i])
            goto fail;
        if (WideCharToMultiByte(CP_UTF8, 0, warg, -1, utf8_argv[i], needed, NULL, NULL) != needed)
            goto fail;
    }

    LocalFree(wide_argv);
    *out_argc = wide_argc;
    *out_argv = utf8_argv;
    return 1;

fail:
    LocalFree(wide_argv);
    free_argv_strings(wide_argc, utf8_argv);
    free(utf8_argv);
    return 0;
}
#endif

int normalize_cli_argv(int argc, char** argv, int* out_argc, char*** out_argv) {
    if (argc < 0 || !argv || !out_argc || !out_argv)
        return 0;

#ifdef _WIN32
    return build_windows_utf8_argv(out_argc, out_argv);
#else
    char** copy = (char**)calloc((size_t)argc, sizeof(char*));
    if (!copy)
        return 0;

    for (int i = 0; i < argc; ++i) {
        copy[i] = duplicate_string(argv[i]);
        if (!copy[i]) {
            free_argv_strings(argc, copy);
            free(copy);
            return 0;
        }
    }

    *out_argc = argc;
    *out_argv = copy;
    return 1;
#endif
}

void free_normalized_cli_argv(int argc, char** argv) {
    if (argc < 0 || !argv)
        return;
    free_argv_strings(argc, argv);
    free(argv);
}
