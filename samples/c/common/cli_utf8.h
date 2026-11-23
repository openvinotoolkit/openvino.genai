// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef CLI_UTF8_H
#define CLI_UTF8_H

#ifdef __cplusplus
extern "C" {
#endif

// Produces a UTF-8 argv array suitable for CLI parsing.
// On Windows, arguments are reconstructed from CommandLineToArgvW and converted
// to UTF-8. On other platforms, arguments are duplicated from argv.
// Returns 1 on success and 0 on allocation or conversion failure.
int normalize_cli_argv(int argc, char** argv, int* out_argc, char*** out_argv);

// Releases argv returned by normalize_cli_argv.
void free_normalized_cli_argv(int argc, char** argv);

#ifdef __cplusplus
}
#endif

#endif  // CLI_UTF8_H
