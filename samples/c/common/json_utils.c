// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "json_utils.h"

static int has_capacity(size_t used, size_t capacity, size_t bytes_to_write) {
    if (capacity == 0 || used >= capacity) {
        return 0;
    }
    return bytes_to_write <= (capacity - used - 1);
}

int json_escape_string(const char* input, char* output, size_t output_size) {
    size_t i = 0;
    size_t j = 0;

    if (!input || !output || output_size == 0) {
        return -1;
    }

    while (input[i] != '\0') {
        unsigned char c = (unsigned char)input[i];
        switch (c) {
            case '"':
            case '\\':
                if (!has_capacity(j, output_size, 2)) {
                    return -1;
                }
                output[j++] = '\\';
                output[j++] = (char)c;
                break;
            case '\b':
            case '\f':
            case '\n':
            case '\r':
            case '\t':
                if (!has_capacity(j, output_size, 2)) {
                    return -1;
                }
                output[j++] = '\\';
                output[j++] = (c == '\b') ? 'b' :
                              (c == '\f') ? 'f' :
                              (c == '\n') ? 'n' :
                              (c == '\r') ? 'r' : 't';
                break;
            default:
                if (c < 0x20) {
                    char hex1;
                    char hex2;
                    if (!has_capacity(j, output_size, 6)) {
                        return -1;
                    }
                    output[j++] = '\\';
                    output[j++] = 'u';
                    output[j++] = '0';
                    output[j++] = '0';
                    hex1 = (char)((c >> 4) & 0x0F);
                    hex2 = (char)(c & 0x0F);
                    output[j++] = (hex1 < 10) ? ('0' + hex1) : ('A' + hex1 - 10);
                    output[j++] = (hex2 < 10) ? ('0' + hex2) : ('A' + hex2 - 10);
                } else {
                    int utf8_len = 1;
                    if ((c & 0xE0) == 0xC0) {
                        utf8_len = 2;
                    } else if ((c & 0xF0) == 0xE0) {
                        utf8_len = 3;
                    } else if ((c & 0xF8) == 0xF0) {
                        utf8_len = 4;
                    }

                    if (utf8_len > 1) {
                        int valid = 1;
                        int k = 0;
                        for (k = 1; k < utf8_len; ++k) {
                            if (input[i + k] == '\0' || (((unsigned char)input[i + k] & 0xC0) != 0x80)) {
                                valid = 0;
                                break;
                            }
                        }
                        if (valid) {
                            if (!has_capacity(j, output_size, (size_t)utf8_len)) {
                                return -1;
                            }
                            for (k = 0; k < utf8_len; ++k) {
                                output[j++] = input[i + k];
                            }
                            i += (size_t)utf8_len - 1;
                        } else {
                            if (!has_capacity(j, output_size, 1)) {
                                return -1;
                            }
                            output[j++] = input[i];
                        }
                    } else {
                        if (!has_capacity(j, output_size, 1)) {
                            return -1;
                        }
                        output[j++] = input[i];
                    }
                }
                break;
        }
        ++i;
    }

    output[j] = '\0';
    return 0;
}
