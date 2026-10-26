// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ov::genai::unicode_helpers {

// One decoded Unicode scalar or one preserved malformed UTF-8 byte.
// code stores the scalar value for valid input and the byte value for malformed input.
struct Utf8Char {
    uint32_t code;
    std::string bytes;
};

// Decodes UTF-8 while preserving the original byte sequence for each result.
// Malformed or truncated input is preserved one byte at a time.
std::vector<Utf8Char> decode_utf8(const std::string& text);

// Returns whether the code point is in one of the CJK ideograph ranges used by the preprocessing logic.
bool is_cjk_char(uint32_t code);

// Returns whether the code point is a letter or number in the subset used by Qwen forced-aligner
// preprocessing. This is not a complete implementation of the Unicode L/N categories.
bool is_unicode_letter_or_number(uint32_t code);

// Returns whether the code point is treated as whitespace by Qwen forced-aligner preprocessing.
bool is_unicode_whitespace(uint32_t code);

}  // namespace ov::genai::unicode_helpers
