// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ov::genai::unicode_helpers {

// A decoded Unicode scalar: its code point and the original UTF-8 bytes it was decoded from.
struct Utf8Char {
    uint32_t code;
    std::string bytes;
};

// Decodes a UTF-8 string into its Unicode scalars. Malformed or truncated byte sequences are
// emitted as individual single-byte scalars; never throws.
// Example: decode_utf8("a\xC2\xA3") -> [{0x61, "a"}, {0xA3, "\xC2\xA3"}]  // 'a', '£'
std::vector<Utf8Char> decode_utf8(const std::string& text);

// Returns whether the code point is a CJK ideograph (Han and CJK compatibility ranges).
// Example: is_cjk_char(0x4E2D) == true  // 中
bool is_cjk_char(uint32_t code);

// Returns whether the code point is a letter or number in the subset used by Qwen forced-aligner
// preprocessing. This is not a complete implementation of the Unicode L/N categories.
bool is_unicode_letter_or_number(uint32_t code);

// Returns whether the code point is treated as whitespace by Qwen forced-aligner preprocessing.
bool is_unicode_whitespace(uint32_t code);

}  // namespace ov::genai::unicode_helpers
