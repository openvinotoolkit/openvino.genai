// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "unicode_helpers.hpp"

#include <cctype>

namespace ov::genai::unicode_helpers {

std::vector<Utf8Char> decode_utf8(const std::string& text) {
    std::vector<Utf8Char> chars;
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        const auto byte = static_cast<unsigned char>(text[i]);
        size_t len = 1;
        uint32_t code = byte;
        if (byte < 0x80) {
            len = 1;
            code = byte;
        } else if ((byte >> 5) == 0x06) {
            len = 2;
            code = byte & 0x1F;
        } else if ((byte >> 4) == 0x0E) {
            len = 3;
            code = byte & 0x0F;
        } else if ((byte >> 3) == 0x1E) {
            len = 4;
            code = byte & 0x07;
        } else {
            len = 1;
            code = byte;
        }
        if (i + len > n) {
            len = 1;
            code = byte;
        }
        for (size_t k = 1; k < len; ++k) {
            code = (code << 6) | (static_cast<unsigned char>(text[i + k]) & 0x3F);
        }
        chars.push_back({code, text.substr(i, len)});
        i += len;
    }
    return chars;
}

bool is_cjk_char(uint32_t code) {
    return (0x4E00 <= code && code <= 0x9FFF) || (0x3400 <= code && code <= 0x4DBF) ||
           (0x20000 <= code && code <= 0x2A6DF) || (0x2A700 <= code && code <= 0x2B73F) ||
           (0x2B740 <= code && code <= 0x2B81F) || (0x2B820 <= code && code <= 0x2CEAF) ||
           (0xF900 <= code && code <= 0xFAFF);
}

bool is_unicode_letter_or_number(uint32_t code) {
    if (code < 0x80) {
        return std::isalnum(static_cast<int>(code)) != 0;
    }
    if (code == 0xD7 || code == 0xF7) {  // multiplication / division sign (Sm)
        return false;
    }
    if (code >= 0xC0 && code <= 0xFF) {  // Latin-1 letters
        return true;
    }
    if (code == 0xAA || code == 0xB5 || code == 0xBA) {  // ordinal / micro letters
        return true;
    }
    if (code >= 0x0100 && code <= 0x024F) {  // Latin Extended-A / Extended-B
        return true;
    }
    if (code >= 0x1E00 && code <= 0x1EFF) {  // Latin Extended Additional
        return true;
    }
    if (code >= 0x0400 && code <= 0x04FF && !(code >= 0x0483 && code <= 0x0489)) {  // Cyrillic (skip combining)
        return true;
    }
    if (code >= 0x0500 && code <= 0x052F) {  // Cyrillic Supplement
        return true;
    }
    if (code >= 0x3040 && code <= 0x30FF) {  // Hiragana / Katakana
        return true;
    }
    if (code >= 0x1100 && code <= 0x11FF) {  // Hangul Jamo
        return true;
    }
    if (code >= 0x3130 && code <= 0x318F) {  // Hangul Compatibility Jamo
        return true;
    }
    if (code >= 0xAC00 && code <= 0xD7A3) {  // Hangul syllables
        return true;
    }
    if (code >= 0xFF10 && code <= 0xFF19) {  // Fullwidth digits
        return true;
    }
    if ((code >= 0xFF21 && code <= 0xFF3A) || (code >= 0xFF41 && code <= 0xFF5A)) {  // Fullwidth Latin letters
        return true;
    }
    if (code >= 0xFF66 && code <= 0xFF9D) {  // Halfwidth Katakana
        return true;
    }
    // Additional Unicode number code points used by the reference preprocessing.
    if (code == 0xB2 || code == 0xB3 || code == 0xB9) {  // superscript two/three/one (No)
        return true;
    }
    if (code >= 0xBC && code <= 0xBE) {  // vulgar fractions 1/4, 1/2, 3/4 (No)
        return true;
    }
    if (code >= 0x2160 && code <= 0x217F) {  // Roman numerals (Nl)
        return true;
    }
    if (code == 0x3007) {  // ideographic number zero (Nl)
        return true;
    }
    return is_cjk_char(code);
}

bool is_unicode_whitespace(uint32_t code) {
    return (0x09 <= code && code <= 0x0D) ||  // TAB, LF, VT, FF, CR
           (0x1C <= code && code <= 0x1F) ||  // FS, GS, RS, US
           code == 0x20 || code == 0x85 || code == 0xA0 || code == 0x1680 ||
           (0x2000 <= code && code <= 0x200A) || code == 0x2028 || code == 0x2029 || code == 0x202F ||
           code == 0x205F || code == 0x3000;
}

}  // namespace ov::genai::unicode_helpers
