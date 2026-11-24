// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "openvino/core/except.hpp"

namespace ov {
namespace genai {
namespace qwen3_tts {

// Turns text that arrives in pieces into segments that can each be tokenized on its own, with the same tokens as in
// the complete text. A tokenizer splits text into pre-tokens first and only merges inside them, so a segment
// boundary has to be a pre-token boundary, and the committer only cuts where that is certain:
//   * before a run of whitespace that starts with a space or a tab (a space is attached to the word that follows
//     it, so the boundary in front of it is always a pre-token boundary), and
//   * before a run that starts with a line break only when the character in front of it is an ASCII letter or
//     digit (a line break after punctuation is part of the punctuation pre-token).
// Whatever follows the last such cut (the whitespace run and the partial word after it) is held back until more text
// shows where the word ends, or end() is called. A held-back part longer than `max_holdback_chars` characters (text
// without spaces, e.g. Chinese) is committed anyway; the tokens at that boundary can then differ from the complete
// text. Bytes that do not yet form a whole UTF-8 character are never committed.
//
// push() and take() are not synchronized with each other; the owner serializes them.
class TextCommitter {
public:
    explicit TextCommitter(size_t max_holdback_chars = 32) : m_max_holdback_chars(max_holdback_chars) {}

    void push(std::string_view text) {
        OPENVINO_ASSERT(!m_ended, "TextCommitter::push() called after end()");
        m_pending.append(text.data(), text.size());
    }

    // No more text will be pushed: everything left becomes committable.
    void end() {
        OPENVINO_ASSERT(!m_ended, "TextCommitter::end() may only be called once");
        m_ended = true;
    }

    bool ended() const {
        return m_ended;
    }

    // The text that is safe to tokenize on its own since the last call, in order ("" if none). After end(), all the
    // text that is left.
    std::string take() {
        const size_t cut = m_ended ? m_pending.size() : safe_cut();
        std::string committed = m_pending.substr(0, cut);
        m_pending.erase(0, cut);
        return committed;
    }

    // Bytes pushed but not committed yet.
    size_t pending_bytes() const {
        return m_pending.size();
    }

private:
    static bool is_space(char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    }

    static bool is_ascii_alnum(char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    }

    static bool is_continuation_byte(char c) {
        return (static_cast<unsigned char>(c) & 0xC0) == 0x80;
    }

    // Length of the longest prefix that ends on a whole UTF-8 character.
    size_t complete_prefix() const {
        size_t end = m_pending.size();
        size_t back = 0;  // continuation bytes at the end
        while (back < end && back < 3 && is_continuation_byte(m_pending[end - 1 - back])) {
            ++back;
        }
        if (back == end) {
            return end;  // malformed: nothing but continuation bytes, do not hold it forever
        }
        const unsigned char lead = static_cast<unsigned char>(m_pending[end - 1 - back]);
        size_t expected = 1;
        if (lead >= 0xF0) {
            expected = 4;
        } else if (lead >= 0xE0) {
            expected = 3;
        } else if (lead >= 0xC0) {
            expected = 2;
        }
        return (back + 1 < expected) ? end - 1 - back : end;
    }

    // A cut in front of the whitespace run that starts at `run_start` (> 0) keeps every pre-token whole.
    bool boundary_ok(size_t run_start) const {
        const char first = m_pending[run_start];
        if (first == ' ' || first == '\t') {
            return true;
        }
        return is_ascii_alnum(m_pending[run_start - 1]);
    }

    size_t safe_cut() const {
        const size_t complete = complete_prefix();
        size_t cut = 0;
        size_t pos = complete;
        while (pos > 0) {
            while (pos > 0 && !is_space(m_pending[pos - 1])) {
                --pos;
            }
            if (pos == 0) {
                break;
            }
            size_t run_start = pos - 1;
            while (run_start > 0 && is_space(m_pending[run_start - 1])) {
                --run_start;
            }
            if (run_start > 0 && boundary_ok(run_start)) {
                cut = run_start;
                break;
            }
            pos = run_start;
        }
        size_t held_chars = 0;
        for (size_t i = cut; i < complete; ++i) {
            held_chars += is_continuation_byte(m_pending[i]) ? 0 : 1;
        }
        return held_chars > m_max_holdback_chars ? complete : cut;
    }

    size_t m_max_holdback_chars;
    std::string m_pending;
    bool m_ended = false;
};

// Tokens of a target text segment, as they appear in the chat-template turn of the complete text. `encode` maps a
// string to token ids. The segment is wrapped in the template's start and end markers and only the tokens in between
// are kept: the usual "assistant\n" in front of the text would merge with a line break at the start of a segment,
// and the segments after the first one do not follow it in the complete text.
inline std::vector<int64_t> tokenize_target_segment(const std::function<std::vector<int64_t>(const std::string&)>& encode,
                                                    const std::string& segment) {
    if (segment.empty()) {
        return {};
    }
    const std::vector<int64_t> ids = encode("<|im_start|>" + segment + "<|im_end|>");
    OPENVINO_ASSERT(ids.size() >= 2, "Qwen3-TTS tokenizer returned too few tokens for a text segment");
    return std::vector<int64_t>(ids.begin() + 1, ids.end() - 1);
}

}  // namespace qwen3_tts
}  // namespace genai
}  // namespace ov
