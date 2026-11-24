// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {
namespace qwen3_tts {

// The text term of the talker input. Every generated frame feeds the talker the sum of the frame's codec embeddings
// and one text term chosen by the generation step:
//   * non_streaming_mode: tts_pad at every step (the whole text is already in the prefill);
//   * streaming layout:   row `step` of trailing_text_hidden ([1, T, H], the text tokens after the first one and the
//                         final tts_eos), then tts_pad once the rows run out;
//   * no trailing rows at all: nothing is added.
// Shared by the SDPA and the continuous-batching drivers, so both pick the text the same way.
//
// The rows are fixed when the object is made, except for an incremental one (incremental()): its rows arrive while
// the talker runs. append() adds rows, finish() says that no more will come. A step whose row has not arrived yet
// (available() is false) must wait; once finished, the steps past the last row get tts_pad as in the fixed case.
class TrailingText {
public:
    TrailingText() = default;

    TrailingText(bool non_streaming_mode, ov::Tensor tts_pad, ov::Tensor trailing_text_hidden)
        : m_non_streaming_mode(non_streaming_mode),
          m_tts_pad(std::move(tts_pad)),
          m_trailing_text_hidden(std::move(trailing_text_hidden)) {
        if (!m_non_streaming_mode && m_trailing_text_hidden) {
            const ov::Shape shape = m_trailing_text_hidden.get_shape();
            OPENVINO_ASSERT(shape.size() == 3, "trailing_text_hidden is expected to have shape [1, T, H]");
            OPENVINO_ASSERT(shape[0] == 1, "trailing_text_hidden batch size is expected to be 1");
        }
    }

    // Streaming layout whose rows are appended later. `tts_pad` is [1, 1, H].
    static TrailingText incremental(ov::Tensor tts_pad) {
        OPENVINO_ASSERT(tts_pad.get_shape().size() == 3 && tts_pad.get_shape()[1] == 1,
                        "tts_pad is expected to have shape [1, 1, H]");
        TrailingText text;
        text.m_incremental = true;
        text.m_hidden = tts_pad.get_shape()[2];
        text.m_tts_pad = std::move(tts_pad);
        return text;
    }

    bool is_incremental() const {
        return m_incremental;
    }

    // Incremental only: appends `count` rows of H floats.
    void append(const float* rows, size_t count) {
        OPENVINO_ASSERT(m_incremental && !m_finished, "TrailingText::append() needs an open incremental text");
        m_rows.insert(m_rows.end(), rows, rows + count * m_hidden);
    }

    // Incremental only: no more rows will be appended.
    void finish() {
        OPENVINO_ASSERT(m_incremental, "TrailingText::finish() needs an incremental text");
        m_finished = true;
    }

    bool finished() const {
        return m_finished;
    }

    // Incremental only: the number of rows appended so far.
    size_t rows() const {
        return m_incremental ? m_rows.size() / m_hidden : 0;
    }

    // Whether the text term of generation step `step` can be added now. Always true unless incremental.
    bool available(size_t step) const {
        return !m_incremental || m_finished || step < rows();
    }

    // Adds the text term of generation step `step` to `embed` (`hidden` floats).
    void add_to(float* embed, size_t hidden, size_t step) const {
        const float* term = nullptr;
        if (m_incremental) {
            OPENVINO_ASSERT(available(step), "TrailingText: the text of step ", step, " has not arrived");
            OPENVINO_ASSERT(hidden == m_hidden, "TrailingText hidden size mismatch");
            term = step < rows() ? m_rows.data() + step * hidden : m_tts_pad.data<const float>();
        } else if (m_non_streaming_mode) {
            term = m_tts_pad.data<const float>();
        } else if (m_trailing_text_hidden) {
            const ov::Shape shape = m_trailing_text_hidden.get_shape();
            OPENVINO_ASSERT(shape[2] == hidden, "trailing_text_hidden hidden size mismatch");
            term = step < shape[1] ? m_trailing_text_hidden.data<const float>() + step * hidden
                                   : m_tts_pad.data<const float>();
        }
        if (term == nullptr) {
            return;
        }
        for (size_t i = 0; i < hidden; ++i) {
            embed[i] += term[i];
        }
    }

private:
    bool m_non_streaming_mode = false;
    ov::Tensor m_tts_pad;
    ov::Tensor m_trailing_text_hidden;
    bool m_incremental = false;
    bool m_finished = false;
    size_t m_hidden = 0;
    std::vector<float> m_rows;
};

}  // namespace qwen3_tts
}  // namespace genai
}  // namespace ov
