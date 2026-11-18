// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstring>
#include <variant>
#include <vector>

#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/runtime/tensor.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_codec_engine.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
namespace qwen3_tts {

// Accumulates codec frames for ONE utterance and decodes them to waveform in small chunks while
// generation is still running. Shared by the SDPA and continuous-batching drivers.
//
// Voice-clone (ICL) reference codes, if any, are kept as an invisible prefix: they are never
// emitted but serve as left context for the first chunks. Every chunk is decoded after up to
// `left_context_frames` earlier frames (see Qwen3TTSCodecEngine::decode_speech_chunk), so emitted
// samples are final and no look-ahead is required.
//
// Chunk schedule: the first chunk has `first_chunk_frames` frames and each following full chunk
// doubles in size up to `max_chunk_frames` (see SpeechGenerationConfig::audio_first_chunk_frames).
class AudioChunker {
public:
    AudioChunker(Qwen3TTSCodecEngine& engine,
                 size_t first_chunk_frames,
                 size_t max_chunk_frames,
                 size_t left_context_frames,
                 const ov::Tensor& ref_codec_ids = {})
        : m_engine(engine),
          m_max_chunk_frames(std::max<size_t>(max_chunk_frames, 1)),
          m_target_frames(std::min(std::max<size_t>(first_chunk_frames, 1), m_max_chunk_frames)),
          m_left_context_frames(left_context_frames) {
        OPENVINO_ASSERT(first_chunk_frames >= 1 && max_chunk_frames >= 1, "audio chunk sizes must be >= 1");
        if (ref_codec_ids) {
            const ov::Shape shape = ref_codec_ids.get_shape();
            OPENVINO_ASSERT(shape.size() == 2 || shape.size() == 3, "ref_codec_ids must have shape [T, G] or [1, T, G]");
            const int64_t* ptr = ref_codec_ids.data<const int64_t>();
            m_codes.assign(ptr, ptr + ref_codec_ids.get_size());
            m_cursor_frame = m_codes.size() / engine.ids().num_code_groups;
        }
    }

    void push_frame(const std::vector<int64_t>& groups) {
        m_codes.insert(m_codes.end(), groups.begin(), groups.end());
    }

    size_t pending_frames() const {
        return m_codes.size() / m_engine.ids().num_code_groups - m_cursor_frame;
    }

    bool chunk_ready() const {
        return pending_frames() >= m_target_frames;
    }

    // Decodes and consumes all pending frames (a full chunk when chunk_ready(), the tail on flush).
    // A full chunk advances the schedule to the next (doubled) target size.
    std::vector<float> take_pending() {
        const size_t end = m_codes.size() / m_engine.ids().num_code_groups;
        if (end == m_cursor_frame) {
            return {};
        }
        const bool full_chunk = (end - m_cursor_frame) >= m_target_frames;
        auto samples = m_engine.decode_speech_chunk(m_codes, m_cursor_frame, end, m_left_context_frames);
        m_cursor_frame = end;
        if (full_chunk) {
            m_target_frames = std::min(m_target_frames * 2, m_max_chunk_frames);
        }
        return samples;
    }

private:
    Qwen3TTSCodecEngine& m_engine;
    size_t m_max_chunk_frames;
    size_t m_target_frames;  // size of the next chunk
    size_t m_left_context_frames;
    std::vector<int64_t> m_codes;  // [ref frames..., generated frames...], frame-major
    size_t m_cursor_frame = 0;     // first frame not yet decoded
};

inline ov::Tensor make_chunk_tensor(const std::vector<float>& samples) {
    // Same layout as the Omni speech streamer: [1, 1, N] float32 PCM.
    ov::Tensor chunk(ov::element::f32, ov::Shape{1, 1, samples.size()});
    if (!samples.empty()) {
        std::memcpy(chunk.data<float>(), samples.data(), samples.size() * sizeof(float));
    }
    return chunk;
}

inline bool is_speech_streamer_active(const OmniSpeechStreamerVariant& streamer) {
    if (const auto* holder = std::get_if<std::shared_ptr<OmniSpeechStreamerBase>>(&streamer)) {
        return static_cast<bool>(*holder);
    }
    if (const auto* fn = std::get_if<std::function<StreamingStatus(const ov::Tensor&)>>(&streamer)) {
        return static_cast<bool>(*fn);
    }
    return false;
}

// Forwards a chunk to the user's streamer; returns the streamer's verdict.
inline StreamingStatus write_to_speech_streamer(const OmniSpeechStreamerVariant& streamer, const ov::Tensor& chunk) {
    return std::visit(
        [&chunk](const auto& arg) -> StreamingStatus {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::function<StreamingStatus(const ov::Tensor&)>>) {
                return arg ? arg(chunk) : StreamingStatus::RUNNING;
            } else if constexpr (std::is_same_v<T, std::shared_ptr<OmniSpeechStreamerBase>>) {
                return arg ? arg->write(chunk) : StreamingStatus::RUNNING;
            } else {
                return StreamingStatus::RUNNING;
            }
        },
        streamer);
}

// Calls OmniSpeechStreamerBase::end() exactly once on every exit path, including exceptions.
class SpeechStreamerEndGuard {
public:
    explicit SpeechStreamerEndGuard(const OmniSpeechStreamerVariant& streamer) : m_streamer(streamer) {}
    SpeechStreamerEndGuard(const SpeechStreamerEndGuard&) = delete;
    SpeechStreamerEndGuard& operator=(const SpeechStreamerEndGuard&) = delete;

    void end() {
        if (m_ended) {
            return;
        }
        m_ended = true;
        if (const auto* holder = std::get_if<std::shared_ptr<OmniSpeechStreamerBase>>(&m_streamer); holder && *holder) {
            (*holder)->end();
        }
    }

    ~SpeechStreamerEndGuard() {
        try {
            end();
        } catch (...) {
            // end() during unwinding must not throw.
        }
    }

private:
    const OmniSpeechStreamerVariant& m_streamer;
    bool m_ended = false;
};

}  // namespace qwen3_tts
}  // namespace genai
}  // namespace ov
