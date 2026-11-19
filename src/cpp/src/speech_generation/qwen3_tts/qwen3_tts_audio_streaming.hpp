// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/runtime/tensor.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_codec_engine.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_stream_decoder.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
namespace qwen3_tts {

// Accumulates codec frames for ONE utterance and decodes them to waveform in small chunks while
// generation is still running. Shared by the SDPA and continuous-batching drivers.
//
// Chunk schedule (stateless decoding only): the first chunk has `first_chunk_frames` frames and each following
// full chunk doubles in size up to `max_chunk_frames` (see SpeechGenerationConfig::audio_first_chunk_frames).
// Stateful decoding emits constant chunks of `first_chunk_frames` (rounded up to whole blocks).
//
// Two decoding strategies, chosen when the engine is created (OV_GENAI_QWEN_TTS_STREAM_DECODER):
//
// * Stateful (explicit-state streaming decoder, default when the model has it). Frames are decoded in
//   blocks of StreamDecoder::BLOCK_FRAMES as soon as a block is complete, carrying the decoder's causal
//   state, so nothing is re-decoded, the samples equal the non-streamed decode, and taking a chunk costs
//   nothing. The chunk size is rounded up to whole blocks. Voice-clone (ICL) reference codes are not emitted;
//   the last `left_context_frames` of them only prime the decoder state (frames that do not fill a whole block
//   share the first block with generated frames, and their audio is discarded).
//
// * Stateless. Voice-clone reference codes are kept as an invisible prefix. Every chunk is decoded after up to
//   `left_context_frames` earlier frames (see Qwen3TTSCodecEngine::decode_speech_chunk), so emitted samples are
//   final and no look-ahead is required.
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
          m_left_context_frames(left_context_frames),
          m_session(engine.create_stream_session()) {
        OPENVINO_ASSERT(first_chunk_frames >= 1 && max_chunk_frames >= 1, "audio chunk sizes must be >= 1");
        const size_t groups = engine.ids().num_code_groups;
        if (m_session) {
            const size_t block = StreamDecoderSession::block_frames();
            // Constant chunk size (no schedule): nothing is re-decoded, so small chunks cost nothing extra and
            // give the most playback slack. The size is audio_first_chunk_frames rounded up to whole blocks.
            m_target_frames = round_up(std::max<size_t>(first_chunk_frames, 1), block);
            m_max_chunk_frames = m_target_frames;
            m_block_codes.reserve(block * groups);
        }
        if (ref_codec_ids) {
            const ov::Shape shape = ref_codec_ids.get_shape();
            OPENVINO_ASSERT(shape.size() == 2 || shape.size() == 3, "ref_codec_ids must have shape [T, G] or [1, T, G]");
            const int64_t* ptr = ref_codec_ids.data<const int64_t>();
            const size_t ref_frames = ref_codec_ids.get_size() / groups;
            if (m_session) {
                prime_with_reference(ptr, ref_frames, groups);
            } else {
                m_codes.assign(ptr, ptr + ref_codec_ids.get_size());
                m_cursor_frame = ref_frames;
            }
        }
    }

    void push_frame(const std::vector<int64_t>& groups) {
        if (m_session) {
            m_block_codes.insert(m_block_codes.end(), groups.begin(), groups.end());
            if (m_block_codes.size() == StreamDecoderSession::block_frames() * groups.size()) {
                decode_block();
            }
            return;
        }
        m_codes.insert(m_codes.end(), groups.begin(), groups.end());
    }

    // Frames that are ready to be emitted.
    size_t pending_frames() const {
        if (m_session) {
            return m_samples.size() / m_engine.decoder_samples_per_frame();
        }
        return m_codes.size() / m_engine.ids().num_code_groups - m_cursor_frame;
    }

    bool chunk_ready() const {
        // Stateful: the very first chunk goes out as soon as the first block is decoded, even when leftover
        // reference frames made that block shorter than the target (minimal time to first audio).
        if (m_session && !m_first_chunk_taken && !m_samples.empty()) {
            return true;
        }
        return pending_frames() >= m_target_frames;
    }

    // Takes the audio of a full chunk (call when chunk_ready()) and advances the schedule to the next
    // (doubled) target size. Stateless: decodes all pending frames; stateful: hands over the decoded blocks.
    std::vector<float> take_pending() {
        if (m_session) {
            if (m_samples.empty()) {
                return {};
            }
            const bool full_chunk = chunk_ready();
            auto samples = std::move(m_samples);
            m_samples.clear();
            m_first_chunk_taken = true;
            if (full_chunk) {
                m_target_frames = std::min(m_target_frames * 2, m_max_chunk_frames);
            }
            return samples;
        }
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

    // End of the utterance: returns everything not yet taken, including a trailing partial block. Stateful: the
    // partial block is zero-padded to a whole block and the padded output is dropped, which is exact because the
    // decoder is causal.
    std::vector<float> flush() {
        if (!m_session) {
            return take_pending();
        }
        const size_t groups = m_engine.ids().num_code_groups;
        const size_t partial_frames = m_block_codes.size() / groups;
        if (partial_frames > m_discard_frames) {  // else only reference frames are pending: nothing to emit
            m_block_codes.resize(StreamDecoderSession::block_frames() * groups, 0);
            auto block = m_session->decode_block(m_block_codes.data());
            const size_t per_frame = m_engine.decoder_samples_per_frame();
            m_samples.insert(m_samples.end(),
                             block.begin() + m_discard_frames * per_frame,
                             block.begin() + partial_frames * per_frame);
        }
        m_block_codes.clear();
        m_discard_frames = 0;
        auto samples = std::move(m_samples);
        m_samples.clear();
        return samples;
    }

    bool uses_stateful_decoder() const {
        return static_cast<bool>(m_session);
    }

private:
    static size_t round_up(size_t value, size_t multiple) {
        return (value + multiple - 1) / multiple * multiple;
    }

    void decode_block() {
        auto block = m_session->decode_block(m_block_codes.data());
        // The first block may start with reference frames whose audio is not emitted.
        m_samples.insert(m_samples.end(), block.begin() + m_discard_frames * m_engine.decoder_samples_per_frame(), block.end());
        m_block_codes.clear();
        m_discard_frames = 0;
    }

    // Runs the last `left_context_frames` reference frames through the decoder to prime its state (or restores the
    // state cached by an earlier request with the same reference); their audio is discarded. Whole blocks are decoded right away, the remaining newest frames wait in the pending block
    // and are completed by the first generated frames.
    void prime_with_reference(const int64_t* ref_codes, size_t ref_frames, size_t groups) {
        const size_t block = StreamDecoderSession::block_frames();
        const size_t frames = std::min(ref_frames, m_left_context_frames);
        const int64_t* ptr = ref_codes + (ref_frames - frames) * groups;
        const size_t whole = frames / block * block;
        if (whole > 0) {
            // The state after the whole blocks depends only on those reference codes: reuse it across requests.
            const uint64_t key = StreamDecoderSession::primed_key(ptr, whole * groups, whole);
            if (auto cached = m_session->find_primed(key)) {
                m_session->restore(*cached);
            } else {
                for (size_t offset = 0; offset < whole; offset += block) {
                    m_session->decode_block(ptr + offset * groups);
                }
                m_session->store_primed(key, m_session->snapshot());
            }
        }
        m_block_codes.assign(ptr + whole * groups, ptr + frames * groups);
        m_discard_frames = frames - whole;
    }

    Qwen3TTSCodecEngine& m_engine;
    size_t m_max_chunk_frames;
    size_t m_target_frames;  // size of the next chunk
    size_t m_left_context_frames;
    std::unique_ptr<StreamDecoderSession> m_session;  // null: stateless decoding
    // Stateless: [ref frames..., generated frames...], frame-major.
    std::vector<int64_t> m_codes;
    size_t m_cursor_frame = 0;  // first frame not yet decoded
    // Stateful: frames of the block being collected, and decoded samples not yet taken.
    std::vector<int64_t> m_block_codes;
    size_t m_discard_frames = 0;  // leading reference frames of the pending block, their audio is dropped
    bool m_first_chunk_taken = false;
    std::vector<float> m_samples;
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
