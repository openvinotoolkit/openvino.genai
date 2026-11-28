// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <openvino/openvino.hpp>

#include "openvino/genai/speech_generation/audio_streamer_base.hpp"
#include "openvino/genai/speech_generation/speech_generation_config.hpp"

namespace ov::genai::qwen3_tts {

/// Descriptor for a static-shape streaming codec decoder IR (e.g. c0_t8, c25_t12, c25_t24).
struct StreamDecoderSpec {
    std::filesystem::path path;
    size_t left_context_frames = 0;
    size_t chunk_frames = 0;
};

/// Discovers optional static streaming codec decoder IRs in `models_path` via
/// `openvino_config.json` (`streaming_codec_decoders` or `qwen3_tts.streaming_decoders.graphs`)
/// and falls back to scanning filenames matching `openvino_codec_decoder_stream_c<C>_t<T>.xml`.
std::vector<StreamDecoderSpec> discover_stream_decoder_specs(const std::filesystem::path& models_path);

/// Holds compiled optional two-tier static streaming codec decoders (first + steady)
/// plus a fallback callable into the standard codec decoder.
class Qwen3TTSStreamDecoderPool {
public:
    struct CompiledTier {
        size_t left_context_frames = 0;
        size_t chunk_frames = 0;
        size_t input_frames = 0;  // left_context_frames + chunk_frames
        bool static_input = false;
        ov::InferRequest request;
    };

    using FallbackChunkDecodeFn =
        std::function<std::vector<float>(const std::vector<int64_t>& window_frame_major,
                                         size_t context_frames,
                                         size_t new_frames)>;

    Qwen3TTSStreamDecoderPool() = default;

    /// Compiles discovered static streaming decoders on `decoder_device` with `decoder_properties`.
    void initialize(const std::filesystem::path& models_path,
                    const std::string& decoder_device,
                    const ov::AnyMap& decoder_properties,
                    uint32_t num_quantizers,
                    uint32_t decode_upsample_rate);

    /// Registers an already-compiled static tier directly (useful for unit tests).
    void register_compiled_tier(size_t left_context_frames,
                                size_t chunk_frames,
                                ov::InferRequest request);

    void set_metadata(uint32_t num_quantizers, uint32_t decode_upsample_rate) {
        m_num_quantizers = num_quantizers;
        m_decode_upsample_rate = decode_upsample_rate;
    }

    bool has_static_tiers() const {
        return !m_tiers.empty();
    }

    size_t tier_count() const {
        return m_tiers.size();
    }

    uint32_t num_quantizers() const {
        return m_num_quantizers;
    }

    uint32_t decode_upsample_rate() const {
        return m_decode_upsample_rate;
    }

    /// Selects the best matching compiled static tier for (left_context_frames, chunk_frames),
    /// or returns nullptr if none fits (caller uses fallback decoder).
    CompiledTier* select_tier(size_t left_context_frames, size_t chunk_frames);

private:
    std::vector<CompiledTier> m_tiers;
    uint32_t m_num_quantizers = 16;
    uint32_t m_decode_upsample_rate = 1920;
};

struct StreamSessionResult {
    std::vector<float> waveform;
    double ttfa_us = 0.0;
    bool has_ttfa = false;
    double total_decode_us = 0.0;
    bool stopped_early = false;
    bool cancelled = false;
};

/// Per-utterance streaming decode session supporting two-tier chunking (first -> steady),
/// right-edge replicate padding for static decoders, dynamic fallback, and an optional
/// asynchronous producer/consumer decode worker thread.
class Qwen3TTSStreamSession {
public:
    Qwen3TTSStreamSession(Qwen3TTSStreamDecoderPool& pool,
                          Qwen3TTSStreamDecoderPool::FallbackChunkDecodeFn fallback_decode,
                          const SpeechGenerationConfig& config,
                          const AudioStreamerVariant& streamer,
                          std::chrono::steady_clock::time_point generation_start,
                          const std::vector<int64_t>& prefix_ref_codes = {});

    ~Qwen3TTSStreamSession();

    Qwen3TTSStreamSession(const Qwen3TTSStreamSession&) = delete;
    Qwen3TTSStreamSession& operator=(const Qwen3TTSStreamSession&) = delete;

    /// Appends one frame of `num_quantizers` codec IDs. Triggers chunk decode when the
    /// active tier's threshold is reached. Returns false if the streamer requested STOP/CANCEL.
    bool push_frame(const int64_t* frame_codes, size_t num_code_groups);

    /// True if the downstream streamer requested STOP or CANCEL.
    bool should_stop() const {
        return m_stop_status.load(std::memory_order_acquire) != static_cast<int>(StreamingStatus::RUNNING);
    }

    /// Flushes any remaining pending frames, joins the worker thread (if async), propagates
    /// any worker exception, invokes streamer end(), and returns accumulated waveform + metrics.
    StreamSessionResult finish();

private:
    struct DecodeTask {
        std::vector<int64_t> window_frame_major;  // [window_frames, Q]
        size_t window_frames = 0;
        size_t context_frames = 0;
        size_t new_frames = 0;
        size_t chunk_index = 0;
        Qwen3TTSStreamDecoderPool::CompiledTier* static_tier = nullptr;
    };

    DecodeTask build_task(bool is_final);
    void enqueue_or_run(DecodeTask&& task);
    void execute_task(const DecodeTask& task);
    void worker_loop();
    void stop_worker_for_cleanup() noexcept;
    void rethrow_worker_error_if_any();

    Qwen3TTSStreamDecoderPool& m_pool;
    Qwen3TTSStreamDecoderPool::FallbackChunkDecodeFn m_fallback_decode;
    const AudioStreamerVariant& m_streamer;
    std::chrono::steady_clock::time_point m_generation_start;

    size_t m_num_quantizers = 16;
    size_t m_upsample_rate = 1920;
    size_t m_first_chunk_frames = 8;
    size_t m_steady_chunk_frames = 12;
    size_t m_first_context_frames = 0;
    size_t m_steady_context_frames = 25;
    bool m_async = true;
    bool m_streaming_active = false;
    bool m_finished = false;

    std::vector<int64_t> m_all_codes;  // [prefix + generated frames, Q]
    size_t m_emitted_frames = 0;
    size_t m_pending_frames = 0;
    size_t m_chunk_index = 0;

    std::vector<float> m_accumulated_waveform;
    double m_ttfa_us = 0.0;
    bool m_has_ttfa = false;
    double m_total_decode_us = 0.0;

    std::atomic<int> m_stop_status{static_cast<int>(StreamingStatus::RUNNING)};
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::deque<DecodeTask> m_queue;
    bool m_stop_worker = false;
    std::exception_ptr m_worker_error;
    std::thread m_worker_thread;
};

}  // namespace ov::genai::qwen3_tts
