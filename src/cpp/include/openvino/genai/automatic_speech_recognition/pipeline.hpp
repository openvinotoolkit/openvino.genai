// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/automatic_speech_recognition/generation_config.hpp"
#include "openvino/genai/automatic_speech_recognition/perf_metrics.hpp"
#include "openvino/genai/llm_pipeline.hpp"

namespace ov::genai {

using AudioInputs = std::variant<std::vector<float>, std::vector<std::vector<float>>>;

/// Time-aligned text chunk — used for both segment-level and word-level timestamps.
/// For segments: text is the decoded segment, token_ids are the tokens in that segment.
/// For words: text is the individual word, token_ids are the tokens composing that word.
struct ASRDecodedResultChunk {
    float start_ts;  // start of chunk in seconds
    float end_ts;    // end of chunk in seconds (-1.0f if not predicted by model)
    std::string text;
    std::vector<int64_t> token_ids;
};

struct OPENVINO_GENAI_EXPORTS ASRDecodedResults {
    std::vector<std::string> texts;
    std::vector<float> scores;
    /**
     * @brief Language associated with each transcription.
     *
     * Each entry contains the detected language when the model supports language identification, or the requested
     * language when generation forces one. An entry is empty when no language was requested and the model does not
     * support language identification, as with Fun-ASR.
     */
    std::vector<std::string> languages;
    ASRPerfMetrics perf_metrics;

    std::optional<std::vector<std::vector<ASRDecodedResultChunk>>> chunks = std::nullopt;
    std::optional<std::vector<std::vector<ASRDecodedResultChunk>>> words = std::nullopt;

    /**
     * @brief Optional model-specific metadata for each transcription.
     *
     * Models that do not provide metadata leave this unset.
     * When present, `metadata[i]` corresponds to `texts[i]`.
     * Each map contains model-defined key/value metadata and may be empty.
     */
    std::optional<std::vector<ov::AnyMap>> metadata = std::nullopt;

    operator std::string() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    };

    operator std::vector<std::string>() const {
        return texts;
    };

    friend std::ostream& operator<<(std::ostream& os, const ASRDecodedResults& dr) {
        OPENVINO_ASSERT(dr.scores.size() == dr.texts.size(), "The number of scores and texts doesn't match.");
        if (dr.texts.empty()) {
            return os;
        }
        if (dr.texts.size() == 1) {
            os << dr.texts[0];
            return os;
        }
        for (size_t i = 0; i < dr.texts.size() - 1; ++i) {
            os << std::to_string(dr.scores[i]) << ": " << dr.texts[i] << '\n';
        }
        return os << std::to_string(dr.scores.back()) << ": " << dr.texts.back();
    }
};

class ASRPipelineImplBase;
class ASRForcedAligner;
class ASRStreamImplBase;

/// Receives transcript deltas. Returning STOP or CANCEL stops generation and discards not yet processed audio.
using ASRTextCallback = std::function<StreamingStatus(std::string)>;

/**
 * @brief Streaming audio transcription session.
 *
 * generate() runs inference once for the whole stream and consumes audio frames submitted by push_audio()
 * until end_audio() is called. A producer thread may call push_audio() and end_audio() while generate() runs
 * in a different thread. All methods are thread-safe. Frames are processed in the order push_audio() calls
 * complete. Every thread must return from push_audio(), end_audio() and generate() before the stream is
 * destroyed or moved.
 */
class OPENVINO_GENAI_EXPORTS ASRStream {
public:
    ASRStream(ASRStream&&) noexcept;
    ASRStream& operator=(ASRStream&&) noexcept;
    ~ASRStream();

    /**
     * @brief Enqueue normalized mono 16 kHz audio samples for generate() without running inference.
     *
     * Blocks while the bounded input queue is full until generate() consumes a frame. Throws after end_audio() or
     * once generate() has stopped: after completion, after the callback requested a stop, or after an error.
     */
    void push_audio(const std::vector<float>& audio_frame);

    /// Signal the end of audio input. generate() processes the queued frames and returns. May be called once.
    void end_audio();

    /// Return the number of generated tokens so far. Safe to call while generate() is running.
    size_t get_num_generated_tokens() const;

    /**
     * @brief Return retained key/value tensor bytes after the latest completed inference chunk.
     *
     * Includes audio and text KV caches, but excludes convolution caches, inference request buffers and
     * temporary allocations. Safe to call while generate() is running. Throws if the model does not support
     * KV-cache size monitoring.
     */
    size_t get_kv_cache_size() const;

    /**
     * @brief Return retained and maximum KV-cache positions for the audio and text windows.
     *
     * Each map value is {retained_positions, maximum_positions}. Retained sizes start at zero and
     * reflect tensor sequence lengths after the latest completed inference chunk and cropping.
     * Safe to call during generation. Throws if window monitoring is unsupported by the model.
     */
    std::map<std::string, std::pair<size_t, size_t>> get_kv_cache_window_sizes() const;

    /**
     * @brief Return cumulative stage durations in milliseconds for the current stream.
     *
     * Keys are audio_inference, text_inference, audio_kv_cache and text_kv_cache. Inference durations cover
     * infer() calls; cache durations cover retained KV tensor allocation, copying, cropping and rephasing.
     * Feature extraction, convolution cache copies, tensor setup, sampling and callbacks are excluded.
     * Safe to call while generate() is running. Throws if stage timing is unsupported by the model.
     * With ov::enable_profiling enabled (PERF_COUNT), additional audio_op/<node_type>/<exec_type> and
     * text_op/<node_type>/<exec_type> keys group the latest plugin real-time counters, in milliseconds.
     * CPU counters are running averages, not per-call or cumulative durations; do not difference or accumulate them.
     * audio_node/<description> and text_node/<description> keys report individual executed Broadcast counters,
     * with node names, execution types, compiled-graph partial shapes (? denotes dynamic) and consumer names.
     * Shapes are not runtime tensor measurements. Unmatched plugin node names are marked explicitly.
     * Plugin counters may overlap or omit overhead and must not be added to wall-clock stage durations.
     */
    std::map<std::string, double> get_stage_durations() const;

    /**
     * @brief Run inference over all audio pushed to the stream and return the complete transcription.
     *
     * May be called once. Blocks until end_audio() is called and the queue is drained, or the callback requests
     * a stop. The callback is invoked on the calling thread with each non-empty transcript delta and must not wait
     * for push_audio() because a full queue isn't drained while the callback runs. Exceptions thrown by inference
     * or the callback propagate from generate() and stop the stream.
     */
    ASRDecodedResults generate(const ASRTextCallback& text_callback = nullptr);

private:
    explicit ASRStream(std::unique_ptr<ASRStreamImplBase> impl);

    std::unique_ptr<ASRStreamImplBase> m_impl;

    friend class ASRPipeline;
};

class OPENVINO_GENAI_EXPORTS ASRPipeline {
    std::unique_ptr<ASRPipelineImplBase> m_impl;
    friend class ASRStreamingSession;

public:
    ASRPipeline(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    ASRPipeline(const std::filesystem::path& models_path, const std::string& device, Properties&&... properties)
        : ASRPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    ~ASRPipeline();

    /**
     * @brief Transcribes a single audio input or a batch of audio inputs.
     *
     * Batch results are returned in input order. The generation configuration
     * applies to all inputs. Models without batching support accept at most
     * one audio input per call, either directly or as a batch of one.
     * Batches must be non-empty. Streaming is not supported for batches
     * larger than one.
     */
    ASRDecodedResults generate(const AudioInputs& audio_inputs,
                               const std::optional<ASRGenerationConfig>& generation_config = std::nullopt,
                               StreamerVariant streamer = std::monostate());

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ASRDecodedResults, Properties...> generate(const AudioInputs& audio_inputs,
                                                                              Properties&&... properties) {
        return generate(audio_inputs, AnyMap{std::forward<Properties>(properties)...});
    }

    ASRDecodedResults generate(const AudioInputs& audio_inputs, const ov::AnyMap& config_map);

    /// Create a stateful input stream when the selected ASR model supports incremental audio.
    ASRStream create_stream(const std::optional<ASRGenerationConfig>& generation_config = std::nullopt);

    Tokenizer get_tokenizer();
    ASRGenerationConfig get_generation_config() const;
    void set_generation_config(const ASRGenerationConfig& config);
};

/**
 * @brief Pre-constructed forced aligner used for word-level timestamps.
 *
 * Constructor-only property carrying an already-built ASRForcedAligner instance, used when word-level
 * timestamps are requested. Currently supported only for Qwen3-ASR.
 */
static constexpr ov::Property<std::shared_ptr<ASRForcedAligner>> forced_aligner{"forced_aligner"};

/// Parameters controlling a single streaming ASR session.
struct OPENVINO_GENAI_EXPORTS ASRStreamingConfig {
    /// Audio duration to accumulate before triggering a decode pass.
    float chunk_size_sec = 2.0f;
    /// Number of initial decode passes run without a prefix (cold-start).
    size_t warmup_chunks = 2;
    /// Tokens rolled back from accumulated output when computing the prefix for the next pass.
    size_t context_rollback_tokens = 5;

    /// Maximum chunks of audio retained in the sliding window before older, already-decoded
    /// audio is dropped from the front of the accumulated buffer. 0 = unbounded (re-encode the
    /// entire session every pass, the legacy behavior). When set, bounds per-pass re-encoding
    /// cost to O(window_chunk_num) instead of O(session length), turning total session cost
    /// from O(n^2) to O(n * window_chunk_num).
    size_t window_chunk_num = 0;

    /// Chunks of already-decoded audio treated as still "unfixed" (not yet safe to drop) once
    /// the sliding window is active. Must be less than window_chunk_num. Independent of
    /// warmup_chunks. Ignored when window_chunk_num == 0.
    size_t window_rollback_chunk_num = 2;

    /// Experimental. When true, disables eviction of the text-history prefix
    /// tied to the audio sliding window (see window_chunk_num), so the decoder prefix grows
    /// unbounded for the life of the session even though the audio window itself still stays
    /// bounded..
    bool unbounded_prefix = false;
};

/// A partial transcription delivered after each decode pass.
struct OPENVINO_GENAI_EXPORTS ASRPartialResult {
    std::string language;
    std::string committed_text;     // all stable text confirmed so far, including new_committed_text
    std::string new_committed_text; // stable text added since the previous partial result
    std::string partial_text;       // trailing region still subject to change
};

/// Move-only RAII handle for a streaming ASR session.
/// Lifetime must not exceed the ASRPipeline that created it.
/// Only one active session per ASRPipeline is supported at a time.
class OPENVINO_GENAI_EXPORTS ASRStreamingSession {
public:
    /// Construct a streaming session on the given pipeline.
    ASRStreamingSession(ASRPipeline& pipeline,
                        const ASRStreamingConfig& streaming_config = {},
                        const std::optional<ASRGenerationConfig>& generation_config = std::nullopt);

    ~ASRStreamingSession();
    ASRStreamingSession(ASRStreamingSession&&) noexcept;
    ASRStreamingSession& operator=(ASRStreamingSession&&) noexcept;
    ASRStreamingSession(const ASRStreamingSession&) = delete;
    ASRStreamingSession& operator=(const ASRStreamingSession&) = delete;

    /// Feed any amount of 16 kHz mono float32 PCM audio.
    /// Returns a partial result if a decode pass was triggered, std::nullopt otherwise.
    std::optional<ASRPartialResult> push_chunk(const std::vector<float>& pcm16k);

    /// Flush remaining buffered audio and return the final partial result.
    /// partial_text is always empty on the returned result.
    /// The session must not be used after this call.
    ASRPartialResult finish();

    // Public so that ASRPipelineImplBase subclasses can inherit from it without friendship.
    class Impl;

private:
    std::unique_ptr<Impl> m_impl;
};

}  // namespace ov::genai
