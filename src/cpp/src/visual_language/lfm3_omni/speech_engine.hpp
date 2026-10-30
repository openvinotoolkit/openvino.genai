// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "circular_buffer_queue.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/omni/talker_speech_config.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "utils.hpp"
#include "visual_language/embedding_model.hpp"

namespace ov::genai {

/// @brief Per-call settings of LFM3OmniSpeechEngine::generate, checked against what the engine implements.
struct LFM3SpeechRequest {
    /// @throws ov::Exception for speech fields LFM3-Omni has no use for and text fields the engine does not implement.
    LFM3SpeechRequest(const OmniTalkerSpeechConfig& speech_config, const GenerationConfig& text_config);

    /// @brief Rejects speaker selection and the talker_* fields: LFM3-Omni has no separate talker.
    static void validate(const OmniTalkerSpeechConfig& speech_config);

    SpeechOutputMode mode;
    /// @brief Text decode: do_sample, temperature, top_k, top_p, min_p, repetition_penalty, rng_seed, stop ids, and
    /// max_new_tokens, which caps every generated slot, speech frames included.
    GenerationConfig text_config;
    /// @brief Speech frame sampling; unset values fall back to generation_config.json.
    std::optional<float> audio_temperature;
    std::optional<size_t> audio_top_k;
    std::optional<float> audio_repetition_penalty;
    size_t max_audio_frames;
    /// @brief Speech frames decoded per streamed chunk.
    size_t audio_chunk_frames;
    size_t audio_rng_seed;
};

struct LFM3SpeechResult {
    /// @brief Generated text tokens, without the <|audio_out|> slot of each speech frame.
    std::vector<int64_t> text_tokens;
    /// @brief Every token added to the context after the prompt: the appended mode marker, if any, then each generated
    /// slot, with <|audio_out|> standing for each speech frame.
    std::vector<int64_t> slot_tokens;
    GenerationStatus status = GenerationStatus::RUNNING;
    GenerationFinishReason finish_reason = GenerationFinishReason::STOP;
    /// @brief Decode timings in the layout of the text pipelines: one new token time per generated slot.
    RawPerfMetrics raw_perf_metrics;
};

/// @brief Interleaved text and speech decode of LFM3-Omni. The language model emits text tokens and speech frames
/// in one stream: the depth decoder samples the codebooks of a frame from the last hidden state, and the frame is fed
/// back through audio_out_embeddings. Borrows the stateful language model and text embeddings of the VLM pipeline.
class LFM3OmniSpeechEngine {
public:
    LFM3OmniSpeechEngine(const std::filesystem::path& models_dir,
                         ov::InferRequest& language,
                         EmbeddingsModel::Ptr embedding,
                         const std::string& device,
                         const ov::AnyMap& properties);

    LFM3OmniSpeechEngine(const ModelsMap& models_map,
                         const std::filesystem::path& config_dir_path,
                         ov::InferRequest& language,
                         EmbeddingsModel::Ptr embedding,
                         const std::string& device,
                         const ov::AnyMap& properties);

    /// @param inputs_embeds Prompt embeddings [1, seq_len, hidden_size].
    /// @param attention_mask Ones over the cached history and the prompt, [1, history + seq_len].
    /// @param last_prompt_token Last prompt token id; a marker already in place is not appended again.
    /// @param speech_streamer Receives the 24 kHz waveform in chunks of request.audio_chunk_frames frames,
    /// [1, 1, samples]; the only output of the speech, so it must be set.
    LFM3SpeechResult generate(const ov::Tensor& inputs_embeds,
                              const ov::Tensor& attention_mask,
                              int64_t last_prompt_token,
                              const LFM3SpeechRequest& request,
                              const std::shared_ptr<StreamerBase>& streamer,
                              const OmniSpeechStreamerVariant& speech_streamer);

private:
    struct ModelConfig {
        size_t num_codebooks = 12;
        size_t codebook_size = 2017;
        int64_t audio_eos_code = 2016;
        std::vector<size_t> maskgit_schedule{4, 4, 4};
        size_t interleaved_n_text = 6;
        size_t interleaved_n_audio = 12;
        int64_t audio_out_token_id = -1;
        int64_t audio_start_token_id = -1;
        int64_t mixed_start_token_id = -1;
        int64_t text_end_token_id = -1;
        /// @brief Media placeholders, never sampled as text.
        std::vector<int64_t> placeholder_token_ids;
        std::optional<float> audio_temperature;
        std::optional<size_t> audio_top_k;
        std::optional<float> audio_repetition_penalty;
        size_t audio_repetition_window = 64;
        size_t text_repetition_window = 32;
        /// @brief Codec output samples per speech frame.
        size_t frame_samples = 1920;
        /// @brief Samples at the end of a decode that still change when later frames arrive: the ISTFT overlap.
        size_t decode_lookahead_samples = 480;
    };

    struct Sampling {
        std::optional<float> temperature;  ///< unset or 0: greedy
        std::optional<size_t> top_k;
        float top_p = 1.0f;
        float min_p = 0.0f;
        float repetition_penalty = 1.0f;
    };

    static ModelConfig read_model_config(const std::filesystem::path& config_dir_path);

    struct StepOutput {
        std::vector<float> logits;
        ov::Tensor hidden_state;  ///< [1, hidden_size], owned copy
    };

    StepOutput language_step(const ov::Tensor& inputs_embeds, size_t attention_length);
    ov::Tensor embed_token(int64_t token_id);
    ov::Tensor embed_frame(const std::vector<int64_t>& codes);
    std::vector<int64_t> sample_frame(const ov::Tensor& hidden_state,
                                      const Sampling& sampling,
                                      const std::vector<std::vector<int64_t>>& history);
    static int64_t sample(std::vector<float>& scores,
                          const Sampling& sampling,
                          const std::vector<int64_t>& history,
                          const std::vector<int64_t>& suppress,
                          std::mt19937& rng);
    /// @brief Waveform [1, (end - begin) * frame_samples] of frames [begin, end).
    ov::Tensor decode_frames(const std::vector<std::vector<int64_t>>& frames, size_t begin, size_t end);

    ModelConfig m_config;
    ov::InferRequest& m_language;
    EmbeddingsModel::Ptr m_embedding;
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_depth_decoder;
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_audio_out_embeddings;
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_audio_decoder;
    // Text and speech draw from separate generators, seeded by text_config.rng_seed and the speech rng_seed.
    std::mt19937 m_text_rng;
    std::mt19937 m_audio_rng;
};

}  // namespace ov::genai
