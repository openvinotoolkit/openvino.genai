// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <list>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/genai/common_types.hpp"
#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/omni/talker.hpp"
#include "openvino/genai/omni/talker_speech_config.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

/// @brief Configuration for Qwen3-Omni speech generation.
struct Qwen3OmniSpeechConfig {
    int64_t codec_bos_id = -1;
    int64_t codec_eos_token_id = -1;
    int64_t codec_pad_id = -1;
    int64_t codec_nothink_id = -1;
    int64_t codec_think_bos_id = -1;
    int64_t codec_think_eos_id = -1;
    int64_t tts_bos_token_id = -1;
    int64_t tts_eos_token_id = -1;
    int64_t tts_pad_token_id = -1;
    int64_t im_start_token_id = -1;
    int64_t system_token_id = -1;
    int64_t user_token_id = -1;
    int64_t assistant_token_id = -1;
    int64_t audio_token_id = -1;
    int64_t image_token_id = -1;
    int64_t video_token_id = -1;
    size_t num_code_groups = 16;
    size_t talker_hidden_size = 1024;
    std::map<std::string, int64_t> speaker_ids;

    // Talker generation parameters (loaded from generation_config.json)
    float talker_temperature = 0.9f;
    size_t talker_top_k = 50;
    float talker_repetition_penalty = 1.0f;
    size_t talker_max_new_tokens = 4096;
    size_t talker_vocab_size = 3072;
    // Token IDs in [vocab_size-1024, vocab_size) except codec_eos are suppressed
    std::vector<int64_t> talker_suppress_tokens;

    // CodePredictor sampling parameters (defaults match reference Qwen3-Omni;
    // overridable via generation_config.json keys: cp_temperature, cp_top_k, cp_repetition_penalty)
    float cp_temperature = 1.0f;
    size_t cp_top_k = 50;
    float cp_repetition_penalty = 1.0f;

    /// @brief Initialize from VLMConfig.
    static Qwen3OmniSpeechConfig from_vlm_config(const VLMConfig& config);
};

/// @brief Speech generation pipeline for Qwen3-Omni.
/// Runs Talker + CodePredictor + Code2Wav after the Thinker completes text generation.
class Qwen3OmniSpeechPipeline {
public:
    Qwen3OmniSpeechPipeline(const std::filesystem::path& model_dir,
                            const VLMConfig& config,
                            const std::string& device,
                            const ov::AnyMap& properties);

    /// @brief Construct from in-memory model IRs (blob deployment / per-submodel device placement).
    /// @param models_map Model name -> (IR string, weights tensor). Keys: "text_embeddings",
    ///                    "talker", "talker_text_embeddings", "talker_projections",
    ///                    "code_predictor", "code2wav".
    /// @param config Parsed VLM config (codec/token IDs, speaker_ids) from config.json.
    /// @param config_dir_path Directory holding generation_config.json (optional sampling defaults).
    /// @param device_mapping Submodel name -> device. Missing entries fall back to `default_device`.
    /// @param default_device Device used for submodels absent from `device_mapping`.
    Qwen3OmniSpeechPipeline(const ModelsMap& models_map,
                            const VLMConfig& config,
                            const std::filesystem::path& config_dir_path,
                            const std::map<std::string, std::string>& device_mapping,
                            const std::string& default_device,
                            const ov::AnyMap& properties);

    /// @brief Check if all speech models were loaded successfully.
    bool is_available() const {
        return m_talker_available;
    }

    /// @brief Generate speech from thinker generation results.
    /// @param full_token_ids All token IDs from the full sequence (prompt + generated).
    /// @param all_intermediate_hidden_states Accumulated layer-14 hidden states [one tensor per step].
    /// @param audio_streamer Callback or OmniSpeechStreamerBase for streaming (monostate = batch mode).
    /// @param talker_speech_config Speech generation knobs. Reads `audio_chunk_frames`, `speaker`
    ///                              (variant: name or embedding tensor), `max_new_tokens`, `rng_seed`,
    ///                              and the `talker_*` / `cp_*` optional sampling overrides. Unset
    ///                              overrides keep the checkpoint defaults from `generation_config.json`.
    /// @return TalkerResults with `waveforms` (the [1, 1, audio_samples] waveform; empty
    ///         on no-op or failure) and `perf_metrics` (talker-side timing + sample count).
    /// @note Not thread-safe per instance — shares the pipeline's owned std::mt19937 and
    ///       ov::InferRequests across talker and CodePredictor sampling.
    TalkerResults generate_speech(const std::vector<int64_t>& full_token_ids,
                                 const std::vector<ov::Tensor>& all_intermediate_hidden_states,
                                 const OmniSpeechStreamerVariant& audio_streamer,
                                 const OmniTalkerSpeechConfig& talker_speech_config);

    /// @brief Return precomputed speaker embedding for the named speaker. Throws if the model
    /// has no `talker_config.speaker_id` or the name doesn't match. Tensor shape is
    /// `[1, 1, talker_hidden_size]`, f32. Use to blend voices: weight-sum two named embeddings
    /// and assign the result to `OmniTalkerSpeechConfig::speaker` (the Tensor alternative).
    ov::Tensor get_speaker_embedding(const std::string& name) const;

    /// @brief List names of speakers available in the loaded model's `talker_config.speaker_id`.
    /// Returns an empty vector when the model exposes no named speakers.
    std::vector<std::string> list_speakers() const;

private:
    Qwen3OmniSpeechConfig m_config;
    bool m_talker_available = false;

    // Thinker text embeddings (for embedding TTS special tokens)
    ov::InferRequest m_thinker_text_embeddings;

    ov::InferRequest m_talker;
    ov::InferRequest m_talker_text_embeddings;
    ov::InferRequest m_talker_projections;  // dual-output: text_projection + hidden_projection
    ov::InferRequest m_code_predictor;
    ov::InferRequest m_code2wav;

    // LRU cache for talker token embeddings to avoid redundant inference.
    // Bounded to kMaxEmbeddingCacheSize entries; evicts least recently used on overflow.
    static constexpr size_t kMaxEmbeddingCacheSize = 256;
    std::list<std::pair<int64_t, ov::Tensor>> m_embedding_lru_list;
    std::unordered_map<int64_t, std::list<std::pair<int64_t, ov::Tensor>>::iterator> m_embedding_lru_map;

    // Pre-allocated scratch buffers reused across generate_speech() calls
    std::vector<float> m_talker_buf;
    ov::Tensor m_cp_embed_sum;
    ov::Tensor m_stack_codes_buf;

    // Constant embeddings pre-computed at pipeline construction. TTS specials are in
    // talker space (thinker embed -> text projection). Codec specials and speaker IDs
    // are in talker-embedding space. All are invariant for the life of the pipeline.
    ov::Tensor m_tts_bos_embed;
    ov::Tensor m_tts_eos_embed;
    ov::Tensor m_tts_pad_embed;
    std::unordered_map<int64_t, ov::Tensor> m_codec_special_embed;
    std::unordered_map<int64_t, ov::Tensor> m_speaker_embed;

    // Lowercased speaker name -> codec id. Built once at ctor time so resolve_speaker_id
    // does not rescan-and-lowercase the whole map on every generate_speech() call.
    std::unordered_map<std::string, int64_t> m_lower_speaker_ids;

    // Owned RNG for deterministic sampling. Reseeded at generate_speech() entry from the
    // caller-supplied rng_seed; shared across talker first-code sampling and all CodePredictor
    // steps so a single seed fully reproduces the audio output.
    std::mt19937 m_rng{0};

    // Per-instance scratch buffers for sample_top_k(). Previously thread_local statics inside
    // the function; now members because the function is no longer static (needs m_rng).
    std::vector<float> m_sample_scaled;
    std::vector<size_t> m_sample_indices;
    std::vector<size_t> m_sample_topk_indices;
    std::vector<float> m_sample_topk_probs;

    /// @brief Post-load initialization shared by both constructors. Validates submodel IO,
    /// detects hidden/vocab sizes, loads generation params from generation_config.json (if
    /// present under `config_dir`), and precomputes constant embeddings. Requires the six
    /// InferRequest members to already be populated.
    void initialize(const std::filesystem::path& config_dir);

    /// @brief Resolve speaker name to codec token ID.
    int64_t resolve_speaker_id(const std::string& speaker) const;

    /// @brief Embed a token via thinker word embeddings (for TTS special tokens).
    /// @return Tensor [1, 1, thinker_hidden_size].
    ov::Tensor embed_thinker_token(int64_t token_id);

    /// @brief Embed a single token via talker text embeddings.
    /// @return Tensor [1, 1, talker_hidden_size].
    ov::Tensor embed_talker_token(int64_t token_id);

    /// @brief Project thinker hidden state through text projection.
    /// @param hidden_state [1, seq_len, thinker_hidden_size]
    /// @return [1, seq_len, talker_hidden_size]
    ov::Tensor project_text(const ov::Tensor& hidden_state);

    /// @brief Project thinker intermediate hidden state through hidden projection.
    /// @param hidden_state [1, seq_len, thinker_hidden_size]
    /// @return [1, seq_len, talker_hidden_size]
    ov::Tensor project_hidden(const ov::Tensor& hidden_state);

    /// @brief Build the talker input embeddings from thinker outputs.
    /// @param speaker_embed Speaker embedding `[1, 1, talker_hidden_size]` summed with tts_pad in
    ///                      the talker prefix. Caller owns shape validation.
    /// @return Pair of (talker_input_embeds, trailing_text_hidden).
    std::pair<ov::Tensor, ov::Tensor> build_talker_input(const std::vector<int64_t>& full_token_ids,
                                                         const std::vector<ov::Tensor>& all_intermediate_hidden_states,
                                                         const ov::Tensor& speaker_embed);

    /// @brief Run the CodePredictor mini-loop for one talker step.
    /// Drives the single-step stateful CodePredictor graph num_code_groups-1 times. Sampling and
    /// codec embedding are in-graph: each call returns the sampled code and its embedding, which is
    /// fed back as the next step's input.
    /// @param talker_hidden_state The last hidden state from talker [1, 1, talker_hidden_size].
    /// @param first_code The first codec code from talker sampling.
    /// @param cp_temperature Resolved CodePredictor temperature for this call.
    /// @param cp_top_k Resolved CodePredictor top-k for this call.
    /// @return Pair of (additional_codes, codec_embeddings_sum) where codec_embeddings_sum is the
    ///         first-code embedding plus all step-specific codec embeddings [1, 1, hidden_size].
    std::pair<std::vector<int64_t>, ov::Tensor> predict_codes(const ov::Tensor& talker_hidden_state,
                                                              int64_t first_code,
                                                              float cp_temperature,
                                                              size_t cp_top_k);

    /// @brief Convert codec codes to waveform.
    ov::Tensor codes_to_wav(const ov::Tensor& codes);

    /// @brief Reset talker KV cache state.
    void reset_talker();

    /// @brief Sample from logits with temperature, top-k, repetition penalty, and token suppression.
    /// @note Member function (mutates m_rng and scratch buffers). Uses this instance's RNG so
    ///       talker and CodePredictor draws come from a single seeded stream.
    int64_t sample_top_k(const float* logits,
                         size_t vocab_size,
                         float temperature,
                         size_t top_k,
                         float repetition_penalty,
                         const std::vector<int64_t>& generated_tokens,
                         const std::vector<int64_t>& suppress_tokens = {});
};

}  // namespace ov::genai
