// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deque>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <set>
#include <vector>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/omni/talker_speech_config.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "sampling/sampling_overrides.hpp"
#include "utils.hpp"
#include "visual_language/embedding_model.hpp"

namespace ov::genai {

/// @brief Reads the token ids under the given keys of an LFM3-Omni config.json; each key must be defined.
std::vector<int64_t> read_lfm3_omni_token_ids(const nlohmann::json& config_json,
                                              std::initializer_list<const char*> names);

/// @brief True if the export has the depth_decoder, audio_out_embeddings and audio_decoder models of speech output.
bool has_lfm3_speech_models(const std::filesystem::path& models_dir);
bool has_lfm3_speech_models(const ModelsMap& models_map);

/// @brief Throws unless the speech models were found when the pipeline was built.
void check_lfm3_speech_models_loaded(bool loaded);

/// @brief Language model input [1, 1, hidden_size] of one token, owned copy.
ov::Tensor embed_lfm3_token(EmbeddingsModel& embedding, int64_t token_id);

/// @brief Prompt embeddings [1, n + 1, hidden_size]: the rows of prompt, then row [1, 1, hidden_size].
ov::Tensor append_lfm3_prompt_row(const ov::Tensor& prompt, const ov::Tensor& row);

/// @brief Per-call settings of LFM3-Omni speech decoding, checked against what the decoder implements.
struct LFM3SpeechRequest {
    /// @throws ov::Exception for speech fields LFM3-Omni has no use for and text fields the decoder does not implement.
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
    /// @brief Earlier frames decoded again with each streamed chunk.
    size_t audio_chunk_left_context_frames;
    size_t audio_rng_seed;
};

struct LFM3SpeechResult {
    /// @brief Generated text tokens, without the <|audio_out|> slot of each speech frame.
    std::vector<int64_t> text_tokens;
    /// @brief Sum of the log-probabilities the sampler reported for the text tokens; 0 for greedy text decode.
    float text_log_prob = 0.0f;
    /// @brief Every token added to the context after the prompt: the appended mode marker, if any, then each generated
    /// slot, with <|audio_out|> standing for each speech frame.
    std::vector<int64_t> slot_tokens;
    GenerationStatus status = GenerationStatus::RUNNING;
    GenerationFinishReason finish_reason = GenerationFinishReason::STOP;
    /// @brief Decode timings in the layout of the text pipelines: one new token time per generated slot.
    RawPerfMetrics raw_perf_metrics;
    /// @brief 24 kHz speech [1, 1, samples]: every sample emitted, as streamed chunks or one final decode. Empty when
    /// no sample was emitted.
    ov::Tensor waveform;
};

/// @brief LFM3-Omni speech settings from config.json, generation_config.json and audio_tokenizer/config.json.
struct LFM3SpeechModelConfig {
    size_t num_codebooks = 12;
    size_t codebook_size = 2017;
    int64_t audio_eos_code = 2016;
    std::vector<size_t> maskgit_schedule{4, 4, 4};
    size_t interleaved_n_text = 6;
    size_t interleaved_n_audio = 12;
    /// @brief Of the language model, from text_config.
    size_t vocab_size = 0;
    size_t hidden_size = 0;
    int64_t audio_out_token_id = -1;
    int64_t audio_start_token_id = -1;
    int64_t mixed_start_token_id = -1;
    int64_t text_end_token_id = -1;
    /// @brief Media placeholders, never sampled as text.
    std::vector<int64_t> placeholder_token_ids;
    std::optional<float> audio_temperature;
    std::optional<size_t> audio_top_k;
    float audio_repetition_penalty = 1.0f;
    size_t audio_repetition_window = 64;
    size_t text_repetition_window = 32;
    /// @brief Codec output samples per speech frame.
    size_t frame_samples = 1920;
    /// @brief Samples at the end of a decode that still change when later frames arrive: the ISTFT overlap.
    size_t decode_lookahead_samples = 480;

    static LFM3SpeechModelConfig read(const std::filesystem::path& config_dir_path);
};

/// @brief Text slots are sampled by the continuous batching sampler with these rules: the media placeholders are
/// suppressed, and the repetition penalty covers the last text_repetition_window text tokens, never <|audio_out|>.
SamplingOverrides lfm3_text_sampling_overrides(const LFM3SpeechModelConfig& config);

/// @brief Sampling of the codes of a speech frame.
struct LFM3SpeechSampling {
    std::optional<float> temperature;  ///< unset or 0: greedy
    std::optional<size_t> top_k;
    float repetition_penalty = 1.0f;

    /// @brief Draws one id from scores, which are changed in place. The order follows the transformers processors:
    /// penalty over history, suppression, then temperature and top-k.
    int64_t sample(std::vector<float>& scores, const std::vector<int64_t>& history, std::mt19937& rng) const;
};

class LFM3SpeechState;

/// @brief The three speech models of LFM3-Omni: depth decoder, audio_out_embeddings and the codec decoder.
class LFM3SpeechModels {
public:
    LFM3SpeechModels(const std::filesystem::path& models_dir, const std::string& device, const ov::AnyMap& properties);
    LFM3SpeechModels(const ModelsMap& models_map,
                     const std::filesystem::path& config_dir_path,
                     const std::string& device,
                     const ov::AnyMap& properties);

    const LFM3SpeechModelConfig& config() const {
        return m_config;
    }

    /// @param hidden_state Language model hidden state the frame follows, [1, hidden_size].
    /// @return The codes of one frame, drawn with the state's sampling settings and speech RNG.
    std::vector<int64_t> sample_frame(const ov::Tensor& hidden_state, LFM3SpeechState& state);

    /// @return Language model input [1, 1, hidden_size] of the frame, owned copy.
    ov::Tensor embed_frame(const std::vector<int64_t>& codes);

    /// @brief Waveform [1, (end - begin) * frame_samples] of frames [begin, end), owned copy.
    ov::Tensor decode_frames(const std::vector<std::vector<int64_t>>& frames, size_t begin, size_t end);

private:
    struct SubModels {
        std::shared_ptr<ov::Model> depth_decoder;
        std::shared_ptr<ov::Model> audio_out_embeddings;
        std::shared_ptr<ov::Model> audio_decoder;
    };

    LFM3SpeechModels(LFM3SpeechModelConfig config,
                     const SubModels& submodels,
                     const std::string& device,
                     const ov::AnyMap& properties);

    LFM3SpeechModelConfig m_config;
    ov::InferRequest m_depth_decoder;
    ov::InferRequest m_audio_out_embeddings;
    ov::InferRequest m_audio_decoder;
};

/// @brief Speech decoding state of one sequence: the text / speech mode machine, the speech repetition history and
/// RNG, the generated frames and the streamed waveform. The caller runs the language model, samples text slots and
/// feeds back each slot.
class LFM3SpeechState {
public:
    /// @param speech_streamer Optional; without it the waveform is decoded once in finish().
    LFM3SpeechState(const LFM3SpeechModelConfig& config,
                    const LFM3SpeechRequest& request,
                    const OmniSpeechStreamerVariant& speech_streamer);

    /// @brief The mode marker to append to the prompt, or -1 when the prompt already ends with it. Speech always
    /// follows its marker in training. An appended marker is recorded as the first slot token.
    int64_t take_prompt_marker(int64_t last_prompt_token);

    /// @param context_length Positions in the context after the prompt and the marker.
    void start(size_t context_length);

    bool next_is_frame() const;
    bool finished() const {
        return m_finished;
    }
    GenerationFinishReason finish_reason() const {
        return m_result.finish_reason;
    }
    /// @brief Slots generated so far; stops counting once finished. The appended prompt marker is not a slot.
    size_t generated_slots() const {
        return m_generated_slots;
    }

    /// @brief Ends the decode from outside the slot loop, e.g. when the caller stops or cancels the request.
    void end_with(GenerationStatus status);

    /// @brief Records a sampled text token and its log-probability; stream_status is the text streamer status that
    /// applies to it.
    void accept_text(int64_t token, float log_prob, StreamingStatus stream_status);

    /// @brief Ends the decode on a STOP or CANCEL of the text streamer, also one that arrives after the decode ended.
    void accept_text_stream_status(StreamingStatus stream_status);

    /// @brief Records the codes of a sampled frame and streams a chunk when one is due.
    /// @return The codes to feed back: all EOS codes for the end-of-speech frame.
    std::vector<int64_t> accept_frame(std::vector<int64_t> codes, LFM3SpeechModels& models);

    /// @brief Flushes the held-back samples unless cancelled, and returns the result with the whole waveform.
    LFM3SpeechResult finish(LFM3SpeechModels& models);

    const LFM3SpeechSampling& audio_sampling() const {
        return m_audio_sampling;
    }
    /// @brief The last audio_repetition_window frames, per codebook.
    std::vector<std::vector<int64_t>> codebook_history() const;
    std::mt19937& audio_rng() {
        return m_audio_rng;
    }

private:
    enum class Mode { TEXT, SPEECH, INTERLEAVED };

    StreamingStatus stream_up_to(size_t end, bool final, LFM3SpeechModels& models);
    /// @param interleaving Whether the slot was generated in interleaved mode.
    void end_slot(bool interleaving);

    const LFM3SpeechModelConfig& m_config;
    const OmniSpeechStreamerVariant& m_speech_streamer;
    bool m_stream_chunks;
    size_t m_max_audio_frames;
    size_t m_audio_chunk_frames;
    size_t m_left_context_frames;
    GenerationConfig m_text_config;
    LFM3SpeechSampling m_audio_sampling;
    std::set<int64_t> m_stop_token_ids;
    // Seeded by the speech rng_seed; text slots draw from the generator of their sampler.
    std::mt19937 m_audio_rng;

    Mode m_mode;
    size_t m_slots_left;
    bool m_speaking = false;
    bool m_text_done = false;
    bool m_finished = false;
    // A speech streamer that returned STOP or CANCEL gets no further write(), so finish() does not flush.
    bool m_speech_streamer_stopped = false;
    size_t m_max_new_slots = 0;
    size_t m_generated_slots = 0;
    std::deque<std::vector<int64_t>> m_audio_history;
    std::vector<std::vector<int64_t>> m_frames;
    std::vector<float> m_samples;
    size_t m_streamed_samples = 0;
    LFM3SpeechResult m_result;
};

}  // namespace ov::genai
