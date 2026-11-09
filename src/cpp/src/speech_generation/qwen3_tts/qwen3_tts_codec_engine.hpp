// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <random>
#include <string>
#include <unordered_map>

#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {

// Shared Qwen3-TTS codec subsystem: every submodel EXCEPT the talker backbone (talker embedding
// table, text embedding/projection, code predictor, speech tokenizer encoder/decoder, speaker
// encoder), plus the per-frame code-predictor decode loop, sampling, prefill-assembly helpers and
// waveform decoding built on top of them. Reused as-is by both the SDPA driver (Qwen3TTSImpl) and
// the continuous-batching driver (Qwen3TTSContinuousBatchingImpl); only the talker backbone's own
// inference mechanism (stateful infer vs. PagedAttention) differs between the two.
class Qwen3TTSCodecEngine {
public:
    struct QwenIds {
        int64_t tts_bos_token_id = -1;
        int64_t tts_eos_token_id = -1;
        int64_t tts_pad_token_id = -1;

        int64_t codec_bos_id = -1;
        int64_t codec_pad_id = -1;
        int64_t codec_eos_token_id = -1;
        int64_t codec_think_id = -1;
        int64_t codec_nothink_id = -1;
        int64_t codec_think_bos_id = -1;
        int64_t codec_think_eos_id = -1;

        size_t num_code_groups = 16;
        size_t talker_vocab_size = 3072;

        std::unordered_map<std::string, int64_t> codec_language_id;
        std::unordered_map<std::string, int64_t> spk_id;
        std::unordered_map<std::string, std::string> spk_is_dialect;
    };

    // Result of projecting tts_bos/tts_eos/tts_pad through the text embedding model.
    struct SpecialEmbeds {
        ov::Tensor bos;
        ov::Tensor eos;
        ov::Tensor pad;
        size_t hidden = 0;
    };

    struct CodecGroupBatchItem {
        ov::Tensor past_hidden;
        int64_t first_codec_token = -1;
        const SpeechGenerationConfig* generation_config = nullptr;
        std::mt19937* rng = nullptr;
    };

    Qwen3TTSCodecEngine(const std::filesystem::path& models_path,
                       const std::string& device,
                       const ov::AnyMap& properties,
                       const Tokenizer& tokenizer);

    const QwenIds& ids() const {
        return m_ids;
    }

    Tokenizer tokenizer() const {
        return m_tokenizer;
    }

    bool is_base_model() const {
        return m_tts_model_type == "base";
    }

    const std::string& tts_model_type() const {
        return m_tts_model_type;
    }

    uint32_t output_sample_rate() const {
        return m_output_sample_rate;
    }

    size_t speaker_embedding_dim() const {
        return m_speaker_embedding_dim;
    }

    size_t talker_hidden_size() const {
        return m_talker_hidden_size;
    }

    ov::InferRequest& talker_embedding() {
        return m_talker_embedding;
    }

    ov::InferRequest& talker_text_embedding() {
        return m_talker_text_embedding;
    }

    // for per-component inference timing (see qwen3_tts::run_and_time); shared across every
    // submodel this engine owns. Callers (Qwen3TTSImpl / Qwen3TTSContinuousBatchingImpl) may add
    // their own talker-specific entries (e.g. talker prefill/generate timing) into the same maps.
    std::unordered_map<std::string, double>& perf_ms() const {
        return m_perf_ms;
    }
    std::unordered_map<std::string, int64_t>& perf_calls() const {
        return m_perf_calls;
    }
    std::unordered_map<std::string, std::string>& perf_device() const {
        return m_perf_device;
    }
    // Prints and clears the accumulated per-component timing (no-op unless
    // OV_GENAI_QWEN_TTS_PERF is set). talker_prefill_tokens is reported alongside the caller's
    // own TALKER_PREFILL_PERF_NAME entry, since that counter is talker-driver-specific.
    void perf_print_and_reset(int64_t talker_prefill_tokens) const;

    ov::Tensor infer_embedding(ov::InferRequest& request, int64_t token_id);
    ov::Tensor infer_embedding_seq(ov::InferRequest& request, const std::vector<int64_t>& token_ids);

    // Single-token step against the static all-heads code predictor. Manages the
    // host-side explicit KV cache and absolute-position counter internally; pass
    // reset_state=true at the start of each per-frame code-group prediction.
    // Returns [1,T,vocab] for stateful predictor or [1,1,vocab] for static predictor.
    ov::Tensor infer_predictor(const ov::Tensor& inputs_embeds, bool reset_state, int64_t step = 0);
    ov::Tensor infer_predictor_embedding(int64_t token_id, int64_t generation_step);
    ov::Tensor infer_predictor_embedding_seq(const std::vector<int64_t>& token_ids, int64_t generation_step);
    // Zero host KV buffers and reset the running absolute-position counter.
    void reset_predictor_state();

    int64_t sample_token_from_logits(const ov::Tensor& logits,
                                     const SpeechGenerationConfig& generation_config,
                                     const std::vector<int64_t>& generated,
                                     const std::vector<bool>& suppressed,
                                     std::mt19937& rng,
                                     size_t batch_index = 0) const;
    // Predicts the remaining `num_code_groups - 1` codec ids for one talker frame, given the
    // talker's last hidden state for that frame and the already-sampled first (group-0) codec id.
    std::vector<int64_t> generate_codec_groups(const ov::Tensor& past_hidden,
                                               int64_t first_codec_token,
                                               const SpeechGenerationConfig& generation_config,
                                               std::mt19937& rng);
    std::vector<std::vector<int64_t>> generate_codec_groups_batch(
        const std::vector<CodecGroupBatchItem>& batch_items);

    std::vector<float> decode_speech_tokenizer(const std::vector<int64_t>& codes);

    std::vector<bool> compute_suppress_tokens() const;
    std::vector<int64_t> make_speaker_and_codec_prefill_ids(const std::string& language,
                                                            const std::string& speaker) const;
    std::string normalize_text_language(const std::string& language) const;
    std::string normalize_speaker(const std::string& speaker) const;
    ov::Tensor normalize_external_speaker_embedding(const ov::Tensor& speaker_embedding, size_t hidden_size) const;
    std::vector<float> normalize_ref_audio_waveform(const ov::Tensor& ref_audio) const;
    ov::Tensor extract_qwen3_speaker_embedding_from_audio(const ov::Tensor& ref_audio) const;
    ov::Tensor extract_qwen3_ref_code_from_audio(const ov::Tensor& ref_audio) const;

    SpecialEmbeds infer_special_embeds();
    ov::Tensor prepend_instruct_embed(ov::Tensor talker_prefill, const std::string& instruct);

    // Result of build_prefill(): the talker's prefill embedding plus the per-frame additive terms
    // decode_from_prefill()-style loops need afterwards (trailing_text_hidden is an empty tensor in
    // non_streaming_mode).
    struct PrefillResult {
        ov::Tensor talker_prefill;
        ov::Tensor tts_pad;
        ov::Tensor trailing_text_hidden;
    };

    struct VoiceClonePrefillResult {
        PrefillResult prefill;
        ov::Tensor ref_codec_ids;
    };

    // Builds the talker prefill embedding for the non-ICL path (CustomVoice / VoiceDesign / Base
    // x-vector). Shared by the SDPA driver (Qwen3TTSImpl) and the continuous-batching driver.
    // `input_ids_tensor` is the already-tokenized "<|im_start|>assistant\n...<|im_start|>assistant\n"
    // turn; `effective_speaker_embedding` must already be resolved (x-vector extracted from ref_audio
    // if needed) by the caller; `language`/`speaker` are the caller's already-normalized values.
    PrefillResult build_prefill(const ov::Tensor& input_ids_tensor,
                               const ov::Tensor& effective_speaker_embedding,
                               const SpeechGenerationConfig& generation_config,
                               const std::string& language,
                               const std::string& speaker,
                               bool base_model);

    VoiceClonePrefillResult build_voice_clone_prefill(const std::string& text,
                                                      const std::string& ref_text,
                                                      const ov::Tensor& speaker_embedding,
                                                      const ov::Tensor& ref_codec_ids,
                                                      const SpeechGenerationConfig& generation_config);

    std::vector<float> decode_speech_tokenizer_with_reference(const ov::Tensor& ref_codec_ids,
                                                              const std::vector<int64_t>& generated_codes);

private:
    void init_config(const std::filesystem::path& models_path);
    ov::Tensor infer_predictor_batch(const ov::Tensor& inputs_embeds, int64_t step);
    // Read static-predictor dims from the IR and (re)allocate host KV buffers.
    void init_static_predictor_meta(const std::shared_ptr<ov::Model>& model);
    // Reshape a dynamically-exported code predictor IR to the required static
    // shapes. Derives the KV window from the stacked-heads `logits` output and
    // the fixed n_kv/head_dim of past_key_values.
    void reshape_predictor_to_static(const std::shared_ptr<ov::Model>& model);

    Tokenizer m_tokenizer;
    QwenIds m_ids;
    std::string m_tts_model_type = "custom_voice";
    size_t m_speaker_embedding_dim = 1024;
    size_t m_talker_hidden_size = 1024;  // talker hidden_size; predictor inputs_embeds width

    ov::InferRequest m_talker_embedding;
    ov::InferRequest m_talker_text_embedding;
    ov::InferRequest m_talker_code_predictor;
    bool m_predictor_static = false;  // true when predictor is converted to stateless+static at runtime
    bool m_predictor_batching_supported = false;
    bool m_predictor_has_beam_idx = false;
    ov::InferRequest m_talker_code_predictor_embedding;

    // Static all-heads code predictor state. The loaded predictor IR must expose
    // explicit past_key_values inputs (a non-static model is rejected at
    // construction); the host drives the KV cache, see infer_predictor.
    size_t m_pred_num_layers = 0;
    size_t m_pred_n_kv = 0;
    size_t m_pred_head_dim = 0;
    size_t m_pred_past_len = 0;   // KV slots in past_key_values inputs
    size_t m_pred_kv_len = 0;     // KV slots in present outputs (= past_len + 1)
    size_t m_pred_num_heads = 0;  // stacked code-group heads in `logits`
    size_t m_pred_vocab = 0;
    size_t m_pred_position = 0;   // running absolute position within a frame
    std::vector<ov::Tensor> m_pred_past_k;  // per-layer host KV buffers
    std::vector<ov::Tensor> m_pred_past_v;
    // Pre-allocated reusable tensors for infer_predictor (avoid repeated allocations)
    ov::Tensor m_pred_attn;   // attention_mask, reused across calls
    ov::Tensor m_pred_pos;    // position_ids, reused across calls
    // Pre-allocated reusable tensors for infer_predictor_embedding (avoid repeated allocations)
    ov::Tensor m_pred_emb_ids;  // token ids input, reused across calls
    ov::Tensor m_pred_emb_step;  // generation_steps input, reused across calls
    ov::InferRequest m_speech_tokenizer_decoder;
    bool m_decoder_static = false;  // true when reshaped to a fixed DECODER_WINDOW_LEN window (NPU/GPU)
    ov::InferRequest m_qwen3_mel_preprocess;
    ov::InferRequest m_speaker_encoder;
    ov::InferRequest m_speech_tokenizer_encoder;

    uint32_t m_output_sample_rate = 24000;
    mutable std::unordered_map<std::string, double> m_perf_ms;
    mutable std::unordered_map<std::string, int64_t> m_perf_calls;
    mutable std::unordered_map<std::string, std::string> m_perf_device;

    uint32_t m_decoder_num_quantizers = 16;
    uint32_t m_decoder_upsample = 1920;
    uint32_t m_speaker_encoder_mel_dim = 128;

    bool m_has_speaker_encoder = false;
    bool m_has_speech_tokenizer_encoder = false;
    bool m_has_qwen3_mel_preprocess = false;
};

}  // namespace genai
}  // namespace ov
