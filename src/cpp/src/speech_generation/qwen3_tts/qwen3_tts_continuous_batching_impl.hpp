// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <random>
#include <unordered_map>

#include "continuous_batching/pipeline_impl.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "sequence_group.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_codec_engine.hpp"
#include "speech_generation/speech_generation_stream.hpp"

namespace ov {
namespace genai {

// Continuous-batching driver for the Qwen3-TTS talker. The talker backbone runs under
// PagedAttention via the base ContinuousBatchingImpl scheduler/cache/model-runner, batching the
// talker's forward pass across multiple concurrent sessions. The code-predictor/sampling/decode
// subsystem (Qwen3TTSCodecEngine, shared with the SDPA driver Qwen3TTSImpl) runs once per frame,
// per sequence, synchronously inside generate_candidates_for_prompt_lookup() -- the hook point
// ContinuousBatchingImpl::step() calls right after sampling and before the automatic
// embedding-table-lookup fallback (append_embeddings()), which this class's per-frame embedding
// injection always preempts (append_embeddings() becomes a no-op once every generated id already
// has a matching entry in get_generated_ids_embeds()).
//
// Scope: only the non-ICL prefill (CustomVoice / VoiceDesign / Base x-vector) is supported here;
// Base ref-audio/ICL voice cloning remains SDPA-only (Text2SpeechPipeline) for now.
class ContinuousBatchingPipeline::Qwen3TTSContinuousBatchingImpl : public ContinuousBatchingPipeline::ContinuousBatchingImpl {
public:
    Qwen3TTSContinuousBatchingImpl(const std::filesystem::path& models_path,
                                   const SchedulerConfig& scheduler_config,
                                   const std::string& device,
                                   const ov::AnyMap& properties,
                                   const Tokenizer& tokenizer);

    SpeechGenerationHandle add_speech_request(uint64_t request_id,
                                              const std::string& text,
                                              const ov::Tensor& speaker_embedding,
                                              const SpeechGenerationConfig& generation_config);

    ov::Shape get_speaker_embedding_shape() const;
    uint32_t get_output_sample_rate() const;

protected:
    void generate_candidates_for_prompt_lookup() override;

private:
    // Per-request state that persists across step() calls. Qwen3-TTS has no equivalent of the
    // token-oriented GenerationStream SequenceGroup already owns, hence the separate map here.
    struct RequestState {
        SpeechGenerationConfig generation_config;
        std::mt19937 rng;
        ov::Tensor tts_pad;
        ov::Tensor trailing_text_hidden;
        std::vector<int64_t> all_codes;
        SpeechGenerationStream::Ptr stream;
    };

    void process_new_frame(const SequenceGroup::Ptr& sequence_group, const Sequence::Ptr& sequence, RequestState& state);
    void finalize_request(const SequenceGroup::Ptr& sequence_group, RequestState& state);

    Qwen3TTSCodecEngine m_engine;
    std::mutex m_request_state_mutex;
    std::unordered_map<uint64_t, RequestState> m_request_state;
};

}  // namespace genai
}  // namespace ov
