// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/omni/decoded_results.hpp"
#include "openvino/genai/omni/speech_generation_config.hpp"
#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"

namespace ov::genai {

/**
 * @brief Public abstract interface for OmniPipeline's speech-output backend.
 *
 * The talker consumes the hidden states accumulated during VLM generation and produces
 * a waveform plus optional streamed audio chunks. Inject a custom subclass into
 * `OmniPipeline(shared_ptr<VLMPipelineBase>, shared_ptr<TalkerBase>)` to swap the
 * default Qwen3-Omni speech stack for an alternative implementation (e.g. a different
 * codec, a different talker model, or a mock for testing).
 *
 * The default implementation `Qwen3OmniTalker` (declared in this header) wraps the
 * existing Qwen3-Omni Talker + CodePredictor + Code2Wav stack.
 */
class OPENVINO_GENAI_EXPORTS TalkerBase {
public:
    virtual ~TalkerBase() = default;

    /// @brief Run speech generation against a VLM result.
    /// @param vlm_result VLM-side text result; must carry hidden states from a generate()
    ///                   call that ran with `speech_config.return_audio == true`. The
    ///                   talker reads `m_hidden_states_data` (final + intermediate
    ///                   hidden states + prompt token ids) from the result.
    /// @param speech_config Generation knobs for the talker (`return_audio`, `speaker`,
    ///                      `speaker_embedding`, `audio_chunk_frames`, `max_new_tokens`,
    ///                      `rng_seed`, `validate()` already enforced).
    /// @param speech_streamer Optional callback or `OmniSpeechStreamerBase` for streaming
    ///                        audio chunks. `monostate` = batch mode (single waveform).
    /// @return OmniDecodedResults with `speech_outputs` populated when `return_audio` is true.
    virtual OmniDecodedResults generate(VLMDecodedResults vlm_result,
                                        const OmniSpeechGenerationConfig& speech_config,
                                        const OmniSpeechStreamerVariant& speech_streamer = std::monostate{}) = 0;

    /// @brief List names of speakers exposed by this talker.
    /// Returns an empty vector when the backend does not enumerate named speakers.
    virtual std::vector<std::string> list_speakers() const = 0;

    /// @brief Return precomputed speaker embedding for the named speaker.
    /// Tensor shape and dtype are backend-defined (Qwen3-Omni: `[1, 1, talker_hidden_size]`, f32).
    /// Use to blend voices: combine two named embeddings and pass the result via
    /// `OmniSpeechGenerationConfig::speaker_embedding`.
    /// @throws when the backend has no named speakers, or `name` doesn't match.
    virtual ov::Tensor get_speaker_embedding(const std::string& name) const = 0;

    /// @brief True when the talker is fully loaded and able to produce audio. False when
    /// optional submodels are missing (e.g. the model directory ships VLM weights only).
    virtual bool is_available() const = 0;
};

/**
 * @brief Default OmniPipeline talker for Qwen3-Omni Talker + CodePredictor + Code2Wav.
 *
 * Loads the speech stack from a model directory (the same directory as the VLM weights
 * for path-based OmniPipeline construction; an arbitrary directory containing the
 * speech submodels for the DI ctor).
 *
 * Public-API thin wrapper around the internal `Qwen3OmniSpeechPipeline`. Construct
 * directly when you want to inject your own VLMPipelineBase but keep the default speech
 * backend:
 *
 *     auto vlm = ov::genai::VLMPipeline::create_base(model_dir, "GPU", props);
 *     auto talker = std::make_shared<ov::genai::Qwen3OmniTalker>(model_dir, "CPU", props);
 *     ov::genai::OmniPipeline pipe(vlm, talker);
 */
class OPENVINO_GENAI_EXPORTS Qwen3OmniTalker : public TalkerBase {
public:
    /// @brief Load Qwen3-Omni speech submodels from a directory containing
    /// `openvino_talker_model.xml`, `openvino_code_predictor_model.xml`,
    /// `openvino_code2wav_model.xml`, plus the talker text-embedding and projection
    /// submodels. The directory must contain `config.json` (used for talker config).
    Qwen3OmniTalker(const std::filesystem::path& model_dir,
                    const std::string& device,
                    const ov::AnyMap& properties = {});

    ~Qwen3OmniTalker() override;
    Qwen3OmniTalker(Qwen3OmniTalker&&) noexcept;
    Qwen3OmniTalker& operator=(Qwen3OmniTalker&&) noexcept;
    Qwen3OmniTalker(const Qwen3OmniTalker&) = delete;
    Qwen3OmniTalker& operator=(const Qwen3OmniTalker&) = delete;

    OmniDecodedResults generate(VLMDecodedResults vlm_result,
                                const OmniSpeechGenerationConfig& speech_config,
                                const OmniSpeechStreamerVariant& speech_streamer = std::monostate{}) override;

    std::vector<std::string> list_speakers() const override;
    ov::Tensor get_speaker_embedding(const std::string& name) const override;
    bool is_available() const override;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace ov::genai
