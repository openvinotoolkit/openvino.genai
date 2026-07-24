// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/common_types.hpp"
#include "openvino/genai/omni/speech_streamer_base.hpp"
#include "openvino/genai/omni/talker_perf_metrics.hpp"
#include "openvino/genai/omni/talker_speech_config.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"

namespace ov::genai {

/**
 * @brief Output of TalkerBase::generate. Holds the talker-side waveform plus its perf metrics.
 * OmniPipelineImpl combines this with VLMDecodedResults from the VLM stage to produce the
 * public OmniDecodedResults.
 *
 * @note This is a preview API and is subject to change.
 */
struct OPENVINO_GENAI_EXPORTS TalkerResults {
    /// Speech output waveforms. Empty when `talker_speech_config.return_audio == false`.
    std::vector<ov::Tensor> waveforms;

    /// Speech-side perf metrics: wall-clock time and sample count.
    /// Populated even when `waveforms` is empty so callers can see how long the no-op path took.
    TalkerPerfMetrics perf_metrics;
};

/**
 * @brief Public abstract interface for OmniPipeline's speech-output backend.
 *
 * The talker consumes the hidden states accumulated during VLM generation and produces
 * a waveform plus optional streamed audio chunks. Inject a custom subclass into
 * `OmniPipeline(shared_ptr<VLMPipeline::VLMBackend>, shared_ptr<TalkerBase>)` to swap the
 * default Qwen3-Omni speech stack for an alternative implementation (e.g. a different
 * codec, a different talker model, or a mock for testing).
 *
 * The default implementation `Talker` (declared in this header) wraps the
 * existing Qwen3-Omni Talker + CodePredictor + Code2Wav stack.
 *
 * @note This is a preview API and is subject to change.
 */
class OPENVINO_GENAI_EXPORTS TalkerBase {
public:
    virtual ~TalkerBase() = default;

    /// @brief Run speech generation against a VLM result.
    /// @param vlm_result VLM-side text result; must carry hidden states from a generate()
    ///                   call that ran with `talker_speech_config.return_audio == true`. The
    ///                   talker reads `intermediate_hidden_states` and `full_token_ids` from the
    ///                   result. Passed by const-ref — the talker does
    ///                   not own the VLM result; OmniPipelineImpl assembles the final
    ///                   `OmniDecodedResults` from both the VLM and talker outputs.
    /// @param talker_speech_config Generation knobs for the talker (`return_audio`, `speaker`
    ///                             (name or embedding), `audio_chunk_frames`, `max_new_tokens`,
    ///                             `rng_seed`, validated before calling).
    /// @param speech_streamer Optional callback or `OmniSpeechStreamerBase` for streaming
    ///                        audio chunks. `monostate` = batch mode (single waveform).
    /// @return TalkerResults with `waveforms` populated when `return_audio` is true and
    ///         `perf_metrics` populated regardless.
    virtual TalkerResults generate(const VLMDecodedResults& vlm_result,
                                  const OmniTalkerSpeechConfig& talker_speech_config,
                                  const OmniSpeechStreamerVariant& speech_streamer = std::monostate{}) = 0;

    /// @brief Property-bag speech generation. `properties` recognizes `speech_streamer` plus any
    /// `OmniTalkerSpeechConfig` field. How unspecified fields are resolved is backend-defined; the
    /// default `Talker` falls back to its stored config (see `Talker::get_speech_config` /
    /// `set_speech_config`). Convenience wrapper over the typed overload for callers that build
    /// config from an ov::AnyMap.
    virtual TalkerResults generate(const VLMDecodedResults& vlm_result, const ov::AnyMap& properties = {}) = 0;

    /// @brief List names of speakers exposed by this talker.
    /// Returns an empty vector when the backend does not enumerate named speakers.
    virtual std::vector<std::string> list_speakers() const = 0;

    /// @brief Return precomputed speaker embedding for the named speaker.
    /// Tensor shape and dtype are backend-defined (Qwen3-Omni: `[1, 1, talker_hidden_size]`, f32).
    /// Use to blend voices: combine two named embeddings and pass the result via
    /// `OmniTalkerSpeechConfig::speaker` (the Tensor alternative of the variant).
    /// @throws when the backend has no named speakers, or `name` doesn't match.
    virtual ov::Tensor get_speaker_embedding(const std::string& name) const = 0;
};

/**
 * @brief Default OmniPipeline talker for Qwen3-Omni Talker + CodePredictor + Code2Wav.
 *
 * Loads the speech stack from a model directory (the same directory as the VLM weights
 * for path-based OmniPipeline construction; an arbitrary directory containing the
 * speech submodels for the DI ctor).
 *
 * Public-API thin wrapper around the internal `Qwen3OmniSpeechPipeline`. Construct
 * directly when you want to inject your own VLM (e.g. placed on a different device) but keep
 * the default speech backend:
 *
 *     auto vlm = std::make_shared<ov::genai::VLMPipeline>(model_dir, "GPU", props);
 *     auto talker = std::make_shared<ov::genai::Talker>(model_dir, "CPU", props);
 *     ov::genai::OmniPipeline pipe(vlm, talker);
 *
 * @note This is a preview API and is subject to change.
 */
class OPENVINO_GENAI_EXPORTS Talker : public TalkerBase {
public:
    /// @brief Load Qwen3-Omni speech submodels from a directory containing
    /// `openvino_talker_model.xml`, `openvino_code_predictor_model.xml`,
    /// `openvino_code2wav_model.xml`, plus the talker text-embedding and projection
    /// submodels. The directory must contain `config.json` (used for talker config).
    Talker(const std::filesystem::path& model_dir,
                    const std::string& device,
                    const ov::AnyMap& properties = {});

    /// @brief Construct from in-memory model IRs (blob deployment / per-submodel device placement).
    /// @param models_map Model name -> (IR string, weights tensor). Keys: `text_embeddings`,
    ///        `talker`, `talker_text_embeddings`, `talker_projections`, `code_predictor`, `code2wav`.
    /// @param config Stored default speech config (see get/set_speech_config).
    /// @param config_dir_path Directory with `config.json` (codec/token IDs, speakers) and optional
    ///        `generation_config.json` (sampling defaults).
    /// @param device_mapping Submodel name -> device. Entries absent from this map fall back to
    ///        "CPU"; submodels absent from `models_map` remain unavailable and fail the availability check.
    /// @param properties Passed to ov::Core::compile_model() for every submodel.
    Talker(const ModelsMap& models_map,
                    const OmniTalkerSpeechConfig& config,
                    const std::filesystem::path& config_dir_path,
                    const std::map<std::string, std::string>& device_mapping,
                    const ov::AnyMap& properties = {});

    ~Talker() override;

    TalkerResults generate(const VLMDecodedResults& vlm_result,
                          const OmniTalkerSpeechConfig& talker_speech_config,
                          const OmniSpeechStreamerVariant& speech_streamer = std::monostate{}) override;

    TalkerResults generate(const VLMDecodedResults& vlm_result, const ov::AnyMap& properties = {}) override;

    std::vector<std::string> list_speakers() const override;
    ov::Tensor get_speaker_embedding(const std::string& name) const override;

    /// @brief Return the talker's stored default speech config. Seeds the AnyMap generate()
    /// overload: fields absent from its `properties` fall back to this config.
    OmniTalkerSpeechConfig get_speech_config() const;

    /// @brief Set the talker's stored default speech config (validated).
    /// @throws when `config` fails OmniTalkerSpeechConfig validation.
    void set_speech_config(const OmniTalkerSpeechConfig& config);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace ov::genai
