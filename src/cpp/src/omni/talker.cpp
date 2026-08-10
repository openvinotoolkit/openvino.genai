// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/omni/talker.hpp"

#include "openvino/core/except.hpp"
#include "openvino/genai/omni/pipeline.hpp"
#include "omni/talker_speech_config_utils.hpp"
#include "utils.hpp"
#include "visual_language/qwen3_omni/speech_pipeline.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

namespace {

/// @brief Flatten per-step hidden states into per-token hidden states.
/// Each step tensor has shape [num_tokens, 1, hidden_size]. Multi-token tensors
/// (from prefill) are split into individual [1, 1, hidden_size] slices.
std::vector<ov::Tensor> split_hidden_states(const std::vector<ov::Tensor>& per_step_hidden_states) {
    std::vector<ov::Tensor> per_token;
    size_t total_tokens = 0;
    for (const auto& step_hs : per_step_hidden_states) {
        total_tokens += step_hs.get_shape()[0];
    }
    per_token.reserve(total_tokens);
    for (const auto& step_hs : per_step_hidden_states) {
        const auto shape = step_hs.get_shape();
        const size_t num_tokens = shape[0];
        for (size_t t = 0; t < num_tokens; t++) {
            const auto [begin, end] = ov::genai::utils::make_roi(shape, 0, t, t + 1);
            ov::Tensor token_hs(step_hs, begin, end);
            per_token.push_back(std::move(token_hs));
        }
    }
    return per_token;
}

}  // namespace

class Talker::Impl {
public:
    Impl(const std::filesystem::path& model_dir, const std::string& device, const ov::AnyMap& properties) {
        VLMConfig vlm_config(model_dir / "config.json");
        m_speech = std::make_unique<Qwen3OmniSpeechPipeline>(model_dir, vlm_config, device, properties);

        OPENVINO_ASSERT(m_speech->is_available(),
                        "Talker: speech generation submodels are missing from ", model_dir.string(),
                        ". If your model has no speech generation capability, use VLMPipeline directly instead of OmniPipeline.");
    }

    Impl(const ModelsMap& models_map,
         const OmniTalkerSpeechConfig& config,
         const std::filesystem::path& config_dir_path,
         const std::map<std::string, std::string>& device_mapping,
         const ov::AnyMap& properties)
        : m_speech_config(config) {
        VLMConfig vlm_config(config_dir_path / "config.json");
        m_speech = std::make_unique<Qwen3OmniSpeechPipeline>(models_map,
                                                             vlm_config,
                                                             config_dir_path,
                                                             device_mapping,
                                                             /*default_device=*/"CPU",
                                                             properties);

        OPENVINO_ASSERT(m_speech->is_available(),
                        "Talker: speech generation submodels are missing from the provided models_map. "
                        "Required keys: text_embeddings, talker, talker_text_embeddings, talker_projections, "
                        "code_predictor, code2wav.");
    }

    TalkerResults generate(const VLMDecodedResults& vlm_result,
                          const OmniTalkerSpeechConfig& talker_speech_config,
                          const OmniSpeechStreamerVariant& speech_streamer) {
        OPENVINO_ASSERT(!vlm_result.intermediate_hidden_states.empty(),
                        "Talker: intermediate hidden states missing on VLMDecodedResults; the VLM must run "
                        "with talker_speech_config.return_audio=true so it accumulates thinker hidden states.");
        OPENVINO_ASSERT(!vlm_result.full_token_ids.empty(),
                        "Talker: full_token_ids missing on VLMDecodedResults; the VLM must run "
                        "with talker_speech_config.return_audio=true.");
        OPENVINO_ASSERT(vlm_result.intermediate_hidden_states.size() == 1 && vlm_result.full_token_ids.size() == 1,
                        "Talker: expected a single return sequence, got ",
                        vlm_result.full_token_ids.size(),
                        "; speech generation consumes one hidden-state stream.");

        auto all_intermediate_hidden_states = split_hidden_states(vlm_result.intermediate_hidden_states[0]);

        // talker_speech_config carries audio_chunk_frames, speaker / speaker_embedding,
        // max_new_tokens, rng_seed, plus the talker_* / cp_* sampling overrides. The pipeline
        // resolves each std::optional<...> against the JSON-loaded checkpoint defaults at the
        // top of generate_speech(); unset overrides keep the checkpoint values.
        return m_speech->generate_speech(vlm_result.full_token_ids[0],
                                         all_intermediate_hidden_states,
                                         speech_streamer,
                                         talker_speech_config);
    }

    // Property-bag form: resolve `properties` against the stored default config, pulling out
    // an optional speech_streamer, then delegate to the typed overload.
    TalkerResults generate(const VLMDecodedResults& vlm_result, const ov::AnyMap& properties) {
        OmniTalkerSpeechConfig config = m_speech_config;
        OmniSpeechStreamerVariant speech_streamer = std::monostate{};
        ov::AnyMap leftover;
        for (const auto& [key, value] : properties) {
            if (key == ov::genai::speech_streamer.name()) {
                speech_streamer = value.as<OmniSpeechStreamerVariant>();
            } else if (key == ov::genai::talker_speech_config.name()) {
                config = value.as<OmniTalkerSpeechConfig>();
            } else {
                leftover.emplace(key, value);
            }
        }
        if (!leftover.empty()) {
            update_omni_talker_speech_config(config, leftover);
        }
        return generate(vlm_result, config, speech_streamer);
    }

    std::vector<std::string> list_speakers() const {
        return m_speech->list_speakers();
    }

    ov::Tensor get_speaker_embedding(const std::string& name) const {
        return m_speech->get_speaker_embedding(name);
    }

    OmniTalkerSpeechConfig get_speech_config() const {
        return m_speech_config;
    }

    void set_speech_config(const OmniTalkerSpeechConfig& config) {
        validate_omni_talker_speech_config(config);
        m_speech_config = config;
    }

private:
    std::unique_ptr<Qwen3OmniSpeechPipeline> m_speech;
    OmniTalkerSpeechConfig m_speech_config;
};

Talker::Talker(const std::filesystem::path& model_dir,
                                 const std::string& device,
                                 const ov::AnyMap& properties)
    : m_impl(std::make_unique<Impl>(model_dir, device, properties)) {}

Talker::Talker(const ModelsMap& models_map,
                                 const OmniTalkerSpeechConfig& config,
                                 const std::filesystem::path& config_dir_path,
                                 const std::map<std::string, std::string>& device_mapping,
                                 const ov::AnyMap& properties)
    : m_impl(std::make_unique<Impl>(models_map, config, config_dir_path, device_mapping, properties)) {}

Talker::~Talker() = default;

TalkerResults Talker::generate(const VLMDecodedResults& vlm_result,
                                        const OmniTalkerSpeechConfig& talker_speech_config,
                                        const OmniSpeechStreamerVariant& speech_streamer) {
    return m_impl->generate(vlm_result, talker_speech_config, speech_streamer);
}

TalkerResults Talker::generate(const VLMDecodedResults& vlm_result, const ov::AnyMap& properties) {
    return m_impl->generate(vlm_result, properties);
}

std::vector<std::string> Talker::list_speakers() const {
    return m_impl->list_speakers();
}

ov::Tensor Talker::get_speaker_embedding(const std::string& name) const {
    return m_impl->get_speaker_embedding(name);
}

OmniTalkerSpeechConfig Talker::get_speech_config() const {
    return m_impl->get_speech_config();
}

void Talker::set_speech_config(const OmniTalkerSpeechConfig& config) {
    m_impl->set_speech_config(config);
}

}  // namespace ov::genai
