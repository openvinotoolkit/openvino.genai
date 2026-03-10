// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <unordered_map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "text2speech_pipeline_impl.hpp"

#if OPENVINO_GENAI_HAS_MISAKI_CPP
#include "misaki/g2p.hpp"
#endif

namespace ov {
namespace genai {

class KokoroRuntime;

class KokoroTTSImpl : public Text2SpeechPipelineImpl {
public:
    KokoroTTSImpl(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties);

    Text2SpeechDecodedResults generate(const std::vector<std::string>& texts,
                                       const ov::Tensor& speaker_embedding,
                                       const SpeechGenerationConfig& generation_config) override;

    Text2SpeechDecodedResults generate_from_phonemes(const std::vector<std::vector<std::string>>& phoneme_chunks,
                                                     const ov::Tensor& speaker_embedding,
                                                     const SpeechGenerationConfig& generation_config) override;

    Text2SpeechDecodedResults generate_from_tokens(const std::vector<std::vector<SpeechToken>>& token_batches,
                                                   const ov::Tensor& speaker_embedding,
                                                   const SpeechGenerationConfig& generation_config) override;

    std::vector<std::vector<std::string>> phonemize(const std::vector<std::string>& texts,
                                                    const SpeechGenerationConfig& generation_config) override;

private:
    Text2SpeechDecodedResults synthesize_from_phoneme_chunks(const std::vector<std::vector<std::string>>& all_phoneme_chunks,
                                                             const ov::Tensor& speaker_embedding,
                                                             const SpeechGenerationConfig& generation_config);

private:
    std::filesystem::path m_models_path;
    ov::InferRequest m_request;
    std::string m_input_ids_name;
    std::string m_ref_s_name;
    std::string m_speed_name;
    size_t m_static_input_ids_length = 0;
    bool m_has_pred_dur_output = false;
    std::shared_ptr<KokoroRuntime> m_runtime;
    std::unordered_map<std::string, std::vector<float>> m_voice_cache;
#if OPENVINO_GENAI_HAS_MISAKI_CPP
    std::unique_ptr<misaki::G2P> m_g2p;
    bool m_fallback_initialized = false;
    std::optional<std::string> m_phonemize_fallback_model_dir;
#endif
};

}  // namespace genai
}  // namespace ov
