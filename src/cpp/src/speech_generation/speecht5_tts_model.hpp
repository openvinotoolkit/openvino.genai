// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <filesystem>
#include <openvino/openvino.hpp>
#include <variant>

#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "speecht5_tts_decoder.hpp"
#include "text2speech_pipeline_impl.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

class SpeechT5TTSImpl : public Text2SpeechPipelineImpl {
public:
    SpeechT5TTSImpl(const std::filesystem::path& models_path,
                    const std::string& device,
                    const ov::AnyMap& properties,
                    const Tokenizer& tokenizer);

    Text2SpeechDecodedResults generate(const std::vector<std::string>& texts,
                                       const ov::Tensor& speaker_embedding,
                                       const SpeechGenerationConfig& generation_config) override;

    SpeechGenerationPerfMetrics get_performance_metrics() override;

private:
    void init_model_config_params(const std::filesystem::path& root_dir);

private:
    ov::InferRequest m_encoder;
    std::shared_ptr<SpeechT5TTSDecoder> m_decoder;
    ov::InferRequest m_postnet;
    ov::InferRequest m_vocoder;
    Tokenizer m_tokenizer;
    uint64_t m_reduction_factor;
    uint64_t m_num_mel_bins;
};

}  // namespace genai
}  // namespace ov
