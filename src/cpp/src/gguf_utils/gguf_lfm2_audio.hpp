// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <vector>

#include "openvino/openvino.hpp"

namespace ov::genai {
ov::Tensor lfm2_spectrum_to_waveform(const ov::Tensor& spectral,
                                     const std::vector<float>& window,
                                     ov::InferRequest& inverse_fft);

}

#ifdef ENABLE_GGUF
#    include <mutex>
#    include <random>

#    include "gguf_multimodal.hpp"
#    include "openvino/genai/omni/decoded_results.hpp"
#    include "visual_language/inputs_embedder.hpp"

namespace ov::genai {
class VisionRegistry;

class GGUFLFM2Speech {
public:
    GGUFLFM2Speech(const std::filesystem::path& language, const std::string& device, ov::AnyMap properties);
    GenerationConfig get_generation_config() const;
    std::string default_system_prompt(bool return_audio) const;
    OmniDecodedResults generate(const ChatHistory& history,
                                const std::vector<ov::Tensor>& audios,
                                const GenerationConfig& config,
                                const OmniTalkerSpeechConfig& speech,
                                const StreamerVariant& streamer,
                                const OmniSpeechStreamerVariant& speech_streamer);

private:
    int64_t sample(const float* logits,
                   size_t count,
                   bool greedy,
                   float temperature,
                   size_t top_k,
                   float top_p,
                   std::mt19937& rng);
    ov::Tensor decode(const std::vector<int32_t>& codes);
    GGUFMultimodalModels m_models;
    std::shared_ptr<InputsEmbedder> m_embedder;
    ov::InferRequest m_language, m_embedding, m_depth, m_feedback, m_codec, m_detokenizer, m_ifft;
    std::vector<float> m_window;
    bool m_interleaved = true;
    size_t m_text_cadence = 6, m_audio_cadence = 12;
    std::mutex m_mutex;
    std::shared_ptr<VisionRegistry> m_registry;
};
}  // namespace ov::genai
#endif
