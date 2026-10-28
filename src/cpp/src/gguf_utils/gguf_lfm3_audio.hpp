// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#ifdef ENABLE_GGUF
#    include <mutex>
#    include <random>

#    include "gguf_multimodal.hpp"
#    include "openvino/genai/omni/decoded_results.hpp"
#    include "openvino/genai/omni/talker_speech_config.hpp"
#    include "visual_language/inputs_embedder.hpp"
#    include "visual_language/lfm3/preprocessing.hpp"

namespace ov::genai {
class VisionRegistry;
class GGUFLFM3Speech {
public:
    GGUFLFM3Speech(const std::filesystem::path& language, const std::string& device, ov::AnyMap properties);
    GenerationConfig get_generation_config() const;
    std::string default_system_prompt(bool return_audio) const;
    OmniDecodedResults generate(const ChatHistory& history,
                                const std::vector<ov::Tensor>& images,
                                const std::vector<ov::Tensor>& videos,
                                const std::vector<VideoMetadata>& metadata,
                                const std::vector<ov::Tensor>& audios,
                                const GenerationConfig& config,
                                const OmniTalkerSpeechConfig& speech,
                                const StreamerVariant& streamer,
                                const OmniSpeechStreamerVariant& speech_streamer);

private:
    ov::Tensor sample_frame(const ov::Tensor& hidden, const OmniTalkerSpeechConfig& speech, std::mt19937& rng);
    ov::Tensor decode(const std::vector<int32_t>& codes);
    void reset();
    GGUFMultimodalModels m_models;
    GenerationConfig m_config;
    std::shared_ptr<InputsEmbedder> m_embedder;
    ov::InferRequest m_language, m_embedding, m_depth, m_feedback, m_codec, m_ifft;
    std::unique_ptr<LFM3ISTFT> m_istft;
    size_t m_codec_position = 0;
    int64_t m_audio_start, m_mixed_start, m_text_end;
    size_t m_text_cadence = 6, m_audio_cadence = 12;
    std::vector<int32_t> m_audio_history;
    std::mutex m_mutex;
    std::shared_ptr<VisionRegistry> m_registry;
};
}  // namespace ov::genai
#endif
