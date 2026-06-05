// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/omni/pipeline.hpp"

#include <optional>

#include "omni/pipeline_impl.hpp"

namespace ov::genai {

OmniPipeline::OmniPipeline(const std::filesystem::path& models_path,
                           const std::string& device,
                           const ov::AnyMap& properties)
    : OmniPipeline(VLMPipeline::create_base(models_path, device, properties),
                   std::make_shared<Qwen3OmniTalker>(models_path, device, properties)) {}

OmniPipeline::OmniPipeline(const std::shared_ptr<VLMPipeline::VLMPipelineBase>& vlm,
                           const std::shared_ptr<TalkerBase>& talker)
    : m_pimpl(std::make_unique<OmniPipelineImpl>(vlm, talker)) {}

OmniPipeline::~OmniPipeline() = default;
OmniPipeline::OmniPipeline(OmniPipeline&&) noexcept = default;
OmniPipeline& OmniPipeline::operator=(OmniPipeline&&) noexcept = default;

OmniDecodedResults OmniPipeline::generate(const std::string& prompt,
                                           const std::vector<ov::Tensor>& images,
                                           const std::vector<ov::Tensor>& videos,
                                           const std::vector<ov::Tensor>& audios,
                                           const GenerationConfig& text_config,
                                           const OmniTalkerSpeechConfig& talker_speech_config,
                                           const StreamerVariant& text_streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(prompt,
                             images,
                             videos,
                             audios,
                             text_config,
                             talker_speech_config,
                             text_streamer,
                             speech_streamer);
}

OmniDecodedResults OmniPipeline::generate(const std::string& prompt,
                                           const std::vector<ov::Tensor>& images,
                                           const GenerationConfig& text_config,
                                           const OmniTalkerSpeechConfig& talker_speech_config,
                                           const StreamerVariant& text_streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(prompt,
                             images,
                             /*videos=*/{},
                             /*audios=*/{},
                             text_config,
                             talker_speech_config,
                             text_streamer,
                             speech_streamer);
}

OmniDecodedResults OmniPipeline::generate(const std::string& prompt,
                                           const std::vector<ov::Tensor>& images,
                                           const std::vector<ov::Tensor>& videos,
                                           const GenerationConfig& text_config,
                                           const OmniTalkerSpeechConfig& talker_speech_config,
                                           const StreamerVariant& text_streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(prompt,
                             images,
                             videos,
                             /*audios=*/{},
                             text_config,
                             talker_speech_config,
                             text_streamer,
                             speech_streamer);
}

OmniDecodedResults OmniPipeline::generate(const std::string& prompt,
                                           const ov::Tensor& image,
                                           const GenerationConfig& text_config,
                                           const OmniTalkerSpeechConfig& talker_speech_config,
                                           const StreamerVariant& text_streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(prompt,
                             std::vector<ov::Tensor>{image},
                             /*videos=*/{},
                             /*audios=*/{},
                             text_config,
                             talker_speech_config,
                             text_streamer,
                             speech_streamer);
}

OmniDecodedResults OmniPipeline::generate(const ChatHistory& history,
                                           const std::vector<ov::Tensor>& images,
                                           const std::vector<ov::Tensor>& videos,
                                           const std::vector<ov::Tensor>& audios,
                                           const GenerationConfig& text_config,
                                           const OmniTalkerSpeechConfig& talker_speech_config,
                                           const StreamerVariant& text_streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(history,
                             images,
                             videos,
                             audios,
                             text_config,
                             talker_speech_config,
                             text_streamer,
                             speech_streamer);
}

OmniDecodedResults OmniPipeline::generate(const ChatHistory& history,
                                           const std::vector<ov::Tensor>& images,
                                           const GenerationConfig& text_config,
                                           const OmniTalkerSpeechConfig& talker_speech_config,
                                           const StreamerVariant& text_streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(history,
                             images,
                             /*videos=*/{},
                             /*audios=*/{},
                             text_config,
                             talker_speech_config,
                             text_streamer,
                             speech_streamer);
}

OmniDecodedResults OmniPipeline::generate(const ChatHistory& history,
                                           const std::vector<ov::Tensor>& images,
                                           const std::vector<ov::Tensor>& videos,
                                           const GenerationConfig& text_config,
                                           const OmniTalkerSpeechConfig& talker_speech_config,
                                           const StreamerVariant& text_streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(history,
                             images,
                             videos,
                             /*audios=*/{},
                             text_config,
                             talker_speech_config,
                             text_streamer,
                             speech_streamer);
}

OmniDecodedResults OmniPipeline::generate(const ChatHistory& history,
                                           const ov::Tensor& image,
                                           const GenerationConfig& text_config,
                                           const OmniTalkerSpeechConfig& talker_speech_config,
                                           const StreamerVariant& text_streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(history,
                             std::vector<ov::Tensor>{image},
                             /*videos=*/{},
                             /*audios=*/{},
                             text_config,
                             talker_speech_config,
                             text_streamer,
                             speech_streamer);
}

namespace {

// Pull recognized OmniPipeline keys out of `config_map`. Anything left over is forwarded to
// BOTH OmniTalkerSpeechConfig::update_generation_config AND (a copy of) the resolved
// GenerationConfig, so users can pass GenerationConfig fields (max_new_tokens, do_sample,
// ...) and OmniTalkerSpeechConfig fields (return_audio, speaker, audio_chunk_frames, ...)
// inline alongside the IO properties. Shared field names (`max_new_tokens`, `rng_seed`)
// land on both configs intentionally — that's the broadcast contract for the AnyMap form.
struct ExtractedOmniProps {
    std::vector<ov::Tensor> images;
    std::vector<ov::Tensor> videos;
    std::vector<ov::Tensor> audios;
    std::optional<GenerationConfig> explicit_text_config;
    std::optional<OmniTalkerSpeechConfig> explicit_talker_speech_config;
    StreamerVariant text_streamer = std::monostate{};
    OmniSpeechStreamerVariant speech_streamer = std::monostate{};
    ov::AnyMap leftover;
};

ExtractedOmniProps extract_omni_props(const ov::AnyMap& config_map) {
    ExtractedOmniProps out;
    for (const auto& [key, value] : config_map) {
        if (key == ov::genai::images.name()) {
            out.images = value.as<std::vector<ov::Tensor>>();
        } else if (key == ov::genai::videos.name()) {
            out.videos = value.as<std::vector<ov::Tensor>>();
        } else if (key == ov::genai::audios.name()) {
            out.audios = value.as<std::vector<ov::Tensor>>();
        } else if (key == ov::genai::text_config.name()) {
            out.explicit_text_config = value.as<GenerationConfig>();
        } else if (key == ov::genai::talker_speech_config.name()) {
            out.explicit_talker_speech_config = value.as<OmniTalkerSpeechConfig>();
        } else if (key == "streamer") {
            out.text_streamer = value.as<StreamerVariant>();
        } else if (key == ov::genai::speech_streamer.name()) {
            out.speech_streamer = value.as<OmniSpeechStreamerVariant>();
        } else {
            out.leftover.emplace(key, value);
        }
    }
    return out;
}

}  // namespace

OmniDecodedResults OmniPipeline::generate(const std::string& prompt, const ov::AnyMap& config_map) {
    auto p = extract_omni_props(config_map);
    GenerationConfig text_config = p.explicit_text_config.value_or(m_pimpl->get_text_generation_config());
    OmniTalkerSpeechConfig talker_speech_config =
        p.explicit_talker_speech_config.value_or(m_pimpl->get_talker_speech_config());
    if (!p.leftover.empty()) {
        text_config.update_generation_config(p.leftover);
        talker_speech_config.update_generation_config(p.leftover);
    }
    return m_pimpl->generate(prompt,
                             p.images,
                             p.videos,
                             p.audios,
                             text_config,
                             talker_speech_config,
                             p.text_streamer,
                             p.speech_streamer);
}

OmniDecodedResults OmniPipeline::generate(const ChatHistory& history, const ov::AnyMap& config_map) {
    auto p = extract_omni_props(config_map);
    GenerationConfig text_config = p.explicit_text_config.value_or(m_pimpl->get_text_generation_config());
    OmniTalkerSpeechConfig talker_speech_config =
        p.explicit_talker_speech_config.value_or(m_pimpl->get_talker_speech_config());
    if (!p.leftover.empty()) {
        text_config.update_generation_config(p.leftover);
        talker_speech_config.update_generation_config(p.leftover);
    }
    return m_pimpl->generate(history,
                             p.images,
                             p.videos,
                             p.audios,
                             text_config,
                             talker_speech_config,
                             p.text_streamer,
                             p.speech_streamer);
}

ov::Tensor OmniPipeline::get_speaker_embedding(const std::string& name) const {
    return m_pimpl->get_speaker_embedding(name);
}

std::vector<std::string> OmniPipeline::list_speakers() const {
    return m_pimpl->list_speakers();
}

GenerationConfig OmniPipeline::get_text_generation_config() const {
    return m_pimpl->get_text_generation_config();
}

void OmniPipeline::set_text_generation_config(const GenerationConfig& new_config) {
    m_pimpl->set_text_generation_config(new_config);
}

OmniTalkerSpeechConfig OmniPipeline::get_talker_speech_config() const {
    return m_pimpl->get_talker_speech_config();
}

void OmniPipeline::set_talker_speech_config(const OmniTalkerSpeechConfig& new_config) {
    m_pimpl->set_talker_speech_config(new_config);
}

std::shared_ptr<VLMPipeline::VLMPipelineBase> OmniPipeline::get_vlm() const {
    return m_pimpl->get_vlm();
}

std::shared_ptr<TalkerBase> OmniPipeline::get_talker() const {
    return m_pimpl->get_talker();
}

}  // namespace ov::genai
