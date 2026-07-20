// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/omni/pipeline.hpp"

#include <optional>

#include "omni/pipeline_impl.hpp"
#include "omni/talker_speech_config_utils.hpp"

namespace ov::genai {

OmniPipeline::OmniPipeline(const std::filesystem::path& models_path,
                           const std::string& device,
                           const ov::AnyMap& properties)
    : OmniPipeline(std::make_shared<VLMPipeline>(models_path, device, properties),
                   std::make_shared<Talker>(models_path, device, properties)) {}

OmniPipeline::OmniPipeline(const std::shared_ptr<VLMPipelineBase>& vlm,
                           const std::shared_ptr<TalkerBase>& talker)
    : m_pimpl(std::make_unique<OmniPipelineImpl>(vlm, talker)) {}

OmniPipeline::~OmniPipeline() = default;

OmniDecodedResults OmniPipeline::generate(const std::string& prompt,
                                           const std::vector<ov::Tensor>& images,
                                           const std::vector<ov::Tensor>& videos,
                                           const std::vector<VideoMetadata>& videos_metadata,
                                           const std::vector<ov::Tensor>& audios,
                                           const GenerationConfig& text_config,
                                           const OmniTalkerSpeechConfig& talker_speech_config,
                                           const StreamerVariant& streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(prompt,
                             images,
                             videos,
                             videos_metadata,
                             audios,
                             text_config,
                             talker_speech_config,
                             streamer,
                             speech_streamer);
}

OmniDecodedResults OmniPipeline::generate(const std::string& prompt,
                                           const std::vector<ov::Tensor>& images,
                                           const GenerationConfig& text_config,
                                           const OmniTalkerSpeechConfig& talker_speech_config,
                                           const StreamerVariant& streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(prompt,
                             images,
                             /*videos=*/{},
                             /*videos_metadata=*/{},
                             /*audios=*/{},
                             text_config,
                             talker_speech_config,
                             streamer,
                             speech_streamer);
}

OmniDecodedResults OmniPipeline::generate(const std::string& prompt,
                                           const std::vector<ov::Tensor>& images,
                                           const std::vector<ov::Tensor>& videos,
                                           const std::vector<VideoMetadata>& videos_metadata,
                                           const GenerationConfig& text_config,
                                           const OmniTalkerSpeechConfig& talker_speech_config,
                                           const StreamerVariant& streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(prompt,
                             images,
                             videos,
                             videos_metadata,
                             /*audios=*/{},
                             text_config,
                             talker_speech_config,
                             streamer,
                             speech_streamer);
}

OmniDecodedResults OmniPipeline::generate(const std::string& prompt,
                                           const ov::Tensor& image,
                                           const GenerationConfig& text_config,
                                           const OmniTalkerSpeechConfig& talker_speech_config,
                                           const StreamerVariant& streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(prompt,
                             std::vector<ov::Tensor>{image},
                             /*videos=*/{},
                             /*videos_metadata=*/{},
                             /*audios=*/{},
                             text_config,
                             talker_speech_config,
                             streamer,
                             speech_streamer);
}

OmniDecodedResults OmniPipeline::generate(const ChatHistory& history,
                                           const std::vector<ov::Tensor>& images,
                                           const std::vector<ov::Tensor>& videos,
                                           const std::vector<VideoMetadata>& videos_metadata,
                                           const std::vector<ov::Tensor>& audios,
                                           const GenerationConfig& text_config,
                                           const OmniTalkerSpeechConfig& talker_speech_config,
                                           const StreamerVariant& streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(history,
                             images,
                             videos,
                             videos_metadata,
                             audios,
                             text_config,
                             talker_speech_config,
                             streamer,
                             speech_streamer);
}

OmniDecodedResults OmniPipeline::generate(const ChatHistory& history,
                                           const std::vector<ov::Tensor>& images,
                                           const GenerationConfig& text_config,
                                           const OmniTalkerSpeechConfig& talker_speech_config,
                                           const StreamerVariant& streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(history,
                             images,
                             /*videos=*/{},
                             /*videos_metadata=*/{},
                             /*audios=*/{},
                             text_config,
                             talker_speech_config,
                             streamer,
                             speech_streamer);
}

OmniDecodedResults OmniPipeline::generate(const ChatHistory& history,
                                           const std::vector<ov::Tensor>& images,
                                           const std::vector<ov::Tensor>& videos,
                                           const std::vector<VideoMetadata>& videos_metadata,
                                           const GenerationConfig& text_config,
                                           const OmniTalkerSpeechConfig& talker_speech_config,
                                           const StreamerVariant& streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(history,
                             images,
                             videos,
                             videos_metadata,
                             /*audios=*/{},
                             text_config,
                             talker_speech_config,
                             streamer,
                             speech_streamer);
}

OmniDecodedResults OmniPipeline::generate(const ChatHistory& history,
                                           const ov::Tensor& image,
                                           const GenerationConfig& text_config,
                                           const OmniTalkerSpeechConfig& talker_speech_config,
                                           const StreamerVariant& streamer,
                                           const OmniSpeechStreamerVariant& speech_streamer) {
    return m_pimpl->generate(history,
                             std::vector<ov::Tensor>{image},
                             /*videos=*/{},
                             /*videos_metadata=*/{},
                             /*audios=*/{},
                             text_config,
                             talker_speech_config,
                             streamer,
                             speech_streamer);
}

namespace {

// Pull recognized OmniPipeline keys out of `config_map`. Anything left over is forwarded to
// BOTH update_omni_talker_speech_config AND (a copy of) the resolved
// GenerationConfig, so users can pass GenerationConfig fields (max_new_tokens, do_sample,
// ...) and OmniTalkerSpeechConfig fields (return_audio, speaker, audio_chunk_frames, ...)
// inline alongside the IO properties. Shared field names (`max_new_tokens`, `rng_seed`)
// land on both configs intentionally — that's the broadcast contract for the AnyMap form.
struct ExtractedOmniProps {
    std::vector<ov::Tensor> images;
    std::vector<ov::Tensor> videos;
    std::vector<VideoMetadata> videos_metadata;
    std::vector<ov::Tensor> audios;
    std::optional<GenerationConfig> explicit_text_config;
    std::optional<OmniTalkerSpeechConfig> explicit_talker_speech_config;
    StreamerVariant streamer = std::monostate{};
    OmniSpeechStreamerVariant speech_streamer = std::monostate{};
    ov::AnyMap leftover;
};

ExtractedOmniProps extract_omni_props(const ov::AnyMap& config_map) {
    ExtractedOmniProps out;
    for (const auto& [property_key, value] : config_map) {
        if (property_key == ov::genai::images.name()) {
            out.images = value.as<std::vector<ov::Tensor>>();
        } else if (property_key == ov::genai::videos.name()) {
            out.videos = value.as<std::vector<ov::Tensor>>();
        } else if (property_key == ov::genai::videos_metadata.name()) {
            out.videos_metadata = value.as<std::vector<VideoMetadata>>();
        } else if (property_key == ov::genai::audios.name()) {
            out.audios = value.as<std::vector<ov::Tensor>>();
        } else if (property_key == ov::genai::text_config.name()) {
            out.explicit_text_config = value.as<GenerationConfig>();
        } else if (property_key == ov::genai::talker_speech_config.name()) {
            out.explicit_talker_speech_config = value.as<OmniTalkerSpeechConfig>();
        } else if (property_key == "streamer") {
            out.streamer = value.as<StreamerVariant>();
        } else if (property_key == ov::genai::speech_streamer.name()) {
            out.speech_streamer = value.as<OmniSpeechStreamerVariant>();
        } else {
            out.leftover.emplace(property_key, value);
        }
    }
    return out;
}

}  // namespace

OmniDecodedResults OmniPipeline::generate(const std::string& prompt, const ov::AnyMap& config_map) {
    auto omni_props = extract_omni_props(config_map);
    GenerationConfig text_config = omni_props.explicit_text_config.value_or(m_pimpl->get_text_generation_config());
    OmniTalkerSpeechConfig talker_speech_config =
        omni_props.explicit_talker_speech_config.value_or(m_pimpl->get_talker_speech_config());
    if (!omni_props.leftover.empty()) {
        text_config.update_generation_config(omni_props.leftover);
        update_omni_talker_speech_config(talker_speech_config, omni_props.leftover);
    }
    return m_pimpl->generate(prompt,
                             omni_props.images,
                             omni_props.videos,
                             omni_props.videos_metadata,
                             omni_props.audios,
                             text_config,
                             talker_speech_config,
                             omni_props.streamer,
                             omni_props.speech_streamer);
}

OmniDecodedResults OmniPipeline::generate(const ChatHistory& history, const ov::AnyMap& config_map) {
    auto omni_props = extract_omni_props(config_map);
    GenerationConfig text_config = omni_props.explicit_text_config.value_or(m_pimpl->get_text_generation_config());
    OmniTalkerSpeechConfig talker_speech_config =
        omni_props.explicit_talker_speech_config.value_or(m_pimpl->get_talker_speech_config());
    if (!omni_props.leftover.empty()) {
        text_config.update_generation_config(omni_props.leftover);
        update_omni_talker_speech_config(talker_speech_config, omni_props.leftover);
    }
    return m_pimpl->generate(history,
                             omni_props.images,
                             omni_props.videos,
                             omni_props.videos_metadata,
                             omni_props.audios,
                             text_config,
                             talker_speech_config,
                             omni_props.streamer,
                             omni_props.speech_streamer);
}

std::shared_ptr<VLMPipelineBase> OmniPipeline::get_vlm() const {
    return m_pimpl->get_vlm();
}

std::shared_ptr<TalkerBase> OmniPipeline::get_talker() const {
    return m_pimpl->get_talker();
}

}  // namespace ov::genai
