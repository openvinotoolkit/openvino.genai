// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/audio_streamer_base.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "utils.hpp"

using namespace ov::genai;

namespace ov::genai {
class ov::genai::VLMPipeline::VLMPipelineBase {
    // Load pipeline time
    float m_load_time_ms = 0;

protected:
    // Audio tensors extracted from AnyMap, forwarded to implementations
    std::vector<ov::Tensor> m_pending_audios;
    // Audio streamer extracted from AnyMap, forwarded to speech pipeline
    AudioStreamerVariant m_pending_audio_streamer = std::monostate{};
    size_t m_pending_audio_chunk_frames = 1;

    // RAII guard to reset audio streamer and pending audios on scope exit (exception-safe)
    struct AudioStreamerGuard {
        AudioStreamerVariant& streamer_ref;
        std::vector<ov::Tensor>& audios_ref;
        ~AudioStreamerGuard() {
            streamer_ref = std::monostate{};
            audios_ref.clear();
        }
    };

    GenerationConfig resolve_generation_config(const ov::AnyMap& config_map) {
        ov::genai::OptionalGenerationConfig optional_config = utils::get_config_from_map(config_map);
        GenerationConfig config = optional_config.value_or(get_generation_config());
        config.update_generation_config(config_map);
        return config;
    }

    struct ExtractedMedia {
        std::vector<ov::Tensor> images;
        std::vector<ov::Tensor> videos;
        std::vector<ov::Tensor> audios;
    };

    static ExtractedMedia extract_media_from_config_map(const ov::AnyMap& config_map) {
        ExtractedMedia result;

        auto image = config_map.find(ov::genai::image.name());
        auto images = config_map.find(ov::genai::images.name());
        auto videos = config_map.find(ov::genai::videos.name());
        auto audios_it = config_map.find(ov::genai::audios.name());

        if (config_map.end() != image) {
            result.images = {image->second.as<ov::Tensor>()};
        }

        if (config_map.end() != images) {
            if (images->second.is<std::vector<ov::Tensor>>()) {
                auto imgs = images->second.as<std::vector<ov::Tensor>>();
                result.images.insert(result.images.end(), imgs.begin(), imgs.end());
            } else if (images->second.is<ov::Tensor>()) {
                result.images.push_back(std::move(images->second.as<ov::Tensor>()));
            } else if (!images->second.empty()) {
                OPENVINO_THROW("Unknown images type.");
            }
        }

        if (config_map.end() != videos) {
            if (videos->second.is<std::vector<ov::Tensor>>()) {
                result.videos = videos->second.as<std::vector<ov::Tensor>>();
            } else if (videos->second.is<ov::Tensor>()) {
                result.videos = {videos->second.as<ov::Tensor>()};
            } else if (!videos->second.empty()) {
                OPENVINO_THROW("Unknown videos type.");
            }
        }

        if (config_map.end() != audios_it) {
            if (audios_it->second.is<std::vector<ov::Tensor>>()) {
                result.audios = audios_it->second.as<std::vector<ov::Tensor>>();
            } else if (audios_it->second.is<ov::Tensor>()) {
                result.audios = {audios_it->second.as<ov::Tensor>()};
            } else if (!audios_it->second.empty()) {
                OPENVINO_THROW("Unknown audios type.");
            }
        }

        return result;
    }

public:

    virtual ~VLMPipelineBase() = default;

    virtual VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) = 0;

    virtual VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) = 0;

    VLMDecodedResults generate(
        const std::string& prompt,
        const ov::AnyMap& config_map
    ) {
        auto media = extract_media_from_config_map(config_map);
        m_pending_audios = std::move(media.audios);
        m_pending_audio_streamer = utils::get_audio_streamer_from_map(config_map);
        GenerationConfig config = resolve_generation_config(config_map);
        m_pending_audio_chunk_frames = config.audio_chunk_frames;
        AudioStreamerGuard guard{m_pending_audio_streamer, m_pending_audios};
        return generate(prompt, media.images, media.videos, config, utils::get_streamer_from_map(config_map));
    }

    virtual VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) = 0;

    virtual VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) = 0;

    VLMDecodedResults generate(
        const ChatHistory& history,
        const ov::AnyMap& config_map
    ) {
        auto media = extract_media_from_config_map(config_map);
        m_pending_audios = std::move(media.audios);
        m_pending_audio_streamer = utils::get_audio_streamer_from_map(config_map);
        GenerationConfig config = resolve_generation_config(config_map);
        m_pending_audio_chunk_frames = config.audio_chunk_frames;
        AudioStreamerGuard guard{m_pending_audio_streamer, m_pending_audios};
        return generate(history, media.images, media.videos, config, utils::get_streamer_from_map(config_map));
    }

    virtual void start_chat(const std::string& system_message) = 0;

    virtual void finish_chat() = 0;

    virtual Tokenizer get_tokenizer() const = 0;

    virtual void set_chat_template(const std::string& new_template) = 0;

    virtual GenerationConfig get_generation_config() const = 0;

    virtual void set_generation_config(const GenerationConfig& new_config) = 0;

    void set_load_time(float load_time_ms) {
        m_load_time_ms = load_time_ms;
    }

    float get_load_time() const {
        return m_load_time_ms;
    }
};
}  // namespace ov::genai
