// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/audio_streamer_base.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "visual_language/vision_properties.hpp"
#include "utils.hpp"

using namespace ov::genai;

namespace ov::genai {
class ov::genai::VLMPipeline::VLMPipelineBase {
    float m_load_time_ms = 0;
    std::string m_attention_backend = SDPA_BACKEND;

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

    static std::vector<ov::Tensor> extract_audios_from_config_map(const ov::AnyMap& config_map) {
        std::vector<ov::Tensor> result;
        const auto audios_it = config_map.find(ov::genai::audios.name());
        if (audios_it == config_map.end()) {
            return result;
        }
        if (audios_it->second.is<std::vector<ov::Tensor>>()) {
            result = audios_it->second.as<std::vector<ov::Tensor>>();
        } else if (audios_it->second.is<ov::Tensor>()) {
            result = {audios_it->second.as<ov::Tensor>()};
        } else if (!audios_it->second.empty()) {
            OPENVINO_THROW("Property 'audios' must be ov::Tensor or std::vector<ov::Tensor>, got unsupported type.");
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

    virtual VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const std::vector<VideoMetadata>& videos_metadata,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) = 0;

    VLMDecodedResults generate(
        const std::string& prompt,
        const ov::AnyMap& config_map
    ) {
        const auto vision_properties = extract_vision_properties(config_map);
        m_pending_audios = extract_audios_from_config_map(config_map);
        m_pending_audio_streamer = utils::get_audio_streamer_from_map(config_map);
        GenerationConfig config = resolve_generation_config(config_map);
        m_pending_audio_chunk_frames = config.audio_chunk_frames;
        AudioStreamerGuard guard{m_pending_audio_streamer, m_pending_audios};
        return generate(
            prompt,
            vision_properties.images.value_or(std::vector<ov::Tensor>{}),
            vision_properties.videos.value_or(std::vector<ov::Tensor>{}),
            vision_properties.videos_metadata.value_or(std::vector<VideoMetadata>{}),
            config,
            utils::get_streamer_from_map(config_map)
        );
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

    virtual VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const std::vector<VideoMetadata>& videos_metadata,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) = 0;

    VLMDecodedResults generate(
        const ChatHistory& history,
        const ov::AnyMap& config_map
    ) {
        const auto vision_properties = extract_vision_properties(config_map);
        m_pending_audios = extract_audios_from_config_map(config_map);
        m_pending_audio_streamer = utils::get_audio_streamer_from_map(config_map);
        GenerationConfig config = resolve_generation_config(config_map);
        m_pending_audio_chunk_frames = config.audio_chunk_frames;
        AudioStreamerGuard guard{m_pending_audio_streamer, m_pending_audios};
        return generate(
            history,
            vision_properties.images.value_or(std::vector<ov::Tensor>{}),
            vision_properties.videos.value_or(std::vector<ov::Tensor>{}),
            vision_properties.videos_metadata.value_or(std::vector<VideoMetadata>{}),
            config,
            utils::get_streamer_from_map(config_map)
        );
    }

    virtual void start_chat(const std::string& system_message) = 0;

    virtual void finish_chat() = 0;

    virtual Tokenizer get_tokenizer() const = 0;

    virtual void set_chat_template(const std::string& new_template) = 0;

    virtual GenerationConfig get_generation_config() const = 0;

    virtual void set_generation_config(const GenerationConfig& new_config) = 0;

    void set_attention_backend(const std::string& attention_backend) {
        m_attention_backend = attention_backend;
    }

    void set_load_time(float load_time_ms) {
        m_load_time_ms = load_time_ms;
    }

    float get_load_time() const {
        return m_load_time_ms;
    }
};
}  // namespace ov::genai
