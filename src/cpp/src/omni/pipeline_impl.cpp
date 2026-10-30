// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "omni/pipeline_impl.hpp"

#include <chrono>
#include <utility>

#include "omni/speech_streamer_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/genai/perf_metrics.hpp"

namespace ov::genai {

namespace {

/// @brief Cross-config validation: when speech output is requested the thinker text decode
/// must use a sampling mode the talker can consume — single hidden-state stream, no beam
/// candidates, no speculative draft tokens.
void enforce_text_config_compatible_with_audio(const GenerationConfig& text_config,
                                               const OmniTalkerSpeechConfig& talker_speech_config) {
    if (!talker_speech_config.return_audio) {
        return;
    }
    OPENVINO_ASSERT(!text_config.is_beam_search(),
                    "OmniPipeline: return_audio is not compatible with beam search (num_beams > 1)");
    OPENVINO_ASSERT(!text_config.is_prompt_lookup() && !text_config.is_assisting_generation(),
                    "OmniPipeline: return_audio is not compatible with prompt lookup or assistant/speculative decoding");
    OPENVINO_ASSERT(text_config.num_return_sequences == 1,
                    "OmniPipeline: return_audio requires num_return_sequences == 1 (got ",
                    text_config.num_return_sequences,
                    "); the talker consumes a single hidden-state stream");
}

/// @brief Keeps every speech chunk and forwards it to the caller's streamer.
class CollectingSpeechStreamer : public OmniSpeechStreamerBase {
public:
    explicit CollectingSpeechStreamer(const OmniSpeechStreamerVariant& streamer)
        : m_streamer(streamer),
          m_active(is_speech_streamer_active(streamer)) {}

    StreamingStatus write(const ov::Tensor& audio_chunk) override {
        const float* samples = audio_chunk.data<const float>();
        m_samples.insert(m_samples.end(), samples, samples + audio_chunk.get_size());
        return m_active ? invoke_speech_streamer(m_streamer, audio_chunk) : StreamingStatus::RUNNING;
    }

    // Called by the speech engine and by the guard in generate_speech_in_stream(); forwards only the first call.
    void end() override {
        if (m_active) {
            m_active = false;
            end_speech_streamer(m_streamer);
        }
    }

    ov::Tensor waveform() const {
        if (m_samples.empty()) {
            return {};
        }
        ov::Tensor waveform(ov::element::f32, {1, 1, m_samples.size()});
        std::copy(m_samples.begin(), m_samples.end(), waveform.data<float>());
        return waveform;
    }

private:
    const OmniSpeechStreamerVariant& m_streamer;
    bool m_active;
    std::vector<float> m_samples;
};

}  // namespace

OmniPipeline::OmniPipelineImpl::OmniPipelineImpl(const std::shared_ptr<VLMPipelineBase>& vlm,
                                                  const std::shared_ptr<TalkerBase>& talker) :
    m_vlm{vlm}, m_talker{talker} {
    OPENVINO_ASSERT(m_vlm, "OmniPipeline: VLM pointer is null");
    OPENVINO_ASSERT(m_talker, "OmniPipeline: talker pointer is null");
    assert_omni_capable();
}

OmniPipeline::OmniPipelineImpl::OmniPipelineImpl(const std::shared_ptr<VLMPipelineBase>& vlm) : m_vlm{vlm} {
    OPENVINO_ASSERT(m_vlm, "OmniPipeline: VLM pointer is null");
}

OmniDecodedResults OmniPipeline::OmniPipelineImpl::generate_speech_in_stream(
    const std::function<VLMDecodedResults(const ov::AnyMap&)>& generate,
    const std::vector<ov::Tensor>& images,
    const std::vector<ov::Tensor>& videos,
    const std::vector<VideoMetadata>& videos_metadata,
    const std::vector<ov::Tensor>& audios,
    const GenerationConfig& text_config,
    const OmniTalkerSpeechConfig& talker_speech_config,
    const StreamerVariant& streamer,
    const OmniSpeechStreamerVariant& speech_streamer) {
    const auto collector = std::make_shared<CollectingSpeechStreamer>(speech_streamer);
    const OmniSpeechStreamerVariant collector_variant = std::shared_ptr<OmniSpeechStreamerBase>(collector);
    // The speech streamer gets end() even if validation, encoding or prefill throws before decoding starts.
    SpeechStreamerGuard collector_guard(collector_variant, true);
    const auto start_time = std::chrono::steady_clock::now();
    VLMDecodedResults vlm_result = generate(ov::AnyMap{
        ov::genai::images(images),
        ov::genai::videos(videos),
        ov::genai::videos_metadata(videos_metadata),
        ov::genai::audios(audios),
        ov::genai::generation_config(text_config),
        ov::genai::streamer(streamer),
        ov::genai::talker_speech_config(talker_speech_config),
        ov::genai::speech_streamer(std::get<std::shared_ptr<OmniSpeechStreamerBase>>(collector_variant)),
    });
    OmniDecodedResults omni_result;
    static_cast<VLMDecodedResults&>(omni_result) = std::move(vlm_result);
    ov::Tensor waveform = collector->waveform();
    if (waveform) {
        omni_result.speech_result.perf_metrics.num_generated_samples = waveform.get_size();
        omni_result.speech_result.waveforms.push_back(std::move(waveform));
    }
    omni_result.speech_result.perf_metrics.generation_time_ms =
        PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start_time) / 1000.0f;
    return omni_result;
}

void OmniPipeline::OmniPipelineImpl::assert_omni_capable() const {
    OPENVINO_ASSERT(
        m_vlm->is_audio_output_enabled(),
        "OmniPipeline(vlm, talker) requires a Qwen3-Omni model with audio output enabled (config.json: "
        "enable_audio_output=true). LFM3-Omni has no separate talker: use OmniPipeline(models_path, device, properties).");
    OPENVINO_ASSERT(
        m_vlm->supports_hidden_states_collection(),
        "OmniPipeline speech output requires the continuous-batching backend, but the loaded VLM uses the SDPA "
        "fallback path. Load the model with attention_backend=PA on a CPU or GPU device (NPU is not supported "
        "for Qwen3-Omni speech output).");
}

OmniDecodedResults OmniPipeline::OmniPipelineImpl::generate(const std::string& prompt,
                                                             const std::vector<ov::Tensor>& images,
                                                             const std::vector<ov::Tensor>& videos,
                                                             const std::vector<VideoMetadata>& videos_metadata,
                                                             const std::vector<ov::Tensor>& audios,
                                                             const GenerationConfig& text_config,
                                                             const OmniTalkerSpeechConfig& talker_speech_config,
                                                             const StreamerVariant& streamer,
                                                             const OmniSpeechStreamerVariant& speech_streamer) {
    validate_omni_talker_speech_config(talker_speech_config);
    enforce_text_config_compatible_with_audio(text_config, talker_speech_config);

    OPENVINO_ASSERT(videos_metadata.empty() || videos_metadata.size() == videos.size(),
                    "OmniPipeline: videos_metadata size (", videos_metadata.size(),
                    ") must match videos size (", videos.size(), ") or be empty");

    if (talker_speech_config.return_audio && !m_talker) {
        return generate_speech_in_stream(
            [&](const ov::AnyMap& properties) {
                return m_vlm->generate(prompt, properties);
            },
            images, videos, videos_metadata, audios, text_config, talker_speech_config, streamer, speech_streamer);
    }
    if (talker_speech_config.return_audio) {
        GenerationConfig text_cfg = text_config;
        text_cfg.return_omni_outputs = true;
        VLMDecodedResults vlm_result =
            m_vlm->generate(prompt, images, videos, audios, videos_metadata, text_cfg, streamer);
        TalkerResults talker_result = m_talker->generate(vlm_result, talker_speech_config, speech_streamer);
        OmniDecodedResults omni_result;
        static_cast<VLMDecodedResults&>(omni_result) = std::move(vlm_result);
        omni_result.speech_result = std::move(talker_result);
        return omni_result;
    }

    // Text-only path: convert VLMDecodedResults to OmniDecodedResults with empty speech_result.
    VLMDecodedResults vlm_result =
        m_vlm->generate(prompt, images, videos, audios, videos_metadata, text_config, streamer);
    OmniDecodedResults omni_result;
    static_cast<VLMDecodedResults&>(omni_result) = std::move(vlm_result);
    return omni_result;
}

OmniDecodedResults OmniPipeline::OmniPipelineImpl::generate(const ChatHistory& history,
                                                             const std::vector<ov::Tensor>& images,
                                                             const std::vector<ov::Tensor>& videos,
                                                             const std::vector<VideoMetadata>& videos_metadata,
                                                             const std::vector<ov::Tensor>& audios,
                                                             const GenerationConfig& text_config,
                                                             const OmniTalkerSpeechConfig& talker_speech_config,
                                                             const StreamerVariant& streamer,
                                                             const OmniSpeechStreamerVariant& speech_streamer) {
    validate_omni_talker_speech_config(talker_speech_config);
    enforce_text_config_compatible_with_audio(text_config, talker_speech_config);

    OPENVINO_ASSERT(videos_metadata.empty() || videos_metadata.size() == videos.size(),
                    "OmniPipeline: videos_metadata size (", videos_metadata.size(),
                    ") must match videos size (", videos.size(), ") or be empty");

    if (talker_speech_config.return_audio && !m_talker) {
        return generate_speech_in_stream(
            [&](const ov::AnyMap& properties) {
                return m_vlm->generate(history, properties);
            },
            images, videos, videos_metadata, audios, text_config, talker_speech_config, streamer, speech_streamer);
    }
    if (talker_speech_config.return_audio) {
        GenerationConfig text_cfg = text_config;
        text_cfg.return_omni_outputs = true;
        // Keep multimodal normalization inside the ChatHistory path. Applying the chat template
        // first and routing the resulting string through the prompt overload would place image
        // and audio tags outside the user message and change the Thinker output.
        VLMDecodedResults vlm_result =
            m_vlm->generate(history, images, videos, audios, videos_metadata, text_cfg, streamer);
        TalkerResults talker_result = m_talker->generate(vlm_result, talker_speech_config, speech_streamer);
        OmniDecodedResults omni_result;
        static_cast<VLMDecodedResults&>(omni_result) = std::move(vlm_result);
        omni_result.speech_result = std::move(talker_result);
        return omni_result;
    }

    // Text-only path: convert VLMDecodedResults to OmniDecodedResults with empty speech_result.
    VLMDecodedResults vlm_result =
        m_vlm->generate(history, images, videos, audios, videos_metadata, text_config, streamer);
    OmniDecodedResults omni_result;
    static_cast<VLMDecodedResults&>(omni_result) = std::move(vlm_result);
    return omni_result;
}

}  // namespace ov::genai
