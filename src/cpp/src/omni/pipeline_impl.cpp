// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "omni/pipeline_impl.hpp"

#include <future>
#include <memory>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/genai/omni/channel.hpp"

namespace ov::genai {

namespace {

/// @brief Bridge the thinker and the talker when the caller opted into streaming. Null otherwise,
/// which keeps the VLM on its batch path: the talker then consumes the finished VLMDecodedResults.
std::shared_ptr<OmniChannel> make_channel_if_streaming(const GenerationConfig& text_config) {
    return text_config.text2audio_stream ? std::make_shared<OmniChannel>() : nullptr;
}

/// @brief Owns the talker half of one generate() call and decides where it runs.
///
/// With a bridge, the talker starts right away on its own thread and reads the bridge while the
/// thinker is still filling it; the thinker keeps the calling thread. finish() joins the two back
/// together. Without a bridge there is nothing to overlap, so finish() runs the talker inline over
/// the finished VLM result.
///
/// The thread is what makes early speech possible at all: with the talker on the caller's thread it
/// would have nowhere to run until the thinker was already done, so no amount of incremental
/// consumption downstream could help. Whether speech actually arrives sooner is then up to the
/// talker, which decides how much of the stream it wants before inferring anything.
///
/// One visible consequence: when streaming, a speech streamer callback is invoked from the talker
/// thread rather than the caller's. The text streamer keeps running on the caller's thread as
/// before, so the two can now fire concurrently.
class TalkerStage {
public:
    TalkerStage(const std::shared_ptr<TalkerBase>& talker,
                std::shared_ptr<OmniChannel> channel,
                const OmniTalkerSpeechConfig& talker_speech_config,
                const OmniSpeechStreamerVariant& speech_streamer)
        : m_talker{talker},
          m_channel{std::move(channel)},
          m_talker_speech_config{talker_speech_config},
          m_speech_streamer{speech_streamer} {
        if (!m_channel) {
            return;
        }
        m_result = std::async(std::launch::async, [this] {
            return m_talker->generate(m_channel, m_talker_speech_config, m_speech_streamer);
        });
    }

    /// @brief Close the bridge, then wait for the talker thread (the future's destructor blocks,
    /// and members die after this body). Ending here covers the paths that never reach finish():
    /// a thinker that threw before it could close its own write end would otherwise leave the
    /// talker blocked on a read that can never be satisfied, and the wait would deadlock. Ending
    /// twice is harmless, and a reader still drains whatever was written before the end.
    ///
    /// The talker then runs a full inference over a truncated stream just to be discarded, which
    /// is wasteful during unwinding but keeps the failure path simple: whatever it throws stays in
    /// the future and never competes with the exception already in flight.
    ~TalkerStage() {
        if (m_channel) {
            m_channel->end();
        }
    }

    /// @brief Hand back the talker's output, waiting for its thread when streaming. Anything the
    /// talker thread threw is rethrown here, on the caller's thread.
    TalkerResults finish(const VLMDecodedResults& vlm_result) {
        if (!m_channel) {
            return m_talker->generate(vlm_result, m_talker_speech_config, m_speech_streamer);
        }
        return m_result.get();
    }

private:
    std::shared_ptr<TalkerBase> m_talker;
    std::shared_ptr<OmniChannel> m_channel;
    // Both outlive the stage: it is a local of the generate() call that owns them, and finish()
    // (or the destructor) joins the thread before that call returns.
    const OmniTalkerSpeechConfig& m_talker_speech_config;
    const OmniSpeechStreamerVariant& m_speech_streamer;
    std::future<TalkerResults> m_result;
};

/// @brief Cross-config validation: when speech output is requested the thinker text decode
/// must use a sampling mode the talker can consume — single hidden-state stream, no beam
/// candidates, no speculative draft tokens.
void enforce_text_config_compatible_with_audio(const GenerationConfig& text_config,
                                               const OmniTalkerSpeechConfig& talker_speech_config) {
    OPENVINO_ASSERT(!text_config.text2audio_stream || talker_speech_config.return_audio,
                    "OmniPipeline: text_config.text2audio_stream streams the thinker's output to the talker, "
                    "so it requires talker_speech_config.return_audio == true");
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

}  // namespace

OmniPipeline::OmniPipelineImpl::OmniPipelineImpl(const std::shared_ptr<VLMPipelineBase>& vlm,
                                                  const std::shared_ptr<TalkerBase>& talker) :
    m_vlm{vlm}, m_talker{talker} {
    OPENVINO_ASSERT(m_vlm, "OmniPipeline: VLM pointer is null");
    OPENVINO_ASSERT(m_talker, "OmniPipeline: talker pointer is null");
    assert_omni_capable();
}

void OmniPipeline::OmniPipelineImpl::assert_omni_capable() const {
    OPENVINO_ASSERT(
        m_vlm->is_audio_output_enabled(),
        "OmniPipeline requires a Qwen3-Omni model with audio output enabled (config.json: enable_audio_output=true)");
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

    if (talker_speech_config.return_audio) {
        GenerationConfig text_cfg = text_config;
        text_cfg.return_omni_outputs = true;
        const std::shared_ptr<OmniChannel> channel = make_channel_if_streaming(text_cfg);
        // Declared before the VLM runs: when streaming, this is what puts the talker on its own
        // thread, and it has to be listening before the thinker starts writing.
        TalkerStage talker_stage(m_talker, channel, talker_speech_config, speech_streamer);
        // Only reach for the streaming overload when there is something to stream: implementing it
        // is optional for a VLM backend, and calling it with a null streamer would throw on a
        // backend that supports batch speech but not streaming.
        VLMDecodedResults vlm_result =
            channel ? m_vlm->generate(prompt, images, videos, audios, videos_metadata, text_cfg, streamer, channel)
                    : m_vlm->generate(prompt, images, videos, audios, videos_metadata, text_cfg, streamer);
        TalkerResults talker_result = talker_stage.finish(vlm_result);
        // TODO: when `channel` is set, vlm_result.intermediate_hidden_states holds a second full
        // copy of what already travelled through it — the CB backend accumulates every step on the
        // sequence *and* forwards it. Clearing it here (and in the ChatHistory overload below)
        // reclaims one copy once the field is no longer part of the streaming contract. The deeper
        // fix is upstream: with a bridge attached the backend shouldn't accumulate at all, which
        // means the forwarder in continuous_batching/pipeline_impl.cpp has to tap the per-step
        // tensor directly instead of reading out of that same accumulation buffer.
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

    if (talker_speech_config.return_audio) {
        GenerationConfig text_cfg = text_config;
        text_cfg.return_omni_outputs = true;
        // Keep multimodal normalization inside the ChatHistory path. Applying the chat template
        // first and routing the resulting string through the prompt overload would place image
        // and audio tags outside the user message and change the Thinker output.
        const std::shared_ptr<OmniChannel> channel = make_channel_if_streaming(text_cfg);
        TalkerStage talker_stage(m_talker, channel, talker_speech_config, speech_streamer);
        // Only reach for the streaming overload when there is something to stream: implementing it
        // is optional for a VLM backend, and calling it with a null streamer would throw on a
        // backend that supports batch speech but not streaming.
        VLMDecodedResults vlm_result =
            channel ? m_vlm->generate(history, images, videos, audios, videos_metadata, text_cfg, streamer, channel)
                    : m_vlm->generate(history, images, videos, audios, videos_metadata, text_cfg, streamer);
        // TODO: same duplicated hidden states as in the prompt overload above; see the note there.
        TalkerResults talker_result = talker_stage.finish(vlm_result);
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
