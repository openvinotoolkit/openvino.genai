// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "text2speech_pipeline_impl.hpp"

#include "omni/speech_streamer_utils.hpp"

namespace ov {
namespace genai {

Text2SpeechDecodedResults Text2SpeechPipelineImpl::generate(const std::vector<std::string>& texts,
                                                            const ov::Tensor& speaker_embedding,
                                                            const SpeechGenerationConfig& generation_config,
                                                            const AudioStreamerVariant& streamer) {
    const bool streaming = is_speech_streamer_active(streamer);
    if (!streaming) {
        return generate(texts, speaker_embedding, generation_config);
    }
    SpeechStreamerGuard guard(streamer, true);
    auto results = generate(texts, speaker_embedding, generation_config);
    for (const auto& speech : results.speeches) {
        if (!speech || speech.get_size() == 0) {
            continue;
        }
        ov::Tensor chunk(ov::element::f32, ov::Shape{1, 1, speech.get_size()});
        std::copy_n(speech.data<const float>(), speech.get_size(), chunk.data<float>());
        const auto status = invoke_speech_streamer(streamer, chunk);
        if (status == StreamingStatus::CANCEL) {
            results.speeches.clear();
            break;
        }
        if (status == StreamingStatus::STOP) {
            break;
        }
    }
    guard.end();
    return results;
}

SpeechGenerationPerfMetrics Text2SpeechPipelineImpl::get_performance_metrics() {
    m_perf_metrics.load_time = m_load_time_ms;
    return m_perf_metrics;
}

void Text2SpeechPipelineImpl::save_load_time(std::chrono::steady_clock::time_point start_time) {
    auto stop_time = std::chrono::steady_clock::now();
    m_load_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
}
}  // namespace genai
}  // namespace ov
