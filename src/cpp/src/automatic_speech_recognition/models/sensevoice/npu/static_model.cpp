// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/models/sensevoice/npu/static_model.hpp"

#include <memory>

#include "automatic_speech_recognition/models/sensevoice/feature_extractor.hpp"
#include "openvino/core/model.hpp"
#include "utils.hpp"

namespace ov::genai {

ov::CompiledModel compile_sensevoice_for_npu(const std::filesystem::path& model_path,
                                             const std::string& device,
                                             std::optional<size_t> window_seconds,
                                             const ov::AnyMap& properties) {
    constexpr size_t default_window_seconds = 30;
    const size_t max_audio_samples =
        window_seconds.value_or(default_window_seconds) * FunASRFeatureExtractor::sampling_rate;
    const size_t max_lfr_frames = FunASRFeatureExtractor::output_frames(max_audio_samples);
    // The model prepends 4 query tokens (language, event/emotion, textnorm) to the encoder sequence.
    // NPU prefers sequence lengths aligned to 16, so the padded input is sized to make the encoder sequence aligned.
    constexpr size_t prefix_query_tokens = 4;
    constexpr size_t sequence_alignment = 16;
    const size_t padded_frames =
        (max_lfr_frames + prefix_query_tokens + sequence_alignment - 1) / sequence_alignment * sequence_alignment -
        prefix_query_tokens;

    ov::Core& core = utils::singleton_core();
    std::shared_ptr<ov::Model> model = core.read_model(model_path, {}, properties);
    model->reshape(
        {{"input_features", ov::PartialShape{1, padded_frames, SenseVoiceSmallFeatureExtractor::feature_size}},
         {"speech_lengths", ov::PartialShape{1}},
         {"language", ov::PartialShape{1}},
         {"textnorm", ov::PartialShape{1}}});

    return core.compile_model(model, device, properties);
}

}  // namespace ov::genai
