// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/speech_generation/audio_streamer_base.hpp"

#include "utils.hpp"

namespace ov {
namespace genai {

AudioStreamerBase::~AudioStreamerBase() = default;

std::pair<std::string, ov::Any> audio_streamer(AudioStreamerVariant streamer) {
    if (std::holds_alternative<std::monostate>(streamer)) {
        return {utils::AUDIO_STREAMER_ARG_NAME, ov::Any()};
    } else if (auto* streamer_obj = std::get_if<std::shared_ptr<AudioStreamerBase>>(&streamer)) {
        return {utils::AUDIO_STREAMER_ARG_NAME, ov::Any::make<std::shared_ptr<AudioStreamerBase>>(*streamer_obj)};
    } else {
        auto callback = std::get<std::function<StreamingStatus(const ov::Tensor&)>>(streamer);
        return {utils::AUDIO_STREAMER_ARG_NAME,
                ov::Any::make<std::function<StreamingStatus(const ov::Tensor&)>>(callback)};
    }
}

}  // namespace genai
}  // namespace ov
