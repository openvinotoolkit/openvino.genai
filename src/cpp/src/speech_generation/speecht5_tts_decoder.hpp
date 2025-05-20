// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "openvino/runtime/core.hpp"

namespace ov::genai {

class SpeechT5TTSDecoder {
public:
    static std::shared_ptr<SpeechT5TTSDecoder> from_path(const std::filesystem::path& models_path,
                                                         const std::string& device,
                                                         const ov::AnyMap& properties);

    SpeechT5TTSDecoder(const std::filesystem::path& models_path,
                       const std::string& device,
                       const ov::AnyMap& properties);

    void start_async(const Tensor& inputs_embeds,
                     const Tensor& speaker_embeddings,
                     const Tensor& encoder_hidden_states,
                     const Tensor& encoder_attention_mask,
                     const Tensor& spectrogram);

    std::tuple<Tensor, Tensor, Tensor, Tensor> wait();

    void reset_state();

    ov::Tensor create_host_tensor(const element::Type element_type, const Shape& shape);

private:
    ov::InferRequest m_request;
    Tensor m_beam_idx_tensor;
};
}  // namespace ov::genai
