// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "openvino/core/model.hpp"
#include "openvino/runtime/infer_request.hpp"

namespace ov::genai {
std::shared_ptr<ov::Model> lfm3_spectrogram();

class LFM3ISTFT {
public:
    explicit LFM3ISTFT(std::vector<float> window);
    ov::Tensor append(const ov::Tensor& spectrum, ov::InferRequest& inverse_fft, bool finalize = false);
    ov::Tensor flush();
    void reset();

private:
    ov::Tensor emit(bool finalize);
    std::vector<float> m_window;
    std::vector<double> m_waveform, m_envelope;
    size_t m_frames = 0, m_emitted = 480;
    bool m_finished = false;
};
}  // namespace ov::genai
