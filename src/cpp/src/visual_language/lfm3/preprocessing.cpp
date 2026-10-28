// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "preprocessing.hpp"

#include <algorithm>
#include <cmath>

#include "openvino/opsets/opset15.hpp"

namespace ov::genai {
std::shared_ptr<ov::Model> lfm3_spectrogram() {
    using namespace ov::opset15;
    constexpr size_t fft = 400, bins = 201, mel_bins = 128;
    constexpr double pi = 3.14159265358979323846;
    auto samples = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{-1});
    std::vector<float> window(fft);
    const auto angular_step = float(2. * pi / fft);
    for (size_t i = 0; i < fft; ++i)
        window[i] = .5f - .5f * std::cos(float(i) * angular_step);
    auto spectrum = std::make_shared<STFT>(samples,
                                           Constant::create(ov::element::f32, {fft}, window),
                                           Constant::create(ov::element::i64, {}, {400}),
                                           Constant::create(ov::element::i64, {}, {160}),
                                           false);
    auto power = std::make_shared<ReduceSum>(std::make_shared<Multiply>(spectrum, spectrum),
                                             Constant::create(ov::element::i64, {1}, {-1}),
                                             false);
    const auto mel_to_hz = [](double mel) {
        return mel < 15. ? mel * 200. / 3. : 1000. * std::exp((mel - 15.) * std::log(6.4) / 27.);
    };
    const double max_mel = 15. + std::log(8.) * 27. / std::log(6.4);
    std::vector<double> edges(mel_bins + 2);
    for (size_t i = 0; i < edges.size(); ++i)
        edges[i] = mel_to_hz(max_mel * i / (mel_bins + 1));
    std::vector<float> filters(mel_bins * bins);
    for (size_t m = 0; m < mel_bins; ++m)
        for (size_t k = 0; k < bins; ++k) {
            const double hz = double(k) * 16000. / fft;
            const double weight = std::max(0.,
                                           std::min((hz - edges[m]) / (edges[m + 1] - edges[m]),
                                                    (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1])));
            filters[m * bins + k] = float(weight * 2. / (edges[m + 2] - edges[m]));
        }
    auto mel =
        std::make_shared<MatMul>(Constant::create(ov::element::f32, {mel_bins, bins}, filters), power, false, true);
    auto log10 = std::make_shared<Multiply>(
        std::make_shared<Log>(std::make_shared<Maximum>(mel, Constant::create(ov::element::f32, {}, {1e-12f}))),
        Constant::create(ov::element::f32, {}, {float(1. / std::log(10.))}));
    auto normalized =
        std::make_shared<Multiply>(std::make_shared<Add>(log10, Constant::create(ov::element::f32, {}, {4.f})),
                                   Constant::create(ov::element::f32, {}, {.25f}));
    return std::make_shared<ov::Model>(ov::OutputVector{normalized}, ov::ParameterVector{samples});
}

LFM3ISTFT::LFM3ISTFT(std::vector<float> window) : m_window(std::move(window)) {
    OPENVINO_ASSERT(m_window.size() == 1280, "LFM3 ISTFT requires a 1280-sample window");
}

void LFM3ISTFT::reset() {
    m_waveform.clear();
    m_envelope.clear();
    m_frames = 0;
    m_emitted = 480;
    m_finished = false;
}

ov::Tensor LFM3ISTFT::append(const ov::Tensor& values, ov::InferRequest& inverse_fft, bool finalize) {
    OPENVINO_ASSERT(!m_finished, "LFM3 ISTFT stream is finalized; reset before continuing");
    const auto shape = values.get_shape();
    OPENVINO_ASSERT(values.get_element_type() == ov::element::f32 && (shape.size() == 3 || shape.size() == 4) &&
                        shape.back() == 1282 && shape[0] == 1 && (shape.size() == 3 || shape[1] == 1),
                    "Invalid LFM3 spectral output");
    const auto frames = shape[shape.size() - 2];
    if (!frames)
        return emit(finalize);
    ov::Tensor spectrum(ov::element::f32, {frames, 641, 2});
    for (size_t t = 0; t < frames; ++t)
        for (size_t k = 0; k < 641; ++k) {
            const auto magnitude = std::exp(values.data<const float>()[t * 1282 + k]);
            const auto angle = values.data<const float>()[t * 1282 + 641 + k];
            spectrum.data<float>()[(t * 641 + k) * 2] = magnitude * std::cos(angle);
            spectrum.data<float>()[(t * 641 + k) * 2 + 1] = magnitude * std::sin(angle);
        }
    inverse_fft.set_input_tensor(spectrum);
    inverse_fft.infer();
    const auto inverse = inverse_fft.get_output_tensor();
    m_waveform.resize((m_frames + frames - 1) * 320 + 1280);
    m_envelope.resize(m_waveform.size());
    for (size_t t = 0; t < frames; ++t)
        for (size_t i = 0; i < 1280; ++i) {
            const auto position = (m_frames + t) * 320 + i;
            m_waveform[position] += inverse.data<const float>()[t * 1280 + i] * m_window[i];
            m_envelope[position] += double(m_window[i]) * m_window[i];
        }
    m_frames += frames;
    return emit(finalize);
}

ov::Tensor LFM3ISTFT::emit(bool finalize) {
    const auto end = m_frames ? m_frames * 320 + (finalize ? 480 : 0) : m_emitted;
    const auto count = end > m_emitted ? end - m_emitted : 0;
    ov::Tensor output(ov::element::f32, {1, count});
    for (size_t i = 0; i < count; ++i) {
        OPENVINO_ASSERT(m_envelope[m_emitted + i] > 1e-11, "LFM3 ISTFT has a zero overlap envelope");
        output.data<float>()[i] = float(m_waveform[m_emitted + i] / m_envelope[m_emitted + i]);
    }
    m_emitted += count;
    m_finished = finalize;
    return output;
}

ov::Tensor LFM3ISTFT::flush() {
    OPENVINO_ASSERT(!m_finished, "LFM3 ISTFT stream is finalized; reset before flushing");
    return emit(true);
}
}  // namespace ov::genai
