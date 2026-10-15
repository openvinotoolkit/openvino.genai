// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <mutex>

#include "gguf_multimodal.hpp"
#include "openvino/opsets/opset15.hpp"
#include "utils.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {
namespace {
// Gemma4 uses a magnitude HTK spectrogram, a 20 ms periodic Hann window,
// and semicausal padding. Whisper's power/Slaney frontend is not interchangeable.
std::shared_ptr<ov::Model> gemma4_spectrogram(size_t mel_bins) {
    using namespace ov::opset15;
    auto samples = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{-1});
    std::vector<float> window(512, 0.f);
    constexpr double pi = 3.14159265358979323846;
    for (size_t i = 0; i < 320; ++i)
        window[i] = .5f - .5f * std::cos(float(2 * pi * i / 320));
    auto spectrum = std::make_shared<STFT>(samples,
                                           Constant::create(ov::element::f32, {512}, window),
                                           Constant::create(ov::element::i64, {}, {512}),
                                           Constant::create(ov::element::i64, {}, {160}),
                                           false);
    auto power = std::make_shared<Multiply>(spectrum, spectrum);
    auto magnitude = std::make_shared<Sqrt>(
        std::make_shared<ReduceSum>(power, Constant::create(ov::element::i64, {1}, {-1}), false));
    std::vector<double> edges(mel_bins + 2);
    const double mel_max = 2595. * std::log10(1. + 8000. / 700.);
    for (size_t i = 0; i < edges.size(); ++i)
        edges[i] = 700. * (std::pow(10., mel_max * double(i) / double(mel_bins + 1) / 2595.) - 1.);
    std::vector<float> filters(mel_bins * 257);
    for (size_t m = 0; m < mel_bins; ++m)
        for (size_t k = 0; k < 257; ++k) {
            const double hz = double(k) * 16000. / 512.;
            filters[m * 257 + k] = std::max(0.,
                                            std::min((hz - edges[m]) / (edges[m + 1] - edges[m]),
                                                     (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1])));
        }
    auto mel =
        std::make_shared<MatMul>(Constant::create(ov::element::f32, {mel_bins, 257}, filters), magnitude, false, true);
    auto logged =
        std::make_shared<Log>(std::make_shared<Maximum>(mel, Constant::create(ov::element::f32, {}, {.001f})));
    auto output = std::make_shared<Unsqueeze>(logged, Constant::create(ov::element::i64, {2}, {0, 1}));
    return std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{samples});
}

std::shared_ptr<ov::Model> lfm2_spectrogram(size_t bins) {
    using namespace ov::opset15;
    auto samples = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{-1});
    std::vector<float> window(512, 0.f);
    constexpr double pi = 3.14159265358979323846;
    for (size_t i = 0; i < 400; ++i)
        window[56 + i] = .5f - .5f * std::cos(float(2 * pi * i / 399));
    auto padded = std::make_shared<Pad>(samples,
                                        Constant::create(ov::element::i64, {1}, {256}),
                                        Constant::create(ov::element::i64, {1}, {256}),
                                        ov::op::PadMode::CONSTANT);
    auto spectrum = std::make_shared<STFT>(padded,
                                           Constant::create(ov::element::f32, {512}, window),
                                           Constant::create(ov::element::i64, {}, {512}),
                                           Constant::create(ov::element::i64, {}, {160}),
                                           false);
    auto power = std::make_shared<ReduceSum>(std::make_shared<Multiply>(spectrum, spectrum),
                                             Constant::create(ov::element::i64, {1}, {-1}),
                                             false);
    const auto hz_to_mel = [](double hz) {
        return hz < 1000. ? hz * 3. / 200. : 15. + std::log(hz / 1000.) * 27. / std::log(6.4);
    };
    const auto mel_to_hz = [](double mel) {
        return mel < 15. ? mel * 200. / 3. : 1000. * std::exp((mel - 15.) * std::log(6.4) / 27.);
    };
    std::vector<double> edges(bins + 2);
    for (size_t i = 0; i < edges.size(); ++i)
        edges[i] = mel_to_hz(hz_to_mel(8000.) * double(i) / double(bins + 1));
    std::vector<float> filters(bins * 257);
    for (size_t m = 0; m < bins; ++m)
        for (size_t k = 0; k < 257; ++k) {
            const double hz = double(k) * 16000. / 512.;
            filters[m * 257 + k] = std::max(0.,
                                            std::min((hz - edges[m]) / (edges[m + 1] - edges[m]),
                                                     (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]))) *
                                   2. / (edges[m + 2] - edges[m]);
        }
    auto mel = std::make_shared<MatMul>(Constant::create(ov::element::f32, {bins, 257}, filters), power, false, true);
    auto logged = std::make_shared<Log>(std::make_shared<Add>(mel, Constant::create(ov::element::f32, {}, {0x1p-24f})));
    return std::make_shared<ov::Model>(ov::OutputVector{logged}, ov::ParameterVector{samples});
}

class GGUFLFM2Audio {
public:
    GGUFLFM2Audio(const std::shared_ptr<ov::Model>& model, const std::string& device, const ov::AnyMap& properties) {
        m_bins = std::stoull(model->get_rt_info<std::string>({"gguf_mmproj", "clip.audio.num_mel_bins"}));
        m_position_width = model->input("pos_emb").get_partial_shape()[3].get_length();
        m_encoder = utils::singleton_core()
                        .compile_model(model, device, utils::get_model_properties(properties, "audio_encoder", device))
                        .create_infer_request();
        m_features = utils::singleton_core()
                         .compile_model(lfm2_spectrogram(m_bins),
                                        "CPU",
                                        {{ov::hint::inference_precision.name(), ov::element::f32}})
                         .create_infer_request();
    }

    std::vector<ov::Tensor> encode(const ov::Tensor& waveform) {
        std::lock_guard<std::mutex> lock(m_mutex);
        OPENVINO_ASSERT(waveform.get_element_type() == ov::element::f32 && waveform.get_size() >= 160,
                        "LFM2 audio requires mono f32 samples at 16000 Hz, at least 160 samples");
        const auto& shape = waveform.get_shape();
        OPENVINO_ASSERT(shape.size() == 1 || (shape.size() == 2 && shape[0] == 1), "Audio must be mono");
        ov::Tensor emphasized(ov::element::f32, {waveform.get_size()});
        const auto* source = waveform.data<const float>();
        auto* target = emphasized.data<float>();
        target[0] = source[0];
        for (size_t i = 1; i < waveform.get_size(); ++i)
            target[i] = source[i] - .97f * source[i - 1];
        m_features.set_input_tensor(emphasized);
        m_features.infer();
        const auto mel = m_features.get_output_tensor();
        const auto valid = waveform.get_size() / 160;
        const auto frames = mel.get_shape()[1];
        ov::Tensor features(ov::element::f32, {1, 1, m_bins, frames});
        for (size_t m = 0; m < m_bins; ++m) {
            const auto* row = mel.data<const float>() + m * frames;
            double mean = 0., variance = 0.;
            for (size_t i = 0; i < valid; ++i)
                mean += row[i];
            mean /= valid;
            for (size_t i = 0; i < valid; ++i)
                variance += (row[i] - mean) * (row[i] - mean);
            const auto stddev = (valid > 1 ? std::sqrt(variance / (valid - 1)) : 0.) + 1e-5;
            for (size_t i = 0; i < frames; ++i)
                features.data<float>()[m * frames + i] = i < valid ? float((row[i] - mean) / stddev) : 0.f;
        }
        const auto tokens = (frames + 7) / 8;
        ov::Tensor positions(ov::element::f32, {1, 1, tokens * 2 - 1, m_position_width});
        for (size_t p = 0; p < tokens * 2 - 1; ++p)
            for (size_t i = 0; i < m_position_width / 2; ++i) {
                const float angle =
                    (int64_t(tokens) - int64_t(p) - 1) * std::exp(-std::log(10000.) * (2 * i) / m_position_width);
                positions.data<float>()[p * m_position_width + 2 * i] = std::sin(angle);
                positions.data<float>()[p * m_position_width + 2 * i + 1] = std::cos(angle);
            }
        m_encoder.set_tensor("features", features);
        m_encoder.set_tensor("pos_emb", positions);
        m_encoder.infer();
        const auto output = m_encoder.get_tensor("audio_features");
        ov::Tensor owned(output.get_element_type(), output.get_shape());
        output.copy_to(owned);
        return {owned};
    }

private:
    size_t m_bins, m_position_width;
    ov::InferRequest m_encoder, m_features;
    std::mutex m_mutex;
};

class GGUFGemmaAudio {
public:
    GGUFGemmaAudio(const std::shared_ptr<ov::Model>& model, const std::string& device, const ov::AnyMap& properties) {
        const auto projector = model->get_rt_info<std::string>({"gguf_mmproj", "audio.projector"});
        m_unified = projector == "gemma4ua";
        OPENVINO_ASSERT(m_unified || projector == "gemma4a", "Unsupported GGUF audio projector: ", projector);
        const auto config = utils::get_model_properties(properties, "audio_encoder", device);
        m_encoder = utils::singleton_core().compile_model(model, device, config).create_infer_request();
        if (!m_unified) {
            const auto bins = std::stoull(model->get_rt_info<std::string>({"gguf_mmproj", "clip.audio.num_mel_bins"}));
            m_features = utils::singleton_core()
                             .compile_model(gemma4_spectrogram(bins),
                                            "CPU",
                                            {{ov::hint::inference_precision.name(), ov::element::f32}})
                             .create_infer_request();
        }
    }

    std::vector<ov::Tensor> encode(const ov::Tensor& waveform) {
        std::lock_guard<std::mutex> lock(m_mutex);
        OPENVINO_ASSERT(waveform.get_element_type() == ov::element::f32 && waveform.get_size() > 0,
                        "Gemma4 audio requires nonempty mono f32 samples at 16000 Hz");
        const auto& shape = waveform.get_shape();
        OPENVINO_ASSERT(shape.size() == 1 || (shape.size() == 2 && shape[0] == 1), "Audio must be mono");
        if (m_unified)
            return {encode_waveform_frames(waveform.data<const float>(), waveform.get_size())};
        // The encoder attends within 30 s chunks of the spectrogram.
        std::vector<ov::Tensor> outputs;
        constexpr size_t chunk = 30 * 16000;
        for (size_t offset = 0; offset < waveform.get_size(); offset += chunk)
            outputs.push_back(encode_spectrogram(waveform.data<const float>() + offset,
                                                 std::min(chunk, waveform.get_size() - offset)));
        return outputs;
    }

private:
    // Raw 16 kHz samples in zero-padded 640-sample frames.
    ov::Tensor encode_waveform_frames(const float* samples, size_t count) {
        ov::Tensor input(ov::element::f32, {1, 1, (count + 639) / 640, 640});
        std::fill_n(input.data<float>(), input.get_size(), 0.f);
        std::copy_n(samples, count, input.data<float>());
        m_encoder.set_tensor("waveform_frames", input);
        return infer();
    }

    ov::Tensor encode_spectrogram(const float* samples, size_t count) {
        const int64_t frames = (int64_t(count) + 160 - 321) / 160 + 1;
        OPENVINO_ASSERT(frames > 0, "Gemma4 audio chunk is too short to form a spectrogram frame");
        const auto padded = std::max(size_t((frames - 1) * 160 + 512), count + 160);
        ov::Tensor input(ov::element::f32, {padded});
        std::fill_n(input.data<float>(), input.get_size(), 0.f);
        std::copy_n(samples, count, input.data<float>() + 160);
        m_features.set_input_tensor(input);
        m_features.infer();
        m_encoder.set_tensor("features", m_features.get_output_tensor());
        return infer();
    }

    ov::Tensor infer() {
        m_encoder.infer();
        const auto output = m_encoder.get_tensor("audio_features");
        ov::Tensor owned(output.get_element_type(), output.get_shape());
        output.copy_to(owned);
        return owned;
    }

    bool m_unified = false;
    ov::InferRequest m_encoder, m_features;
    std::mutex m_mutex;
};
}  // namespace

void attach_gguf_audio_encoder(InputsEmbedder& embedder,
                               const GGUFMultimodalModels& models,
                               const std::string& device,
                               const ov::AnyMap& properties) {
    if (models.audio) {
        if (models.audio->get_rt_info<std::string>({"gguf_mmproj", "audio.projector"}) == "lfm2a") {
            auto encoder = std::make_shared<GGUFLFM2Audio>(models.audio, device, properties);
            embedder.set_audio_encoder([encoder](const ov::Tensor& audio) {
                return encoder->encode(audio);
            });
            return;
        }
        auto encoder = std::make_shared<GGUFGemmaAudio>(models.audio, device, properties);
        embedder.set_audio_encoder([encoder](const ov::Tensor& audio) {
            return encoder->encode(audio);
        });
    }
}
}  // namespace ov::genai
