// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/audio_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>

#include "openvino/opsets/opset15.hpp"
#include "utils.hpp"

namespace ov::genai {
namespace {
constexpr size_t fft_size = 512, fft_bins = fft_size / 2 + 1, hop = 160;
constexpr double pi = 3.14159265358979323846;

// Power spectrum of a 512-point STFT with a 160-sample hop: [frames, 257].
std::shared_ptr<ov::Node> stft_power(const ov::Output<ov::Node>& samples, const std::vector<float>& window) {
    using namespace ov::opset15;
    auto spectrum = std::make_shared<STFT>(samples,
                                           Constant::create(ov::element::f32, {fft_size}, window),
                                           Constant::create(ov::element::i64, {}, {int64_t(fft_size)}),
                                           Constant::create(ov::element::i64, {}, {int64_t(hop)}),
                                           false);
    return std::make_shared<ReduceSum>(std::make_shared<Multiply>(spectrum, spectrum),
                                       Constant::create(ov::element::i64, {1}, {-1}),
                                       false);
}

// Mel bands as [bins, 257] triangles over `edges` (bins + 2 Hz values), applied to a [frames, 257] spectrum.
std::shared_ptr<ov::Node> mel_projection(const std::vector<double>& edges,
                                         const ov::Output<ov::Node>& spectrum,
                                         bool area_normalized) {
    using namespace ov::opset15;
    const size_t bins = edges.size() - 2;
    std::vector<float> filters(bins * fft_bins);
    for (size_t m = 0; m < bins; ++m)
        for (size_t k = 0; k < fft_bins; ++k) {
            const double hz = double(k) * 16000. / fft_size;
            const double weight = std::max(0.,
                                           std::min((hz - edges[m]) / (edges[m + 1] - edges[m]),
                                                    (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1])));
            filters[m * fft_bins + k] = float(area_normalized ? weight * 2. / (edges[m + 2] - edges[m]) : weight);
        }
    return std::make_shared<MatMul>(Constant::create(ov::element::f32, {bins, fft_bins}, filters),
                                    spectrum,
                                    false,
                                    true);
}

std::shared_ptr<ov::Model> spectrogram_model(const std::shared_ptr<ov::op::v0::Parameter>& samples,
                                             const std::shared_ptr<ov::Node>& output) {
    return std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{samples});
}

// Gemma4 uses a magnitude HTK spectrogram, a 20 ms periodic Hann window,
// and semicausal padding. Whisper's power/Slaney frontend is not interchangeable.
std::shared_ptr<ov::Model> gemma4_spectrogram(size_t mel_bins) {
    using namespace ov::opset15;
    auto samples = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{-1});
    std::vector<float> window(fft_size, 0.f);
    for (size_t i = 0; i < 320; ++i)
        window[i] = .5f - .5f * std::cos(float(2 * pi * i / 320));
    auto magnitude = std::make_shared<Sqrt>(stft_power(samples, window));
    std::vector<double> edges(mel_bins + 2);
    const double mel_max = 2595. * std::log10(1. + 8000. / 700.);
    for (size_t i = 0; i < edges.size(); ++i)
        edges[i] = 700. * (std::pow(10., mel_max * double(i) / double(mel_bins + 1) / 2595.) - 1.);
    auto mel = mel_projection(edges, magnitude, false);
    auto logged =
        std::make_shared<Log>(std::make_shared<Maximum>(mel, Constant::create(ov::element::f32, {}, {.001f})));
    return spectrogram_model(samples,
                             std::make_shared<Unsqueeze>(logged, Constant::create(ov::element::i64, {2}, {0, 1})));
}

std::shared_ptr<ov::Model> lfm2_spectrogram(size_t bins) {
    using namespace ov::opset15;
    auto samples = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{-1});
    std::vector<float> window(fft_size, 0.f);
    for (size_t i = 0; i < 400; ++i)
        window[56 + i] = .5f - .5f * std::cos(float(2 * pi * i / 399));
    auto padded = std::make_shared<Pad>(samples,
                                        Constant::create(ov::element::i64, {1}, {256}),
                                        Constant::create(ov::element::i64, {1}, {256}),
                                        ov::op::PadMode::CONSTANT);
    const auto hz_to_mel = [](double hz) {
        return hz < 1000. ? hz * 3. / 200. : 15. + std::log(hz / 1000.) * 27. / std::log(6.4);
    };
    const auto mel_to_hz = [](double mel) {
        return mel < 15. ? mel * 200. / 3. : 1000. * std::exp((mel - 15.) * std::log(6.4) / 27.);
    };
    std::vector<double> edges(bins + 2);
    for (size_t i = 0; i < edges.size(); ++i)
        edges[i] = mel_to_hz(hz_to_mel(8000.) * double(i) / double(bins + 1));
    auto mel = mel_projection(edges, stft_power(padded, window), true);
    return spectrogram_model(
        samples,
        std::make_shared<Log>(std::make_shared<Add>(mel, Constant::create(ov::element::f32, {}, {0x1p-24f}))));
}

// Compiled encoder plus its serialized infer request; subclasses add the feature extraction.
class AudioEncoderBase {
public:
    AudioEncoderBase(const std::shared_ptr<ov::Model>& model, const std::string& device, const ov::AnyMap& properties)
        : m_encoder(utils::singleton_core()
                        .compile_model(model, device, utils::get_model_properties(properties, "audio_encoder", device))
                        .create_infer_request()) {}

    virtual ~AudioEncoderBase() = default;

    std::vector<ov::Tensor> encode(const ov::Tensor& waveform) {
        std::lock_guard<std::mutex> lock(m_mutex);
        OPENVINO_ASSERT(waveform.get_element_type() == ov::element::f32 && waveform.get_size() >= min_samples(),
                        "Audio requires mono f32 samples at 16000 Hz, at least ",
                        min_samples(),
                        " samples");
        const auto& shape = waveform.get_shape();
        OPENVINO_ASSERT(shape.size() == 1 || (shape.size() == 2 && shape[0] == 1), "Audio must be mono");
        return encode_waveform(waveform.data<const float>(), waveform.get_size());
    }

protected:
    virtual size_t min_samples() const = 0;
    virtual std::vector<ov::Tensor> encode_waveform(const float* samples, size_t count) = 0;

    ov::Tensor infer() {
        m_encoder.infer();
        const auto output = m_encoder.get_tensor("audio_features");
        ov::Tensor owned(output.get_element_type(), output.get_shape());
        output.copy_to(owned);
        return owned;
    }

    static ov::InferRequest compile_features(const std::shared_ptr<ov::Model>& model) {
        return utils::singleton_core()
            .compile_model(model, "CPU", {{ov::hint::inference_precision.name(), ov::element::f32}})
            .create_infer_request();
    }

    ov::InferRequest m_encoder, m_features;

private:
    std::mutex m_mutex;
};

class LFM2AudioEncoder : public AudioEncoderBase {
public:
    LFM2AudioEncoder(const std::shared_ptr<ov::Model>& model,
                     size_t mel_bins,
                     const std::string& device,
                     const ov::AnyMap& properties)
        : AudioEncoderBase(model, device, properties),
          m_bins(mel_bins),
          m_position_width(model->input("pos_emb").get_partial_shape()[3].get_length()) {
        m_features = compile_features(lfm2_spectrogram(m_bins));
    }

private:
    size_t min_samples() const override {
        return hop;
    }

    std::vector<ov::Tensor> encode_waveform(const float* source, size_t count) override {
        ov::Tensor emphasized(ov::element::f32, {count});
        auto* target = emphasized.data<float>();
        target[0] = source[0];
        for (size_t i = 1; i < count; ++i)
            target[i] = source[i] - .97f * source[i - 1];
        m_features.set_input_tensor(emphasized);
        m_features.infer();
        const auto mel = m_features.get_output_tensor();
        const auto valid = count / hop;
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
        return {infer()};
    }

    size_t m_bins, m_position_width;
};

class Gemma4AudioEncoder : public AudioEncoderBase {
public:
    Gemma4AudioEncoder(const std::shared_ptr<ov::Model>& model,
                       const AudioEncoderConfig& config,
                       const std::string& device,
                       const ov::AnyMap& properties)
        : AudioEncoderBase(model, device, properties),
          m_unified(config.type == AudioEncoderType::GEMMA4_UNIFIED) {
        if (!m_unified)
            m_features = compile_features(gemma4_spectrogram(config.num_mel_bins));
    }

private:
    size_t min_samples() const override {
        return 1;
    }

    std::vector<ov::Tensor> encode_waveform(const float* samples, size_t count) override {
        if (m_unified)
            return {encode_waveform_frames(samples, count)};
        // The encoder attends within 30 s chunks of the spectrogram.
        std::vector<ov::Tensor> outputs;
        constexpr size_t chunk = 30 * 16000;
        for (size_t offset = 0; offset < count; offset += chunk)
            outputs.push_back(encode_spectrogram(samples + offset, std::min(chunk, count - offset)));
        return outputs;
    }

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

    bool m_unified;
};
}  // namespace

AudioEncode create_audio_encoder(const std::shared_ptr<ov::Model>& model,
                                 const AudioEncoderConfig& config,
                                 const std::string& device,
                                 const ov::AnyMap& properties) {
    std::shared_ptr<AudioEncoderBase> encoder;
    switch (config.type) {
    case AudioEncoderType::LFM2:
        encoder = std::make_shared<LFM2AudioEncoder>(model, config.num_mel_bins, device, properties);
        break;
    case AudioEncoderType::GEMMA4:
    case AudioEncoderType::GEMMA4_UNIFIED:
        encoder = std::make_shared<Gemma4AudioEncoder>(model, config, device, properties);
        break;
    }
    return [encoder](const ov::Tensor& audio) {
        return encoder->encode(audio);
    };
}
}  // namespace ov::genai
