// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_tts_model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>

#include <nlohmann/json.hpp>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset15.hpp>

#include "json_utils.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "utils.hpp"

namespace {

using json = nlohmann::json;

constexpr const char* TALKER_LANGUAGE_NAME = "openvino_talker_language_model.xml";
constexpr const char* TALKER_EMBEDDING_NAME = "openvino_talker_embedding_model.xml";
constexpr const char* TALKER_TEXT_EMBEDDING_NAME = "openvino_talker_text_embedding_model.xml";
constexpr const char* TALKER_TEXT_PROJECTION_NAME = "openvino_talker_text_projection_model.xml";
constexpr const char* TALKER_CODE_PREDICTOR_NAME = "openvino_talker_code_predictor_model.xml";
constexpr const char* TALKER_CODE_PREDICTOR_EMBEDDING_NAME = "openvino_talker_code_predictor_embedding_model.xml";
constexpr const char* SPEECH_TOKENIZER_DECODER_NAME = "openvino_speech_tokenizer_decoder_model.xml";
constexpr const char* SPEAKER_ENCODER_NAME = "openvino_speaker_encoder_model.xml";
constexpr const char* SPEECH_TOKENIZER_ENCODER_NAME = "openvino_speech_tokenizer_encoder_model.xml";

constexpr int64_t DECODER_TRACE_LEN = 325;
constexpr int64_t DECODER_CHUNK_SIZE = 300;
constexpr int64_t DECODER_LEFT_CONTEXT = 25;
constexpr int64_t DECODER_OFFSET = 555;
constexpr float PI_F = 3.14159265358979323846f;

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

ov::InferRequest compile_request(const std::filesystem::path& model_path,
                                 const std::string& model_name,
                                 const std::string& device,
                                 const ov::AnyMap& properties) {
    ov::Core& core = ov::genai::utils::singleton_core();
    auto compiled_model = core.compile_model(model_path, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, model_name.c_str());
    return compiled_model.create_infer_request();
}

ov::InferRequest compile_request(const std::shared_ptr<ov::Model>& model,
                                 const std::string& model_name,
                                 const std::string& device,
                                 const ov::AnyMap& properties) {
    ov::Core& core = ov::genai::utils::singleton_core();
    auto compiled_model = core.compile_model(model, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, model_name.c_str());
    return compiled_model.create_infer_request();
}

ov::Tensor clone_tensor(const ov::Tensor& src) {
    ov::Tensor dst(src.get_element_type(), src.get_shape());
    std::memcpy(dst.data(), src.data(), src.get_byte_size());
    return dst;
}

bool qwen_debug_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("OV_GENAI_QWEN_TTS_DEBUG");
        if (env == nullptr) {
            return false;
        }
        const std::string v = to_lower(std::string(env));
        return v == "1" || v == "true" || v == "yes" || v == "on";
    }();
    return enabled;
}

std::string shape_to_string(const ov::Shape& shape) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << shape[i];
    }
    os << "]";
    return os.str();
}

void debug_print_tensor(const std::string& stage, const std::string& name, const ov::Tensor& tensor) {
    if (!qwen_debug_enabled()) {
        return;
    }
    std::cout << "[QWEN_DEBUG] " << stage << " tensor='" << name << "'"
              << " type=" << tensor.get_element_type()
              << " shape=" << shape_to_string(tensor.get_shape())
              << " size=" << tensor.get_size() << std::endl;
}

void debug_print_model_io(const std::string& stage, ov::InferRequest& request) {
    if (!qwen_debug_enabled()) {
        return;
    }

    auto compiled = request.get_compiled_model();
    std::cout << "[QWEN_DEBUG] " << stage << " model IO:" << std::endl;
    for (const auto& input : compiled.inputs()) {
        std::string name;
        try {
            const auto& names = input.get_names();
            name = names.empty() ? "(unnamed)" : *names.begin();
        } catch (...) {
            name = "(unnamed)";
        }
        std::cout << "[QWEN_DEBUG]   input name='" << name << "' type="
                  << input.get_element_type() << " pshape=" << input.get_partial_shape() << std::endl;
    }
    for (const auto& output : compiled.outputs()) {
        std::string name;
        try {
            const auto& names = output.get_names();
            name = names.empty() ? "(unnamed)" : *names.begin();
        } catch (...) {
            name = "(unnamed)";
        }
        std::cout << "[QWEN_DEBUG]   output name='" << name << "' type="
                  << output.get_element_type() << " pshape=" << output.get_partial_shape() << std::endl;
    }
}

void debug_print_stage(const std::string& stage) {
    if (!qwen_debug_enabled()) {
        return;
    }
    std::cout << "[QWEN_DEBUG] " << stage << std::endl;
}

float hertz_to_mel(const float freq) {
    constexpr float min_log_hertz = 1000.0f;
    constexpr float min_log_mel = 15.0f;
    const float logstep = 27.0f / std::log(6.4f);
    float mel = 3.0f * freq / 200.0f;

    if (freq >= min_log_hertz) {
        mel = min_log_mel + std::log(freq / min_log_hertz) * logstep;
    }
    return mel;
}

float mel_to_hertz(const float mel) {
    constexpr float min_log_hertz = 1000.0f;
    constexpr float min_log_mel = 15.0f;
    const float logstep = std::log(6.4f) / 27.0f;
    float freq = 200.0f * mel / 3.0f;

    if (mel >= min_log_mel) {
        freq = min_log_hertz * std::exp(logstep * (mel - min_log_mel));
    }
    return freq;
}

std::vector<std::vector<float>> create_triangular_filter_bank(const std::vector<float>& fft_freqs,
                                                              const std::vector<float>& filter_freqs) {
    std::vector<float> filter_diff(filter_freqs.size() - 1);
    for (size_t i = 0; i < filter_diff.size(); ++i) {
        filter_diff[i] = filter_freqs[i + 1] - filter_freqs[i];
    }

    std::vector<std::vector<float>> slopes(fft_freqs.size(), std::vector<float>(filter_freqs.size()));
    for (size_t row = 0; row < slopes.size(); ++row) {
        for (size_t col = 0; col < slopes[0].size(); ++col) {
            slopes[row][col] = filter_freqs[col] - fft_freqs[row];
        }
    }

    std::vector<std::vector<float>> down_slopes(fft_freqs.size(), std::vector<float>(filter_freqs.size() - 2));
    for (size_t row = 0; row < down_slopes.size(); ++row) {
        for (size_t col = 0; col < down_slopes[0].size(); ++col) {
            down_slopes[row][col] = -slopes[row][col] / filter_diff[col];
        }
    }

    std::vector<std::vector<float>> up_slopes(fft_freqs.size(), std::vector<float>(filter_freqs.size() - 2));
    for (size_t row = 0; row < up_slopes.size(); ++row) {
        for (size_t col = 0; col < up_slopes[0].size(); ++col) {
            up_slopes[row][col] = slopes[row][col + 2] / filter_diff[col + 1];
        }
    }

    std::vector<std::vector<float>> result(fft_freqs.size(), std::vector<float>(filter_freqs.size() - 2));
    for (size_t row = 0; row < result.size(); ++row) {
        for (size_t col = 0; col < result[0].size(); ++col) {
            result[row][col] = std::max(0.0f, std::min(down_slopes[row][col], up_slopes[row][col]));
        }
    }

    return result;
}

std::vector<std::vector<float>> mel_filter_bank(const int64_t num_frequency_bins,
                                                const int64_t num_mel_filters,
                                                const int64_t sampling_rate,
                                                const float min_frequency,
                                                const float max_frequency) {
    OPENVINO_ASSERT(max_frequency <= (sampling_rate / 2.0f),
                    "max_frequency should be less or equal sampling_rate / 2");

    const float mel_min = hertz_to_mel(min_frequency);
    const float mel_max = hertz_to_mel(max_frequency);

    const float mel_freqs_step = (mel_max - mel_min) / static_cast<float>(num_mel_filters + 1);
    std::vector<float> filter_freqs(num_mel_filters + 2);
    for (size_t i = 0; i < filter_freqs.size(); ++i) {
        filter_freqs[i] = mel_to_hertz(mel_min + static_cast<float>(i) * mel_freqs_step);
    }

    std::vector<float> fft_freqs(num_frequency_bins);
    const float fft_freq_step = (sampling_rate / 2.0f) / static_cast<float>(num_frequency_bins - 1);
    for (size_t i = 0; i < fft_freqs.size(); ++i) {
        fft_freqs[i] = static_cast<float>(i) * fft_freq_step;
    }

    auto mel_filters = create_triangular_filter_bank(fft_freqs, filter_freqs);
    std::vector<float> enorm(num_mel_filters);
    for (size_t i = 0; i < enorm.size(); ++i) {
        enorm[i] = 2.0f / (filter_freqs[i + 2] - filter_freqs[i]);
    }

    for (size_t row = 0; row < mel_filters.size(); ++row) {
        for (size_t col = 0; col < mel_filters[0].size(); ++col) {
            mel_filters[row][col] *= enorm[col];
        }
    }
    return mel_filters;
}

std::vector<float> hann_window(size_t length) {
    std::vector<float> out(length, 0.0f);
    for (size_t i = 0; i < length; ++i) {
        out[i] = 0.5f * (1.0f - std::cos(2.0f * PI_F * static_cast<float>(i) / static_cast<float>(length)));
    }
    return out;
}

std::shared_ptr<ov::Model> build_qwen3_mel_preprocess_model(size_t mel_dim) {
    constexpr int64_t n_fft = 1024;
    constexpr int64_t hop_size = 256;
    constexpr int64_t frame_size = 1024;
    constexpr int64_t padding = (n_fft - hop_size) / 2;

    auto waveform = std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::PartialShape{-1, -1});
    waveform->set_friendly_name("waveform");
    waveform->output(0).set_names({"waveform"});

    auto axis_1 = ov::opset15::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto unsqueezed = std::make_shared<ov::opset15::Unsqueeze>(waveform, axis_1);  // [B, 1, T]

    auto pads_begin = ov::opset15::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 0, padding});
    auto pads_end = ov::opset15::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 0, padding});
    auto pad_value = ov::opset15::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{0.0f});
    auto padded = std::make_shared<ov::opset15::Pad>(unsqueezed,
                                                     pads_begin,
                                                     pads_end,
                                                     pad_value,
                                                     ov::op::PadMode::REFLECT);

    auto squeezed = std::make_shared<ov::opset15::Squeeze>(padded, axis_1);  // [B, T']

    const auto hann = hann_window(static_cast<size_t>(frame_size));
    auto window = ov::opset15::Constant::create(ov::element::f32,
                                                ov::Shape{static_cast<size_t>(frame_size)},
                                                hann);
    auto frame_size_c = ov::opset15::Constant::create(ov::element::i32,
                                                      ov::Shape{},
                                                      std::vector<int32_t>{static_cast<int32_t>(frame_size)});
    auto frame_step_c = ov::opset15::Constant::create(ov::element::i32,
                                                      ov::Shape{},
                                                      std::vector<int32_t>{static_cast<int32_t>(hop_size)});

    auto stft = std::make_shared<ov::opset15::STFT>(squeezed, window, frame_size_c, frame_step_c, false);

    auto power_2 = ov::opset15::Constant::create(ov::element::f32,
                                                 ov::Shape{1, 1, 1, 1},
                                                 std::vector<float>{2.0f});
    auto squared = std::make_shared<ov::opset15::Power>(stft, power_2);
    auto imag_axis = ov::opset15::Constant::create(ov::element::i64,
                                                   ov::Shape{},
                                                   std::vector<int64_t>{-1});
    auto power_sum = std::make_shared<ov::opset15::ReduceSum>(squared, imag_axis, false);
    auto magnitude = std::make_shared<ov::opset15::Sqrt>(power_sum);  // [B, F, Frames]

    const auto mel_filter_2d = mel_filter_bank(1 + n_fft / 2,
                                               static_cast<int64_t>(mel_dim),
                                               24000,
                                               0.0f,
                                               12000.0f);
    std::vector<float> mel_filter_flat;
    mel_filter_flat.reserve(mel_dim * (1 + n_fft / 2));
    for (size_t m = 0; m < mel_dim; ++m) {
        for (size_t f = 0; f < (1 + n_fft / 2); ++f) {
            mel_filter_flat.push_back(mel_filter_2d[f][m]);
        }
    }
    auto mel_filter = ov::opset15::Constant::create(ov::element::f32,
                                                    ov::Shape{1, mel_dim, static_cast<size_t>(1 + n_fft / 2)},
                                                    mel_filter_flat);

    auto mel = std::make_shared<ov::opset15::MatMul>(mel_filter, magnitude, false, true);  // [B, M, Frames]
    auto min_clip = ov::opset15::Constant::create(ov::element::f32,
                                                  ov::Shape{1, 1, 1},
                                                  std::vector<float>{1e-5f});
    auto clipped = std::make_shared<ov::opset15::Maximum>(mel, min_clip);
    auto log_mel = std::make_shared<ov::opset15::Log>(clipped);

    auto order = ov::opset15::Constant::create(ov::element::i64,
                                               ov::Shape{3},
                                               std::vector<int64_t>{0, 2, 1});
    auto output = std::make_shared<ov::opset15::Transpose>(log_mel, order);  // [B, Frames, M]
    output->set_friendly_name("log_mel_features");

    auto result = std::make_shared<ov::opset15::Result>(output);
    result->set_friendly_name("log_mel_features");

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{waveform}, "qwen3_mel_preprocess");
}

}  // namespace

namespace ov {
namespace genai {

Qwen3TTSImpl::Qwen3TTSImpl(const std::filesystem::path& models_path,
                           const std::string& device,
                           const ov::AnyMap& properties,
                           const Tokenizer& tokenizer)
    : m_models_path(models_path),
      m_device(device),
      m_tokenizer(tokenizer) {
    init_config(models_path);

    m_talker = compile_request(models_path / TALKER_LANGUAGE_NAME, "qwen3_tts talker", device, properties);
    m_talker_embedding = compile_request(models_path / TALKER_EMBEDDING_NAME, "qwen3_tts talker embedding", device, properties);
    m_talker_text_embedding = compile_request(models_path / TALKER_TEXT_EMBEDDING_NAME, "qwen3_tts text embedding", device, properties);
    m_talker_text_projection = compile_request(models_path / TALKER_TEXT_PROJECTION_NAME, "qwen3_tts text projection", device, properties);
    m_talker_code_predictor = compile_request(models_path / TALKER_CODE_PREDICTOR_NAME, "qwen3_tts code predictor", device, properties);
    m_talker_code_predictor_embedding = compile_request(models_path / TALKER_CODE_PREDICTOR_EMBEDDING_NAME,
                                                        "qwen3_tts code predictor embedding",
                                                        device,
                                                        properties);
    m_speech_tokenizer_decoder = compile_request(models_path / "speech_tokenizer" / SPEECH_TOKENIZER_DECODER_NAME,
                                                 "qwen3_tts speech tokenizer decoder",
                                                 device,
                                                 properties);

    const auto speaker_encoder_path = models_path / SPEAKER_ENCODER_NAME;
    if (std::filesystem::exists(speaker_encoder_path)) {
        m_speaker_encoder = compile_request(speaker_encoder_path, "qwen3_tts speaker encoder", device, properties);
        m_has_speaker_encoder = true;

        auto mel_model = build_qwen3_mel_preprocess_model(m_speaker_encoder_mel_dim);
        m_qwen3_mel_preprocess = compile_request(mel_model, "qwen3_tts mel preprocess", device, properties);
        m_has_qwen3_mel_preprocess = true;
    }

    const auto speech_tokenizer_encoder_path = models_path / "speech_tokenizer" / SPEECH_TOKENIZER_ENCODER_NAME;
    if (std::filesystem::exists(speech_tokenizer_encoder_path)) {
        m_speech_tokenizer_encoder =
            compile_request(speech_tokenizer_encoder_path, "qwen3_tts speech tokenizer encoder", device, properties);
        m_has_speech_tokenizer_encoder = true;
    }

    debug_print_stage("Qwen3TTSImpl initialized");
    debug_print_model_io("talker", m_talker);
    debug_print_model_io("talker_embedding", m_talker_embedding);
    debug_print_model_io("talker_text_embedding", m_talker_text_embedding);
    debug_print_model_io("talker_text_projection", m_talker_text_projection);
    debug_print_model_io("talker_code_predictor", m_talker_code_predictor);
    debug_print_model_io("talker_code_predictor_embedding", m_talker_code_predictor_embedding);
    debug_print_model_io("speech_tokenizer_decoder", m_speech_tokenizer_decoder);
    if (m_has_speaker_encoder) {
        debug_print_model_io("speaker_encoder", m_speaker_encoder);
    }
    if (m_has_speech_tokenizer_encoder) {
        debug_print_model_io("speech_tokenizer_encoder", m_speech_tokenizer_encoder);
    }
    if (m_has_qwen3_mel_preprocess) {
        debug_print_model_io("mel_preprocess", m_qwen3_mel_preprocess);
    }
}

void Qwen3TTSImpl::init_config(const std::filesystem::path& models_path) {
    const auto config_path = models_path / "config.json";
    std::ifstream config_stream(config_path);
    OPENVINO_ASSERT(config_stream.is_open(), "Failed to open ", config_path);

    json config = json::parse(config_stream);

    m_tts_model_type = to_lower(config.value("tts_model_type", std::string("custom_voice")));

    m_ids.tts_bos_token_id = config.value("tts_bos_token_id", -1);
    m_ids.tts_eos_token_id = config.value("tts_eos_token_id", -1);
    m_ids.tts_pad_token_id = config.value("tts_pad_token_id", -1);

    const json talker = config.at("talker_config");
    m_ids.codec_bos_id = talker.value("codec_bos_id", -1);
    m_ids.codec_pad_id = talker.value("codec_pad_id", -1);
    m_ids.codec_eos_token_id = talker.value("codec_eos_token_id", -1);
    m_ids.codec_think_id = talker.value("codec_think_id", -1);
    m_ids.codec_nothink_id = talker.value("codec_nothink_id", -1);
    m_ids.codec_think_bos_id = talker.value("codec_think_bos_id", -1);
    m_ids.codec_think_eos_id = talker.value("codec_think_eos_id", -1);
    m_ids.num_code_groups = talker.value("num_code_groups", 16);
    m_ids.talker_vocab_size = talker.value("vocab_size", 3072);

    if (config.contains("speaker_encoder_config") && config["speaker_encoder_config"].is_object()) {
        m_speaker_embedding_dim = config["speaker_encoder_config"].value("enc_dim", static_cast<size_t>(talker.value("hidden_size", 1024)));
        m_speaker_encoder_sample_rate =
            config["speaker_encoder_config"].value("sample_rate", static_cast<uint32_t>(24000));
        m_speaker_encoder_mel_dim =
            config["speaker_encoder_config"].value("mel_dim", static_cast<uint32_t>(128));
    } else {
        m_speaker_embedding_dim = talker.value("hidden_size", 1024);
    }

    if (talker.contains("codec_language_id") && talker["codec_language_id"].is_object()) {
        for (auto it = talker["codec_language_id"].begin(); it != talker["codec_language_id"].end(); ++it) {
            m_ids.codec_language_id.emplace(to_lower(it.key()), it.value().get<int64_t>());
        }
    }

    if (talker.contains("spk_id") && talker["spk_id"].is_object()) {
        for (auto it = talker["spk_id"].begin(); it != talker["spk_id"].end(); ++it) {
            m_ids.spk_id.emplace(to_lower(it.key()), it.value().get<int64_t>());
        }
    }

    if (talker.contains("spk_is_dialect") && talker["spk_is_dialect"].is_object()) {
        for (auto it = talker["spk_is_dialect"].begin(); it != talker["spk_is_dialect"].end(); ++it) {
            if (it.value().is_string()) {
                m_ids.spk_is_dialect.emplace(to_lower(it.key()), to_lower(it.value().get<std::string>()));
            }
        }
    }

    const auto speech_tokenizer_config_path = models_path / "speech_tokenizer" / "config.json";
    if (std::filesystem::exists(speech_tokenizer_config_path)) {
        std::ifstream speech_cfg_stream(speech_tokenizer_config_path);
        if (speech_cfg_stream.is_open()) {
            json speech_cfg = json::parse(speech_cfg_stream);
            m_output_sample_rate = speech_cfg.value("output_sample_rate", static_cast<uint32_t>(24000));
            m_decoder_upsample = speech_cfg.value("decode_upsample_rate", static_cast<uint32_t>(1920));
            m_speech_tokenizer_input_sample_rate = speech_cfg.value("input_sample_rate", static_cast<uint32_t>(24000));
            if (speech_cfg.contains("decoder_config") && speech_cfg["decoder_config"].is_object()) {
                m_decoder_num_quantizers = speech_cfg["decoder_config"].value("num_quantizers", static_cast<uint32_t>(16));
            }
        }
    }
}

bool Qwen3TTSImpl::is_base_model() const {
    return m_tts_model_type == "base";
}

ov::Tensor Qwen3TTSImpl::normalize_external_speaker_embedding(const ov::Tensor& speaker_embedding,
                                                              size_t hidden_size) const {
    OPENVINO_ASSERT(!speaker_embedding.get_shape().empty(),
                    "Qwen3 Base expects speaker_embedding tensor with rank 1, 2 or 3");

    const auto in_shape = speaker_embedding.get_shape();
    ov::Tensor out(ov::element::f32, ov::Shape{1, 1, hidden_size});
    float* out_ptr = out.data<float>();

    auto assign_from_flat = [&](const float* src, size_t src_size) {
        OPENVINO_ASSERT(src_size == hidden_size,
                        "Qwen3 Base speaker_embedding size mismatch. Expected ",
                        hidden_size,
                        " values, got ",
                        src_size);
        std::copy_n(src, hidden_size, out_ptr);
    };

    if (speaker_embedding.get_element_type() == ov::element::f32) {
        const float* src = speaker_embedding.data<const float>();
        if (in_shape.size() == 1) {
            assign_from_flat(src, in_shape[0]);
        } else if (in_shape.size() == 2) {
            OPENVINO_ASSERT(in_shape[0] == 1,
                            "Qwen3 Base speaker_embedding rank-2 tensor must have shape [1, D]");
            assign_from_flat(src, in_shape[1]);
        } else if (in_shape.size() == 3) {
            OPENVINO_ASSERT(in_shape[0] == 1,
                            "Qwen3 Base speaker_embedding rank-3 tensor must have batch dimension 1");
            OPENVINO_ASSERT(in_shape[1] == 1,
                            "Qwen3 Base speaker_embedding rank-3 tensor must have shape [1, 1, D]");
            assign_from_flat(src, in_shape[2]);
        } else {
            OPENVINO_THROW("Qwen3 Base speaker_embedding rank must be 1, 2 or 3");
        }
        return out;
    }

    if (speaker_embedding.get_element_type() == ov::element::f16) {
        const ov::float16* src = speaker_embedding.data<const ov::float16>();
        const size_t src_size = speaker_embedding.get_size();
        OPENVINO_ASSERT(src_size == hidden_size,
                        "Qwen3 Base speaker_embedding size mismatch. Expected ",
                        hidden_size,
                        " values, got ",
                        src_size);
        for (size_t i = 0; i < hidden_size; ++i) {
            out_ptr[i] = static_cast<float>(src[i]);
        }
        return out;
    }

    OPENVINO_THROW("Qwen3 Base speaker_embedding must be f32 or f16");
}

std::vector<float> Qwen3TTSImpl::normalize_ref_audio_waveform(const ov::Tensor& ref_audio) const {
    OPENVINO_ASSERT(ref_audio, "qwen_ref_audio must be a non-empty tensor");
    OPENVINO_ASSERT(ref_audio.get_element_type() == ov::element::f32,
                    "qwen_ref_audio must be float32 tensor with shape [T], [1, T], or [1, 1, T]");

    const auto shape = ref_audio.get_shape();
    OPENVINO_ASSERT(!shape.empty(), "qwen_ref_audio tensor rank must be 1, 2, or 3");

    size_t num_samples = 0;
    if (shape.size() == 1) {
        num_samples = shape[0];
    } else if (shape.size() == 2) {
        OPENVINO_ASSERT(shape[0] == 1, "qwen_ref_audio rank-2 tensor must have shape [1, T]");
        num_samples = shape[1];
    } else if (shape.size() == 3) {
        OPENVINO_ASSERT(shape[0] == 1 && shape[1] == 1,
                        "qwen_ref_audio rank-3 tensor must have shape [1, 1, T]");
        num_samples = shape[2];
    } else {
        OPENVINO_THROW("qwen_ref_audio tensor rank must be 1, 2, or 3");
    }

    OPENVINO_ASSERT(num_samples > 0, "qwen_ref_audio must contain at least one sample");

    const float* src = ref_audio.data<const float>();
    std::vector<float> waveform(num_samples, 0.0f);
    std::copy_n(src, num_samples, waveform.data());
    return waveform;
}

ov::Tensor Qwen3TTSImpl::extract_qwen3_speaker_embedding_from_audio(const ov::Tensor& ref_audio) const {
    OPENVINO_ASSERT(m_has_speaker_encoder,
                    "qwen_ref_audio requires 'openvino_speaker_encoder_model.xml' in the model directory");
    OPENVINO_ASSERT(m_has_qwen3_mel_preprocess,
                    "qwen_ref_audio requires internal mel preprocessing model initialization");
    OPENVINO_ASSERT(m_speaker_encoder_sample_rate == 24000,
                    "Qwen3 internal ref-audio extraction assumes 24000 Hz speaker encoder sample rate");

    const std::vector<float> waveform = normalize_ref_audio_waveform(ref_audio);

    ov::Tensor audio_input(ov::element::f32, ov::Shape{1, waveform.size()});
    std::copy(waveform.begin(), waveform.end(), audio_input.data<float>());

    auto mel_req = m_qwen3_mel_preprocess;
    mel_req.set_input_tensor(0, audio_input);
    mel_req.infer();
    ov::Tensor mels = clone_tensor(mel_req.get_output_tensor(0));
    OPENVINO_ASSERT(mels.get_element_type() == ov::element::f32,
                    "Internal mel preprocessing output must be float32");
    OPENVINO_ASSERT(mels.get_shape().size() == 3 && mels.get_shape()[0] == 1 && mels.get_shape()[2] == m_speaker_encoder_mel_dim,
                    "Internal mel preprocessing output must be [1, T, ",
                    m_speaker_encoder_mel_dim,
                    "], got ",
                    shape_to_string(mels.get_shape()));

    auto req = m_speaker_encoder;
    req.set_input_tensor(0, mels);
    req.infer();

    const ov::Tensor raw_out = req.get_output_tensor(0);
    OPENVINO_ASSERT(raw_out.get_element_type() == ov::element::f32,
                    "Speaker encoder output must be float32");
    OPENVINO_ASSERT(raw_out.get_size() == m_speaker_embedding_dim,
                    "Unexpected speaker encoder output size. Got ",
                    raw_out.get_size(),
                    ", expected ",
                    m_speaker_embedding_dim);

    ov::Tensor embedding(ov::element::f32, ov::Shape{m_speaker_embedding_dim});
    std::copy_n(raw_out.data<const float>(), m_speaker_embedding_dim, embedding.data<float>());
    return embedding;
}

ov::Tensor Qwen3TTSImpl::extract_qwen3_ref_code_from_audio(const ov::Tensor& ref_audio) const {
    OPENVINO_ASSERT(m_has_speech_tokenizer_encoder,
                    "qwen_ref_audio ICL mode requires 'speech_tokenizer/openvino_speech_tokenizer_encoder_model.xml'");
    OPENVINO_ASSERT(m_speech_tokenizer_input_sample_rate == 24000,
                    "Qwen3 internal ref-audio extraction assumes 24000 Hz speech tokenizer input sample rate");

    const std::vector<float> waveform = normalize_ref_audio_waveform(ref_audio);

    ov::Tensor audio_input(ov::element::f32, ov::Shape{1, 1, waveform.size()});
    std::copy(waveform.begin(), waveform.end(), audio_input.data<float>());

    auto req = m_speech_tokenizer_encoder;
    req.set_input_tensor(0, audio_input);
    req.infer();

    const ov::Tensor out = req.get_output_tensor(0);
    OPENVINO_ASSERT(out.get_element_type() == ov::element::i64 || out.get_element_type() == ov::element::i32,
                    "Speech tokenizer encoder output must be i64 or i32");

    const ov::Shape shape = out.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 || shape.size() == 3,
                    "Unexpected ref_code output rank. Expected 2 or 3, got ",
                    shape.size());

    auto out_i64 = [&](size_t idx) -> int64_t {
        if (out.get_element_type() == ov::element::i64) {
            return out.data<const int64_t>()[idx];
        }
        return static_cast<int64_t>(out.data<const int32_t>()[idx]);
    };

    if (shape.size() == 2) {
        const size_t d0 = shape[0];
        const size_t d1 = shape[1];
        if (d1 == m_ids.num_code_groups) {
            ov::Tensor ref_code(ov::element::i64, ov::Shape{d0, d1});
            for (size_t i = 0; i < out.get_size(); ++i) {
                ref_code.data<int64_t>()[i] = out_i64(i);
            }
            return ref_code;
        }
        OPENVINO_ASSERT(d0 == m_ids.num_code_groups,
                        "Unexpected ref_code 2D shape ",
                        shape_to_string(shape),
                        ", expected [T,G] or [G,T] with G=",
                        m_ids.num_code_groups);
        ov::Tensor ref_code(ov::element::i64, ov::Shape{d1, d0});
        for (size_t t = 0; t < d1; ++t) {
            for (size_t g = 0; g < d0; ++g) {
                ref_code.data<int64_t>()[t * d0 + g] = out_i64(g * d1 + t);
            }
        }
        return ref_code;
    }

    const size_t b = shape[0];
    const size_t d1 = shape[1];
    const size_t d2 = shape[2];
    OPENVINO_ASSERT(b == 1, "Unexpected ref_code batch dimension. Expected 1, got ", b);

    if (d2 == m_ids.num_code_groups) {
        ov::Tensor ref_code(ov::element::i64, ov::Shape{1, d1, d2});
        for (size_t i = 0; i < out.get_size(); ++i) {
            ref_code.data<int64_t>()[i] = out_i64(i);
        }
        return ref_code;
    }

    OPENVINO_ASSERT(d1 == m_ids.num_code_groups,
                    "Unexpected ref_code 3D shape ",
                    shape_to_string(shape),
                    ", expected [1,T,G] or [1,G,T] with G=",
                    m_ids.num_code_groups);

    ov::Tensor ref_code(ov::element::i64, ov::Shape{1, d2, d1});
    for (size_t t = 0; t < d2; ++t) {
        for (size_t g = 0; g < d1; ++g) {
            ref_code.data<int64_t>()[t * d1 + g] = out_i64(g * d2 + t);
        }
    }
    return ref_code;
}

ov::Tensor Qwen3TTSImpl::infer_embedding(ov::InferRequest& request, int64_t token_id) {
    return infer_embedding_seq(request, std::vector<int64_t>{token_id});
}

ov::Tensor Qwen3TTSImpl::infer_embedding_seq(ov::InferRequest& request, const std::vector<int64_t>& token_ids) {
    ov::Tensor ids(ov::element::i64, ov::Shape{1, token_ids.size()});
    std::copy(token_ids.begin(), token_ids.end(), ids.data<int64_t>());
    debug_print_tensor("infer_embedding_seq", "input_ids", ids);
    request.set_input_tensor(ids);
    request.infer();
    auto out = clone_tensor(request.get_output_tensor(0));
    debug_print_tensor("infer_embedding_seq", "output_0", out);
    return out;
}

ov::Tensor Qwen3TTSImpl::infer_text_projection(const ov::Tensor& hidden_states) {
    debug_print_tensor("infer_text_projection", "hidden_states", hidden_states);
    m_talker_text_projection.set_input_tensor(hidden_states);
    m_talker_text_projection.infer();
    auto out = clone_tensor(m_talker_text_projection.get_output_tensor(0));
    debug_print_tensor("infer_text_projection", "output_0", out);
    return out;
}

ov::Tensor Qwen3TTSImpl::infer_talker(const ov::Tensor& inputs_embeds,
                                      const ov::Tensor& attention_mask,
                                      const ov::Tensor& position_ids,
                                      bool reset_state) {
    if (reset_state) {
        debug_print_stage("infer_talker reset_state=true");
        m_talker.reset_state();
    }

    debug_print_tensor("infer_talker", "inputs_embeds", inputs_embeds);
    debug_print_tensor("infer_talker", "attention_mask", attention_mask);
    debug_print_tensor("infer_talker", "position_ids", position_ids);
    m_talker.set_tensor("inputs_embeds", inputs_embeds);
    m_talker.set_tensor("attention_mask", attention_mask);
    m_talker.set_tensor("position_ids", position_ids);

    if (m_talker.get_compiled_model().inputs().size() > 3) {
        ov::Tensor beam_idx(ov::element::i32, ov::Shape{1});
        beam_idx.data<int32_t>()[0] = 0;
        if (m_talker.get_compiled_model().input(3).get_any_name().find("beam") != std::string::npos) {
            m_talker.set_input_tensor(3, beam_idx);
        }
    }

    m_talker.infer();
    auto logits = clone_tensor(m_talker.get_tensor("logits"));
    debug_print_tensor("infer_talker", "logits", logits);
    return logits;
}

ov::Tensor Qwen3TTSImpl::infer_talker_hidden(const ov::Tensor& inputs_embeds,
                                             const ov::Tensor& attention_mask,
                                             const ov::Tensor& position_ids,
                                             bool reset_state) {
    infer_talker(inputs_embeds, attention_mask, position_ids, reset_state);
    return clone_tensor(m_talker.get_tensor("hidden_states"));
}

ov::Tensor Qwen3TTSImpl::infer_predictor(const ov::Tensor& inputs_embeds,
                                         const ov::Tensor& attention_mask,
                                         const ov::Tensor& position_ids,
                                         int64_t generation_step,
                                         bool reset_state) {
    if (reset_state) {
        debug_print_stage("infer_predictor reset_state=true");
        m_talker_code_predictor.reset_state();
    }

    debug_print_tensor("infer_predictor", "inputs_embeds", inputs_embeds);
    debug_print_tensor("infer_predictor", "attention_mask", attention_mask);
    debug_print_tensor("infer_predictor", "position_ids", position_ids);
    m_talker_code_predictor.set_tensor("inputs_embeds", inputs_embeds);
    m_talker_code_predictor.set_tensor("attention_mask", attention_mask);
    m_talker_code_predictor.set_tensor("position_ids", position_ids);
    ov::Tensor generation_steps(ov::element::i64, ov::Shape{});
    generation_steps.data<int64_t>()[0] = generation_step;
    debug_print_tensor("infer_predictor", "generation_steps", generation_steps);
    m_talker_code_predictor.set_tensor("generation_steps", generation_steps);

    if (m_talker_code_predictor.get_compiled_model().inputs().size() > 4) {
        ov::Tensor beam_idx(ov::element::i32, ov::Shape{1});
        beam_idx.data<int32_t>()[0] = 0;
        if (m_talker_code_predictor.get_compiled_model().input(4).get_any_name().find("beam") != std::string::npos) {
            m_talker_code_predictor.set_input_tensor(4, beam_idx);
        }
    }

    m_talker_code_predictor.infer();
    auto logits = clone_tensor(m_talker_code_predictor.get_tensor("logits"));
    debug_print_tensor("infer_predictor", "logits", logits);
    return logits;
}

ov::Tensor Qwen3TTSImpl::infer_predictor_hidden(const ov::Tensor& inputs_embeds,
                                                const ov::Tensor& attention_mask,
                                                const ov::Tensor& position_ids,
                                                int64_t generation_step,
                                                bool reset_state) {
    infer_predictor(inputs_embeds, attention_mask, position_ids, generation_step, reset_state);
    return clone_tensor(m_talker_code_predictor.get_tensor("mid_residual_hiddens"));
}

ov::Tensor Qwen3TTSImpl::infer_predictor_embedding(int64_t token_id, int64_t generation_step) {
    ov::Tensor ids(ov::element::i64, ov::Shape{1, 1});
    ids.data<int64_t>()[0] = token_id;
    ov::Tensor generation_steps(ov::element::i64, ov::Shape{});
    generation_steps.data<int64_t>()[0] = generation_step;

    debug_print_tensor("infer_predictor_embedding", "input_ids", ids);
    debug_print_tensor("infer_predictor_embedding", "generation_steps", generation_steps);
    m_talker_code_predictor_embedding.set_tensor("input_ids", ids);
    m_talker_code_predictor_embedding.set_tensor("generation_steps", generation_steps);
    m_talker_code_predictor_embedding.infer();
    auto out = clone_tensor(m_talker_code_predictor_embedding.get_output_tensor(0));
    debug_print_tensor("infer_predictor_embedding", "output_0", out);
    return out;
}

ov::Tensor Qwen3TTSImpl::infer_predictor_embedding_seq(const std::vector<int64_t>& token_ids, int64_t generation_step) {
    if (token_ids.empty()) {
        return ov::Tensor(ov::element::f32, ov::Shape{1, 0, m_speaker_embedding_dim});
    }

    ov::Tensor out(ov::element::f32, ov::Shape{1, token_ids.size(), m_speaker_embedding_dim});
    for (size_t i = 0; i < token_ids.size(); ++i) {
        auto emb = infer_predictor_embedding(token_ids[i], generation_step);
        OPENVINO_ASSERT(emb.get_shape().size() == 3 && emb.get_shape()[1] == 1,
                        "Qwen3 predictor embedding must produce shape [1, 1, D]");
        std::copy_n(emb.data<const float>(), m_speaker_embedding_dim, out.data<float>() + i * m_speaker_embedding_dim);
    }
    return out;
}

int64_t Qwen3TTSImpl::sample_token_from_logits(const ov::Tensor& logits,
                                               const SpeechGenerationConfig& generation_config,
                                               const std::vector<int64_t>& generated,
                                             const std::vector<bool>& suppressed,
                                             std::mt19937& rng) const {
    const auto shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "Expected logits shape [B, T, V]");
    const size_t vocab = shape[2];
    const float* ptr = logits.data<const float>();
    const size_t offset = (shape[1] - 1) * vocab;

    std::vector<float> scores(vocab);
    for (size_t i = 0; i < vocab; ++i) {
        float s = ptr[offset + i];
        if (i < suppressed.size() && suppressed[i]) {
            s = -std::numeric_limits<float>::infinity();
        }
        scores[i] = s;
    }

    if (generation_config.repetition_penalty != 1.0f && !generated.empty()) {
        for (int64_t t : generated) {
            if (t >= 0 && static_cast<size_t>(t) < scores.size()) {
                scores[static_cast<size_t>(t)] /= generation_config.repetition_penalty;
            }
        }
    }

    const float temperature = generation_config.temperature;
    if (temperature != 1.0f && temperature > 0.0f) {
        for (auto& s : scores) {
            s /= temperature;
        }
    }

    if (!generation_config.do_sample) {
        return static_cast<int64_t>(std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())));
    }

        // Stochastic sampling: seed parameter only used here (ignored when do_sample=false with argmax)
    std::vector<size_t> idx(vocab);
    std::iota(idx.begin(), idx.end(), static_cast<size_t>(0));
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return scores[a] > scores[b];
    });

    size_t keep_k = idx.size();
    if (generation_config.top_k != std::numeric_limits<size_t>::max()) {
        keep_k = std::min(keep_k, generation_config.top_k);
    }

    std::vector<size_t> kept;
    kept.reserve(keep_k);
    for (size_t i = 0; i < keep_k; ++i) {
        if (std::isfinite(scores[idx[i]])) {
            kept.push_back(idx[i]);
        }
    }

    if (kept.empty()) {
        return static_cast<int64_t>(idx[0]);
    }

    std::vector<float> probs;
    probs.reserve(kept.size());
    float max_score = -std::numeric_limits<float>::infinity();
    for (size_t id : kept) {
        max_score = std::max(max_score, scores[id]);
    }
    for (size_t id : kept) {
        probs.push_back(std::exp(scores[id] - max_score));
    }

    float norm = std::accumulate(probs.begin(), probs.end(), 0.0f);
    if (norm <= 0.0f) {
        return static_cast<int64_t>(kept[0]);
    }

    for (auto& p : probs) {
        p /= norm;
    }

    if (generation_config.top_p < 1.0f) {
        float cumulative = 0.0f;
        size_t last = 0;
        for (; last < probs.size(); ++last) {
            cumulative += probs[last];
            if (cumulative >= generation_config.top_p) {
                break;
            }
        }
        probs.resize(last + 1);
        kept.resize(last + 1);
        float renorm = std::accumulate(probs.begin(), probs.end(), 0.0f);
        if (renorm > 0.0f) {
            for (auto& p : probs) {
                p /= renorm;
            }
        }
    }

    std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
    return static_cast<int64_t>(kept[dist(rng)]);
}

std::vector<int64_t> Qwen3TTSImpl::generate_codec_groups(const ov::Tensor& past_hidden,
                                                         int64_t first_codec_token,
                                                         const SpeechGenerationConfig& generation_config,
                                                         std::mt19937& rng) {
    SpeechGenerationConfig predictor_config = generation_config;
    predictor_config.do_sample = generation_config.subtalker_dosample;
    predictor_config.top_k = generation_config.subtalker_top_k;
    predictor_config.top_p = generation_config.subtalker_top_p;
    predictor_config.temperature = generation_config.subtalker_temperature;
    predictor_config.repetition_penalty = 1.0f;

    std::vector<int64_t> groups;
    groups.reserve(m_ids.num_code_groups);
    groups.push_back(first_codec_token);

    // Minimal residual codec generation that mirrors the helper's code predictor flow.
    auto first_id_hidden = infer_embedding(m_talker_embedding, first_codec_token);
    ov::Shape ph_shape = past_hidden.get_shape();
    ov::Shape hid_shape = first_id_hidden.get_shape();
    OPENVINO_ASSERT(ph_shape.size() == 3 && hid_shape.size() == 3, "Unexpected hidden shape in code predictor prefill");

    ov::Shape prefill_shape{1, ph_shape[1] + hid_shape[1], ph_shape[2]};
    ov::Tensor prefill(ov::element::f32, prefill_shape);
    const size_t ph_bytes = past_hidden.get_byte_size();
    const size_t hid_bytes = first_id_hidden.get_byte_size();
    std::memcpy(prefill.data(), past_hidden.data(), ph_bytes);
    std::memcpy(static_cast<uint8_t*>(prefill.data()) + ph_bytes, first_id_hidden.data(), hid_bytes);

    auto prefill_mask = make_attention_mask(prefill_shape[1]);
    auto prefill_pos = make_predictor_position_ids(0, prefill_shape[1]);
    auto logits = infer_predictor(prefill, prefill_mask, prefill_pos, 0, true);

    std::vector<int64_t> generated;
    generated.reserve(m_ids.num_code_groups - 1);

    std::vector<bool> predictor_suppressed(2048, false);
    int64_t next = sample_token_from_logits(logits, predictor_config, generated, predictor_suppressed, rng);
    generated.push_back(next);

    size_t absolute_pos = prefill_shape[1];
    for (size_t g = 1; g < m_ids.num_code_groups - 1; ++g) {
        auto emb = infer_predictor_embedding(next, static_cast<int64_t>(g - 1));
        auto attn = make_attention_mask(absolute_pos + 1);
        auto pos = make_predictor_position_ids(absolute_pos, 1);
        auto lg = infer_predictor(emb, attn, pos, static_cast<int64_t>(g), false);
        next = sample_token_from_logits(lg, predictor_config, generated, predictor_suppressed, rng);
        generated.push_back(next);
        ++absolute_pos;
    }

    groups.insert(groups.end(), generated.begin(), generated.end());
    while (groups.size() < m_ids.num_code_groups) {
        groups.push_back(m_ids.codec_pad_id);
    }
    return groups;
}

Text2SpeechDecodedResults Qwen3TTSImpl::decode_from_prefill(const ov::Tensor& talker_prefill,
                                                            const ov::Tensor& tts_pad,
                                                            const SpeechGenerationConfig& generation_config,
                                                            const std::vector<bool>& suppress_tokens) {
    Text2SpeechDecodedResults result;
    result.output_sample_rate = m_output_sample_rate;

    auto prefill_mask = make_attention_mask(talker_prefill.get_shape()[1]);
    auto prefill_pos = make_position_ids_prefill(talker_prefill.get_shape()[1]);
    debug_print_tensor("generate", "talker_prefill", talker_prefill);
    debug_print_tensor("generate", "prefill_mask", prefill_mask);
    debug_print_tensor("generate", "prefill_pos", prefill_pos);

    auto logits = infer_talker(talker_prefill, prefill_mask, prefill_pos, true);
    auto hidden_states = clone_tensor(m_talker.get_tensor("hidden_states"));

    std::vector<int64_t> generated_main;
    std::vector<int64_t> all_codes;
    size_t max_steps = generation_config.get_max_new_tokens();
    if (max_steps == SIZE_MAX) {
        max_steps = 2048;
    }
    generated_main.reserve(std::min(max_steps, size_t(8192)));
    all_codes.reserve(std::min(max_steps, size_t(8192)) * m_ids.num_code_groups);

    std::mt19937 rng(generation_config.seed != 0 ? generation_config.seed : std::random_device{}());
    size_t absolute_pos = talker_prefill.get_shape()[1];

    for (size_t step = 0; step < max_steps; ++step) {
        int64_t token = sample_token_from_logits(logits, generation_config, generated_main, suppress_tokens, rng);
        if (qwen_debug_enabled() && (step < 5 || step % 32 == 0)) {
            std::cout << "[QWEN_DEBUG] decode step=" << step << " token=" << token << std::endl;
        }
        if (token == m_ids.codec_eos_token_id) {
            debug_print_stage("main decoder produced EOS");
            break;
        }
        generated_main.push_back(token);

        ov::Tensor past_hidden(ov::element::f32, ov::Shape{1, 1, hidden_states.get_shape()[2]});
        const float* hs_ptr = hidden_states.data<const float>();
        const size_t hs_len = hidden_states.get_shape()[1];
        std::copy_n(hs_ptr + (hs_len - 1) * hidden_states.get_shape()[2], hidden_states.get_shape()[2], past_hidden.data<float>());

        auto groups = generate_codec_groups(past_hidden, token, generation_config, rng);
        if (qwen_debug_enabled() && step < 3) {
            std::cout << "[QWEN_DEBUG] step=" << step << " main_token=" << token
                      << " codec_group0=" << groups.front()
                      << " codec_group_last=" << groups.back() << std::endl;
            if (step == 0) {
                std::cout << "[QWEN_DEBUG] step=0 codec_row=";
                for (size_t i = 0; i < groups.size(); ++i) {
                    if (i > 0) {
                        std::cout << ",";
                    }
                    std::cout << groups[i];
                }
                std::cout << std::endl;
            }
        }
        all_codes.insert(all_codes.end(), groups.begin(), groups.end());

        auto token_embed = infer_embedding(m_talker_embedding, groups[0]);
        float* token_embed_ptr = token_embed.data<float>();
        for (size_t g = 1; g < groups.size(); ++g) {
            auto pred_emb = infer_predictor_embedding(groups[g], static_cast<int64_t>(g - 1));
            const float* pred_ptr = pred_emb.data<const float>();
            for (size_t h_i = 0; h_i < token_embed.get_shape()[2]; ++h_i) {
                token_embed_ptr[h_i] += pred_ptr[h_i];
            }
        }
        if (generation_config.non_streaming_mode) {
            const float* tts_pad_ptr = tts_pad.data<const float>();
            for (size_t h_i = 0; h_i < token_embed.get_shape()[2]; ++h_i) {
                token_embed_ptr[h_i] += tts_pad_ptr[h_i];
            }
        }

        auto attn = make_attention_mask(absolute_pos + 1);
        auto pos = make_position_ids_decode(absolute_pos);
        logits = infer_talker(token_embed, attn, pos, false);
        hidden_states = clone_tensor(m_talker.get_tensor("hidden_states"));
        ++absolute_pos;
    }

    auto waveform = decode_speech_tokenizer(all_codes);
    if (qwen_debug_enabled()) {
        std::cout << "[QWEN_DEBUG] decoded code groups=" << all_codes.size()
                  << " waveform_samples=" << waveform.size() << std::endl;
    }
    ov::Tensor wav_tensor(ov::element::f32, ov::Shape{waveform.size()});
    if (!waveform.empty()) {
        std::copy(waveform.begin(), waveform.end(), wav_tensor.data<float>());
    }

    result.perf_metrics.num_generated_samples += wav_tensor.get_size();
    result.speeches.push_back(std::move(wav_tensor));
    result.perf_metrics.evaluate_statistics();
    m_perf_metrics = result.perf_metrics;
    return result;
}

ov::Tensor Qwen3TTSImpl::make_attention_mask(size_t length) {
    ov::Tensor mask(ov::element::i64, ov::Shape{1, length});
    std::fill_n(mask.data<int64_t>(), length, static_cast<int64_t>(1));
    return mask;
}

ov::Tensor Qwen3TTSImpl::make_position_ids_prefill(size_t length) {
    ov::Tensor pos(ov::element::i64, ov::Shape{3, 1, length});
    int64_t* data = pos.data<int64_t>();
    for (size_t i = 0; i < length; ++i) {
        data[i] = static_cast<int64_t>(i);
        data[length + i] = static_cast<int64_t>(i);
        data[2 * length + i] = static_cast<int64_t>(i);
    }
    return pos;
}

ov::Tensor Qwen3TTSImpl::make_position_ids_decode(size_t absolute_position) {
    ov::Tensor pos(ov::element::i64, ov::Shape{3, 1, 1});
    int64_t* data = pos.data<int64_t>();
    data[0] = static_cast<int64_t>(absolute_position);
    data[1] = static_cast<int64_t>(absolute_position);
    data[2] = static_cast<int64_t>(absolute_position);
    return pos;
}

ov::Tensor Qwen3TTSImpl::make_predictor_position_ids(size_t start_position, size_t length) {
    // Code predictor uses 2D position IDs: [batch_size=1, seq_len=length]
    ov::Tensor pos(ov::element::i64, ov::Shape{1, length});
    int64_t* data = pos.data<int64_t>();
    for (size_t i = 0; i < length; ++i) {
        data[i] = static_cast<int64_t>(start_position + i);
    }
    return pos;
}
std::string Qwen3TTSImpl::normalize_text_language(const std::string& language) const {
    if (language.empty()) {
        return "auto";
    }
    return to_lower(language);
}

std::string Qwen3TTSImpl::normalize_speaker(const std::string& speaker) const {
    return to_lower(speaker);
}

std::vector<float> Qwen3TTSImpl::decode_speech_tokenizer(const std::vector<int64_t>& codes) {
    if (codes.empty()) {
        return {};
    }

    OPENVINO_ASSERT(codes.size() % m_decoder_num_quantizers == 0,
                    "Qwen codec tensor is malformed: expected [T, num_quantizers] flattening");

    const size_t num_frames = codes.size() / m_decoder_num_quantizers;
    std::vector<float> chunks_audio;
    chunks_audio.reserve(num_frames * m_decoder_upsample);

    size_t start = 0;
    while (start < num_frames) {
        const size_t end = std::min(start + static_cast<size_t>(DECODER_CHUNK_SIZE), num_frames);
        const size_t ctx = (start > static_cast<size_t>(DECODER_LEFT_CONTEXT)) ? static_cast<size_t>(DECODER_LEFT_CONTEXT) : start;
        const size_t chunk_start = start - ctx;
        const size_t chunk_len = end - chunk_start;

        std::vector<int64_t> chunk_codes(chunk_len * m_decoder_num_quantizers);
        const size_t src_offset = chunk_start * m_decoder_num_quantizers;
        std::copy_n(codes.begin() + static_cast<std::ptrdiff_t>(src_offset),
                    chunk_codes.size(),
                    chunk_codes.begin());

        if (chunk_len < static_cast<size_t>(DECODER_TRACE_LEN)) {
            chunk_codes.resize(static_cast<size_t>(DECODER_TRACE_LEN) * m_decoder_num_quantizers, 0);
        }

        ov::Tensor audio_codes(ov::element::i64,
                               ov::Shape{1, chunk_codes.size() / m_decoder_num_quantizers, m_decoder_num_quantizers});
        std::copy(chunk_codes.begin(), chunk_codes.end(), audio_codes.data<int64_t>());

        m_speech_tokenizer_decoder.set_tensor("audio_codes", audio_codes);
        m_speech_tokenizer_decoder.infer();

        const auto out = m_speech_tokenizer_decoder.get_tensor("audio_values");
        OPENVINO_ASSERT(out.get_element_type() == ov::element::f32,
                        "Speech tokenizer decoder output is expected to be f32");

        const float* out_ptr = out.data<const float>();
        const size_t out_size = out.get_size();

        const size_t total_valid = chunk_len * m_decoder_upsample > static_cast<size_t>(DECODER_OFFSET)
                                       ? chunk_len * m_decoder_upsample - static_cast<size_t>(DECODER_OFFSET)
                                       : 0;
        const size_t context_samples = ctx * m_decoder_upsample;

        if (total_valid > context_samples && total_valid <= out_size) {
            chunks_audio.insert(chunks_audio.end(), out_ptr + static_cast<std::ptrdiff_t>(context_samples), out_ptr + static_cast<std::ptrdiff_t>(total_valid));
        }

        start = end;
    }

    return chunks_audio;
}

Text2SpeechDecodedResults Qwen3TTSImpl::generate(const std::vector<std::string>& texts,
                                                  const ov::Tensor& speaker_embedding,
                                                  const SpeechGenerationConfig& generation_config) {
    debug_print_stage("generate() entered");
    if (qwen_debug_enabled()) {
        std::cout << "[QWEN_DEBUG] request config: language='" << generation_config.language
                  << "' speaker='" << generation_config.speaker
                  << "' instruct_len=" << generation_config.instruct.size()
                  << " non_streaming_mode=" << (generation_config.non_streaming_mode ? "true" : "false")
                  << " max_new_tokens=" << generation_config.get_max_new_tokens()
                  << std::endl;
    }

    Text2SpeechDecodedResults result;
    result.output_sample_rate = m_output_sample_rate;

    auto generation_start = std::chrono::steady_clock::now();

    std::vector<bool> suppress_tokens(m_ids.talker_vocab_size, false);
    const size_t suppress_begin = m_ids.talker_vocab_size > 1024 ? (m_ids.talker_vocab_size - 1024) : 0;
    for (size_t i = suppress_begin; i < m_ids.talker_vocab_size; ++i) {
        if (static_cast<int64_t>(i) != m_ids.codec_eos_token_id) {
            suppress_tokens[i] = true;
        }
    }

    const std::string language = normalize_text_language(generation_config.language);
    const std::string speaker = normalize_speaker(generation_config.speaker);
    const bool base_model = is_base_model();

    const bool has_qwen_voice_clone_props =
        base_model && (generation_config.qwen_x_vector_only_mode ||
                       !generation_config.qwen_ref_text.empty() ||
                       static_cast<bool>(generation_config.qwen_ref_audio) ||
                       static_cast<bool>(generation_config.qwen_ref_code));

    if (has_qwen_voice_clone_props) {
        ov::Tensor resolved_speaker_embedding = speaker_embedding;
        if (!resolved_speaker_embedding && generation_config.qwen_ref_audio) {
            OPENVINO_ASSERT(m_speaker_encoder_sample_rate == 24000,
                            "qwen_ref_audio assumes 24000 Hz waveform input. OV GenAI does not resample");
            resolved_speaker_embedding = extract_qwen3_speaker_embedding_from_audio(generation_config.qwen_ref_audio);
        }

        OPENVINO_ASSERT(resolved_speaker_embedding,
                        "Qwen3 Base voice cloning via generate(...) requires either speaker_embedding or qwen_ref_audio");

        ov::Tensor resolved_ref_code = generation_config.qwen_ref_code;
        if (!generation_config.qwen_x_vector_only_mode && !resolved_ref_code && generation_config.qwen_ref_audio) {
            OPENVINO_ASSERT(m_speech_tokenizer_input_sample_rate == 24000,
                            "qwen_ref_audio assumes 24000 Hz waveform input. OV GenAI does not resample");
            resolved_ref_code = extract_qwen3_ref_code_from_audio(generation_config.qwen_ref_audio);
        }

        Qwen3VoiceClonePrompt prompt;
        prompt.ref_spk_embedding = resolved_speaker_embedding;
        prompt.ref_text = generation_config.qwen_ref_text;
        prompt.ref_code = resolved_ref_code;
        prompt.x_vector_only_mode = generation_config.qwen_x_vector_only_mode;

        for (const auto& text : texts) {
            auto decoded = generate_voice_clone(text, prompt, generation_config);
            result.speeches.insert(result.speeches.end(), decoded.speeches.begin(), decoded.speeches.end());
            result.perf_metrics.num_generated_samples += decoded.perf_metrics.num_generated_samples;
        }

        auto generation_end = std::chrono::steady_clock::now();
        result.perf_metrics.raw_metrics.generate_durations.emplace_back(
            PerfMetrics::get_microsec(generation_end - generation_start));
        result.perf_metrics.evaluate_statistics();
        m_perf_metrics = result.perf_metrics;
        return result;
    }

    for (const auto& text : texts) {
        if (qwen_debug_enabled()) {
            std::cout << "[QWEN_DEBUG] text prompt len=" << text.size() << std::endl;
        }
        auto tokenization_start = std::chrono::steady_clock::now();
        const std::string assistant_text = "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
        auto input = m_tokenizer.encode(assistant_text);
        auto input_ids_tensor = input.input_ids;
        auto tokenization_end = std::chrono::steady_clock::now();
        result.perf_metrics.raw_metrics.tokenization_durations.emplace_back(
            PerfMetrics::get_microsec(tokenization_end - tokenization_start));

        OPENVINO_ASSERT(input_ids_tensor.get_element_type() == ov::element::i64,
                        "Qwen3-TTS tokenizer must produce i64 input_ids tensor");

        const auto input_shape = input_ids_tensor.get_shape();
        OPENVINO_ASSERT(input_shape.size() == 2 && input_shape[0] == 1, "Expected input_ids shape [1, T]");
        debug_print_tensor("generate", "tokenizer.input_ids", input_ids_tensor);
        const size_t input_len = input_shape[1];
        const int64_t* input_ids = input_ids_tensor.data<const int64_t>();

        std::vector<int64_t> speaker_and_codec_prefill;
        int64_t language_id = -1;
        auto lang_it = m_ids.codec_language_id.find(language);
        if (language != "auto" && lang_it != m_ids.codec_language_id.end()) {
            language_id = lang_it->second;
        }

        if (language_id == -1 && !speaker.empty()) {
            auto spk_dialect = m_ids.spk_is_dialect.find(speaker);
            if (spk_dialect != m_ids.spk_is_dialect.end()) {
                auto dialect_it = m_ids.codec_language_id.find(spk_dialect->second);
                if (dialect_it != m_ids.codec_language_id.end()) {
                    language_id = dialect_it->second;
                }
            }
        }

        if (language_id == -1) {
            speaker_and_codec_prefill = {m_ids.codec_nothink_id, m_ids.codec_think_bos_id, m_ids.codec_think_eos_id};
        } else {
            speaker_and_codec_prefill = {m_ids.codec_think_id, m_ids.codec_think_bos_id, language_id, m_ids.codec_think_eos_id};
        }
        auto codec_prefill_embed0 = infer_embedding_seq(m_talker_embedding, speaker_and_codec_prefill);
        auto codec_prefill_embed1 = infer_embedding_seq(m_talker_embedding, {m_ids.codec_pad_id, m_ids.codec_bos_id});

        int64_t speaker_token_id = -1;
        if (!base_model && !speaker.empty()) {
            auto spk_it = m_ids.spk_id.find(speaker);
            if (spk_it != m_ids.spk_id.end()) {
                speaker_token_id = spk_it->second;
            }
        }
        if (qwen_debug_enabled()) {
            std::cout << "[QWEN_DEBUG] language_id=" << language_id
                      << " speaker_token_id=" << speaker_token_id
                      << " codec_prefill_tokens=" << speaker_and_codec_prefill.size()
                      << " tts_model_type='" << m_tts_model_type << "'"
                      << std::endl;
        }

        ov::Tensor speaker_embed;
        bool has_speaker_embed = false;
        if (base_model) {
            OPENVINO_ASSERT(speaker_embedding,
                            "Qwen3 Base requires speaker_embedding input. "
                            "Provide a speaker embedding tensor shaped [D], [1,D], or [1,1,D].");
            speaker_embed = normalize_external_speaker_embedding(speaker_embedding, codec_prefill_embed0.get_shape()[2]);
            has_speaker_embed = true;
        } else if (speaker_token_id != -1) {
            speaker_embed = infer_embedding(m_talker_embedding, speaker_token_id);
            has_speaker_embed = true;
        }

        auto special_text_embed = infer_embedding_seq(m_talker_text_embedding,
                                                      {m_ids.tts_bos_token_id, m_ids.tts_eos_token_id, m_ids.tts_pad_token_id});
        auto special_projected = infer_text_projection(special_text_embed);

        const size_t hidden = special_projected.get_shape()[2];
        ov::Tensor tts_bos(ov::element::f32, ov::Shape{1, 1, hidden});
        ov::Tensor tts_eos(ov::element::f32, ov::Shape{1, 1, hidden});
        ov::Tensor tts_pad(ov::element::f32, ov::Shape{1, 1, hidden});
        const float* sp = special_projected.data<const float>();
        std::copy_n(sp + 0 * hidden, hidden, tts_bos.data<float>());
        std::copy_n(sp + 1 * hidden, hidden, tts_eos.data<float>());
        std::copy_n(sp + 2 * hidden, hidden, tts_pad.data<float>());

        auto concat_embed = [&](const std::vector<ov::Tensor>& tensors) {
            size_t total_len = 0;
            size_t h = tensors.front().get_shape()[2];
            for (const auto& t : tensors) {
                total_len += t.get_shape()[1];
            }
            ov::Tensor out(ov::element::f32, ov::Shape{1, total_len, h});
            float* out_ptr = out.data<float>();
            size_t cursor = 0;
            for (const auto& t : tensors) {
                const size_t sz = t.get_size();
                std::copy_n(t.data<const float>(), sz, out_ptr + cursor);
                cursor += sz;
            }
            return out;
        };

        ov::Tensor codec_input_embedding;
        if (has_speaker_embed) {
            codec_input_embedding = concat_embed({codec_prefill_embed0, speaker_embed, codec_prefill_embed1});
        } else {
            codec_input_embedding = concat_embed({codec_prefill_embed0, codec_prefill_embed1});
        }

        const std::vector<int64_t> role_tokens(input_ids, input_ids + std::min<size_t>(3, input_len));
        auto role_embed = infer_text_projection(infer_embedding_seq(m_talker_text_embedding, role_tokens));

        const size_t codec_len = codec_input_embedding.get_shape()[1];
        ov::Tensor tts_pad_expand(ov::element::f32, ov::Shape{1, codec_len - 2, hidden});
        for (size_t i = 0; i < codec_len - 2; ++i) {
            std::copy_n(tts_pad.data<const float>(), hidden, tts_pad_expand.data<float>() + i * hidden);
        }
        ov::Tensor left_codec_sum(ov::element::f32, ov::Shape{1, codec_len - 1, hidden});
        const float* codec_ptr = codec_input_embedding.data<const float>();
        float* left_ptr = left_codec_sum.data<float>();
        for (size_t i = 0; i < codec_len - 2; ++i) {
            const float* a = tts_pad_expand.data<const float>() + i * hidden;
            const float* b = codec_ptr + i * hidden;
            for (size_t j = 0; j < hidden; ++j) {
                left_ptr[i * hidden + j] = a[j] + b[j];
            }
        }
        {
            const float* a = tts_bos.data<const float>();
            const float* b = codec_ptr + (codec_len - 2) * hidden;
            for (size_t j = 0; j < hidden; ++j) {
                left_ptr[(codec_len - 2) * hidden + j] = a[j] + b[j];
            }
        }

        auto text_embed_full = infer_text_projection(infer_embedding_seq(
            m_talker_text_embedding,
            std::vector<int64_t>(input_ids + std::min<size_t>(3, input_len), input_ids + input_len)));

        // Remove trailing control tokens in the same spirit as helper slicing [:, 3:-5].
        size_t text_tokens_len = text_embed_full.get_shape()[1];
        size_t trimmed_len = text_tokens_len > 5 ? text_tokens_len - 5 : text_tokens_len;
        ov::Tensor text_embed_trimmed(ov::element::f32, ov::Shape{1, trimmed_len, hidden});
        if (trimmed_len > 0) {
            std::copy_n(text_embed_full.data<const float>(), text_embed_trimmed.get_size(), text_embed_trimmed.data<float>());
        }

        ov::Tensor talker_prefill;
        if (generation_config.non_streaming_mode) {
            debug_print_stage("prompt assembly mode: non_streaming");
            ov::Tensor text_with_eos(ov::element::f32, ov::Shape{1, trimmed_len + 1, hidden});
            if (trimmed_len > 0) {
                std::copy_n(text_embed_trimmed.data<const float>(), text_embed_trimmed.get_size(), text_with_eos.data<float>());
            }
            std::copy_n(tts_eos.data<const float>(), hidden, text_with_eos.data<float>() + trimmed_len * hidden);

            auto codec_pad_for_text = infer_embedding_seq(m_talker_embedding, std::vector<int64_t>(trimmed_len + 1, m_ids.codec_pad_id));
            ov::Tensor text_side(ov::element::f32, ov::Shape{1, trimmed_len + 1, hidden});
            for (size_t i = 0; i < (trimmed_len + 1) * hidden; ++i) {
                text_side.data<float>()[i] = text_with_eos.data<const float>()[i] + codec_pad_for_text.data<const float>()[i];
            }

            auto codec_bos_embed = infer_embedding(m_talker_embedding, m_ids.codec_bos_id);
            ov::Tensor pad_plus_codec_bos(ov::element::f32, ov::Shape{1, 1, hidden});
            for (size_t j = 0; j < hidden; ++j) {
                pad_plus_codec_bos.data<float>()[j] = tts_pad.data<const float>()[j] + codec_bos_embed.data<const float>()[j];
            }

            talker_prefill = concat_embed({role_embed, left_codec_sum, text_side, pad_plus_codec_bos});
        } else {
            debug_print_stage("prompt assembly mode: streaming");
            // Streaming-style prompt assembly follows helper logic:
            // align text and codec prefill by min(text_len, codec_len) and carry no explicit
            // text-side + codec-pad block in prefill.
            const size_t codec_prefill_len = left_codec_sum.get_shape()[1];
            const size_t overlap_len = std::min(trimmed_len, codec_prefill_len);

            ov::Tensor stream_mix(ov::element::f32, ov::Shape{1, overlap_len, hidden});
            for (size_t i = 0; i < overlap_len * hidden; ++i) {
                stream_mix.data<float>()[i] = text_embed_trimmed.data<const float>()[i] + left_codec_sum.data<const float>()[i];
            }

            if (trimmed_len > codec_prefill_len) {
                // Text dominates: only overlap section is part of prompt prefill.
                talker_prefill = concat_embed({role_embed, stream_mix});
            } else if (trimmed_len < codec_prefill_len) {
                // Codec dominates: pad text with tts_pad and continue with remaining codec prefix.
                const size_t pad_count = codec_prefill_len - trimmed_len;
                ov::Tensor text_pad(ov::element::f32, ov::Shape{1, pad_count, hidden});
                for (size_t i = 0; i < pad_count; ++i) {
                    std::copy_n(tts_pad.data<const float>(), hidden, text_pad.data<float>() + i * hidden);
                }

                ov::Tensor mixed_with_pad(ov::element::f32, ov::Shape{1, codec_prefill_len, hidden});
                if (overlap_len > 0) {
                    std::copy_n(stream_mix.data<const float>(), stream_mix.get_size(), mixed_with_pad.data<float>());
                }
                for (size_t i = 0; i < pad_count * hidden; ++i) {
                    mixed_with_pad.data<float>()[overlap_len * hidden + i] =
                        text_pad.data<const float>()[i] + left_codec_sum.data<const float>()[overlap_len * hidden + i];
                }

                talker_prefill = concat_embed({role_embed, mixed_with_pad});
            } else {
                talker_prefill = concat_embed({role_embed, stream_mix});
            }
        }

        if (!generation_config.instruct.empty()) {
            debug_print_stage("adding instruct embedding");
            const std::string instruct_text = "<|im_start|>user\n" + generation_config.instruct + "<|im_end|>\n";
            auto instruct_ids = m_tokenizer.encode(instruct_text).input_ids;
            const int64_t* instr = instruct_ids.data<const int64_t>();
            const size_t instr_len = instruct_ids.get_shape()[1];
            auto instruct_embed = infer_text_projection(infer_embedding_seq(m_talker_text_embedding, std::vector<int64_t>(instr, instr + instr_len)));
            talker_prefill = concat_embed({instruct_embed, talker_prefill});
        }

        auto decoded = decode_from_prefill(talker_prefill, tts_pad, generation_config, suppress_tokens);
        result.speeches.insert(result.speeches.end(), decoded.speeches.begin(), decoded.speeches.end());
        result.perf_metrics.num_generated_samples += decoded.perf_metrics.num_generated_samples;
    }

    auto generation_end = std::chrono::steady_clock::now();
    result.perf_metrics.raw_metrics.generate_durations.emplace_back(
        PerfMetrics::get_microsec(generation_end - generation_start));
    result.perf_metrics.evaluate_statistics();
    m_perf_metrics = result.perf_metrics;
    return result;
}

Text2SpeechDecodedResults Qwen3TTSImpl::generate_voice_clone(const std::string& text,
                                                             const Qwen3VoiceClonePrompt& prompt,
                                                             const SpeechGenerationConfig& generation_config) {
    OPENVINO_ASSERT(is_base_model(), "Qwen3 voice-clone prompt generation is supported only for Base models");
    if (prompt.x_vector_only_mode) {
        SpeechGenerationConfig cfg = generation_config;
        cfg.qwen_ref_text.clear();
        cfg.qwen_ref_code = ov::Tensor();
        cfg.qwen_x_vector_only_mode = false;
        return generate(std::vector<std::string>{text}, prompt.ref_spk_embedding, cfg);
    }

    OPENVINO_ASSERT(!prompt.ref_text.empty(), "Qwen3 voice-clone ICL mode requires a non-empty reference transcript");
    OPENVINO_ASSERT(!prompt.ref_code.get_shape().empty(), "Qwen3 voice-clone ICL mode requires reference codec ids");
    OPENVINO_ASSERT(prompt.ref_code.get_shape().size() == 2 || prompt.ref_code.get_shape().size() == 3,
                    "Qwen3 voice-clone prompt expects ref_code shaped [T, G] or [1, T, G]");

    const std::string language = normalize_text_language(generation_config.language);
    const std::string speaker = normalize_speaker(generation_config.speaker);

    std::vector<bool> suppress_tokens(m_ids.talker_vocab_size, false);
    const size_t suppress_begin = m_ids.talker_vocab_size > 1024 ? (m_ids.talker_vocab_size - 1024) : 0;
    for (size_t i = suppress_begin; i < m_ids.talker_vocab_size; ++i) {
        if (static_cast<int64_t>(i) != m_ids.codec_eos_token_id) {
            suppress_tokens[i] = true;
        }
    }

    auto result = Text2SpeechDecodedResults{};
    result.output_sample_rate = m_output_sample_rate;

    const std::string assistant_text = "<|im_start|>assistant\n" + prompt.ref_text + text + "<|im_end|>\n<|im_start|>assistant\n";
    auto input = m_tokenizer.encode(assistant_text);
    auto input_ids_tensor = input.input_ids;
    OPENVINO_ASSERT(input_ids_tensor.get_element_type() == ov::element::i64,
                    "Qwen3-TTS tokenizer must produce i64 input_ids tensor");
    OPENVINO_ASSERT(input_ids_tensor.get_shape().size() == 2 && input_ids_tensor.get_shape()[0] == 1,
                    "Expected input_ids shape [1, T]");

    const size_t input_len = input_ids_tensor.get_shape()[1];
    const int64_t* input_ids = input_ids_tensor.data<const int64_t>();

    int64_t language_id = -1;
    auto lang_it = m_ids.codec_language_id.find(language);
    if (language != "auto" && lang_it != m_ids.codec_language_id.end()) {
        language_id = lang_it->second;
    }
    if (language_id == -1 && !speaker.empty()) {
        auto spk_dialect = m_ids.spk_is_dialect.find(speaker);
        if (spk_dialect != m_ids.spk_is_dialect.end()) {
            auto dialect_it = m_ids.codec_language_id.find(spk_dialect->second);
            if (dialect_it != m_ids.codec_language_id.end()) {
                language_id = dialect_it->second;
            }
        }
    }

    std::vector<int64_t> speaker_and_codec_prefill;
    if (language_id == -1) {
        speaker_and_codec_prefill = {m_ids.codec_nothink_id, m_ids.codec_think_bos_id, m_ids.codec_think_eos_id};
    } else {
        speaker_and_codec_prefill = {m_ids.codec_think_id, m_ids.codec_think_bos_id, language_id, m_ids.codec_think_eos_id};
    }

    auto codec_prefill_embed0 = infer_embedding_seq(m_talker_embedding, speaker_and_codec_prefill);
    // Split codec_pad and codec_bos: they occupy different structural roles in the prefill.
    auto codec_pad_embed1 = infer_embedding_seq(m_talker_embedding, {m_ids.codec_pad_id});
    auto codec_bos_embed = infer_embedding_seq(m_talker_embedding, {m_ids.codec_bos_id});
    auto special_text_embed = infer_embedding_seq(m_talker_text_embedding,
                                                  {m_ids.tts_bos_token_id, m_ids.tts_eos_token_id, m_ids.tts_pad_token_id});
    auto special_projected = infer_text_projection(special_text_embed);

    const size_t hidden = special_projected.get_shape()[2];
    ov::Tensor tts_bos(ov::element::f32, ov::Shape{1, 1, hidden});
    ov::Tensor tts_eos(ov::element::f32, ov::Shape{1, 1, hidden});
    ov::Tensor tts_pad(ov::element::f32, ov::Shape{1, 1, hidden});
    const float* sp = special_projected.data<const float>();
    std::copy_n(sp + 0 * hidden, hidden, tts_bos.data<float>());
    std::copy_n(sp + 1 * hidden, hidden, tts_eos.data<float>());
    std::copy_n(sp + 2 * hidden, hidden, tts_pad.data<float>());

    auto concat_embed = [&](const std::vector<ov::Tensor>& tensors) {
        size_t total_len = 0;
        size_t h = tensors.front().get_shape()[2];
        for (const auto& t : tensors) {
            total_len += t.get_shape()[1];
        }
        ov::Tensor out(ov::element::f32, ov::Shape{1, total_len, h});
        float* out_ptr = out.data<float>();
        size_t cursor = 0;
        for (const auto& t : tensors) {
            std::copy_n(t.data<const float>(), t.get_size(), out_ptr + cursor);
            cursor += t.get_size();
        }
        return out;
    };

    ov::Tensor speaker_embed = normalize_external_speaker_embedding(prompt.ref_spk_embedding, codec_prefill_embed0.get_shape()[2]);
    // codec_input_no_bos = [codec_prefill0, speaker, codec_pad] — codec_bos is placed separately at start of codec_side
    ov::Tensor codec_input_no_bos = concat_embed({codec_prefill_embed0, speaker_embed, codec_pad_embed1});

    ov::Shape ref_code_shape = prompt.ref_code.get_shape();
    const bool ref_code_rank3 = ref_code_shape.size() == 3;
    const size_t ref_len = ref_code_rank3 ? ref_code_shape[1] : ref_code_shape[0];
    const size_t num_code_groups = ref_code_rank3 ? ref_code_shape[2] : ref_code_shape[1];
    OPENVINO_ASSERT(num_code_groups == m_ids.num_code_groups,
                    "Qwen3 voice-clone ref_code group count mismatch. Expected ",
                    m_ids.num_code_groups,
                    ", got ",
                    num_code_groups);

    ov::Tensor ref_code_embed(ov::element::f32, ov::Shape{1, ref_len, hidden});
    std::fill_n(ref_code_embed.data<float>(), ref_code_embed.get_size(), 0.0f);
    for (size_t g = 0; g < num_code_groups; ++g) {
        std::vector<int64_t> group_ids(ref_len);
        const int64_t* codes = prompt.ref_code.data<const int64_t>();
        for (size_t t = 0; t < ref_len; ++t) {
            group_ids[t] = ref_code_rank3 ? codes[t * num_code_groups + g] : codes[t * num_code_groups + g];
        }

        ov::Tensor group_embed = (g == 0) ? infer_embedding_seq(m_talker_embedding, group_ids)
                                          : infer_predictor_embedding_seq(group_ids, static_cast<int64_t>(g - 1));
        const float* group_ptr = group_embed.data<const float>();
        float* ref_ptr = ref_code_embed.data<float>();
        for (size_t i = 0; i < ref_len * hidden; ++i) {
            ref_ptr[i] += group_ptr[i];
        }
    }

    // Role embed: first 3 tokens = <|im_start|>assistant\n
    const std::vector<int64_t> role_tokens(input_ids, input_ids + std::min<size_t>(3, input_len));
    auto role_embed = infer_text_projection(infer_embedding_seq(m_talker_text_embedding, role_tokens));

    // text_embed: ref_text + target_text, strip role (first 3) and trailing structural tokens (last 5:
    // <|im_end|>\n<|im_start|>assistant\n), then append tts_eos.
    auto ref_text_embed = infer_text_projection(infer_embedding_seq(
        m_talker_text_embedding,
        std::vector<int64_t>(input_ids + std::min<size_t>(3, input_len), input_ids + input_len)));
    const size_t ref_text_tokens_len = ref_text_embed.get_shape()[1];
    const size_t ref_trimmed_len = ref_text_tokens_len > 5 ? ref_text_tokens_len - 5 : ref_text_tokens_len;
    ov::Tensor ref_text_trimmed(ov::element::f32, ov::Shape{1, ref_trimmed_len, hidden});
    if (ref_trimmed_len > 0) {
        std::copy_n(ref_text_embed.data<const float>(), ref_text_trimmed.get_size(), ref_text_trimmed.data<float>());
    }
    // Append tts_eos to text (matches Python: text_embed = torch.cat([text_embed, tts_eos_embed], dim=1))
    ov::Tensor text_embed_with_eos = concat_embed({ref_text_trimmed, tts_eos});
    const size_t text_eos_len = text_embed_with_eos.get_shape()[1];

    std::cerr << "[DEBUG] ICL path: text_eos_len=" << text_eos_len << " ref_len=" << ref_len << " hidden=" << hidden << std::endl;

    // Part 1: codec_input_part — blend codec_input_no_bos with [tts_pad × (len-2), tts_bos]
    // Matches Python's: [(tts_pad*(N-2) + tts_bos)] + codec_input_embedding[:-1]
    const size_t ci_len = codec_input_no_bos.get_shape()[1];
    ov::Tensor codec_input_part(ov::element::f32, ov::Shape{1, ci_len, hidden});
    for (size_t i = 0; i < ci_len; ++i) {
        const float* src = codec_input_no_bos.data<const float>() + i * hidden;
        float* dst = codec_input_part.data<float>() + i * hidden;
        const float* blend = (i == ci_len - 1) ? tts_bos.data<const float>() : tts_pad.data<const float>();
        for (size_t h = 0; h < hidden; ++h) {
            dst[h] = src[h] + blend[h];
        }
    }

    // Part 2: text_side — (text_embed + tts_eos) blended with codec_pad per-position
    auto codec_pad_for_text = infer_embedding_seq(m_talker_embedding, std::vector<int64_t>(text_eos_len, m_ids.codec_pad_id));
    ov::Tensor text_side(ov::element::f32, ov::Shape{1, text_eos_len, hidden});
    for (size_t i = 0; i < text_side.get_size(); ++i) {
        text_side.data<float>()[i] = text_embed_with_eos.data<const float>()[i] + codec_pad_for_text.data<const float>()[i];
    }

    // Part 3: codec_side — (codec_bos + ref_codes) blended with tts_pad per-position
    ov::Tensor codec_bos_plus_ref = concat_embed({codec_bos_embed, ref_code_embed});
    const size_t codec_side_len = codec_bos_plus_ref.get_shape()[1];
    ov::Tensor codec_side(ov::element::f32, ov::Shape{1, codec_side_len, hidden});
    for (size_t i = 0; i < codec_side.get_size(); ++i) {
        codec_side.data<float>()[i] = codec_bos_plus_ref.data<const float>()[i] + tts_pad.data<const float>()[i % hidden];
    }

    // Full prefill: role | codec_input_part | text_side | codec_side
    // Matches Python non-streaming ICL: role + codec_prefill/speaker + (text+tts_eos+cpad) + (cbos+ref_codes+tpad)
    ov::Tensor talker_prefill = concat_embed({role_embed, codec_input_part, text_side, codec_side});

    if (!generation_config.instruct.empty()) {
        const std::string instruct_text = "<|im_start|>user\n" + generation_config.instruct + "<|im_end|>\n";
        auto instruct_ids = m_tokenizer.encode(instruct_text).input_ids;
        const int64_t* instr = instruct_ids.data<const int64_t>();
        const size_t instr_len = instruct_ids.get_shape()[1];
        auto instruct_embed = infer_text_projection(infer_embedding_seq(m_talker_text_embedding, std::vector<int64_t>(instr, instr + instr_len)));
        talker_prefill = concat_embed({instruct_embed, talker_prefill});
    }

    // Force non-streaming decode behavior: always add tts_pad to each generated token embed.
    // This matches Python's trailing_text_hidden = tts_pad_embed for ICL mode when text_len <= codec_len.
    SpeechGenerationConfig vc_config = generation_config;
    vc_config.non_streaming_mode = true;
    return decode_from_prefill(talker_prefill, tts_pad, vc_config, suppress_tokens);
}

ov::Shape Qwen3TTSImpl::get_speaker_embedding_shape() const {
    if (is_base_model()) {
        return ov::Shape{1, 1, m_speaker_embedding_dim};
    }
    // Qwen3 CustomVoice does not consume external speaker embeddings in the GenAI API.
    return ov::Shape{1};
}

}  // namespace genai
}  // namespace ov
