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
#include <iomanip>
#include <sstream>

#include <nlohmann/json.hpp>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset15.hpp>

#include "json_utils.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "sampling/logit_processor.hpp"
#include "sampling/logit_transformers.hpp"
#include "utils.hpp"

namespace {

constexpr const char* TALKER_LANGUAGE_NAME = "openvino_talker_language_model.xml";
constexpr const char* TALKER_EMBEDDING_NAME = "openvino_talker_embedding_model.xml";
constexpr const char* TALKER_TEXT_EMBEDDING_NAME = "openvino_talker_text_embedding_model.xml";
constexpr const char* TALKER_TEXT_PROJECTION_NAME = "openvino_talker_text_projection_model.xml";
constexpr const char* TALKER_CODE_PREDICTOR_NAME = "openvino_talker_code_predictor_model.xml";
constexpr const char* TALKER_CODE_PREDICTOR_EMBEDDING_NAME = "openvino_talker_code_predictor_embedding_model.xml";
constexpr const char* SPEECH_TOKENIZER_DECODER_NAME = "openvino_speech_tokenizer_decoder_model.xml";
constexpr const char* SPEAKER_ENCODER_NAME = "openvino_speaker_encoder_model.xml";
constexpr const char* SPEECH_TOKENIZER_ENCODER_NAME = "openvino_speech_tokenizer_encoder_model.xml";

constexpr int64_t DECODER_TRACE_LEN = 256;
constexpr int64_t DECODER_CHUNK_SIZE = 231;
constexpr int64_t DECODER_LEFT_CONTEXT = 25;
constexpr int64_t DECODER_OFFSET = 555;
constexpr float PI_F = 3.14159265358979323846f;

// Component roles used for per-component device routing. Each Qwen3-TTS
// submodel is a separate IR, so (mirroring the VLM pipeline) the pipeline can
// place each one on a different device while still accepting a single primary
// `device` argument.
namespace roles {
constexpr const char* TALKER = "talker";
constexpr const char* TALKER_EMBEDDING = "talker_embedding";
constexpr const char* TALKER_TEXT_EMBEDDING = "talker_text_embedding";
constexpr const char* TALKER_TEXT_PROJECTION = "talker_text_projection";
constexpr const char* CODE_PREDICTOR = "code_predictor";
constexpr const char* CODE_PREDICTOR_EMBEDDING = "code_predictor_embedding";
constexpr const char* SPEECH_TOKENIZER_DECODER = "speech_tokenizer_decoder";
constexpr const char* SPEECH_TOKENIZER_ENCODER = "speech_tokenizer_encoder";
constexpr const char* SPEAKER_ENCODER = "speaker_encoder";
constexpr const char* MEL_PREPROCESS = "mel_preprocess";
}  // namespace roles

// Default device policy for a component. The Qwen3-TTS pipeline can place each submodel on a different device,
// and so here we define the default device for each role.
std::string default_device_for_role(const std::string& base_device, bool is_npu, const std::string& role) {
    // Keep predictor embedding on CPU (short, data-dependent lookup path).
    if (role == roles::CODE_PREDICTOR_EMBEDDING) {
        return "CPU";
    }

    // There seems to be accuracy issues running code predictor on GPU, so force it to CPU for now.
    if( base_device.find("GPU") != std::string::npos && role == roles::CODE_PREDICTOR) {
        return "CPU";
    }

    if (!is_npu) {
        return base_device;
    }

    // NPU can run talker, code predictor, and speech-tokenizer decoder.
    if (role == roles::TALKER || role == roles::CODE_PREDICTOR || role == roles::SPEECH_TOKENIZER_DECODER) {
        return base_device;
    }

    return "CPU";
}

// Resolve the (device, properties) pair for a component following the VLM
// `ov::device::properties` convention: when the caller supplies a per-device
// property map each component picks up the sub-map for the device it actually
// lands on; otherwise the same top-level properties are reused for every
// component.
std::pair<std::string, ov::AnyMap> resolve_component_target(const std::string& base_device,
                                                            bool is_npu,
                                                            const ov::AnyMap& device_properties,
                                                            const ov::AnyMap& base_properties,
                                                            const std::string& role) {
    const std::string device = default_device_for_role(base_device, is_npu, role);
    if (device_properties.empty()) {
        return {device, base_properties};
    }
    auto it = device_properties.find(device);
    ov::AnyMap properties = (it != device_properties.end()) ? it->second.as<ov::AnyMap>() : ov::AnyMap{};
    return {device, properties};
}

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

// Gate for per-component inference timing. Enable with OV_GENAI_QWEN_TTS_PERF=1.
bool qwen_perf_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("OV_GENAI_QWEN_TTS_PERF");
        if (!env) return false;
        const std::string v = to_lower(std::string(env));
        return v == "1" || v == "true" || v == "yes" || v == "on";
    }();
    return enabled;
}

// Conditionally time a callable and accumulate into the caller's perf maps.
template<typename F>
inline void run_and_time(F&& f,
                         const std::string& name,
                         std::unordered_map<std::string, double>& ms,
                         std::unordered_map<std::string, int64_t>& calls) {
    if (!qwen_perf_enabled()) {
        f();
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    f();
    ms[name] += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    ++calls[name];
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

    m_is_npu = ov::genai::utils::is_npu_requested(device, properties);

    // Split off the optional per-device property map (VLM `ov::device::properties`
    // convention). The remaining top-level properties act as the default for
    // every component when no per-device map is supplied.
    ov::AnyMap base_properties = properties;
    ov::AnyMap device_properties =
        ov::genai::utils::pop_or_default<ov::AnyMap>(base_properties, ov::device::properties.name(), ov::AnyMap{});

    auto compile_for = [&](const auto& model_source, const char* model_name, const std::string& role) {
        auto [target_device, target_properties] =
            resolve_component_target(device, m_is_npu, device_properties, base_properties, role);
        return compile_request(model_source, model_name, target_device, target_properties);
    };

    auto device_of = [&](ov::InferRequest& request, const std::string& fallback_device) {
        try {
            auto devices = request.get_compiled_model().get_property(ov::execution_devices);
            if (!devices.empty()) {
                return devices.front();
            }
        } catch (...) {
        }
        return fallback_device;
    };

    // The talker stays on the requested device. On NPU it is routed through the
    // NPUW LLM compile path (stateful decoder -> static); every other device
    // keeps the generic path.
    {
        auto [talker_device, talker_properties] =
            resolve_component_target(device, m_is_npu, device_properties, base_properties, roles::TALKER);
        if (m_is_npu) {
            // Apply TTS-specific static-shape sizing unless the caller already set it.
            constexpr int64_t DEFAULT_MAX_PROMPT_LEN = 1024;
            constexpr int64_t DEFAULT_MIN_RESPONSE_LEN = 128;
            ov::AnyMap npu_talker_properties = talker_properties;
            npu_talker_properties.emplace("MAX_PROMPT_LEN", DEFAULT_MAX_PROMPT_LEN);
            npu_talker_properties.emplace("MIN_RESPONSE_LEN", DEFAULT_MIN_RESPONSE_LEN);

            auto talker_model = ov::genai::utils::singleton_core().read_model(models_path / TALKER_LANGUAGE_NAME);
            const auto kv_pos = ov::genai::utils::get_kv_axes_pos(talker_model);
            ov::CompiledModel compiled;
            ov::genai::utils::KVDesc kv_desc;
            std::tie(compiled, kv_desc) =
                ov::genai::utils::compile_decoder_for_npu(talker_model, npu_talker_properties, kv_pos);
            ov::genai::utils::print_compiled_model_properties(compiled, "qwen3_tts talker (NPU)");
            m_talker = compiled.create_infer_request();
            m_perf_device["talker_prefill"] = device_of(m_talker, "NPU");
            m_perf_device["talker_generate"] = m_perf_device["talker_prefill"];
        } else {
            m_talker =
                compile_request(models_path / TALKER_LANGUAGE_NAME, "qwen3_tts talker", talker_device, talker_properties);
            m_perf_device["talker_prefill"] = device_of(m_talker, talker_device);
            m_perf_device["talker_generate"] = m_perf_device["talker_prefill"];
        }
    }

    m_talker_embedding =
        compile_for(models_path / TALKER_EMBEDDING_NAME, "qwen3_tts talker embedding", roles::TALKER_EMBEDDING);
    m_perf_device["talker_embedding"] = device_of(m_talker_embedding, default_device_for_role(device, m_is_npu, roles::TALKER_EMBEDDING));
    m_talker_text_embedding =
        compile_for(models_path / TALKER_TEXT_EMBEDDING_NAME, "qwen3_tts text embedding", roles::TALKER_TEXT_EMBEDDING);
    m_perf_device["talker_text_embedding"] = device_of(m_talker_text_embedding, default_device_for_role(device, m_is_npu, roles::TALKER_TEXT_EMBEDDING));
    m_talker_text_projection = compile_for(models_path / TALKER_TEXT_PROJECTION_NAME,
                                           "qwen3_tts text projection",
                                           roles::TALKER_TEXT_PROJECTION);
    m_perf_device["text_projection"] = device_of(m_talker_text_projection, default_device_for_role(device, m_is_npu, roles::TALKER_TEXT_PROJECTION));
    {
        auto [predictor_device, predictor_properties] =
            resolve_component_target(device, m_is_npu, device_properties, base_properties, roles::CODE_PREDICTOR);

        // Detect the static all-heads code predictor variant: it exposes explicit
        // KV-cache inputs (past_key_values.0.key) and a stacked all-heads `logits`
        // output, instead of the stateful NPUW decoder contract. The static model
        // is a plain fixed-shape graph, so it must be compiled directly (NOT through
        // the NPUW LLM path) on every device. This is the only supported predictor
        // layout; see convert.py / qwen_3_tts_helper.py (static_code_predictor).
        auto predictor_model = ov::genai::utils::singleton_core().read_model(models_path / TALKER_CODE_PREDICTOR_NAME);
        bool predictor_static = false;
        for (const auto& in : predictor_model->inputs()) {
            if (in.get_any_name().rfind("past_key_values", 0) == 0) {
                predictor_static = true;
                break;
            }
        }

        if (predictor_static) {
            std::cout << "Using static all-heads code predictor variant (explicit KV cache)" << std::endl;
            reshape_predictor_to_static(predictor_model);
            // Plain static compile (works identically on NPU/GPU/CPU). Host manages
            // the explicit KV cache; see infer_predictor / generate_codec_groups.
            init_static_predictor_meta(predictor_model);
            m_talker_code_predictor = compile_request(predictor_model,
                                                      "qwen3_tts code predictor (static all-heads)",
                                                      predictor_device,
                                                      predictor_properties);
            m_perf_device["code_predictor"] = device_of(m_talker_code_predictor, predictor_device);

            // Cache KV tensors infer request.
            // These are set once and reused via memset/memcpy without further set_tensor calls.
            for (size_t i = 0; i < m_pred_num_layers; ++i) {
                m_pred_past_k.emplace_back(m_talker_code_predictor.get_tensor("past_key_values." + std::to_string(i) + ".key"));
                m_pred_past_v.emplace_back(m_talker_code_predictor.get_tensor("past_key_values." + std::to_string(i) + ".value"));
            }
            m_pred_attn = m_talker_code_predictor.get_tensor("attention_mask");
            m_pred_pos = m_talker_code_predictor.get_tensor("position_ids");
        } else {
            // The legacy stateful/NPUW predictor path has been removed; only the
            // static all-heads layout is supported. Fail fast here with an
            // actionable message instead of deep inside infer_predictor.
            OPENVINO_THROW("Unsupported code predictor model: expected the static all-heads variant with "
                           "explicit KV-cache inputs (past_key_values.*). Re-export with the static code "
                           "predictor (convert.py / qwen_3_tts_helper.py: static_code_predictor=True).");
        }
    }
    m_talker_code_predictor_embedding = compile_for(models_path / TALKER_CODE_PREDICTOR_EMBEDDING_NAME,
                                                    "qwen3_tts code predictor embedding",
                                                    roles::CODE_PREDICTOR_EMBEDDING);
    m_perf_device["code_predictor_embedding"] =
        device_of(m_talker_code_predictor_embedding, default_device_for_role(device, m_is_npu, roles::CODE_PREDICTOR_EMBEDDING));

    // Bind fixed-shape input tensors for the code-predictor embedding once. Unlike
    // the code predictor (reshaped to static), this embedding model may be dynamic,
    // so a get_tensor() default can have zero-sized dims and set_shape() on it does
    // not reliably re-bind the new shape into the request. We instead own buffers
    // with the exact shapes the original per-call path used (input_ids=[1,1],
    // generation_steps=scalar) and set_tensor them a single time here. Subsequent
    // inferences only rewrite the data in place -- no per-call set_tensor.
    m_pred_emb_ids = ov::Tensor(ov::element::i64, ov::Shape{1, 1});
    m_pred_emb_step = ov::Tensor(ov::element::i64, ov::Shape{});
    m_talker_code_predictor_embedding.set_tensor("input_ids", m_pred_emb_ids);
    m_talker_code_predictor_embedding.set_tensor("generation_steps", m_pred_emb_step);
    {
        // The speech-tokenizer decoder is always fed fixed-size [1, DECODER_TRACE_LEN,
        // num_quantizers] code chunks (decode_speech_tokenizer zero-pads shorter
        // chunks up to DECODER_TRACE_LEN), so on accelerators that require static
        // shapes (e.g. NPU) we reshape it before compiling. CPU/GPU keep the model
        // as exported.
        auto [decoder_device, decoder_properties] =
            resolve_component_target(device, m_is_npu, device_properties, base_properties, roles::SPEECH_TOKENIZER_DECODER);
        const auto decoder_path = models_path / "speech_tokenizer" / SPEECH_TOKENIZER_DECODER_NAME;
        if (decoder_device.find("NPU") != std::string::npos) {
            auto decoder_model = ov::genai::utils::singleton_core().read_model(decoder_path);
            const ov::PartialShape static_codes{1,
                                                static_cast<int64_t>(DECODER_TRACE_LEN),
                                                static_cast<int64_t>(m_decoder_num_quantizers)};
            std::cout << "Reshaping speech-tokenizer decoder to static shape " << static_codes << std::endl;
            decoder_model->reshape({{"audio_codes", static_codes}});
            m_speech_tokenizer_decoder = compile_request(decoder_model,
                                                         "qwen3_tts speech tokenizer decoder",
                                                         decoder_device,
                                                         decoder_properties);
            m_perf_device["speech_tokenizer_decoder"] = device_of(m_speech_tokenizer_decoder, decoder_device);
            // warm-up.
            m_speech_tokenizer_decoder.infer();
        } else {
            m_speech_tokenizer_decoder = compile_request(decoder_path,
                                                         "qwen3_tts speech tokenizer decoder",
                                                         decoder_device,
                                                         decoder_properties);
            m_perf_device["speech_tokenizer_decoder"] = device_of(m_speech_tokenizer_decoder, decoder_device);
        }
    }

    const auto speaker_encoder_path = models_path / SPEAKER_ENCODER_NAME;
    if (std::filesystem::exists(speaker_encoder_path)) {
        m_speaker_encoder = compile_for(speaker_encoder_path, "qwen3_tts speaker encoder", roles::SPEAKER_ENCODER);
        m_perf_device["speaker_encoder"] = device_of(m_speaker_encoder, default_device_for_role(device, m_is_npu, roles::SPEAKER_ENCODER));
        m_has_speaker_encoder = true;

        auto mel_model = build_qwen3_mel_preprocess_model(m_speaker_encoder_mel_dim);
        m_qwen3_mel_preprocess = compile_for(mel_model, "qwen3_tts mel preprocess", roles::MEL_PREPROCESS);
        m_perf_device["mel_preprocess"] = device_of(m_qwen3_mel_preprocess, default_device_for_role(device, m_is_npu, roles::MEL_PREPROCESS));
        m_has_qwen3_mel_preprocess = true;
    }

    const auto speech_tokenizer_encoder_path = models_path / "speech_tokenizer" / SPEECH_TOKENIZER_ENCODER_NAME;
    if (std::filesystem::exists(speech_tokenizer_encoder_path)) {
        m_speech_tokenizer_encoder = compile_for(speech_tokenizer_encoder_path,
                                                 "qwen3_tts speech tokenizer encoder",
                                                 roles::SPEECH_TOKENIZER_ENCODER);
        m_perf_device["speech_tokenizer_encoder"] =
            device_of(m_speech_tokenizer_encoder, default_device_for_role(device, m_is_npu, roles::SPEECH_TOKENIZER_ENCODER));
        m_has_speech_tokenizer_encoder = true;
    }

}

void Qwen3TTSImpl::init_config(const std::filesystem::path& models_path) {
    const auto config_path = models_path / "config.json";
    std::ifstream config_stream(config_path);
    OPENVINO_ASSERT(config_stream.is_open(), "Failed to open ", config_path);

    nlohmann::json config = nlohmann::json::parse(config_stream);

    m_tts_model_type = to_lower(config.value("tts_model_type", std::string("custom_voice")));

    m_ids.tts_bos_token_id = config.value("tts_bos_token_id", -1);
    m_ids.tts_eos_token_id = config.value("tts_eos_token_id", -1);
    m_ids.tts_pad_token_id = config.value("tts_pad_token_id", -1);

    const nlohmann::json talker = config.at("talker_config");
    m_ids.codec_bos_id = talker.value("codec_bos_id", -1);
    m_ids.codec_pad_id = talker.value("codec_pad_id", -1);
    m_ids.codec_eos_token_id = talker.value("codec_eos_token_id", -1);
    m_ids.codec_think_id = talker.value("codec_think_id", -1);
    m_ids.codec_nothink_id = talker.value("codec_nothink_id", -1);
    m_ids.codec_think_bos_id = talker.value("codec_think_bos_id", -1);
    m_ids.codec_think_eos_id = talker.value("codec_think_eos_id", -1);
    m_ids.num_code_groups = talker.value("num_code_groups", 16);
    m_ids.talker_vocab_size = talker.value("vocab_size", 3072);
    m_talker_hidden_size = talker.value("hidden_size", 1024);

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
            nlohmann::json speech_cfg = nlohmann::json::parse(speech_cfg_stream);
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
     run_and_time([&]{ mel_req.infer(); }, "mel_preprocess", m_perf_ms, m_perf_calls);
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
     run_and_time([&]{ req.infer(); }, "speaker_encoder", m_perf_ms, m_perf_calls);

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
     run_and_time([&]{ req.infer(); }, "speech_tokenizer_encoder", m_perf_ms, m_perf_calls);

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
    request.set_input_tensor(ids);
    request.infer();
    auto out = clone_tensor(request.get_output_tensor(0));
    return out;
}

ov::Tensor Qwen3TTSImpl::infer_text_projection(const ov::Tensor& hidden_states) {
    m_talker_text_projection.set_input_tensor(hidden_states);
    run_and_time([&]{ m_talker_text_projection.infer(); }, "text_projection", m_perf_ms, m_perf_calls);
    auto out = clone_tensor(m_talker_text_projection.get_output_tensor(0));
    return out;
}

ov::Tensor Qwen3TTSImpl::infer_talker(const ov::Tensor& inputs_embeds,
                                      const ov::Tensor& attention_mask,
                                      const ov::Tensor& position_ids,
                                      bool reset_state) {
    if (reset_state) {
        m_talker.reset_state();
    }

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

    if (reset_state) {
        const auto shape = inputs_embeds.get_shape();
        if (shape.size() >= 2) {
            m_talker_prefill_tokens += static_cast<int64_t>(shape[1]);
        }
        run_and_time([&]{ m_talker.infer(); }, "talker_prefill", m_perf_ms, m_perf_calls);
    } else {
        run_and_time([&]{ m_talker.infer(); }, "talker_generate", m_perf_ms, m_perf_calls);
    }
    // Return the request's own logits output tensor (no clone). Valid until the
    // next m_talker.infer(). The decode loop samples from it before issuing the
    // next talker infer, so this is safe. Do NOT reorder a talker infer ahead of
    // consuming a previously returned logits tensor.
    return m_talker.get_tensor("logits");
}

ov::Tensor Qwen3TTSImpl::infer_talker_hidden(const ov::Tensor& inputs_embeds,
                                             const ov::Tensor& attention_mask,
                                             const ov::Tensor& position_ids,
                                             bool reset_state) {
    infer_talker(inputs_embeds, attention_mask, position_ids, reset_state);
    return clone_tensor(m_talker.get_tensor("hidden_states"));
}

// Read fixed dimensions from the static all-heads code predictor IR and
// allocate the host-side KV cache buffers. Called once after read_model when the
// static variant is detected.
//
// Static contract (see Notebooks/export_code_predictor_optionA.py --all-heads and
// Notebooks/validate_optionA_full.py):
//   inputs : inputs_embeds [1,1,H], attention_mask [1,kv_len], position_ids [1,1],
//            past_key_values.{i}.{key,value} [1,n_kv,past_len,head_dim]  (i in 0..L-1)
//   outputs: logits [num_heads,1,1,V] (all code-group heads stacked),
//            present.{i}.{key,value}   [1,n_kv,kv_len,head_dim] (or [1,n_kv,1,head_dim]
//                                       when the exporter slices to just the new token)
// where kv_len = past_len + 1. The graph hard-wires cache_position = [past_len],
// so the freshly produced token always lands at the last present slot.

// Newer exporters emit the all-heads code predictor with fully dynamic shapes
// (e.g. inputs_embeds [?,?,?], past_key_values [?,n_kv,?,head_dim], logits
// [num_heads,?,?,V]). The runtime host-KV logic (init_static_predictor_meta /
// infer_predictor) requires a single fixed shape, so specialize the IR here. The
// export still pins the structurally-fixed dims (n_kv, head_dim on the KV inputs;
// num_heads, V on logits); the KV window is derived as past_len = num_heads and
// kv_len = past_len + 1, matching the static export contract above. No-op when the
// IR is already static.
void Qwen3TTSImpl::reshape_predictor_to_static(const std::shared_ptr<ov::Model>& model) {
    bool is_dynamic = false;
    for (const auto& in : model->inputs()) {
        if (in.get_partial_shape().is_dynamic()) {
            is_dynamic = true;
            break;
        }
    }
    if (!is_dynamic) {
        return;  // already static (reshaped at export time)
    }

    // Fixed dims the dynamic export still pins.
    size_t n_kv = 0, head_dim = 0, num_heads = 0;
    for (const auto& in : model->inputs()) {
        const auto name = in.get_any_name();
        if (name.rfind("past_key_values.", 0) == 0 && name.find(".key") != std::string::npos) {
            const auto& ps = in.get_partial_shape();  // [?, n_kv, ?, head_dim]
            OPENVINO_ASSERT(ps.rank().is_static() && ps.size() == 4 && ps[1].is_static() && ps[3].is_static(),
                            "Unexpected code predictor past_key_values shape: ", ps);
            n_kv = static_cast<size_t>(ps[1].get_length());
            head_dim = static_cast<size_t>(ps[3].get_length());
            break;
        }
    }
    for (const auto& out : model->outputs()) {
        if (out.get_any_name() == "logits") {
            const auto& ps = out.get_partial_shape();  // [num_heads, ?, ?, V]
            OPENVINO_ASSERT(ps.rank().is_static() && ps.size() == 4 && ps[0].is_static(),
                            "Unexpected code predictor logits shape: ", ps);
            num_heads = static_cast<size_t>(ps[0].get_length());
            break;
        }
    }
    OPENVINO_ASSERT(n_kv > 0 && head_dim > 0 && num_heads > 0,
                    "Could not derive static code predictor dims (n_kv=",
                    n_kv, " head_dim=", head_dim, " num_heads=", num_heads, ")");

    const int64_t past_len = static_cast<int64_t>(num_heads);  // KV window = one slot per prior head
    const int64_t kv_len = past_len + 1;
    const int64_t hidden = static_cast<int64_t>(m_talker_hidden_size);

    std::map<std::string, ov::PartialShape> shapes;
    for (const auto& in : model->inputs()) {
        const auto name = in.get_any_name();
        if (name == "inputs_embeds") {
            shapes[name] = ov::PartialShape{1, 1, hidden};
        } else if (name == "attention_mask") {
            shapes[name] = ov::PartialShape{1, kv_len};
        } else if (name == "position_ids") {
            shapes[name] = ov::PartialShape{1, 1};
        } else if (name.rfind("past_key_values.", 0) == 0) {
            shapes[name] = ov::PartialShape{1, static_cast<int64_t>(n_kv), past_len, static_cast<int64_t>(head_dim)};
        }
    }

    std::cout << "Reshaping dynamic code predictor to static: inputs_embeds=[1,1," << hidden
              << "] attention_mask=[1," << kv_len << "] past_key_values=[1," << n_kv << "," << past_len << ","
              << head_dim << "]" << std::endl;
    model->reshape(shapes);
}

void Qwen3TTSImpl::init_static_predictor_meta(const std::shared_ptr<ov::Model>& model) {
    m_pred_num_layers = 0;
    for (const auto& in : model->inputs()) {
        const auto name = in.get_any_name();
        if (name == "attention_mask") {
            m_pred_kv_len = static_cast<size_t>(in.get_shape()[1]);
        } else if (name.rfind("past_key_values.", 0) == 0 && name.find(".key") != std::string::npos) {
            const auto& s = in.get_shape();  // [1, n_kv, past_len, head_dim]
            m_pred_n_kv = static_cast<size_t>(s[1]);
            m_pred_past_len = static_cast<size_t>(s[2]);
            m_pred_head_dim = static_cast<size_t>(s[3]);
            ++m_pred_num_layers;
        }
    }
    for (const auto& out : model->outputs()) {
        if (out.get_any_name() == "logits") {
            m_pred_num_heads = static_cast<size_t>(out.get_shape()[0]);  // [num_heads,1,1,V]
            m_pred_vocab = static_cast<size_t>(out.get_shape()[3]);
        }
    }
    OPENVINO_ASSERT(m_pred_num_layers > 0 && m_pred_kv_len == m_pred_past_len + 1,
                    "Unexpected static code predictor shapes: layers=",
                    m_pred_num_layers, " kv_len=", m_pred_kv_len, " past_len=", m_pred_past_len);

    // Allocate persistent host KV buffers; reset_predictor_state() zero-fills them.
    // Tensors will be cached from InferRequest after compilation.
    m_pred_past_k.clear();
    m_pred_past_v.clear();
    m_pred_past_k.reserve(m_pred_num_layers);
    m_pred_past_v.reserve(m_pred_num_layers);
    reset_predictor_state();

    std::cout << "static code predictor: layers=" << m_pred_num_layers
              << " n_kv=" << m_pred_n_kv << " head_dim=" << m_pred_head_dim
              << " past_len=" << m_pred_past_len << " kv_len=" << m_pred_kv_len
              << " heads=" << m_pred_num_heads << " vocab=" << m_pred_vocab << std::endl;
}

// Zero the host KV buffers and reset the running absolute-position counter. Call
// at the start of each per-frame code-group prediction (the predictor cache is
// local to a single talker step).
void Qwen3TTSImpl::reset_predictor_state() {
    for (size_t i = 0; i < m_pred_past_k.size(); ++i) {
        std::memset(m_pred_past_k[i].data(), 0, m_pred_past_k[i].get_byte_size());
        std::memset(m_pred_past_v[i].data(), 0, m_pred_past_v[i].get_byte_size());
    }
    m_pred_position = 0;
}

ov::Tensor Qwen3TTSImpl::infer_predictor(const ov::Tensor& inputs_embeds, bool reset_state) {
    if (reset_state) {
        reset_predictor_state();
    }

    const size_t p = m_pred_position;  // absolute position of this single token
    OPENVINO_ASSERT(p < m_pred_past_len + 1,
                    "Static code predictor exceeded its KV window (position ",
                    p, " >= kv_len ", m_pred_kv_len, ")");

    m_talker_code_predictor.set_tensor("inputs_embeds", inputs_embeds);

    // Update attention_mask data in-place (tensor already set on infer request).
    // Valid kv slots are the p real past tokens (slots 0..p-1) plus the current
    // token, which the graph appends at slot index past_len.
    int64_t* attn_ptr = m_pred_attn.data<int64_t>();
    std::fill_n(attn_ptr, m_pred_kv_len, static_cast<int64_t>(0));
    for (size_t j = 0; j < p; ++j) {
        attn_ptr[j] = 1;
    }
    attn_ptr[m_pred_past_len] = 1;  // current token slot (= kv_len - 1)

    // Update position_ids data in-place (tensor already set on infer request).
    m_pred_pos.data<int64_t>()[0] = static_cast<int64_t>(p);

    // Past KV tensors already set on infer request; no need to call set_tensor.
    run_and_time([&]{ m_talker_code_predictor.infer(); }, "code_predictor", m_perf_ms, m_perf_calls);

    // Copy the freshly produced token into host past slot p so the next step
    // sees positions 0..p left-aligned. The new token is always the LAST slot of
    // each `present` output. The exporter may emit `present` either full
    // ([1,n_kv,kv_len,head_dim]) or sliced to just the new token
    // ([1,n_kv,1,head_dim]); deriving the slot count from the tensor shape
    // supports both -- we only ever read that last slot regardless.
    if (p < m_pred_past_len) {
        const size_t row = m_pred_head_dim;                 // floats per (head, slot)
        const size_t slot_stride = m_pred_past_len * row;   // host past: [.,.,past_len,head_dim]
        for (size_t i = 0; i < m_pred_num_layers; ++i) {
            const auto& pk = m_talker_code_predictor.get_tensor("present." + std::to_string(i) + ".key");
            const auto& pv = m_talker_code_predictor.get_tensor("present." + std::to_string(i) + ".value");
            const size_t present_slots = pk.get_shape()[2];          // kv_len (full) or 1 (sliced)
            const size_t present_slot_stride = present_slots * row;
            const size_t src_slot = present_slots - 1;               // new token = last slot
            const float* pk_ptr = pk.data<const float>();
            const float* pv_ptr = pv.data<const float>();
            float* dk_ptr = m_pred_past_k[i].data<float>();
            float* dv_ptr = m_pred_past_v[i].data<float>();
            for (size_t h = 0; h < m_pred_n_kv; ++h) {
                std::memcpy(dk_ptr + h * slot_stride + p * row,
                            pk_ptr + h * present_slot_stride + src_slot * row,
                            row * sizeof(float));
                std::memcpy(dv_ptr + h * slot_stride + p * row,
                            pv_ptr + h * present_slot_stride + src_slot * row,
                            row * sizeof(float));
            }
        }
    }
    ++m_pred_position;

    return m_talker_code_predictor.get_tensor("logits");  // [num_heads,1,1,V]
}

// Slice a single code-group head out of the stacked all-heads logits
// [num_heads, 1, 1, V] into a [1, 1, V] tensor that sample_token_from_logits
// expects. `head` is the 0-based MTP head index (= code_group - 1).
// The head slice is the outermost dimension, so its V elements are contiguous
// in memory; we wrap that memory in a zero-copy 3D view (no allocation/copy).
// Safe because the caller keeps `all_logits` alive while the result is used.
ov::Tensor Qwen3TTSImpl::select_predictor_head(const ov::Tensor& all_logits, size_t head) const {
    const auto& shape = all_logits.get_shape();  // [num_heads,1,1,V]
    OPENVINO_ASSERT(shape.size() == 4 && head < shape[0],
                    "select_predictor_head: bad shape/head (heads=", shape[0], " head=", head, ")");
    const size_t vocab = shape[3];
    const auto et = all_logits.get_element_type();
    auto* base = static_cast<uint8_t*>(const_cast<void*>(all_logits.data()));
    void* head_ptr = base + head * vocab * et.size();
    return ov::Tensor(et, ov::Shape{1, 1, vocab}, head_ptr);  // zero-copy view [1,1,V]
}

ov::Tensor Qwen3TTSImpl::infer_predictor_embedding(int64_t token_id, int64_t generation_step) {
    // Update pre-allocated id tensors in-place (tensors already set on infer request).
    m_pred_emb_ids.data<int64_t>()[0] = token_id;
    m_pred_emb_step.data<int64_t>()[0] = generation_step;

    run_and_time([&]{ m_talker_code_predictor_embedding.infer(); }, "code_predictor_embedding", m_perf_ms, m_perf_calls);
    // Return the request's own output tensor (no clone). Valid until the next
    // m_talker_code_predictor_embedding.infer(). Every caller consumes it (copies
    // or uses it as the predictor input) before the next embedding infer.
    return m_talker_code_predictor_embedding.get_output_tensor(0);
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

    // Reuse shared GenAI sampling transforms (repetition penalty, temperature,
    // top-k/top-p/min-p, structured transformers if configured later).
    ov::genai::Logits logits_view(scores.data(), vocab);
    ov::genai::LogitProcessor logit_processor(generation_config, generated);
    logit_processor.update_generated_len(generated.size());
    logit_processor.apply(logits_view);

    auto greedy_pick = [&]() -> int64_t {
        if (logits_view.is_vector_initialized()) {
            const auto it = std::max_element(logits_view.m_vector.begin(), logits_view.m_vector.end(),
                                             [](const ov::genai::Token& lhs, const ov::genai::Token& rhs) {
                                                 return lhs.m_log_prob < rhs.m_log_prob;
                                             });
            return (it != logits_view.m_vector.end()) ? it->m_index : 0;
        }
        return static_cast<int64_t>(std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())));
    };

    if (!generation_config.do_sample) {
        return greedy_pick();
    }

    auto sample_from_probs = [&](const std::vector<float>& probs,
                                 const std::vector<int64_t>& token_ids) -> int64_t {
        OPENVINO_ASSERT(!probs.empty() && probs.size() == token_ids.size(),
                        "Invalid probability vector for sampling");
        std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
        return token_ids[dist(rng)];
    };

    if (logits_view.m_defer_expf) {
        OPENVINO_ASSERT(logits_view.is_vector_initialized(),
                        "defer_expf path requires initialized logits vector");
        float max_val = logits_view.m_vector[0].m_log_prob;
        for (size_t i = 1; i < logits_view.m_size; ++i) {
            max_val = std::max(max_val, logits_view.m_vector[i].m_log_prob);
        }

        std::vector<float> weights;
        std::vector<int64_t> ids;
        weights.reserve(logits_view.m_size);
        ids.reserve(logits_view.m_size);
        float sum_w = 0.0f;
        for (size_t i = 0; i < logits_view.m_size; ++i) {
            float w = std::exp(logits_view.m_vector[i].m_log_prob - max_val);
            if (!std::isfinite(w) || w < 0.0f) {
                w = 0.0f;
            }
            weights.push_back(w);
            ids.push_back(logits_view.m_vector[i].m_index);
            sum_w += w;
        }

        if (!(sum_w > 0.0f && std::isfinite(sum_w))) {
            return ids.empty() ? greedy_pick() : ids.front();
        }
        return sample_from_probs(weights, ids);
    }

    std::vector<float> probs;
    std::vector<int64_t> ids;
    if (logits_view.is_vector_initialized()) {
        probs.reserve(logits_view.m_size);
        ids.reserve(logits_view.m_size);
        for (size_t i = 0; i < logits_view.m_size; ++i) {
            const auto& t = logits_view.m_vector[i];
            if (std::isfinite(t.m_log_prob) && t.m_log_prob > 0.0f) {
                probs.push_back(t.m_log_prob);
                ids.push_back(t.m_index);
            }
        }
    } else {
        probs.reserve(vocab);
        ids.reserve(vocab);
        for (size_t i = 0; i < vocab; ++i) {
            const float p = scores[i];
            if (std::isfinite(p) && p > 0.0f) {
                probs.push_back(p);
                ids.push_back(static_cast<int64_t>(i));
            }
        }
    }

    if (probs.empty()) {
        return greedy_pick();
    }

    return sample_from_probs(probs, ids);
}

std::vector<int64_t> Qwen3TTSImpl::generate_codec_groups(const ov::Tensor& past_hidden,
                                                         int64_t first_codec_token,
                                                         const SpeechGenerationConfig& generation_config,
                                                         std::mt19937& rng) {
    // Measure wall-clock time and inference time separately to compute overhead per-call.
    auto wall_start = std::chrono::high_resolution_clock::now();
    auto code_predictor_ms_before = m_perf_ms["code_predictor"];

    SpeechGenerationConfig predictor_config = generation_config;
    predictor_config.do_sample = generation_config.subtalker_dosample;
    predictor_config.top_k = generation_config.subtalker_top_k;
    predictor_config.top_p = generation_config.subtalker_top_p;
    predictor_config.temperature = generation_config.subtalker_temperature;
    predictor_config.repetition_penalty = 1.0f;

    std::vector<int64_t> groups;
    groups.reserve(m_ids.num_code_groups);
    groups.push_back(first_codec_token);

    // Residual codec generation against the static all-heads code predictor.
    //
    // The static model is single-token with an explicit fixed KV cache; infer_predictor
    // manages the host-side cache and absolute-position counter. We therefore feed the
    // 2-token prefill (talker past_hidden then the first-codec embedding) ONE TOKEN AT A
    // TIME instead of a single multi-token prefill, then continue one token per group.
    //
    // The model emits ALL code-group heads stacked as logits [num_heads,1,1,V]; we pick
    // head index (group-1) on the host via select_predictor_head. Group `g` is predicted
    // from the hidden state produced AFTER consuming token (g-1)'s embedding.

    // === First embedding lookup ===
    ov::Tensor first_id_hidden;
    run_and_time(
        [&]() { first_id_hidden = infer_embedding(m_talker_embedding, first_codec_token); },
        "codec_groups_first_embedding",
        m_perf_ms,
        m_perf_calls);

    ov::Shape ph_shape = past_hidden.get_shape();
    ov::Shape hid_shape = first_id_hidden.get_shape();
    OPENVINO_ASSERT(ph_shape.size() == 3 && hid_shape.size() == 3, "Unexpected hidden shape in code predictor prefill");
    OPENVINO_ASSERT(hid_shape[1] == 1, "Code predictor first-id embedding must be a single token");

    const size_t hidden = ph_shape[2];
    const size_t prefill_tokens = ph_shape[1] + hid_shape[1];  // talker context + first-id (normally 2)

    auto feed_token = [&](const float* src, bool reset) -> ov::Tensor {
        // Use pre-allocated inputs_embeds tensor from infer_request,
        // instead of allocating our own unaligned memory. Copy data directly into it.
        ov::Tensor tok = m_talker_code_predictor.get_tensor("inputs_embeds");
        std::memcpy(tok.data(), src, hidden * sizeof(float));
        return infer_predictor(tok, reset);
    };

    // Prefill: feed each context token, resetting the KV cache on the first one. Only the
    // logits from the LAST prefill token (which sees the full context) predict group 1.
    const float* ph_ptr = past_hidden.data<const float>();
    const float* hid_ptr = first_id_hidden.data<const float>();
    ov::Tensor logits;
    run_and_time(
        [&]() {
            for (size_t t = 0; t < prefill_tokens; ++t) {
                const bool is_last_ctx = (t + 1 == ph_shape[1]);  // last talker-context token
                const float* src = (t < ph_shape[1]) ? (ph_ptr + t * hidden) : hid_ptr;
                logits = feed_token(src, /*reset=*/t == 0);
                (void)is_last_ctx;
            }
        },
        "codec_groups_prefill_loop",
        m_perf_ms,
        m_perf_calls);

    std::vector<int64_t> generated;
    generated.reserve(m_ids.num_code_groups - 1);

    std::vector<bool> predictor_suppressed(2048, false);

    // === Prefill sampling: select head 0 and sample group 1 ===
    int64_t next;
    run_and_time(
        [&]() {
            auto head0 = select_predictor_head(logits, 0);
            next = sample_token_from_logits(head0, predictor_config, generated, predictor_suppressed, rng);
        },
        "codec_groups_prefill_sampling",
        m_perf_ms,
        m_perf_calls);
    generated.push_back(next);

    // === Residual loops: embedding + head selection + sampling per group ===
    run_and_time(
        [&]() {
            for (size_t g = 1; g < m_ids.num_code_groups - 1; ++g) {
                // Embed the previous residual token, advance one position, predict group g+1
                // from head index g.
                auto emb = infer_predictor_embedding(next, static_cast<int64_t>(g - 1));
                OPENVINO_ASSERT(emb.get_shape().size() == 3 && emb.get_shape()[1] == 1,
                                "Code predictor residual embedding must be a single token");
                auto all_lg = infer_predictor(emb, /*reset=*/false);
                auto head_lg = select_predictor_head(all_lg, g);
                next = sample_token_from_logits(head_lg, predictor_config, generated, predictor_suppressed, rng);
                generated.push_back(next);
            }
        },
        "codec_groups_residual_loop",
        m_perf_ms,
        m_perf_calls);

    groups.insert(groups.end(), generated.begin(), generated.end());
    while (groups.size() < m_ids.num_code_groups) {
        groups.push_back(m_ids.codec_pad_id);
    }

    // Record timing: total wall-clock time and nested inference time separately.
    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    double code_predictor_ms_after = m_perf_ms["code_predictor"];
    double inference_ms = code_predictor_ms_after - code_predictor_ms_before;
    double overhead_ms = wall_ms - inference_ms;

    m_perf_ms["generate_codec_groups"] += wall_ms;
    m_perf_calls["generate_codec_groups"]++;
    m_perf_ms["codec_groups_overhead"] += overhead_ms;
    m_perf_calls["codec_groups_overhead"]++;

    return groups;
}

Text2SpeechDecodedResults Qwen3TTSImpl::decode_from_prefill(const ov::Tensor& talker_prefill,
                                                            const ov::Tensor& tts_pad,
                                                            const ov::Tensor& trailing_text_hidden,
                                                            const SpeechGenerationConfig& generation_config,
                                                            const std::vector<bool>& suppress_tokens) {
    Text2SpeechDecodedResults result;
    result.output_sample_rate = m_output_sample_rate;

    auto prefill_mask = make_attention_mask(talker_prefill.get_shape()[1]);
    auto prefill_pos = make_position_ids_prefill(talker_prefill.get_shape()[1]);
    auto logits = infer_talker(talker_prefill, prefill_mask, prefill_pos, true);
    // Hold the talker's own hidden_states output (no clone). We only read its last
    // row into past_hidden below, which happens before the next m_talker.infer().
    auto hidden_states = m_talker.get_tensor("hidden_states");

    std::vector<int64_t> generated_main;
    std::vector<int64_t> all_codes;
    size_t max_steps = generation_config.get_max_new_tokens();
    if (max_steps == SIZE_MAX) {
        max_steps = 2048;
    }
    generated_main.reserve(std::min(max_steps, size_t(8192)));
    all_codes.reserve(std::min(max_steps, size_t(8192)) * m_ids.num_code_groups);

    std::mt19937 rng(generation_config.rng_seed != 0 ? static_cast<uint32_t>(generation_config.rng_seed) : std::random_device{}());
    size_t absolute_pos = talker_prefill.get_shape()[1];
    for (size_t step = 0; step < max_steps; ++step) {
        int64_t token = sample_token_from_logits(logits, generation_config, generated_main, suppress_tokens, rng);
        if (token == m_ids.codec_eos_token_id) {
            break;
        }
        generated_main.push_back(token);

        // TODO, we can use slice here I think.
        ov::Tensor past_hidden(ov::element::f32, ov::Shape{1, 1, hidden_states.get_shape()[2]});
        const float* hs_ptr = hidden_states.data<const float>();
        const size_t hs_len = hidden_states.get_shape()[1];
        std::copy_n(hs_ptr + (hs_len - 1) * hidden_states.get_shape()[2], hidden_states.get_shape()[2], past_hidden.data<float>());

        auto groups = generate_codec_groups(past_hidden, token, generation_config, rng);
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
        } else if (trailing_text_hidden) {
            OPENVINO_ASSERT(trailing_text_hidden.get_shape().size() == 3,
                            "trailing_text_hidden is expected to have shape [1, T, H]");
            OPENVINO_ASSERT(trailing_text_hidden.get_shape()[0] == 1,
                            "trailing_text_hidden batch size is expected to be 1");
            OPENVINO_ASSERT(trailing_text_hidden.get_shape()[2] == token_embed.get_shape()[2],
                            "trailing_text_hidden hidden size mismatch");
            const size_t trailing_len = trailing_text_hidden.get_shape()[1];
            const float* trailing_ptr = trailing_text_hidden.data<const float>();
            const float* add_ptr = nullptr;
            if (step < trailing_len) {
                add_ptr = trailing_ptr + step * token_embed.get_shape()[2];
            } else {
                add_ptr = tts_pad.data<const float>();
            }
            for (size_t h_i = 0; h_i < token_embed.get_shape()[2]; ++h_i) {
                token_embed_ptr[h_i] += add_ptr[h_i];
            }
        }

        auto attn = make_attention_mask(absolute_pos + 1);
        auto pos = make_position_ids_decode(absolute_pos);
        logits = infer_talker(token_embed, attn, pos, false);
        hidden_states = m_talker.get_tensor("hidden_states");
        ++absolute_pos;
    }

    auto waveform = decode_speech_tokenizer(all_codes);
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
        run_and_time([&]{ m_speech_tokenizer_decoder.infer(); }, "speech_tokenizer_decoder", m_perf_ms, m_perf_calls);

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

    if (m_tts_model_type == "voice_design") {
        OPENVINO_ASSERT(generation_config.speaker.empty(),
                        "Qwen3 VoiceDesign does not support 'speaker'. Remove speaker and use 'instruct'.");
        OPENVINO_ASSERT(!speaker_embedding,
                        "Qwen3 VoiceDesign does not accept external speaker_embedding. Use language/instruct only.");
    }

    if (!base_model &&
        (!generation_config.voice_clone_ref_text.empty() || static_cast<bool>(generation_config.voice_clone_ref_audio) ||
         static_cast<bool>(generation_config.voice_clone_ref_codec_ids))) {
        OPENVINO_THROW("voice_clone_ref_text/voice_clone_ref_audio/voice_clone_ref_codec_ids are supported only for Qwen3 Base models");
    }

    // Derive x-vector-only vs ICL mode from the inputs provided.
    // ICL mode requires ref_text (transcript) or pre-computed ref_codec_ids.
    // ref_audio alone implies x-vector-only mode (embedding extraction only, no ICL prompt).
    const bool x_vector_only_mode = base_model &&
        generation_config.voice_clone_ref_text.empty() &&
        !static_cast<bool>(generation_config.voice_clone_ref_codec_ids);

    ov::Tensor effective_speaker_embedding = speaker_embedding;
    if (base_model && x_vector_only_mode && !effective_speaker_embedding && generation_config.voice_clone_ref_audio) {
        OPENVINO_ASSERT(m_speaker_encoder_sample_rate == 24000,
                        "voice_clone_ref_audio assumes 24000 Hz waveform input. OV GenAI does not resample");
        effective_speaker_embedding = extract_qwen3_speaker_embedding_from_audio(generation_config.voice_clone_ref_audio);
        // Falls through to the normal synthesis loop with effective_speaker_embedding.
    }

    // Surface the resolved speaker embedding so callers can persist and reuse it (x-vector mode).
    if (base_model && effective_speaker_embedding) {
        result.speaker_embedding = effective_speaker_embedding;
    }

    // ICL voice-clone path: triggered when ref_text or ref_codec_ids is provided.
    const bool has_qwen_voice_clone_props =
        base_model &&
        !x_vector_only_mode &&
        (!generation_config.voice_clone_ref_text.empty() ||
         static_cast<bool>(generation_config.voice_clone_ref_audio) ||
         static_cast<bool>(generation_config.voice_clone_ref_codec_ids));

    if (has_qwen_voice_clone_props) {
        ov::Tensor resolved_speaker_embedding = effective_speaker_embedding;
        if (!resolved_speaker_embedding && generation_config.voice_clone_ref_audio) {
            OPENVINO_ASSERT(m_speaker_encoder_sample_rate == 24000,
                            "voice_clone_ref_audio assumes 24000 Hz waveform input. OV GenAI does not resample");
            resolved_speaker_embedding = extract_qwen3_speaker_embedding_from_audio(generation_config.voice_clone_ref_audio);
        }

        OPENVINO_ASSERT(resolved_speaker_embedding,
                        "Qwen3 Base voice cloning via generate(...) requires either speaker_embedding or voice_clone_ref_audio");

        ov::Tensor resolved_ref_code = generation_config.voice_clone_ref_codec_ids;
        if (!resolved_ref_code && generation_config.voice_clone_ref_audio) {
            OPENVINO_ASSERT(m_speech_tokenizer_input_sample_rate == 24000,
                            "voice_clone_ref_audio assumes 24000 Hz waveform input. OV GenAI does not resample");
            resolved_ref_code = extract_qwen3_ref_code_from_audio(generation_config.voice_clone_ref_audio);
        }

        Qwen3VoiceClonePrompt prompt;
        prompt.ref_spk_embedding = resolved_speaker_embedding;
        prompt.ref_text = generation_config.voice_clone_ref_text;
        prompt.ref_code = resolved_ref_code;

        // Surface the resolved clone artifacts so callers can persist and reuse them (ICL mode).
        result.speaker_embedding = resolved_speaker_embedding;
        result.voice_clone_ref_codec_ids = resolved_ref_code;

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
        perf_print_and_reset();
        return result;
    }

    for (const auto& text : texts) {
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
            } else if (m_tts_model_type == "custom_voice") {
                OPENVINO_THROW("Unsupported Qwen3 CustomVoice speaker '",
                               generation_config.speaker,
                               "'. Use a speaker id from the model config.");
            }
        }
        ov::Tensor speaker_embed;
        bool has_speaker_embed = false;
        if (base_model) {
            OPENVINO_ASSERT(effective_speaker_embedding,
                            "Qwen3 Base requires speaker_embedding input. "
                            "Provide a speaker embedding tensor shaped [D], [1,D], or [1,1,D].");
            speaker_embed = normalize_external_speaker_embedding(effective_speaker_embedding, codec_prefill_embed0.get_shape()[2]);
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
        ov::Tensor trailing_text_hidden;
        if (generation_config.non_streaming_mode) {
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
            // Mirror HF streaming prompt assembly for the non-ICL voice-clone path.
            // prefill = role + left_codec_sum + (first_text_token + last_codec_token)
            // trailing_text_hidden = remaining_text_tokens + tts_eos
            OPENVINO_ASSERT(trimmed_len > 0,
                            "Streaming mode expects at least one target text token after trimming");

            ov::Tensor first_text_plus_last_codec(ov::element::f32, ov::Shape{1, 1, hidden});
            const float* text0 = text_embed_trimmed.data<const float>();
            const float* codec_last = codec_input_embedding.data<const float>() + (codec_len - 1) * hidden;
            for (size_t j = 0; j < hidden; ++j) {
                first_text_plus_last_codec.data<float>()[j] = text0[j] + codec_last[j];
            }
            talker_prefill = concat_embed({role_embed, left_codec_sum, first_text_plus_last_codec});

            const size_t trailing_text_tokens = (trimmed_len > 1 ? (trimmed_len - 1) : 0);
            ov::Tensor trailing_text_without_eos(ov::element::f32, ov::Shape{1, trailing_text_tokens, hidden});
            if (trailing_text_tokens > 0) {
                const float* src = text_embed_trimmed.data<const float>() + hidden;
                std::copy_n(src,
                            trailing_text_without_eos.get_size(),
                            trailing_text_without_eos.data<float>());
            }
            trailing_text_hidden = concat_embed({trailing_text_without_eos, tts_eos});
        }

        if (!generation_config.instruct.empty()) {
            const std::string instruct_text = "<|im_start|>user\n" + generation_config.instruct + "<|im_end|>\n";
            auto instruct_ids = m_tokenizer.encode(instruct_text).input_ids;
            const int64_t* instr = instruct_ids.data<const int64_t>();
            const size_t instr_len = instruct_ids.get_shape()[1];
            auto instruct_embed = infer_text_projection(infer_embedding_seq(m_talker_text_embedding, std::vector<int64_t>(instr, instr + instr_len)));
            talker_prefill = concat_embed({instruct_embed, talker_prefill});
        }

        auto decoded = decode_from_prefill(talker_prefill, tts_pad, trailing_text_hidden, generation_config, suppress_tokens);
        result.speeches.insert(result.speeches.end(), decoded.speeches.begin(), decoded.speeches.end());
        result.perf_metrics.num_generated_samples += decoded.perf_metrics.num_generated_samples;
    }

    auto generation_end = std::chrono::steady_clock::now();
    result.perf_metrics.raw_metrics.generate_durations.emplace_back(
        PerfMetrics::get_microsec(generation_end - generation_start));
    result.perf_metrics.evaluate_statistics();
    m_perf_metrics = result.perf_metrics;
    perf_print_and_reset();
    return result;
}

Text2SpeechDecodedResults Qwen3TTSImpl::generate_voice_clone(const std::string& text,
                                                             const Qwen3VoiceClonePrompt& prompt,
                                                             const SpeechGenerationConfig& generation_config) {
    OPENVINO_ASSERT(is_base_model(), "Qwen3 voice-clone prompt generation is supported only for Base models");
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

    // Tokenize the reference and target turns SEPARATELY, then concatenate token ids,
    // mirroring upstream qwen_tts (ref_id[:, 3:-2] ++ text_id[:, 3:-5]). Gluing the raw
    // strings before tokenization could merge tokens across the ref/target seam (BPE
    // boundary effects) and diverge from the reference.
    const std::string ref_turn = "<|im_start|>assistant\n" + prompt.ref_text + "<|im_end|>\n";
    const std::string target_turn = "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";

    auto ref_input_ids = m_tokenizer.encode(ref_turn).input_ids;
    auto target_input_ids = m_tokenizer.encode(target_turn).input_ids;
    OPENVINO_ASSERT(ref_input_ids.get_element_type() == ov::element::i64 &&
                        target_input_ids.get_element_type() == ov::element::i64,
                    "Qwen3-TTS tokenizer must produce i64 input_ids tensor");
    OPENVINO_ASSERT(ref_input_ids.get_shape().size() == 2 && ref_input_ids.get_shape()[0] == 1 &&
                        target_input_ids.get_shape().size() == 2 && target_input_ids.get_shape()[0] == 1,
                    "Expected input_ids shape [1, T]");

    // Each turn is stripped of its 3-token role prefix (<|im_start|>assistant\n) and its
    // trailing structural tokens: the reference turn drops <|im_end|>\n (2 tokens); the
    // target turn drops <|im_end|>\n<|im_start|>assistant\n (5 tokens).
    const size_t ref_ids_len = ref_input_ids.get_shape()[1];
    const size_t target_ids_len = target_input_ids.get_shape()[1];
    OPENVINO_ASSERT(ref_ids_len > 5, "Qwen3 voice-clone reference transcript tokenized too short");
    OPENVINO_ASSERT(target_ids_len > 8, "Qwen3 voice-clone target text tokenized too short");
    const int64_t* ref_ids = ref_input_ids.data<const int64_t>();
    const int64_t* target_ids = target_input_ids.data<const int64_t>();

    // Role prefix taken from the target turn (matches upstream input_id[:, :3]).
    const std::vector<int64_t> role_ids(target_ids, target_ids + 3);
    // Content = ref_content ++ target_content (token ids), tokenized independently above.
    std::vector<int64_t> icl_content_ids;
    icl_content_ids.reserve((ref_ids_len - 5) + (target_ids_len - 8));
    icl_content_ids.insert(icl_content_ids.end(), ref_ids + 3, ref_ids + (ref_ids_len - 2));
    icl_content_ids.insert(icl_content_ids.end(), target_ids + 3, target_ids + (target_ids_len - 5));

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
    auto role_embed = infer_text_projection(infer_embedding_seq(m_talker_text_embedding, role_ids));

    // text_embed: embed(ref_content ++ target_content), then append tts_eos
    // (matches Python: text_embed = text_projection(embed(cat([ref_id, text_id]))); cat([text_embed, tts_eos])).
    auto content_embed = infer_text_projection(infer_embedding_seq(m_talker_text_embedding, icl_content_ids));
    ov::Tensor text_embed_with_eos = concat_embed({content_embed, tts_eos});
    const size_t text_eos_len = text_embed_with_eos.get_shape()[1];

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

    // Part 2: codec_side source = (codec_bos + ref_codes)
    ov::Tensor codec_bos_plus_ref = concat_embed({codec_bos_embed, ref_code_embed});
    const size_t codec_side_len = codec_bos_plus_ref.get_shape()[1];
    ov::Tensor icl_input_embed;
    ov::Tensor trailing_text_hidden;
    if (generation_config.non_streaming_mode) {
        // HF ICL non-streaming: (text_with_eos + codec_pad) + (codec_bos_plus_ref + tts_pad)
        auto codec_pad_for_text = infer_embedding_seq(m_talker_embedding, std::vector<int64_t>(text_eos_len, m_ids.codec_pad_id));
        ov::Tensor text_side(ov::element::f32, ov::Shape{1, text_eos_len, hidden});
        for (size_t i = 0; i < text_side.get_size(); ++i) {
            text_side.data<float>()[i] = text_embed_with_eos.data<const float>()[i] + codec_pad_for_text.data<const float>()[i];
        }

        ov::Tensor codec_side(ov::element::f32, ov::Shape{1, codec_side_len, hidden});
        for (size_t i = 0; i < codec_side_len; ++i) {
            const float* src = codec_bos_plus_ref.data<const float>() + i * hidden;
            float* dst = codec_side.data<float>() + i * hidden;
            const float* pad = tts_pad.data<const float>();
            for (size_t h = 0; h < hidden; ++h) {
                dst[h] = src[h] + pad[h];
            }
        }

        icl_input_embed = concat_embed({text_side, codec_side});
    } else {
        // HF ICL streaming: align text_with_eos and codec_bos_plus_ref by min(text_len, codec_len),
        // and carry remaining text as trailing_text_hidden (or tts_pad when text is shorter).
        if (text_eos_len > codec_side_len) {
            ov::Tensor overlap(ov::element::f32, ov::Shape{1, codec_side_len, hidden});
            for (size_t i = 0; i < codec_side_len; ++i) {
                const float* text_ptr = text_embed_with_eos.data<const float>() + i * hidden;
                const float* codec_ptr = codec_bos_plus_ref.data<const float>() + i * hidden;
                float* out = overlap.data<float>() + i * hidden;
                for (size_t h = 0; h < hidden; ++h) {
                    out[h] = text_ptr[h] + codec_ptr[h];
                }
            }
            icl_input_embed = overlap;

            const size_t trailing_len = text_eos_len - codec_side_len;
            trailing_text_hidden = ov::Tensor(ov::element::f32, ov::Shape{1, trailing_len, hidden});
            std::copy_n(text_embed_with_eos.data<const float>() + codec_side_len * hidden,
                        trailing_text_hidden.get_size(),
                        trailing_text_hidden.data<float>());
        } else {
            ov::Tensor text_padded(ov::element::f32, ov::Shape{1, codec_side_len, hidden});
            if (text_eos_len > 0) {
                std::copy_n(text_embed_with_eos.data<const float>(),
                            text_embed_with_eos.get_size(),
                            text_padded.data<float>());
            }
            for (size_t i = text_eos_len; i < codec_side_len; ++i) {
                std::copy_n(tts_pad.data<const float>(), hidden, text_padded.data<float>() + i * hidden);
            }

            ov::Tensor overlap(ov::element::f32, ov::Shape{1, codec_side_len, hidden});
            for (size_t i = 0; i < codec_side_len; ++i) {
                const float* text_ptr = text_padded.data<const float>() + i * hidden;
                const float* codec_ptr = codec_bos_plus_ref.data<const float>() + i * hidden;
                float* out = overlap.data<float>() + i * hidden;
                for (size_t h = 0; h < hidden; ++h) {
                    out[h] = text_ptr[h] + codec_ptr[h];
                }
            }
            icl_input_embed = overlap;
            trailing_text_hidden = tts_pad;
        }
    }

    // Full prefill: role | codec_input_part | icl_input_embed
    ov::Tensor talker_prefill = concat_embed({role_embed, codec_input_part, icl_input_embed});

    if (!generation_config.instruct.empty()) {
        const std::string instruct_text = "<|im_start|>user\n" + generation_config.instruct + "<|im_end|>\n";
        auto instruct_ids = m_tokenizer.encode(instruct_text).input_ids;
        const int64_t* instr = instruct_ids.data<const int64_t>();
        const size_t instr_len = instruct_ids.get_shape()[1];
        auto instruct_embed = infer_text_projection(infer_embedding_seq(m_talker_text_embedding, std::vector<int64_t>(instr, instr + instr_len)));
        talker_prefill = concat_embed({instruct_embed, talker_prefill});
    }

    return decode_from_prefill(talker_prefill,
                               tts_pad,
                               trailing_text_hidden,
                               generation_config,
                               suppress_tokens);
}

void Qwen3TTSImpl::perf_print_and_reset() const {
    if (!qwen_perf_enabled() || m_perf_ms.empty()) {
        m_perf_ms.clear();
        m_perf_calls.clear();
        m_talker_prefill_tokens = 0;
        return;
    }
    // Sort by descending total time so the hot path appears first.
    std::vector<std::pair<std::string, double>> entries(m_perf_ms.begin(), m_perf_ms.end());
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::cout << "[QWEN_PERF] Component inference timing (ms):" << std::endl;
    for (const auto& [name, total_ms] : entries) {
        const int64_t count = m_perf_calls.count(name) ? m_perf_calls.at(name) : 1;
        const double avg_ms = total_ms / static_cast<double>(count);
        const std::string device = m_perf_device.count(name) ? m_perf_device.at(name) : "unknown";
        const std::string extra =
            (name == "talker_prefill")
                ? (", prefill_tokens=" + std::to_string(m_talker_prefill_tokens))
                : std::string{};
        std::cout << "[QWEN_PERF]   "
                  << std::left << std::setw(30) << name << " [" << std::setw(10) << device << "] : "
                  << std::fixed << std::setprecision(1) << std::setw(9) << total_ms << " ms"
                  << "  (x" << count
                  << ", avg " << std::setw(7) << avg_ms << " ms"
                  << extra << ")" << std::endl;
    }
    m_perf_ms.clear();
    m_perf_calls.clear();
    m_talker_prefill_tokens = 0;
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
