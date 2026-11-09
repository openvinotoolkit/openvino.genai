// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speech_generation/qwen3_tts/qwen3_tts_codec_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>

#include <nlohmann/json.hpp>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset15.hpp>

#include "json_utils.hpp"
#include "sampling/logit_processor.hpp"
#include "sampling/logit_transformers.hpp"
#include "sampling/sampler.hpp"
#include "speech_generation/qwen3_tts/mel_preprocess.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_utils.hpp"
#include "utils.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"

using namespace ov::genai::qwen3_tts;

namespace ov {
namespace genai {

Qwen3TTSCodecEngine::Qwen3TTSCodecEngine(const std::filesystem::path& models_path,
                                         const std::string& device,
                                         const ov::AnyMap& properties,
                                         const Tokenizer& tokenizer)
    : m_tokenizer(tokenizer) {
    init_config(models_path);

    // Resolve all component properties through the established layered mechanism:
    // globals + DEVICE_PROPERTIES[device] + MODEL_PROPERTIES[role].
    ov::AnyMap base_properties = properties;

    auto compile_for = [&](const auto& model_source, const std::string& role) {
        auto [target_device, target_properties] = resolve_component_target(device, base_properties, role);
        m_perf_device[role] = target_device;
        return compile_request(model_source, role, target_device, target_properties);
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

    m_talker_embedding = compile_for(models_path / TALKER_EMBEDDINGS_NAME, roles::TALKER_EMBEDDING);
    m_talker_text_embedding = compile_for(models_path / TEXT_EMBEDDINGS_NAME, roles::TALKER_TEXT_EMBEDDING);
    {
        auto [predictor_device, predictor_properties] = resolve_component_target(device, base_properties, roles::CODE_PREDICTOR);

        if (predictor_device.find("GPU") != std::string::npos &&
            predictor_properties.find("ACTIVATIONS_SCALE_FACTOR") == predictor_properties.end()) {
            predictor_properties["ACTIVATIONS_SCALE_FACTOR"] = 8.0f;
        }

        const auto predictor_path = models_path / CODE_PREDICTOR_NAME;
        auto predictor_model = ov::genai::utils::singleton_core().read_model(predictor_path);
        const bool predictor_on_npu = predictor_device.find("NPU") != std::string::npos;
        if (predictor_on_npu) {
            // NPU path for code predictor: expose step as [1], convert to stateless,
            // then freeze to static so explicit KV tensors can be host-managed.
            expose_scalar_input_as_vector1(predictor_model, "step");
            ov::pass::StatefulToStateless().run_on_model(predictor_model);
            predictor_model->validate_nodes_and_infer_types();
            reshape_predictor_to_static(predictor_model);
            m_predictor_static = true;
            init_static_predictor_meta(predictor_model);

            m_talker_code_predictor = compile_request(predictor_model, roles::CODE_PREDICTOR, predictor_device, predictor_properties);

            // Cache KV tensors infer request.
            for (size_t i = 0; i < m_pred_num_layers; ++i) {
                m_pred_past_k.emplace_back(m_talker_code_predictor.get_tensor("past_key_values." + std::to_string(i) + ".key"));
                m_pred_past_v.emplace_back(m_talker_code_predictor.get_tensor("past_key_values." + std::to_string(i) + ".value"));
                OPENVINO_ASSERT(m_pred_past_k.back().get_element_type() == ov::element::f32,
                                "Static code predictor past_key_values.", i, ".key must be f32");
                OPENVINO_ASSERT(m_pred_past_v.back().get_element_type() == ov::element::f32,
                                "Static code predictor past_key_values.", i, ".value must be f32");
            }
            m_pred_attn = m_talker_code_predictor.get_tensor("attention_mask");
            m_pred_pos = m_talker_code_predictor.get_tensor("position_ids");
        } else {
            m_talker_code_predictor = compile_request(predictor_model, roles::CODE_PREDICTOR, predictor_device, predictor_properties);
            m_predictor_static = false;
        }
        m_perf_device[roles::CODE_PREDICTOR] = predictor_device;

        m_predictor_has_beam_idx = false;
        for (const auto& in : m_talker_code_predictor.get_compiled_model().inputs()) {
            if (in.get_any_name() == "beam_idx") {
                m_predictor_has_beam_idx = true;
                break;
            }
        }
        if (m_predictor_has_beam_idx) {
            ov::Tensor beam_idx(ov::element::i32, ov::Shape{1});
            beam_idx.data<int32_t>()[0] = 0;
            m_talker_code_predictor.set_tensor("beam_idx", beam_idx);
        }

        auto supports_dynamic_batch = [&](const std::string& input_name) {
            for (const auto& input : m_talker_code_predictor.get_compiled_model().inputs()) {
                if (input.get_any_name() != input_name) {
                    continue;
                }
                const auto shape = input.get_partial_shape();
                return shape.rank().is_static() && shape.rank().get_length() > 0 && shape[0].is_dynamic();
            }
            return false;
        };
        m_predictor_batching_supported = !m_predictor_static &&
                                         supports_dynamic_batch("inputs_embeds") &&
                                         supports_dynamic_batch("attention_mask") &&
                                         supports_dynamic_batch("position_ids") &&
                                         (!m_predictor_has_beam_idx || supports_dynamic_batch("beam_idx"));
    }
    m_talker_code_predictor_embedding = compile_for(models_path / CODE_PREDICTOR_EMBEDDINGS_NAME, roles::CODE_PREDICTOR_EMBEDDING);

    // Bind fixed-shape input tensors for the code-predictor embedding once. Unlike
    // the code predictor (reshaped to static), this embedding model may be dynamic,
    // so a get_tensor() default can have zero-sized dims and set_shape() on it does
    // not reliably re-bind the new shape into the request. We instead own buffers
    // with the exact shapes the original per-call path used (input_ids=[1,1],
    // step=scalar) and set_tensor them a single time here. Subsequent
    // inferences only rewrite the data in place -- no per-call set_tensor.
    m_pred_emb_ids = ov::Tensor(ov::element::i64, ov::Shape{1, 1});
    m_pred_emb_step = ov::Tensor(ov::element::i64, ov::Shape{});
    m_talker_code_predictor_embedding.set_tensor("input_ids", m_pred_emb_ids);
    m_talker_code_predictor_embedding.set_tensor("step", m_pred_emb_step);
    {
        auto [decoder_device, decoder_properties] = resolve_component_target(device, base_properties, roles::SPEECH_TOKENIZER_DECODER);
        const auto decoder_path = models_path / CODEC_DECODER_NAME;
        const bool use_static_decoder = decoder_device.find("NPU") != std::string::npos ||
                                        decoder_device.find("GPU") != std::string::npos;
        if (use_static_decoder) {
            auto decoder_model = ov::genai::utils::singleton_core().read_model(decoder_path);
            const ov::PartialShape static_codes{1,
                                                static_cast<int64_t>(m_decoder_num_quantizers),
                                                static_cast<int64_t>(DECODER_WINDOW_LEN)};

            decoder_model->reshape({{"audio_codes", static_codes}});
            m_speech_tokenizer_decoder = compile_request(decoder_model, roles::SPEECH_TOKENIZER_DECODER, decoder_device, decoder_properties);
            m_decoder_static = true;
        } else {
            m_speech_tokenizer_decoder = compile_request(decoder_path, roles::SPEECH_TOKENIZER_DECODER, decoder_device, decoder_properties);
        }
        m_perf_device[roles::SPEECH_TOKENIZER_DECODER] = device_of(m_speech_tokenizer_decoder, decoder_device);
    }

    const auto speaker_encoder_path = models_path / SPEAKER_ENCODER_NAME;
    if (std::filesystem::exists(speaker_encoder_path)) {
        m_speaker_encoder = compile_for(speaker_encoder_path, roles::SPEAKER_ENCODER);
        m_has_speaker_encoder = true;

        auto mel_model = build_qwen3_mel_preprocess_model(m_speaker_encoder_mel_dim);
        m_qwen3_mel_preprocess = compile_for(mel_model, roles::MEL_PREPROCESS);
        m_has_qwen3_mel_preprocess = true;
    }

    const auto speech_tokenizer_encoder_path = models_path / CODEC_ENCODER_NAME;
    if (std::filesystem::exists(speech_tokenizer_encoder_path)) {
        m_speech_tokenizer_encoder = compile_for(speech_tokenizer_encoder_path, roles::SPEECH_TOKENIZER_ENCODER);
        m_has_speech_tokenizer_encoder = true;
    }
}

void Qwen3TTSCodecEngine::init_config(const std::filesystem::path& models_path) {
    const auto config_path = models_path / "config.json";
    std::ifstream config_stream(config_path);
    OPENVINO_ASSERT(config_stream.is_open(), "Failed to open ", config_path);

    nlohmann::json config = nlohmann::json::parse(config_stream);

    OPENVINO_ASSERT(config.contains("tts_model_type") && config.at("tts_model_type").is_string(),
                    "Qwen3-TTS config must contain string field 'tts_model_type'");
    const std::string tts_model_type = to_lower(config.at("tts_model_type").get<std::string>());
    OPENVINO_ASSERT(tts_model_type == "base" ||
                        tts_model_type == "custom_voice" ||
                        tts_model_type == "voice_design",
                    "Unsupported tts_model_type: '",
                    tts_model_type,
                    "'. Supported values are: 'base', 'custom_voice', 'voice_design'.");
    m_tts_model_type = tts_model_type;

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
        const uint32_t speaker_encoder_sample_rate =
            config["speaker_encoder_config"].value("sample_rate", static_cast<uint32_t>(24000));
        // The internal mel-preprocessing pipeline (build_qwen3_mel_preprocess_model) hardcodes a
        // 24000 Hz mel filterbank rather than reading this value, so a model declaring a different
        // rate would silently get mismatched mel features.
        OPENVINO_ASSERT(speaker_encoder_sample_rate == 24000,
                        "Unsupported speaker_encoder_config.sample_rate: ",
                        speaker_encoder_sample_rate,
                        ". OV GenAI's internal Qwen3 mel-preprocessing pipeline only supports 24000 Hz.");
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
            const uint32_t speech_tokenizer_input_sample_rate =
                speech_cfg.value("input_sample_rate", static_cast<uint32_t>(24000));
            // extract_qwen3_ref_code_from_audio feeds the encoder waveform directly, with no resampling.
            OPENVINO_ASSERT(speech_tokenizer_input_sample_rate == 24000,
                            "Unsupported speech_tokenizer input_sample_rate: ",
                            speech_tokenizer_input_sample_rate,
                            ". OV GenAI's internal Qwen3 ref-audio encoding only supports 24000 Hz.");
            if (speech_cfg.contains("decoder_config") && speech_cfg["decoder_config"].is_object()) {
                m_decoder_num_quantizers = speech_cfg["decoder_config"].value("num_quantizers", static_cast<uint32_t>(16));
            }
        }
    }
}

ov::Tensor Qwen3TTSCodecEngine::normalize_external_speaker_embedding(const ov::Tensor& speaker_embedding,
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

    OPENVINO_THROW("Qwen3 Base speaker_embedding must be f32");
}

std::vector<float> Qwen3TTSCodecEngine::normalize_ref_audio_waveform(const ov::Tensor& ref_audio) const {
    OPENVINO_ASSERT(ref_audio, "ref_audio must be a non-empty tensor");
    OPENVINO_ASSERT(ref_audio.get_element_type() == ov::element::f32,
                    "ref_audio must be float32 tensor with shape [T], [1, T], or [1, 1, T]");

    const auto shape = ref_audio.get_shape();
    OPENVINO_ASSERT(!shape.empty(), "ref_audio tensor rank must be 1, 2, or 3");

    size_t num_samples = 0;
    if (shape.size() == 1) {
        num_samples = shape[0];
    } else if (shape.size() == 2) {
        OPENVINO_ASSERT(shape[0] == 1, "ref_audio rank-2 tensor must have shape [1, T]");
        num_samples = shape[1];
    } else if (shape.size() == 3) {
        OPENVINO_ASSERT(shape[0] == 1 && shape[1] == 1,
                        "ref_audio rank-3 tensor must have shape [1, 1, T]");
        num_samples = shape[2];
    } else {
        OPENVINO_THROW("ref_audio tensor rank must be 1, 2, or 3");
    }

    OPENVINO_ASSERT(num_samples > 0, "ref_audio must contain at least one sample");

    const float* src = ref_audio.data<const float>();
    std::vector<float> waveform(num_samples, 0.0f);
    std::copy_n(src, num_samples, waveform.data());
    return waveform;
}

ov::Tensor Qwen3TTSCodecEngine::extract_qwen3_speaker_embedding_from_audio(const ov::Tensor& ref_audio) const {
    OPENVINO_ASSERT(m_has_speaker_encoder,
                    "ref_audio requires 'openvino_speaker_encoder.xml' in the model directory");
    OPENVINO_ASSERT(m_has_qwen3_mel_preprocess,
                    "ref_audio requires internal mel preprocessing model initialization");

    const std::vector<float> waveform = normalize_ref_audio_waveform(ref_audio);

    ov::Tensor audio_input(ov::element::f32, ov::Shape{1, waveform.size()});
    std::copy(waveform.begin(), waveform.end(), audio_input.data<float>());

    auto mel_req = m_qwen3_mel_preprocess;
    mel_req.set_input_tensor(0, audio_input);
    run_and_time([&]{ mel_req.infer(); }, roles::MEL_PREPROCESS, m_perf_ms, m_perf_calls);
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
    run_and_time([&]{ req.infer(); }, roles::SPEAKER_ENCODER, m_perf_ms, m_perf_calls);

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

ov::Tensor Qwen3TTSCodecEngine::extract_qwen3_ref_code_from_audio(const ov::Tensor& ref_audio) const {
    OPENVINO_ASSERT(m_has_speech_tokenizer_encoder,
                    "ref_audio ICL mode requires 'openvino_codec_encoder.xml'");

    const std::vector<float> waveform = normalize_ref_audio_waveform(ref_audio);

    ov::Tensor audio_input(ov::element::f32, ov::Shape{1, 1, waveform.size()});
    std::copy(waveform.begin(), waveform.end(), audio_input.data<float>());

    auto req = m_speech_tokenizer_encoder;
    req.set_input_tensor(0, audio_input);
    run_and_time([&]{ req.infer(); }, roles::SPEECH_TOKENIZER_ENCODER, m_perf_ms, m_perf_calls);

    const ov::Tensor out = req.get_output_tensor(0);
    OPENVINO_ASSERT(out.get_element_type() == ov::element::i64,
                    "Speech tokenizer encoder output must be i64");

    const ov::Shape shape = out.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "Unexpected ref_code output rank. Expected 3, got ", shape.size());
    const int64_t* out_i64 = out.data<const int64_t>();

    const size_t b = shape[0];
    const size_t d1 = shape[1];
    const size_t d2 = shape[2];
    OPENVINO_ASSERT(b == 1, "Unexpected ref_code batch dimension. Expected 1, got ", b);
    OPENVINO_ASSERT(d1 == m_ids.num_code_groups,
                    "Unexpected ref_code 3D shape ",
                    shape_to_string(shape),
                    ", expected [1,G,T] with G=",
                    m_ids.num_code_groups);

    ov::Tensor ref_code(ov::element::i64, ov::Shape{1, d2, d1});
    for (size_t t = 0; t < d2; ++t) {
        for (size_t g = 0; g < d1; ++g) {
            ref_code.data<int64_t>()[t * d1 + g] = out_i64[g * d2 + t];
        }
    }
    return ref_code;
}

ov::Tensor Qwen3TTSCodecEngine::infer_embedding(ov::InferRequest& request, int64_t token_id) {
    return infer_embedding_seq(request, std::vector<int64_t>{token_id});
}

ov::Tensor Qwen3TTSCodecEngine::infer_embedding_seq(ov::InferRequest& request, const std::vector<int64_t>& token_ids) {
    ov::Tensor ids(ov::element::i64, ov::Shape{1, token_ids.size()});
    std::copy(token_ids.begin(), token_ids.end(), ids.data<int64_t>());
    request.set_input_tensor(ids);

    const char* perf_stage = "talker_embedding_unknown";
    if (&request == &m_talker_embedding) {
        perf_stage = roles::TALKER_EMBEDDING;
    } else if (&request == &m_talker_text_embedding) {
        perf_stage = roles::TALKER_TEXT_EMBEDDING;
    }
    run_and_time([&] { request.infer(); }, perf_stage, m_perf_ms, m_perf_calls);

    // Cloned because callers routinely hold the result across several further
    // infers of this same request (e.g. building a multi-part prefill from several
    // embedding lookups before concatenating them).
    auto out = clone_tensor(request.get_output_tensor(0));
    const ov::Shape shape = out.get_shape();
    OPENVINO_ASSERT(shape.size() == 3,
                    "Qwen3 embedding output is expected to be rank-3 [1,T,H] or [T,1,H], got ",
                    shape);

    if (shape[0] == 1) {
        return out;
    }

    OPENVINO_ASSERT(shape[1] == 1,
                    "Qwen3 embedding output is expected to be [1,T,H] or [T,1,H], got ",
                    shape);

    // Some exports/devices return [T, 1, H]. Canonicalize to [1, T, H] so all
    // prompt assembly code can remain layout-agnostic.
    ov::Tensor transposed(out.get_element_type(), ov::Shape{1, shape[0], shape[2]});
    const size_t t = shape[0];
    const size_t h = shape[2];
    const float* src = out.data<const float>();
    float* dst = transposed.data<float>();
    for (size_t i = 0; i < t; ++i) {
        std::copy_n(src + i * h, h, dst + i * h);
    }
    return transposed;
}

// Read fixed dimensions from the static all-heads code predictor IR and
// allocate the host-side KV cache buffers. Called once after read_model when the
// static variant is detected.
//
void Qwen3TTSCodecEngine::reshape_predictor_to_static(const std::shared_ptr<ov::Model>& model) {
    bool is_dynamic = false;
    for (const auto& in : model->inputs()) {
        if (in.get_partial_shape().is_dynamic()) {
            is_dynamic = true;
            break;
        }
    }
    OPENVINO_ASSERT(is_dynamic, "Code predictor model is expected to be dynamic");

    // Fixed dims the dynamic export still pins.
    size_t n_kv = 0, head_dim = 0, past_len_hint = 0;
    for (const auto& in : model->inputs()) {
        const auto name = in.get_any_name();
        if (name == "attention_mask") {
            const auto& ps = in.get_partial_shape();
            OPENVINO_ASSERT(ps.rank().is_static() && ps.size() == 4,
                            "Code predictor attention_mask must be rank-4, got ",
                            ps);
        }
        if (name.rfind("past_key_values.", 0) == 0 && name.find(".key") != std::string::npos) {
            const auto& ps = in.get_partial_shape();  // [?, n_kv, ?, head_dim]
            OPENVINO_ASSERT(ps.rank().is_static() && ps.size() == 4 && ps[1].is_static() && ps[3].is_static(),
                            "Unexpected code predictor past_key_values shape: ", ps);
            n_kv = static_cast<size_t>(ps[1].get_length());
            head_dim = static_cast<size_t>(ps[3].get_length());
            if (ps[2].is_static()) {
                past_len_hint = static_cast<size_t>(ps[2].get_length());
            }
            break;
        }
    }
    for (const auto& out : model->outputs()) {
        if (out.get_any_name() == "logits") {
            const auto& ps = out.get_partial_shape();
            if (ps.rank().is_static()) {
                OPENVINO_ASSERT(ps.size() != 4,
                                "Unsupported static code predictor logits layout [num_heads, ..., vocab] on NPU path. "
                                "Export logits as rank-3 [B, T, V] to use static predictor mode");
            }
            break;
        }
    }
    const size_t past_len_slots = (past_len_hint > 0) ? past_len_hint
                                                      : m_ids.num_code_groups;
    OPENVINO_ASSERT(n_kv > 0 && head_dim > 0 && past_len_slots > 0,
                    "Could not derive static code predictor dims (n_kv=",
                    n_kv, " head_dim=", head_dim, " past_len_slots=", past_len_slots, ")");

    const int64_t past_len = static_cast<int64_t>(past_len_slots);  // KV window = one slot per prior step
    const int64_t kv_len = past_len + 1;
    const int64_t hidden = static_cast<int64_t>(m_talker_hidden_size);

    std::map<std::string, ov::PartialShape> shapes;
    for (const auto& in : model->inputs()) {
        const auto name = in.get_any_name();
        if (name == "inputs_embeds") {
            shapes[name] = ov::PartialShape{1, 1, hidden};
        } else if (name == "attention_mask") {
            shapes[name] = ov::PartialShape{1, 1, 1, kv_len};
        } else if (name == "position_ids") {
            shapes[name] = ov::PartialShape{1, 1};
        } else if (name == "step") {
            // Converted predictor may expose step as scalar [] or vector [1].
            const auto& ps = in.get_partial_shape();
            if (ps.rank().is_static() && ps.size() == 1) {
                shapes[name] = ov::PartialShape{1};
            } else {
                shapes[name] = ov::PartialShape{};
            }
        } else if (name.rfind("past_key_values.", 0) == 0) {
            shapes[name] = ov::PartialShape{1, static_cast<int64_t>(n_kv), past_len, static_cast<int64_t>(head_dim)};
        }
    }

    model->reshape(shapes);
}

void Qwen3TTSCodecEngine::init_static_predictor_meta(const std::shared_ptr<ov::Model>& model) {
    m_pred_num_layers = 0;
    for (const auto& in : model->inputs()) {
        const auto name = in.get_any_name();
        if (name == "attention_mask") {
            const auto& s = in.get_shape();
            OPENVINO_ASSERT(s.size() == 4,
                            "Unexpected static code predictor attention_mask rank: ", s.size());
            m_pred_kv_len = static_cast<size_t>(s[s.size() - 1]);
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
            const auto& s = out.get_shape();
            OPENVINO_ASSERT(s.size() == 3,
                            "Unexpected static code predictor logits rank: ", s.size(),
                            ". Expected rank-3 [B, T, V]");
            m_pred_num_heads = 1;  // [1,1,V]
            m_pred_vocab = static_cast<size_t>(s[2]);
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
}

// Zero the host KV buffers and reset the running absolute-position counter. Call
// at the start of each per-frame code-group prediction (the predictor cache is
// local to a single talker step).
void Qwen3TTSCodecEngine::reset_predictor_state() {
    if (!m_predictor_static) {
        m_talker_code_predictor.reset_state();
        m_pred_position = 0;
        return;
    }

    for (size_t i = 0; i < m_pred_past_k.size(); ++i) {
        std::memset(m_pred_past_k[i].data(), 0, m_pred_past_k[i].get_byte_size());
        std::memset(m_pred_past_v[i].data(), 0, m_pred_past_v[i].get_byte_size());
    }
    m_pred_position = 0;
}

ov::Tensor Qwen3TTSCodecEngine::infer_predictor(const ov::Tensor& inputs_embeds, bool reset_state, int64_t step) {
    if (reset_state) {
        reset_predictor_state();
    }

    if (!m_predictor_static) {
        const auto shape = inputs_embeds.get_shape();
        OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1,
                        "Stateful code predictor expects inputs_embeds shape [1, T, H]");
        const size_t seq_len = shape[1];
        OPENVINO_ASSERT(seq_len > 0, "Stateful code predictor received empty sequence");

        m_talker_code_predictor.set_tensor("inputs_embeds", inputs_embeds);
        if (seq_len == 1) {
            m_talker_code_predictor.set_tensor("attention_mask", make_decode_attention_mask_4d(m_pred_position + 1));
        } else {
            m_talker_code_predictor.set_tensor("attention_mask", make_causal_attention_mask_4d(seq_len));
        }
        m_talker_code_predictor.set_tensor("position_ids", make_predictor_position_ids(m_pred_position, seq_len));
        auto step_tensor = m_talker_code_predictor.get_tensor("step");
        const auto& step_shape = step_tensor.get_shape();
        OPENVINO_ASSERT(step_shape.empty() || (step_shape.size() == 1 && step_shape[0] == 1),
                "Unsupported code predictor step shape: ", step_shape);
        step_tensor.data<int64_t>()[0] = step;

        run_and_time([&]{ m_talker_code_predictor.infer(); }, roles::CODE_PREDICTOR, m_perf_ms, m_perf_calls);
        m_pred_position += seq_len;
        return m_talker_code_predictor.get_tensor("logits");  // [1, T, V]
    }

    const size_t p = m_pred_position;  // absolute position of this single token
    OPENVINO_ASSERT(p < m_pred_past_len + 1,
                    "Static code predictor exceeded its KV window (position ",
                    p, " >= kv_len ", m_pred_kv_len, ")");

    m_talker_code_predictor.set_tensor("inputs_embeds", inputs_embeds);

    // Update attention_mask data in-place (tensor already set on infer request).
    // Valid kv slots are the p real past tokens (slots 0..p-1) plus the current
    // token, which the graph appends at slot index past_len.
    const auto& attn_shape = m_pred_attn.get_shape();
    OPENVINO_ASSERT(attn_shape.size() == 4,
                    "Unsupported static code predictor attention_mask rank: ",
                    attn_shape.size());
    OPENVINO_ASSERT(m_pred_attn.get_element_type() == ov::element::f32,
                    "Static code predictor attention_mask must be f32");
    float* attn_ptr = m_pred_attn.data<float>();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::fill_n(attn_ptr, m_pred_kv_len, neg_inf);
    for (size_t j = 0; j < p; ++j) {
        attn_ptr[j] = 0.0f;
    }
    attn_ptr[m_pred_past_len] = 0.0f;  // current token slot (= kv_len - 1)

    // Update position_ids data in-place (tensor already set on infer request).
    m_pred_pos.data<int64_t>()[0] = static_cast<int64_t>(p);

    auto step_tensor = m_talker_code_predictor.get_tensor("step");
    const auto& step_shape = step_tensor.get_shape();
    OPENVINO_ASSERT(step_shape.empty() || (step_shape.size() == 1 && step_shape[0] == 1),
                    "Unsupported static code predictor step shape: ", step_shape);
    step_tensor.data<int64_t>()[0] = step;
    if (m_predictor_has_beam_idx) {
        auto beam_idx = m_talker_code_predictor.get_tensor("beam_idx");
        beam_idx.data<int32_t>()[0] = 0;
    }

    // Past KV tensors already set on infer request; no need to call set_tensor.
    run_and_time([&]{ m_talker_code_predictor.infer(); }, roles::CODE_PREDICTOR, m_perf_ms, m_perf_calls);

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
            OPENVINO_ASSERT(pk.get_element_type() == ov::element::f32,
                            "Static code predictor present.", i, ".key must be f32");
            OPENVINO_ASSERT(pv.get_element_type() == ov::element::f32,
                            "Static code predictor present.", i, ".value must be f32");
            OPENVINO_ASSERT(pk.get_element_type() == m_pred_past_k[i].get_element_type(),
                            "Element type mismatch between present.", i, ".key and past_key_values.", i, ".key");
            OPENVINO_ASSERT(pv.get_element_type() == m_pred_past_v[i].get_element_type(),
                            "Element type mismatch between present.", i, ".value and past_key_values.", i, ".value");
            const size_t present_slots = pk.get_shape()[2];          // kv_len (full) or 1 (sliced)
            OPENVINO_ASSERT(present_slots > 0,
                            "Static code predictor present.", i, " must contain at least one KV slot");
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

    return m_talker_code_predictor.get_tensor("logits");  // static: [1,1,V]
}

ov::Tensor Qwen3TTSCodecEngine::infer_predictor_batch(const ov::Tensor& inputs_embeds, int64_t step) {
    OPENVINO_ASSERT(!m_predictor_static && m_predictor_batching_supported,
                    "Batched code predictor inference requires dynamic batch inputs and state");
    const ov::Shape input_shape = inputs_embeds.get_shape();
    OPENVINO_ASSERT(input_shape.size() == 3 && input_shape[0] > 1 && input_shape[2] == m_talker_hidden_size,
                    "Batched code predictor inputs_embeds must have shape [B>1, T, ",
                    m_talker_hidden_size,
                    "], got ",
                    input_shape);
    const size_t batch_size = input_shape[0];
    const size_t seq_len = input_shape[1];
    OPENVINO_ASSERT(seq_len > 0, "Batched code predictor received an empty sequence");

    m_talker_code_predictor.set_tensor("inputs_embeds", inputs_embeds);

    ov::Tensor attention_mask;
    if (seq_len == 1) {
        attention_mask = ov::Tensor(ov::element::f32,
                                    ov::Shape{batch_size, 1, 1, m_pred_position + 1});
        std::fill_n(attention_mask.data<float>(), attention_mask.get_size(), 0.0f);
    } else {
        attention_mask = ov::Tensor(ov::element::f32, ov::Shape{batch_size, 1, seq_len, seq_len});
        const ov::Tensor single_mask = make_causal_attention_mask_4d(seq_len);
        for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            std::copy_n(single_mask.data<const float>(),
                        single_mask.get_size(),
                        attention_mask.data<float>() + batch_idx * single_mask.get_size());
        }
    }
    m_talker_code_predictor.set_tensor("attention_mask", attention_mask);

    ov::Tensor position_ids(ov::element::i64, ov::Shape{batch_size, seq_len});
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (size_t pos_idx = 0; pos_idx < seq_len; ++pos_idx) {
            position_ids.data<int64_t>()[batch_idx * seq_len + pos_idx] =
                static_cast<int64_t>(m_pred_position + pos_idx);
        }
    }
    m_talker_code_predictor.set_tensor("position_ids", position_ids);

    ov::Tensor step_tensor = m_talker_code_predictor.get_tensor("step");
    const ov::Shape step_shape = step_tensor.get_shape();
    OPENVINO_ASSERT(step_shape.empty() || (step_shape.size() == 1 && step_shape[0] == 1),
                    "Unsupported code predictor step shape: ",
                    step_shape);
    step_tensor.data<int64_t>()[0] = step;

    if (m_predictor_has_beam_idx) {
        ov::Tensor beam_idx(ov::element::i32, ov::Shape{batch_size});
        for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            beam_idx.data<int32_t>()[batch_idx] = static_cast<int32_t>(batch_idx);
        }
        m_talker_code_predictor.set_tensor("beam_idx", beam_idx);
    }

    run_and_time([&] { m_talker_code_predictor.infer(); }, roles::CODE_PREDICTOR, m_perf_ms, m_perf_calls);
    m_pred_position += seq_len;
    return m_talker_code_predictor.get_tensor("logits");
}

ov::Tensor Qwen3TTSCodecEngine::infer_predictor_embedding(int64_t token_id, int64_t generation_step) {
    // Update pre-allocated id tensors in-place (tensors already set on infer request).
    m_pred_emb_ids.data<int64_t>()[0] = token_id;
    m_pred_emb_step.data<int64_t>()[0] = generation_step;

    run_and_time([&]{ m_talker_code_predictor_embedding.infer(); }, roles::CODE_PREDICTOR_EMBEDDING, m_perf_ms, m_perf_calls);
    // Return the request's own output tensor (no clone). Valid until the next
    // m_talker_code_predictor_embedding.infer(). Every caller consumes it (copies
    // or uses it as the predictor input) before the next embedding infer.
    return m_talker_code_predictor_embedding.get_output_tensor(0);
}

ov::Tensor Qwen3TTSCodecEngine::infer_predictor_embedding_seq(const std::vector<int64_t>& token_ids, int64_t generation_step) {
    if (token_ids.empty()) {
        return ov::Tensor(ov::element::f32, ov::Shape{1, 0, m_talker_hidden_size});
    }

    ov::Tensor out(ov::element::f32, ov::Shape{1, token_ids.size(), m_talker_hidden_size});
    for (size_t i = 0; i < token_ids.size(); ++i) {
        auto emb = infer_predictor_embedding(token_ids[i], generation_step);
        OPENVINO_ASSERT(emb.get_shape().size() == 3 && emb.get_shape()[0] == 1 && emb.get_shape()[1] == 1,
                        "Qwen3 predictor embedding must produce shape [1, 1, D]");
        OPENVINO_ASSERT(emb.get_shape()[2] == m_talker_hidden_size,
                        "Qwen3 predictor embedding hidden size mismatch. Got ",
                        emb.get_shape()[2],
                        ", expected ",
                        m_talker_hidden_size);
        std::copy_n(emb.data<const float>(), m_talker_hidden_size, out.data<float>() + i * m_talker_hidden_size);
    }
    return out;
}

int64_t Qwen3TTSCodecEngine::sample_token_from_logits(const ov::Tensor& logits,
                                                      const SpeechGenerationConfig& generation_config,
                                                      const std::vector<int64_t>& generated,
                                                      const std::vector<bool>& suppressed,
                                                      std::mt19937& rng,
                                                      size_t batch_index) const {
    const auto shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && batch_index < shape[0],
                    "Expected logits shape [B, T, V] with batch index < B, got ",
                    shape,
                    " and batch index ",
                    batch_index);
    OPENVINO_ASSERT(logits.get_element_type() == ov::element::f32, "Qwen3 logits must be f32 for sampling");
    const size_t vocab = shape[2];
    const float* ptr = logits.data<const float>();
    const size_t offset = (batch_index * shape[1] + shape[1] - 1) * vocab;

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

    return ov::genai::sample_multinomial(logits_view, 1, rng).front().m_index;
}

std::vector<int64_t> Qwen3TTSCodecEngine::generate_codec_groups(const ov::Tensor& past_hidden,
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

    auto sample_remaining_groups = [&](int64_t next,
                                       std::vector<int64_t>& generated,
                                       const std::vector<bool>& predictor_suppressed) {
        for (size_t g = 1; g < m_ids.num_code_groups - 1; ++g) {
            auto emb = infer_predictor_embedding(next, static_cast<int64_t>(g - 1));
            OPENVINO_ASSERT(emb.get_shape().size() == 3 && emb.get_shape()[1] == 1,
                            "Code predictor residual embedding must be a single token");
            auto lg = infer_predictor(emb, /*reset_state=*/false, /*step=*/static_cast<int64_t>(g));
            next = sample_token_from_logits(lg, predictor_config, generated, predictor_suppressed, rng);
            generated.push_back(next);
        }
    };

    auto append_and_pad_groups = [&](const std::vector<int64_t>& generated) {
        groups.insert(groups.end(), generated.begin(), generated.end());
        while (groups.size() < m_ids.num_code_groups) {
            groups.push_back(m_ids.codec_pad_id);
        }
    };

    if (!m_predictor_static) {
        std::vector<int64_t> generated;
        generated.reserve(m_ids.num_code_groups - 1);

        ov::Tensor first_id_hidden = infer_embedding(m_talker_embedding, first_codec_token);
        OPENVINO_ASSERT(first_id_hidden.get_shape().size() == 3 && first_id_hidden.get_shape()[1] == 1,
                        "Code predictor first-id embedding must be a single token");

        ov::Tensor prefill = concat_embed({past_hidden, first_id_hidden});
        auto logits = infer_predictor(prefill, /*reset_state=*/true, /*step=*/0);  // [1, T, V]
        const auto logits_shape = logits.get_shape();
        OPENVINO_ASSERT(logits_shape.size() == 3,
                "Expected code predictor logits shape [B, T, V], got rank ",
                logits_shape.size());
        std::vector<bool> predictor_suppressed(logits_shape[2], false);
        int64_t next = sample_token_from_logits(logits, predictor_config, generated, predictor_suppressed, rng);
        generated.push_back(next);

        sample_remaining_groups(next, generated, predictor_suppressed);
        append_and_pad_groups(generated);

        return groups;
    }

    // Residual codec generation against the static code predictor.
    //
    // The static model is single-token with an explicit fixed KV cache; infer_predictor
    // manages the host-side cache and absolute-position counter. We therefore feed the
    // 2-token prefill (talker past_hidden then the first-codec embedding) ONE TOKEN AT A
    // TIME instead of a single multi-token prefill, then continue one token per group.
    // Group `g` is predicted from the hidden state produced AFTER consuming
    // token (g-1)'s embedding.

    // === First embedding lookup ===
    ov::Tensor first_id_hidden = infer_embedding(m_talker_embedding, first_codec_token);

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
    for (size_t t = 0; t < prefill_tokens; ++t) {
        const bool is_last_ctx = (t + 1 == ph_shape[1]);  // last talker-context token
        const float* src = (t < ph_shape[1]) ? (ph_ptr + t * hidden) : hid_ptr;
        logits = feed_token(src, /*reset=*/t == 0);
        (void)is_last_ctx;
    }

    std::vector<int64_t> generated;
    generated.reserve(m_ids.num_code_groups - 1);

    // === Prefill sampling ===
    const auto logits_shape = logits.get_shape();
    OPENVINO_ASSERT(logits_shape.size() == 3,
                    "Expected code predictor logits shape [B, T, V], got rank ",
                    logits_shape.size());
    std::vector<bool> predictor_suppressed(logits_shape[2], false);
    int64_t next = sample_token_from_logits(logits, predictor_config, generated, predictor_suppressed, rng);
    generated.push_back(next);

    sample_remaining_groups(next, generated, predictor_suppressed);
    append_and_pad_groups(generated);

    return groups;
}

std::vector<std::vector<int64_t>> Qwen3TTSCodecEngine::generate_codec_groups_batch(
    const std::vector<CodecGroupBatchItem>& batch_items) {
    OPENVINO_ASSERT(!batch_items.empty(), "Code-group batch must contain at least one frame");

    if (batch_items.size() == 1 ||
        !qwen_cb_subtalker_batching_enabled() ||
        m_predictor_static ||
        !m_predictor_batching_supported) {
        std::vector<std::vector<int64_t>> groups;
        groups.reserve(batch_items.size());
        for (const CodecGroupBatchItem& item : batch_items) {
            OPENVINO_ASSERT(item.generation_config && item.rng,
                            "Code-group batch item must include generation config and RNG");
            groups.push_back(generate_codec_groups(item.past_hidden,
                                                   item.first_codec_token,
                                                   *item.generation_config,
                                                   *item.rng));
        }
        return groups;
    }

    const size_t batch_size = batch_items.size();
    const size_t hidden_size = m_talker_hidden_size;
    std::vector<SpeechGenerationConfig> predictor_configs;
    std::vector<std::vector<int64_t>> groups(batch_size);
    std::vector<std::vector<int64_t>> generated(batch_size);
    predictor_configs.reserve(batch_size);

    ov::Tensor prefill(ov::element::f32, ov::Shape{batch_size, 2, hidden_size});
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const CodecGroupBatchItem& item = batch_items[batch_idx];
        OPENVINO_ASSERT(item.generation_config && item.rng,
                        "Code-group batch item must include generation config and RNG");
        const ov::Shape past_shape = item.past_hidden.get_shape();
        OPENVINO_ASSERT(past_shape == ov::Shape({1, 1, hidden_size}),
                        "Batched code predictor past_hidden must have shape [1, 1, ",
                        hidden_size,
                        "], got ",
                        past_shape);

        SpeechGenerationConfig predictor_config = *item.generation_config;
        predictor_config.do_sample = item.generation_config->subtalker_dosample;
        predictor_config.top_k = item.generation_config->subtalker_top_k;
        predictor_config.top_p = item.generation_config->subtalker_top_p;
        predictor_config.temperature = item.generation_config->subtalker_temperature;
        predictor_config.repetition_penalty = 1.0f;
        predictor_configs.push_back(std::move(predictor_config));

        groups[batch_idx].reserve(m_ids.num_code_groups);
        groups[batch_idx].push_back(item.first_codec_token);
        generated[batch_idx].reserve(m_ids.num_code_groups - 1);

        const ov::Tensor first_id_hidden = infer_embedding(m_talker_embedding, item.first_codec_token);
        OPENVINO_ASSERT(first_id_hidden.get_shape() == ov::Shape({1, 1, hidden_size}),
                        "Code predictor first-id embedding must have shape [1, 1, ",
                        hidden_size,
                        "], got ",
                        first_id_hidden.get_shape());
        std::copy_n(item.past_hidden.data<const float>(),
                    hidden_size,
                    prefill.data<float>() + batch_idx * 2 * hidden_size);
        std::copy_n(first_id_hidden.data<const float>(),
                    hidden_size,
                    prefill.data<float>() + (batch_idx * 2 + 1) * hidden_size);
    }

    reset_predictor_state();
    ov::Tensor logits = infer_predictor_batch(prefill, /*step=*/0);
    const ov::Shape logits_shape = logits.get_shape();
    OPENVINO_ASSERT(logits_shape.size() == 3 && logits_shape[0] == batch_size,
                    "Batched code predictor logits must have shape [B, T, V], got ",
                    logits_shape);
    const std::vector<bool> predictor_suppressed(logits_shape[2], false);

    std::vector<int64_t> next_tokens(batch_size);
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        next_tokens[batch_idx] = sample_token_from_logits(logits,
                                                          predictor_configs[batch_idx],
                                                          generated[batch_idx],
                                                          predictor_suppressed,
                                                          *batch_items[batch_idx].rng,
                                                          batch_idx);
        generated[batch_idx].push_back(next_tokens[batch_idx]);
        groups[batch_idx].push_back(next_tokens[batch_idx]);
    }

    for (size_t group_idx = 1; group_idx < m_ids.num_code_groups - 1; ++group_idx) {
        ov::Tensor predictor_inputs(ov::element::f32, ov::Shape{batch_size, 1, hidden_size});
        for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            const ov::Tensor embedding = infer_predictor_embedding(next_tokens[batch_idx],
                                                                   static_cast<int64_t>(group_idx - 1));
            OPENVINO_ASSERT(embedding.get_shape() == ov::Shape({1, 1, hidden_size}),
                            "Code predictor residual embedding must have shape [1, 1, ",
                            hidden_size,
                            "], got ",
                            embedding.get_shape());
            std::copy_n(embedding.data<const float>(),
                        hidden_size,
                        predictor_inputs.data<float>() + batch_idx * hidden_size);
        }

        logits = infer_predictor_batch(predictor_inputs, static_cast<int64_t>(group_idx));
        const ov::Shape step_logits_shape = logits.get_shape();
        OPENVINO_ASSERT(step_logits_shape.size() == 3 && step_logits_shape[0] == batch_size,
                        "Batched code predictor logits must have shape [B, T, V], got ",
                        step_logits_shape);
        for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            next_tokens[batch_idx] = sample_token_from_logits(logits,
                                                              predictor_configs[batch_idx],
                                                              generated[batch_idx],
                                                              predictor_suppressed,
                                                              *batch_items[batch_idx].rng,
                                                              batch_idx);
            generated[batch_idx].push_back(next_tokens[batch_idx]);
            groups[batch_idx].push_back(next_tokens[batch_idx]);
        }
    }

    for (std::vector<int64_t>& frame_groups : groups) {
        while (frame_groups.size() < m_ids.num_code_groups) {
            frame_groups.push_back(m_ids.codec_pad_id);
        }
    }

    if (m_predictor_has_beam_idx) {
        ov::Tensor beam_idx(ov::element::i32, ov::Shape{1});
        beam_idx.data<int32_t>()[0] = 0;
        m_talker_code_predictor.set_tensor("beam_idx", beam_idx);
    }

    return groups;
}

std::vector<float> Qwen3TTSCodecEngine::decode_speech_tokenizer(const std::vector<int64_t>& codes) {
    if (codes.empty()) {
        return {};
    }

    OPENVINO_ASSERT(codes.size() % m_decoder_num_quantizers == 0,
                    "Qwen codec tensor is malformed: expected [T, num_quantizers] flattening");

    const size_t num_frames = codes.size() / m_decoder_num_quantizers;

    if (!m_decoder_static) {
        // Dynamic model: feed the whole [1, num_quantizers, num_frames] sequence in one shot
        // and take the whole output, exactly like optimum-intel/native HF. The chunked,
        // padded-window, offset-trimmed path below exists only for the NPU static decoder.
        ov::Tensor audio_codes(ov::element::i64, ov::Shape{1, m_decoder_num_quantizers, num_frames});
        int64_t* dst = audio_codes.data<int64_t>();
        for (size_t t = 0; t < num_frames; ++t) {
            for (size_t q = 0; q < m_decoder_num_quantizers; ++q) {
                dst[q * num_frames + t] = codes[t * m_decoder_num_quantizers + q];
            }
        }

        m_speech_tokenizer_decoder.set_tensor("audio_codes", audio_codes);
        run_and_time([&]{ m_speech_tokenizer_decoder.infer(); }, roles::SPEECH_TOKENIZER_DECODER, m_perf_ms, m_perf_calls);

        const auto out = m_speech_tokenizer_decoder.get_tensor("waveform");
        OPENVINO_ASSERT(out.get_element_type() == ov::element::f32,
                        "Speech tokenizer decoder output is expected to be f32");
        const float* out_ptr = out.data<const float>();
        return std::vector<float>(out_ptr, out_ptr + out.get_size());
    }

    std::vector<float> chunks_audio;
    chunks_audio.reserve(num_frames * m_decoder_upsample);

    size_t start = 0;
    while (start < num_frames) {
        const size_t end = std::min(start + static_cast<size_t>(DECODER_CHUNK_SIZE), num_frames);
        const size_t ctx = (start > static_cast<size_t>(DECODER_LEFT_CONTEXT)) ? static_cast<size_t>(DECODER_LEFT_CONTEXT) : start;
        const size_t chunk_start = start - ctx;
        const size_t chunk_len = end - chunk_start;

        const size_t src_offset = chunk_start * m_decoder_num_quantizers;
        const size_t padded_len = static_cast<size_t>(DECODER_WINDOW_LEN);
        OPENVINO_ASSERT(chunk_len <= padded_len, "Speech tokenizer decoder chunk length exceeds the padded trace length");
        ov::Tensor audio_codes;

        // New layout [B,Q,T]: always allocate and transpose from frame-major source.
        audio_codes = ov::Tensor(ov::element::i64, ov::Shape{1, m_decoder_num_quantizers, padded_len});
        int64_t* dst = audio_codes.data<int64_t>();
        std::fill_n(dst, m_decoder_num_quantizers * padded_len, int64_t{0});
        const int64_t* src = codes.data() + src_offset;
        for (size_t t = 0; t < chunk_len; ++t) {
            for (size_t q = 0; q < m_decoder_num_quantizers; ++q) {
                dst[q * padded_len + t] = src[t * m_decoder_num_quantizers + q];
            }
        }

        // Fill the rest of the window with real look-ahead codes (if available) instead of
        // zeros: a real-vs-zero-padded-code boundary confuses the decoder near that edge, so
        // only the true tail of the whole sequence (no more codes left at all) is ever padded.
        const size_t lookahead_len = std::min(padded_len - chunk_len, num_frames - end);
        if (lookahead_len > 0) {
            const int64_t* lookahead_src = codes.data() + end * m_decoder_num_quantizers;
            for (size_t t = 0; t < lookahead_len; ++t) {
                for (size_t q = 0; q < m_decoder_num_quantizers; ++q) {
                    dst[q * padded_len + (chunk_len + t)] = lookahead_src[t * m_decoder_num_quantizers + q];
                }
            }
        }

        m_speech_tokenizer_decoder.set_tensor("audio_codes", audio_codes);
        run_and_time([&]{ m_speech_tokenizer_decoder.infer(); }, roles::SPEECH_TOKENIZER_DECODER, m_perf_ms, m_perf_calls);
        const auto out = m_speech_tokenizer_decoder.get_tensor("waveform");
        OPENVINO_ASSERT(out.get_element_type() == ov::element::f32,
                        "Speech tokenizer decoder output is expected to be f32");

        const float* out_ptr = out.data<const float>();
        const size_t out_size = out.get_size();
        const size_t valid_samples = chunk_len * m_decoder_upsample;
        OPENVINO_ASSERT(valid_samples <= out_size,
                        "Speech tokenizer decoder produced fewer samples than the chunk's real code length");
        const size_t context_samples = ctx * m_decoder_upsample;

        // Split the ctx-frame overlap into three zones so each chunk gets a region where it is
        // fully trusted, mirrored around a middle blend zone: the first (oldest) left_drop frames
        // keep the previous chunk's decode untouched (this chunk's decode of them has too little
        // left-side context), the last (nearest the boundary) right_drop frames take this chunk's
        // decode outright (the previous chunk's decode of them had no right-side context), and
        // blend_frames in between are linearly crossfaded.
        const size_t blend_frames = std::min<size_t>(static_cast<size_t>(DECODER_BLEND_FRAMES), ctx);
        const size_t leftover_frames = ctx - blend_frames;
        const size_t left_drop_frames = (leftover_frames + 1) / 2;
        const size_t right_drop_frames = leftover_frames - left_drop_frames;
        const size_t blend_samples = blend_frames * m_decoder_upsample;
        const size_t right_drop_samples = right_drop_frames * m_decoder_upsample;

        if (blend_samples > 0 && chunks_audio.size() >= blend_samples + right_drop_samples) {
            const size_t blend_start_local = context_samples - blend_samples - right_drop_samples;
            float* blend_tail = chunks_audio.data() + (chunks_audio.size() - blend_samples - right_drop_samples);
            for (size_t i = 0; i < blend_samples; ++i) {
                // Closed-interval ramp: the last blended sample is fully this chunk's own
                // regeneration, the first is fully the previous chunk's already-appended output.
                const float w = (blend_samples > 1) ? static_cast<float>(i) / static_cast<float>(blend_samples - 1)
                                                     : 1.0f;
                blend_tail[i] = blend_tail[i] * (1.0f - w) + out_ptr[blend_start_local + i] * w;
            }
        }
        if (right_drop_samples > 0 && chunks_audio.size() >= right_drop_samples) {
            float* new_tail = chunks_audio.data() + (chunks_audio.size() - right_drop_samples);
            std::copy_n(out_ptr + (context_samples - right_drop_samples), right_drop_samples, new_tail);
        }

        chunks_audio.insert(chunks_audio.end(),
                            out_ptr + static_cast<std::ptrdiff_t>(context_samples),
                            out_ptr + static_cast<std::ptrdiff_t>(valid_samples));

        start = end;
    }

    return chunks_audio;
}

std::vector<bool> Qwen3TTSCodecEngine::compute_suppress_tokens() const {
    std::vector<bool> suppress_tokens(m_ids.talker_vocab_size, false);
    const size_t suppress_begin = m_ids.talker_vocab_size > 1024 ? (m_ids.talker_vocab_size - 1024) : 0;
    for (size_t i = suppress_begin; i < m_ids.talker_vocab_size; ++i) {
        if (static_cast<int64_t>(i) != m_ids.codec_eos_token_id) {
            suppress_tokens[i] = true;
        }
    }
    return suppress_tokens;
}

std::vector<int64_t> Qwen3TTSCodecEngine::make_speaker_and_codec_prefill_ids(const std::string& language,
                                                                             const std::string& speaker) const {
    int64_t language_id = -1;
    auto lang_it = m_ids.codec_language_id.find(language);
    if (language != "auto" && lang_it != m_ids.codec_language_id.end()) {
        language_id = lang_it->second;
    }

    if ((language == "auto" || language == "chinese") && !speaker.empty()) {
        auto spk_dialect = m_ids.spk_is_dialect.find(speaker);
        if (spk_dialect != m_ids.spk_is_dialect.end()) {
            auto dialect_it = m_ids.codec_language_id.find(spk_dialect->second);
            if (dialect_it != m_ids.codec_language_id.end()) {
                language_id = dialect_it->second;
            }
        }
    }

    if (language_id == -1) {
        return {m_ids.codec_nothink_id, m_ids.codec_think_bos_id, m_ids.codec_think_eos_id};
    }
    return {m_ids.codec_think_id, m_ids.codec_think_bos_id, language_id, m_ids.codec_think_eos_id};
}

Qwen3TTSCodecEngine::SpecialEmbeds Qwen3TTSCodecEngine::infer_special_embeds() {
    auto special_projected = infer_embedding_seq(m_talker_text_embedding,
                                                {m_ids.tts_bos_token_id, m_ids.tts_eos_token_id, m_ids.tts_pad_token_id});
    OPENVINO_ASSERT(special_projected.get_shape().size() == 3 && special_projected.get_shape()[0] == 1 &&
                        special_projected.get_shape()[1] == 3,
                    "Special token embedding must have shape [1, 3, H], got ",
                    shape_to_string(special_projected.get_shape()));

    SpecialEmbeds embeds;
    embeds.hidden = special_projected.get_shape()[2];
    embeds.bos = ov::Tensor(special_projected, ov::Coordinate{0, 0, 0}, ov::Coordinate{1, 1, embeds.hidden});
    embeds.eos = ov::Tensor(special_projected, ov::Coordinate{0, 1, 0}, ov::Coordinate{1, 2, embeds.hidden});
    embeds.pad = ov::Tensor(special_projected, ov::Coordinate{0, 2, 0}, ov::Coordinate{1, 3, embeds.hidden});
    return embeds;
}

ov::Tensor Qwen3TTSCodecEngine::prepend_instruct_embed(ov::Tensor talker_prefill, const std::string& instruct) {
    if (instruct.empty()) {
        return talker_prefill;
    }
    const std::string instruct_text = "<|im_start|>user\n" + instruct + "<|im_end|>\n";
    auto instruct_ids = m_tokenizer.encode(instruct_text).input_ids;
    const int64_t* instr = instruct_ids.data<const int64_t>();
    const size_t instr_len = instruct_ids.get_shape()[1];
    auto instruct_embed = infer_embedding_seq(m_talker_text_embedding, std::vector<int64_t>(instr, instr + instr_len));
    return concat_embed({instruct_embed, talker_prefill});
}

std::string Qwen3TTSCodecEngine::normalize_text_language(const std::string& language) const {
    if (language.empty()) {
        return "auto";
    }

    const std::string normalized_language = to_lower(language);
    if (normalized_language == "auto") {
        return normalized_language;
    }

    if (is_dialect_language(normalized_language) ||
        m_ids.codec_language_id.find(normalized_language) == m_ids.codec_language_id.end()) {
        OPENVINO_THROW("Unsupported language: ['",
                       language,
                       "']. Supported: [",
                       format_supported_languages(m_ids.codec_language_id),
                       "]");
    }

    return normalized_language;
}

std::string Qwen3TTSCodecEngine::normalize_speaker(const std::string& speaker) const {
    return to_lower(speaker);
}

Qwen3TTSCodecEngine::VoiceClonePrefillResult Qwen3TTSCodecEngine::build_voice_clone_prefill(
    const std::string& text,
    const std::string& ref_text,
    const ov::Tensor& speaker_embedding,
    const ov::Tensor& ref_codec_ids,
    const SpeechGenerationConfig& generation_config) {
    OPENVINO_ASSERT(is_base_model(), "Qwen3 voice-clone prompt generation is supported only for Base models");
    OPENVINO_ASSERT(!ref_text.empty(), "Qwen3 voice-clone ICL mode requires a non-empty reference transcript");
    OPENVINO_ASSERT(speaker_embedding, "Qwen3 voice-clone ICL mode requires a speaker embedding");
    OPENVINO_ASSERT(ref_codec_ids, "Qwen3 voice-clone ICL mode requires reference codec ids");
    OPENVINO_ASSERT(ref_codec_ids.get_element_type() == ov::element::i64,
                    "Qwen3 voice-clone ref_code tensor must have element type i64");
    OPENVINO_ASSERT(ref_codec_ids.get_shape().size() == 2 || ref_codec_ids.get_shape().size() == 3,
                    "Qwen3 voice-clone prompt expects ref_code shaped [T, G] or [1, T, G]");

    const std::string language = normalize_text_language(generation_config.language);
    const std::string speaker = normalize_speaker(generation_config.speaker);

    const std::string ref_turn = "<|im_start|>assistant\n" + ref_text + "<|im_end|>\n";
    const std::string target_turn = "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";

    const ov::Tensor ref_input_ids = tokenizer().encode(ref_turn).input_ids;
    const ov::Tensor target_input_ids = tokenizer().encode(target_turn).input_ids;
    OPENVINO_ASSERT(ref_input_ids.get_element_type() == ov::element::i64 &&
                        target_input_ids.get_element_type() == ov::element::i64,
                    "Qwen3-TTS tokenizer must produce i64 input_ids tensor");
    OPENVINO_ASSERT(ref_input_ids.get_shape().size() == 2 && ref_input_ids.get_shape()[0] == 1 &&
                        target_input_ids.get_shape().size() == 2 && target_input_ids.get_shape()[0] == 1,
                    "Expected input_ids shape [1, T]");

    const size_t ref_ids_len = ref_input_ids.get_shape()[1];
    const size_t target_ids_len = target_input_ids.get_shape()[1];
    OPENVINO_ASSERT(ref_ids_len > 5, "Qwen3 voice-clone reference transcript tokenized too short");
    OPENVINO_ASSERT(target_ids_len > 8, "Qwen3 voice-clone target text tokenized too short");
    const int64_t* ref_ids = ref_input_ids.data<const int64_t>();
    const int64_t* target_ids = target_input_ids.data<const int64_t>();

    const std::vector<int64_t> role_ids(target_ids, target_ids + 3);
    std::vector<int64_t> icl_content_ids;
    icl_content_ids.reserve((ref_ids_len - 5) + (target_ids_len - 8));
    icl_content_ids.insert(icl_content_ids.end(), ref_ids + 3, ref_ids + (ref_ids_len - 2));
    icl_content_ids.insert(icl_content_ids.end(), target_ids + 3, target_ids + (target_ids_len - 5));

    const auto speaker_and_codec_prefill = make_speaker_and_codec_prefill_ids(language, speaker);
    const ov::Tensor codec_prefill_embed0 = infer_embedding_seq(m_talker_embedding, speaker_and_codec_prefill);
    const ov::Tensor codec_pad_embed = infer_embedding_seq(m_talker_embedding, {m_ids.codec_pad_id});
    const ov::Tensor codec_bos_embed = infer_embedding_seq(m_talker_embedding, {m_ids.codec_bos_id});
    const SpecialEmbeds special_embeds = infer_special_embeds();
    const size_t hidden = special_embeds.hidden;

    const ov::Tensor normalized_speaker_embedding =
        normalize_external_speaker_embedding(speaker_embedding, codec_prefill_embed0.get_shape()[2]);
    const ov::Tensor codec_input_no_bos = concat_embed({codec_prefill_embed0,
                                                        normalized_speaker_embedding,
                                                        codec_pad_embed});
    OPENVINO_ASSERT(codec_input_no_bos.get_shape()[2] == hidden,
                    "Qwen3 talker embedding hidden size (", codec_input_no_bos.get_shape()[2],
                    ") must match talker text embedding hidden size (", hidden, ")");

    const ov::Shape ref_code_shape = ref_codec_ids.get_shape();
    const bool ref_code_rank3 = ref_code_shape.size() == 3;
    OPENVINO_ASSERT(!ref_code_rank3 || ref_code_shape[0] == 1,
                    "Qwen3 voice-clone ref_code rank-3 tensor must have batch dimension 1");
    const size_t ref_len = ref_code_rank3 ? ref_code_shape[1] : ref_code_shape[0];
    const size_t num_code_groups = ref_code_rank3 ? ref_code_shape[2] : ref_code_shape[1];
    OPENVINO_ASSERT(ref_len > 0, "Qwen3 voice-clone ref_code must contain at least one frame");
    OPENVINO_ASSERT(num_code_groups == m_ids.num_code_groups,
                    "Qwen3 voice-clone ref_code group count mismatch. Expected ",
                    m_ids.num_code_groups,
                    ", got ",
                    num_code_groups);

    ov::Tensor ref_code_embed(ov::element::f32, ov::Shape{1, ref_len, hidden});
    std::fill_n(ref_code_embed.data<float>(), ref_code_embed.get_size(), 0.0f);
    const int64_t* codes = ref_codec_ids.data<const int64_t>();
    for (size_t group = 0; group < num_code_groups; ++group) {
        std::vector<int64_t> group_ids(ref_len);
        for (size_t frame = 0; frame < ref_len; ++frame) {
            group_ids[frame] = codes[frame * num_code_groups + group];
        }

        const ov::Tensor group_embed = group == 0
            ? infer_embedding_seq(m_talker_embedding, group_ids)
            : infer_predictor_embedding_seq(group_ids, static_cast<int64_t>(group - 1));
        OPENVINO_ASSERT(group_embed.get_shape() == ov::Shape({1, ref_len, hidden}),
                        "Qwen3 voice-clone ref_code group embedding shape mismatch. Expected [1,",
                        ref_len,
                        ",",
                        hidden,
                        "], got ",
                        group_embed.get_shape());
        const float* group_ptr = group_embed.data<const float>();
        float* ref_ptr = ref_code_embed.data<float>();
        for (size_t i = 0; i < ref_len * hidden; ++i) {
            ref_ptr[i] += group_ptr[i];
        }
    }

    const ov::Tensor role_embed = infer_embedding_seq(m_talker_text_embedding, role_ids);
    const ov::Tensor content_embed = infer_embedding_seq(m_talker_text_embedding, icl_content_ids);
    const ov::Tensor text_embed_with_eos = concat_embed({content_embed, special_embeds.eos});
    const size_t text_eos_len = text_embed_with_eos.get_shape()[1];

    const size_t codec_input_len = codec_input_no_bos.get_shape()[1];
    ov::Tensor codec_input_part(ov::element::f32, ov::Shape{1, codec_input_len, hidden});
    for (size_t i = 0; i < codec_input_len; ++i) {
        const float* source = codec_input_no_bos.data<const float>() + i * hidden;
        float* destination = codec_input_part.data<float>() + i * hidden;
        const float* blend = (i == codec_input_len - 1)
            ? special_embeds.bos.data<const float>()
            : special_embeds.pad.data<const float>();
        for (size_t h = 0; h < hidden; ++h) {
            destination[h] = source[h] + blend[h];
        }
    }

    const ov::Tensor codec_bos_plus_ref = concat_embed({codec_bos_embed, ref_code_embed});
    const size_t codec_side_len = codec_bos_plus_ref.get_shape()[1];
    ov::Tensor icl_input_embed;
    ov::Tensor trailing_text_hidden;
    if (generation_config.non_streaming_mode) {
        const ov::Tensor codec_pad_for_text = infer_embedding_seq(
            m_talker_embedding,
            std::vector<int64_t>(text_eos_len, m_ids.codec_pad_id));
        ov::Tensor text_side(ov::element::f32, ov::Shape{1, text_eos_len, hidden});
        for (size_t i = 0; i < text_side.get_size(); ++i) {
            text_side.data<float>()[i] = text_embed_with_eos.data<const float>()[i] +
                                         codec_pad_for_text.data<const float>()[i];
        }

        ov::Tensor codec_side(ov::element::f32, ov::Shape{1, codec_side_len, hidden});
        for (size_t i = 0; i < codec_side_len; ++i) {
            const float* source = codec_bos_plus_ref.data<const float>() + i * hidden;
            float* destination = codec_side.data<float>() + i * hidden;
            const float* pad = special_embeds.pad.data<const float>();
            for (size_t h = 0; h < hidden; ++h) {
                destination[h] = source[h] + pad[h];
            }
        }

        icl_input_embed = concat_embed({text_side, codec_side});
    } else if (text_eos_len > codec_side_len) {
        ov::Tensor overlap(ov::element::f32, ov::Shape{1, codec_side_len, hidden});
        for (size_t i = 0; i < codec_side_len; ++i) {
            const float* text_ptr = text_embed_with_eos.data<const float>() + i * hidden;
            const float* codec_ptr = codec_bos_plus_ref.data<const float>() + i * hidden;
            float* destination = overlap.data<float>() + i * hidden;
            for (size_t h = 0; h < hidden; ++h) {
                destination[h] = text_ptr[h] + codec_ptr[h];
            }
        }
        icl_input_embed = overlap;
        trailing_text_hidden = ov::Tensor(text_embed_with_eos,
                                          ov::Coordinate{0, codec_side_len, 0},
                                          ov::Coordinate{1, text_eos_len, hidden});
    } else {
        ov::Tensor text_padded(ov::element::f32, ov::Shape{1, codec_side_len, hidden});
        if (text_eos_len > 0) {
            std::copy_n(text_embed_with_eos.data<const float>(),
                        text_embed_with_eos.get_size(),
                        text_padded.data<float>());
        }
        for (size_t i = text_eos_len; i < codec_side_len; ++i) {
            std::copy_n(special_embeds.pad.data<const float>(),
                        hidden,
                        text_padded.data<float>() + i * hidden);
        }

        ov::Tensor overlap(ov::element::f32, ov::Shape{1, codec_side_len, hidden});
        for (size_t i = 0; i < codec_side_len; ++i) {
            const float* text_ptr = text_padded.data<const float>() + i * hidden;
            const float* codec_ptr = codec_bos_plus_ref.data<const float>() + i * hidden;
            float* destination = overlap.data<float>() + i * hidden;
            for (size_t h = 0; h < hidden; ++h) {
                destination[h] = text_ptr[h] + codec_ptr[h];
            }
        }
        icl_input_embed = overlap;
        trailing_text_hidden = special_embeds.pad;
    }

    VoiceClonePrefillResult result;
    result.prefill.talker_prefill = prepend_instruct_embed(
        concat_embed({role_embed, codec_input_part, icl_input_embed}), generation_config.instruct);
    result.prefill.tts_pad = special_embeds.pad;
    result.prefill.trailing_text_hidden = trailing_text_hidden;
    result.ref_codec_ids = ref_codec_ids;
    return result;
}

std::vector<float> Qwen3TTSCodecEngine::decode_speech_tokenizer_with_reference(
    const ov::Tensor& ref_codec_ids,
    const std::vector<int64_t>& generated_codes) {
    OPENVINO_ASSERT(ref_codec_ids.get_element_type() == ov::element::i64,
                    "Qwen3 Base ref_codec_ids tensor must have element type i64");
    const ov::Shape ref_shape = ref_codec_ids.get_shape();
    OPENVINO_ASSERT(ref_shape.size() == 2 || ref_shape.size() == 3,
                    "Qwen3 Base ref_codec_ids tensor is expected to have shape [T, G] or [1, T, G]");
    const bool rank3 = ref_shape.size() == 3;
    OPENVINO_ASSERT(!rank3 || ref_shape[0] == 1,
                    "Qwen3 Base ref_codec_ids rank-3 tensor must have batch dimension 1");

    const size_t ref_frames = rank3 ? ref_shape[1] : ref_shape[0];
    const size_t ref_groups = rank3 ? ref_shape[2] : ref_shape[1];
    OPENVINO_ASSERT(ref_groups == m_ids.num_code_groups,
                    "Qwen3 Base ref_codec_ids group count mismatch. Expected ",
                    m_ids.num_code_groups,
                    ", got ",
                    ref_groups);

    std::vector<int64_t> codes;
    const int64_t* ref_ptr = ref_codec_ids.data<const int64_t>();
    codes.reserve(ref_frames * ref_groups + generated_codes.size());
    for (size_t frame = 0; frame < ref_frames; ++frame) {
        for (size_t group = 0; group < ref_groups; ++group) {
            codes.push_back(ref_ptr[frame * ref_groups + group]);
        }
    }
    codes.insert(codes.end(), generated_codes.begin(), generated_codes.end());

    std::vector<float> waveform = decode_speech_tokenizer(codes);
    const size_t generated_frames = generated_codes.size() / m_ids.num_code_groups;
    const size_t total_frames = ref_frames + generated_frames;
    const size_t cut_samples = static_cast<size_t>(
        (static_cast<double>(ref_frames) / static_cast<double>(std::max<size_t>(total_frames, 1))) *
        static_cast<double>(waveform.size()));
    if (cut_samples >= waveform.size()) {
        waveform.clear();
    } else {
        waveform.erase(waveform.begin(), waveform.begin() + static_cast<std::ptrdiff_t>(cut_samples));
    }
    return waveform;
}

Qwen3TTSCodecEngine::PrefillResult Qwen3TTSCodecEngine::build_prefill(const ov::Tensor& input_ids_tensor,
                                                                      const ov::Tensor& effective_speaker_embedding,
                                                                      const SpeechGenerationConfig& generation_config,
                                                                      const std::string& language,
                                                                      const std::string& speaker,
                                                                      bool base_model) {
    OPENVINO_ASSERT(input_ids_tensor.get_element_type() == ov::element::i64,
                    "Qwen3-TTS tokenizer must produce i64 input_ids tensor");

    const auto input_shape = input_ids_tensor.get_shape();
    OPENVINO_ASSERT(input_shape.size() == 2 && input_shape[0] == 1, "Expected input_ids shape [1, T]");
    const size_t input_len = input_shape[1];
    const int64_t* input_ids = input_ids_tensor.data<const int64_t>();

    auto speaker_and_codec_prefill = make_speaker_and_codec_prefill_ids(language, speaker);
    auto codec_prefill_embed0 = infer_embedding_seq(m_talker_embedding, speaker_and_codec_prefill);
    auto codec_prefill_embed1 = infer_embedding_seq(m_talker_embedding, {m_ids.codec_pad_id, m_ids.codec_bos_id});

    int64_t speaker_token_id = -1;
    if (!base_model && !speaker.empty()) {
        auto spk_it = m_ids.spk_id.find(speaker);
        if (spk_it != m_ids.spk_id.end()) {
            speaker_token_id = spk_it->second;
        } else if (m_tts_model_type == "custom_voice") {
            OPENVINO_THROW("Unsupported Qwen3 CustomVoice speaker '",
                           speaker,
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

    auto special_embeds = infer_special_embeds();
    const size_t hidden = special_embeds.hidden;
    const ov::Tensor& tts_bos = special_embeds.bos;
    const ov::Tensor& tts_eos = special_embeds.eos;
    const ov::Tensor& tts_pad = special_embeds.pad;

    ov::Tensor codec_input_embedding;
    if (has_speaker_embed) {
        codec_input_embedding = concat_embed({codec_prefill_embed0, speaker_embed, codec_prefill_embed1});
    } else {
        codec_input_embedding = concat_embed({codec_prefill_embed0, codec_prefill_embed1});
    }
    OPENVINO_ASSERT(codec_input_embedding.get_shape()[2] == hidden,
                    "Qwen3 talker embedding hidden size (", codec_input_embedding.get_shape()[2],
                    ") must match talker text embedding hidden size (", hidden, ")");

    const std::vector<int64_t> role_tokens(input_ids, input_ids + std::min<size_t>(3, input_len));
    auto role_embed = infer_embedding_seq(m_talker_text_embedding, role_tokens);

    const size_t codec_len = codec_input_embedding.get_shape()[1];
    OPENVINO_ASSERT(codec_len >= 2, "codec_input_embedding must contain at least codec_pad_id and codec_bos_id");
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

    auto text_embed_full = infer_embedding_seq(
        m_talker_text_embedding,
        std::vector<int64_t>(input_ids + std::min<size_t>(3, input_len), input_ids + input_len));

    // Match upstream qwen_tts/core/models/modeling_qwen3_tts.py, which uses
    // text span slicing input_id[:, 3:-5] for this target turn. For short prompts,
    // that slice is empty, so keep trimmed_len at zero instead of preserving control tokens.
    size_t text_tokens_len = text_embed_full.get_shape()[1];
    size_t trimmed_len = text_tokens_len > 5 ? text_tokens_len - 5 : 0;
    ov::Tensor text_embed_trimmed = (trimmed_len > 0)
        ? ov::Tensor(text_embed_full, ov::Coordinate{0, 0, 0}, ov::Coordinate{1, trimmed_len, hidden})
        : ov::Tensor(ov::element::f32, ov::Shape{1, 0, hidden});

    PrefillResult result;
    result.tts_pad = tts_pad;
    if (generation_config.non_streaming_mode) {
        ov::Tensor text_with_eos = concat_embed({text_embed_trimmed, tts_eos});

        auto codec_pad_for_text = infer_embedding_seq(m_talker_embedding,
                                                      std::vector<int64_t>(trimmed_len + 1, m_ids.codec_pad_id));
        ov::Tensor text_side(ov::element::f32, ov::Shape{1, trimmed_len + 1, hidden});
        for (size_t i = 0; i < (trimmed_len + 1) * hidden; ++i) {
            text_side.data<float>()[i] = text_with_eos.data<const float>()[i] + codec_pad_for_text.data<const float>()[i];
        }

        auto codec_bos_embed = infer_embedding(m_talker_embedding, m_ids.codec_bos_id);
        ov::Tensor pad_plus_codec_bos(ov::element::f32, ov::Shape{1, 1, hidden});
        for (size_t j = 0; j < hidden; ++j) {
            pad_plus_codec_bos.data<float>()[j] = tts_pad.data<const float>()[j] + codec_bos_embed.data<const float>()[j];
        }

        result.talker_prefill = concat_embed({role_embed, left_codec_sum, text_side, pad_plus_codec_bos});
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
        result.talker_prefill = concat_embed({role_embed, left_codec_sum, first_text_plus_last_codec});

        const size_t trailing_text_tokens = (trimmed_len > 1 ? (trimmed_len - 1) : 0);
        ov::Tensor trailing_text_without_eos(ov::element::f32, ov::Shape{1, trailing_text_tokens, hidden});
        if (trailing_text_tokens > 0) {
            const float* src = text_embed_trimmed.data<const float>() + hidden;
            std::copy_n(src,
                        trailing_text_without_eos.get_size(),
                        trailing_text_without_eos.data<float>());
        }
        result.trailing_text_hidden = concat_embed({trailing_text_without_eos, tts_eos});
    }

    result.talker_prefill = prepend_instruct_embed(result.talker_prefill, generation_config.instruct);
    return result;
}

void Qwen3TTSCodecEngine::perf_print_and_reset(int64_t talker_prefill_tokens) const {
    if (!qwen_perf_enabled() || m_perf_ms.empty()) {
        m_perf_ms.clear();
        m_perf_calls.clear();
        return;
    }
    // Sort by descending total time so the hot path appears first.
    std::vector<std::pair<std::string, double>> entries(m_perf_ms.begin(), m_perf_ms.end());
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    double total_accumulated_ms = 0.0;
    for (const auto& [_, total_ms] : entries) {
        total_accumulated_ms += total_ms;
    }
    std::cout << "[QWEN3_TTS_PERF] Per-component inference timing (ms):" << std::endl;
    for (const auto& [name, total_ms] : entries) {
        const int64_t count = m_perf_calls.count(name) ? m_perf_calls.at(name) : 1;
        const double avg_ms = total_ms / static_cast<double>(count);
        const std::string device = m_perf_device.count(name) ? m_perf_device.at(name) : "unknown";
        const std::string extra =
            (name == TALKER_PREFILL_PERF_NAME)
                ? (", prefill_tokens=" + std::to_string(talker_prefill_tokens))
                : std::string{};
        std::cout << "[QWEN3_TTS_PERF]   "
                  << std::left << std::setw(30) << name << " [" << std::setw(10) << device << "] : "
                  << std::fixed << std::setprecision(1) << std::setw(9) << total_ms << " ms"
                  << "  (x" << count
                  << ", avg " << std::setw(7) << avg_ms << " ms"
                  << extra << ")" << std::endl;
    }
    std::cout << "[QWEN3_TTS_PERF] Total accumulated component time : "
              << std::fixed << std::setprecision(1) << total_accumulated_ms << " ms" << std::endl;
    m_perf_ms.clear();
    m_perf_calls.clear();
}

}  // namespace genai
}  // namespace ov
