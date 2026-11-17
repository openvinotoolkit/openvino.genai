// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset15.hpp>

#include "utils.hpp"

// Shared, stateless helpers used by both the SDPA (Qwen3TTSImpl) and continuous-batching
// (Qwen3TTSContinuousBatchingImpl) Qwen3-TTS drivers, and by Qwen3TTSCodecEngine.
namespace ov {
namespace genai {
namespace qwen3_tts {

constexpr const char* TEXT_EMBEDDINGS_NAME = "openvino_text_embeddings.xml";  // projection baked into rows
constexpr const char* CODEC_DECODER_NAME = "openvino_codec_decoder.xml";      // [B,Q,T] layout, waveform output
constexpr const char* SPEAKER_ENCODER_NAME = "openvino_speaker_encoder.xml";
constexpr const char* CODE_PREDICTOR_EMBEDDINGS_NAME = "openvino_code_predictor_embeddings.xml";  // uses 'step' input name
constexpr const char* TALKER_EMBEDDINGS_NAME = "openvino_talker_embeddings.xml";
constexpr const char* TALKER_NAME = "openvino_talker_model.xml";  // 4D float attention_mask, last_hidden_state output
constexpr const char* CODE_PREDICTOR_NAME = "openvino_code_predictor_model.xml";
constexpr const char* CODEC_ENCODER_NAME = "openvino_codec_encoder.xml";

// Constants used for static-window decoder chunking (NPU/GPU).
constexpr int64_t DECODER_CHUNK_SIZE = 231;      // effective (non-overlapping) codes fed per chunk
constexpr int64_t DECODER_LEFT_CONTEXT = 25;     // prior-chunk codes prepended for continuity
constexpr int64_t DECODER_WINDOW_LEN = DECODER_CHUNK_SIZE + DECODER_LEFT_CONTEXT;  // static decoder window
constexpr int64_t DECODER_BLEND_FRAMES = 13;      // middle region of DECODER_LEFT_CONTEXT that is cross-faded.
static_assert(DECODER_LEFT_CONTEXT >= DECODER_BLEND_FRAMES,
              "DECODER_LEFT_CONTEXT must be >= DECODER_BLEND_FRAMES");

// Component roles used for per-component device routing. Each Qwen3-TTS
// submodel is a separate IR, so (mirroring the VLM pipeline) the pipeline can
// place each one on a different device while still accepting a single primary
// `device` argument.
namespace roles {
constexpr const char* TALKER = "talker_model";
constexpr const char* TALKER_EMBEDDING = "talker_embeddings";
constexpr const char* TALKER_TEXT_EMBEDDING = "text_embeddings";
constexpr const char* CODE_PREDICTOR = "code_predictor_model";
constexpr const char* CODE_PREDICTOR_EMBEDDING = "code_predictor_embeddings";
constexpr const char* SPEECH_TOKENIZER_DECODER = "codec_decoder";
constexpr const char* SPEECH_TOKENIZER_ENCODER = "codec_encoder";
constexpr const char* SPEAKER_ENCODER = "speaker_encoder";
constexpr const char* MEL_PREPROCESS = "mel_preprocess";
}  // namespace roles

// talker models splits performance reporting between prefill & generate
constexpr const char* TALKER_PREFILL_PERF_NAME = "talker_model(prefill)";
constexpr const char* TALKER_GENERATE_PERF_NAME = "talker_model(generate)";

// Default device policy for a component. The Qwen3-TTS pipeline can place each submodel on a different device,
// and so here we define the default device for each role.
inline std::string default_device_for_role(const std::string& base_device, const std::string& role) {
    // Keep predictor embedding on CPU (short, data-dependent lookup path).
    if (role == roles::CODE_PREDICTOR_EMBEDDING) {
        return "CPU";
    }

    if (base_device.find("NPU") != std::string::npos) {
        // NPU can run talker, code predictor, and speech-tokenizer decoder.
        if (role == roles::TALKER || role == roles::CODE_PREDICTOR || role == roles::SPEECH_TOKENIZER_DECODER) {
            return base_device;
        }

        return "CPU";
    }

    return base_device;
}

// Resolve the (device, properties) pair for a component following the VLM
// `ov::device::properties` convention: when the caller supplies a per-device
// property map each component picks up the sub-map for the device it actually
// lands on; otherwise the same top-level properties are reused for every
// component.
inline std::pair<std::string, ov::AnyMap> resolve_component_target(const std::string& base_device,
                                                                    const ov::AnyMap& base_properties,
                                                                    const std::string& role) {
    const std::string device = default_device_for_role(base_device, role);
    return {device, ov::genai::utils::get_model_properties(base_properties, role, device)};
}

inline std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

inline bool is_dialect_language(const std::string& language) {
    return language.find("dialect") != std::string::npos;
}

inline std::string format_supported_languages(const std::unordered_map<std::string, int64_t>& codec_language_id) {
    std::vector<std::string> supported_languages;
    supported_languages.reserve(codec_language_id.size() + 1);
    supported_languages.emplace_back("auto");
    for (const auto& [language, _] : codec_language_id) {
        if (is_dialect_language(language)) {
            continue;
        }
        supported_languages.emplace_back(language);
    }
    std::sort(supported_languages.begin(), supported_languages.end());
    supported_languages.erase(std::unique(supported_languages.begin(), supported_languages.end()), supported_languages.end());

    std::ostringstream stream;
    for (size_t index = 0; index < supported_languages.size(); ++index) {
        if (index > 0) {
            stream << ", ";
        }
        stream << "'" << supported_languages[index] << "'";
    }
    return stream.str();
}

inline ov::InferRequest compile_request(const std::filesystem::path& model_path,
                                        const std::string& model_name,
                                        const std::string& device,
                                        const ov::AnyMap& properties) {
    ov::Core& core = ov::genai::utils::singleton_core();
    auto compiled_model = core.compile_model(model_path, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, model_name.c_str());
    return compiled_model.create_infer_request();
}

inline ov::InferRequest compile_request(const std::shared_ptr<ov::Model>& model,
                                        const std::string& model_name,
                                        const std::string& device,
                                        const ov::AnyMap& properties) {
    ov::Core& core = ov::genai::utils::singleton_core();
    auto compiled_model = core.compile_model(model, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, model_name.c_str());
    return compiled_model.create_infer_request();
}

// The code predictor exposes 'step' parameter as scalar (shape of []).
// Currently, this is problematic for NPU. This function converts this
// scalar input to a vector of shape [1].
inline void expose_scalar_input_as_vector1(const std::shared_ptr<ov::Model>& model, const std::string& input_name) {
    OPENVINO_ASSERT(model != nullptr, "Model is null");
    if (!ov::genai::utils::has_input(model, input_name)) {
        return;
    }

    auto old_input = model->input(input_name);
    auto old_param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(old_input.get_node_shared_ptr());
    OPENVINO_ASSERT(old_param != nullptr, "Input '", input_name, "' is not a Parameter node");

    const auto old_shape = old_param->get_partial_shape();
    if (old_shape.rank().is_static() && old_shape.rank().get_length() == 1) {
        return;  // already vector input
    }
    OPENVINO_ASSERT(!old_shape.rank().is_static() || old_shape.rank().get_length() == 0,
                    "Input '", input_name, "' is expected to be scalar or dynamic, got ", old_shape);

    auto new_param = std::make_shared<ov::opset15::Parameter>(old_param->get_element_type(), ov::PartialShape{1});
    new_param->set_friendly_name(input_name);
    new_param->output(0).get_tensor().set_names({input_name});

    auto squeeze_axis = ov::opset15::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto squeezed = std::make_shared<ov::opset15::Squeeze>(new_param, squeeze_axis);

    auto consumers = old_param->output(0).get_target_inputs();
    for (auto& consumer : consumers) {
        consumer.replace_source_output(squeezed->output(0));
    }

    model->remove_parameter(old_param);
    model->add_parameters({new_param});
    model->validate_nodes_and_infer_types();
}

inline ov::Tensor clone_tensor(const ov::Tensor& src) {
    ov::Tensor dst(src.get_element_type(), src.get_shape());
    src.copy_to(dst);
    return dst;
}

// Concatenate embedding tensors along the sequence dimension.
// Accepts either [1, T, H] (canonical) or [T, 1, H] (layout variant seen on
// some exports/devices) and treats both as T sequence rows with hidden size H.
inline ov::Tensor concat_embed(const std::vector<ov::Tensor>& tensors) {
    OPENVINO_ASSERT(!tensors.empty(), "concat_embed requires at least one tensor");
    OPENVINO_ASSERT(tensors.front().get_shape().size() == 3,
                    "concat_embed expects tensors of shape [1, T, H] or [T, 1, H], got ", tensors.front().get_shape());

    const ov::Shape first_shape = tensors.front().get_shape();
    OPENVINO_ASSERT(first_shape[0] == 1 || first_shape[1] == 1,
                    "concat_embed expects tensors of shape [1, T, H] or [T, 1, H], got ", first_shape);

    const size_t h = tensors.front().get_shape()[2];
    size_t total_len = 0;
    for (const auto& t : tensors) {
        const ov::Shape shape = t.get_shape();
        OPENVINO_ASSERT(shape.size() == 3 && shape[2] == h,
                        "concat_embed expects tensors of shape [1, T, ", h, "] or [T, 1, ", h, "], got ",
                        shape);
        OPENVINO_ASSERT(shape[0] == 1 || shape[1] == 1,
                        "concat_embed expects tensors of shape [1, T, ", h, "] or [T, 1, ", h, "], got ",
                        shape);
        total_len += (shape[0] == 1) ? shape[1] : shape[0];
    }
    ov::Tensor out(ov::element::f32, ov::Shape{1, total_len, h});
    float* out_ptr = out.data<float>();
    size_t cursor = 0;
    for (const auto& t : tensors) {
        const ov::Shape shape = t.get_shape();
        const size_t t_len = (shape[0] == 1) ? shape[1] : shape[0];
        const size_t sz = t_len * h;
        std::copy_n(t.data<const float>(), sz, out_ptr + cursor);
        cursor += sz;
    }
    return out;
}

// Gate for per-component inference timing. Enable with OV_GENAI_QWEN_TTS_PERF=1.
inline bool qwen_perf_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("OV_GENAI_QWEN_TTS_PERF");
        if (!env) return false;
        const std::string v = to_lower(std::string(env));
        return v == "1" || v == "true" || v == "yes" || v == "on";
    }();
    return enabled;
}

// Set OV_GENAI_QWEN_TTS_CB_SUBTALKER_BATCHING=0 to use serial predictor inference.
inline bool qwen_cb_subtalker_batching_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("OV_GENAI_QWEN_TTS_CB_SUBTALKER_BATCHING");
        if (!env) return true;
        const std::string value = to_lower(std::string(env));
        return value != "0" && value != "false" && value != "no" && value != "off";
    }();
    return enabled;
}

// Conditionally time a callable and accumulate into the caller's perf maps.
template <typename F>
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
    ms[name] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++calls[name];
}

inline std::string shape_to_string(const ov::Shape& shape) {
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

inline ov::Tensor make_causal_attention_mask_4d(size_t length) {
    ov::Tensor mask(ov::element::f32, ov::Shape{1, 1, length, length});
    float* data = mask.data<float>();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (size_t q = 0; q < length; ++q) {
        for (size_t k = 0; k < length; ++k) {
            data[q * length + k] = (k <= q) ? 0.0f : neg_inf;
        }
    }
    return mask;
}

inline ov::Tensor make_decode_attention_mask_4d(size_t kv_len) {
    ov::Tensor mask(ov::element::f32, ov::Shape{1, 1, 1, kv_len});
    std::fill_n(mask.data<float>(), kv_len, 0.0f);
    return mask;
}

// Talker-style M-RoPE position ids: 3 identical rows (no vision/temporal streams in Qwen3-TTS).
inline ov::Tensor make_position_ids_prefill(size_t length) {
    ov::Tensor pos(ov::element::i64, ov::Shape{3, 1, length});
    int64_t* data = pos.data<int64_t>();
    for (size_t i = 0; i < length; ++i) {
        data[i] = static_cast<int64_t>(i);
        data[length + i] = static_cast<int64_t>(i);
        data[2 * length + i] = static_cast<int64_t>(i);
    }
    return pos;
}

inline ov::Tensor make_position_ids_decode(size_t absolute_position) {
    ov::Tensor pos(ov::element::i64, ov::Shape{3, 1, 1});
    int64_t* data = pos.data<int64_t>();
    data[0] = static_cast<int64_t>(absolute_position);
    data[1] = static_cast<int64_t>(absolute_position);
    data[2] = static_cast<int64_t>(absolute_position);
    return pos;
}

inline ov::Tensor make_predictor_position_ids(size_t start_position, size_t length) {
    // Code predictor uses 2D position IDs: [batch_size=1, seq_len=length]
    ov::Tensor pos(ov::element::i64, ov::Shape{1, length});
    int64_t* data = pos.data<int64_t>();
    for (size_t i = 0; i < length; ++i) {
        data[i] = static_cast<int64_t>(start_position + i);
    }
    return pos;
}

}  // namespace qwen3_tts
}  // namespace genai
}  // namespace ov
