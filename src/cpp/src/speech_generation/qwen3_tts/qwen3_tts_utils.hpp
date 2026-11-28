// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset15.hpp>

#include "utils.hpp"

// Shared, stateless helpers for Qwen3-TTS model and codec components. Keeping these utilities
// separate from the driver lets alternate backends reuse the same tensor and device logic.
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
constexpr const char* CODE_PREDICTOR_FUSED_NAME = "openvino_code_predictor_fused.xml";
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
constexpr const char* CODE_PREDICTOR_FUSED = "code_predictor_fused";
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
        // NPU can run talker, code predictor (fused or legacy), and speech-tokenizer decoder.
        if (role == roles::TALKER || role == roles::CODE_PREDICTOR || role == roles::CODE_PREDICTOR_FUSED ||
            role == roles::SPEECH_TOKENIZER_DECODER) {
            return base_device;
        }

        return "CPU";
    }

    return base_device;
}

inline std::optional<std::string> lookup_device_mapping(const ov::AnyMap& base_properties, const std::string& role) {
    auto it = base_properties.find("device_mapping");
    if (it == base_properties.end() || it->second.empty()) {
        return std::nullopt;
    }

    auto find_in_map = [&](const auto& mapping) -> std::optional<std::string> {
        auto role_it = mapping.find(role);
        if (role_it != mapping.end()) {
            if constexpr (std::is_same_v<std::decay_t<decltype(role_it->second)>, std::string>) {
                if (!role_it->second.empty()) {
                    return role_it->second;
                }
            } else {
                const std::string val = role_it->second.template as<std::string>();
                if (!val.empty()) {
                    return val;
                }
            }
        }
        // Allow "code_predictor_model" mapping entry to also apply to "code_predictor_fused".
        if (role == roles::CODE_PREDICTOR_FUSED) {
            auto alias_it = mapping.find(roles::CODE_PREDICTOR);
            if (alias_it != mapping.end()) {
                if constexpr (std::is_same_v<std::decay_t<decltype(alias_it->second)>, std::string>) {
                    if (!alias_it->second.empty()) {
                        return alias_it->second;
                    }
                } else {
                    const std::string val = alias_it->second.template as<std::string>();
                    if (!val.empty()) {
                        return val;
                    }
                }
            }
        }
        return std::nullopt;
    };

    if (it->second.is<std::map<std::string, std::string>>()) {
        return find_in_map(it->second.as<std::map<std::string, std::string>>());
    }
    if (it->second.is<ov::AnyMap>()) {
        return find_in_map(it->second.as<ov::AnyMap>());
    }
    OPENVINO_THROW("'device_mapping' property must be a map<string, string> or ov::AnyMap");
}

// Remove GenAI speech-pipeline routing meta-properties before forwarding properties to plugins.
inline void strip_speech_routing_properties(ov::AnyMap& properties) {
    properties.erase("device_mapping");
    properties.erase("decoder_device");
    properties.erase("encoder_device");
    properties.erase("attention_backend");
    properties.erase("scheduler_config");
}

inline bool is_non_llm_codec_role(const std::string& role) {
    return role == roles::SPEECH_TOKENIZER_DECODER || role == roles::SPEECH_TOKENIZER_ENCODER ||
           role == roles::SPEAKER_ENCODER || role == roles::MEL_PREPROCESS;
}

inline std::set<std::string> collect_explicit_component_property_keys(const ov::AnyMap& base_properties,
                                                                      const std::string& device,
                                                                      const std::string& role) {
    std::set<std::string> explicit_keys;
    auto dp_it = base_properties.find(ov::device::properties.name());
    if (dp_it != base_properties.end() && dp_it->second.is<ov::AnyMap>()) {
        const auto& dp_map = dp_it->second.as<ov::AnyMap>();
        auto dev_it = dp_map.find(device);
        if (dev_it != dp_map.end() && dev_it->second.is<ov::AnyMap>()) {
            for (const auto& [k, _] : dev_it->second.as<ov::AnyMap>()) {
                explicit_keys.insert(k);
            }
        }
    }
    auto mp_it = base_properties.find(ov::genai::utils::PER_MODEL_PROPERTIES);
    if (mp_it != base_properties.end() && mp_it->second.is<ov::AnyMap>()) {
        const auto& mp_map = mp_it->second.as<ov::AnyMap>();
        auto role_it = mp_map.find(role);
        if (role_it != mp_map.end() && role_it->second.is<ov::AnyMap>()) {
            for (const auto& [k, _] : role_it->second.as<ov::AnyMap>()) {
                explicit_keys.insert(k);
            }
        }
    }
    return explicit_keys;
}

// Resolve the (device, properties) pair for a component following the VLM
// `ov::device::properties` convention plus speech-specific heterogeneous overrides:
//   1. `device_mapping[role]` (highest precedence)
//   2. `decoder_device` (for `codec_decoder`) / `encoder_device` (for `codec_encoder`, `speaker_encoder`, `mel_preprocess`)
//   3. `default_device_for_role(base_device, role)`
// GenAI routing keys are stripped, and inherited GPU-only / LLM-only top-level hints are
// filtered out when targeting non-GPU devices or non-LLM codec/encoder components unless
// explicitly set under `ov::device::properties(device, ...)` or `MODEL_PROPERTIES[role]`.
inline std::pair<std::string, ov::AnyMap> resolve_component_target(const std::string& base_device,
                                                                    const ov::AnyMap& base_properties,
                                                                    const std::string& role) {
    std::string device;
    if (auto mapped = lookup_device_mapping(base_properties, role); mapped.has_value()) {
        device = *mapped;
    } else if (role == roles::SPEECH_TOKENIZER_DECODER) {
        auto it = base_properties.find("decoder_device");
        if (it != base_properties.end() && !it->second.empty()) {
            device = it->second.as<std::string>();
        }
    } else if (role == roles::SPEECH_TOKENIZER_ENCODER || role == roles::SPEAKER_ENCODER ||
               role == roles::MEL_PREPROCESS) {
        auto it = base_properties.find("encoder_device");
        if (it != base_properties.end() && !it->second.empty()) {
            device = it->second.as<std::string>();
        }
    }

    if (device.empty()) {
        device = default_device_for_role(base_device, role);
    }

    ov::AnyMap resolved_props = ov::genai::utils::get_model_properties(base_properties, role, device);
    strip_speech_routing_properties(resolved_props);

    const bool target_is_gpu = device.find("GPU") != std::string::npos;
    const bool non_llm_role = is_non_llm_codec_role(role);
    if (!target_is_gpu || non_llm_role) {
        const auto explicit_keys = collect_explicit_component_property_keys(base_properties, device, role);
        for (auto it = resolved_props.begin(); it != resolved_props.end();) {
            const std::string& key = it->first;
            const bool is_explicit = explicit_keys.count(key) > 0;
            const bool drop_gpu_only =
                !target_is_gpu && !is_explicit &&
                (key == "ACTIVATIONS_SCALE_FACTOR" || key.rfind("GPU_", 0) == 0);
            const bool drop_llm_only =
                non_llm_role && !is_explicit &&
                (key == ov::hint::kv_cache_precision.name() ||
                 key == ov::hint::dynamic_quantization_group_size.name());
            if (drop_gpu_only || drop_llm_only) {
                it = resolved_props.erase(it);
            } else {
                ++it;
            }
        }
    }

    return {device, std::move(resolved_props)};
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

// Concatenate tensors of shape [1, T_i, H] along the sequence dimension.
inline ov::Tensor concat_embed(const std::vector<ov::Tensor>& tensors) {
    OPENVINO_ASSERT(!tensors.empty(), "concat_embed requires at least one tensor");
    OPENVINO_ASSERT(tensors.front().get_shape().size() == 3,
                    "concat_embed expects tensors of shape [1, T, H], got ", tensors.front().get_shape());
    const size_t h = tensors.front().get_shape()[2];
    size_t total_len = 0;
    for (const auto& t : tensors) {
        OPENVINO_ASSERT(t.get_shape().size() == 3 && t.get_shape()[0] == 1 && t.get_shape()[2] == h,
                        "concat_embed expects tensors of shape [1, T, ", h, "], got ", t.get_shape());
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
