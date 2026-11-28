// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speech_generation/qwen3_tts/qwen3_tts_fused_predictor.hpp"

#include <algorithm>
#include <cstring>

#include "speech_generation/qwen3_tts/qwen3_tts_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
namespace qwen3_tts {

bool Qwen3TTSFusedCodePredictor::initialize(const std::filesystem::path& model_path,
                                            const std::string& device,
                                            const ov::AnyMap& properties,
                                            size_t num_code_groups,
                                            size_t hidden_size) {
    if (!std::filesystem::exists(model_path)) {
        m_available = false;
        return false;
    }
    auto model = ov::genai::utils::singleton_core().read_model(model_path);
    return initialize_from_model(model, device, properties, num_code_groups, hidden_size);
}

bool Qwen3TTSFusedCodePredictor::initialize_from_model(const std::shared_ptr<ov::Model>& model,
                                                       const std::string& device,
                                                       const ov::AnyMap& properties,
                                                       size_t num_code_groups,
                                                       size_t hidden_size) {
    OPENVINO_ASSERT(model != nullptr, "Fused code predictor model is null");
    OPENVINO_ASSERT(num_code_groups >= 2, "Fused code predictor requires num_code_groups >= 2");
    OPENVINO_ASSERT(hidden_size > 0, "Fused code predictor requires hidden_size > 0");

    m_num_code_groups = num_code_groups;
    m_hidden_size = hidden_size;
    m_device = device;
    m_has_tts_pad_embed = false;
    m_has_uniform_noise = false;
    m_has_temperature = false;
    m_codes_include_first = true;
    m_use_remote_next_embed = false;
    m_sub_vocab_size = 2048;

    bool has_past_hidden = false;
    bool has_first_code = false;
    for (const auto& input : model->inputs()) {
        const std::string name = input.get_any_name();
        if (name == "past_hidden") {
            has_past_hidden = true;
        } else if (name == "first_code") {
            has_first_code = true;
        } else if (name == "tts_pad_embed") {
            m_has_tts_pad_embed = true;
        } else if (name == "uniform_noise") {
            m_has_uniform_noise = true;
            const auto& ps = input.get_partial_shape();
            if (ps.rank().is_static() && ps.rank().get_length() == 3 && ps[2].is_static()) {
                m_sub_vocab_size = static_cast<size_t>(ps[2].get_length());
            }
        } else if (name == "temperature") {
            m_has_temperature = true;
        }
    }
    OPENVINO_ASSERT(has_past_hidden && has_first_code,
                    "Fused code predictor IR must declare inputs 'past_hidden' and 'first_code'");

    bool has_codes = false;
    bool has_next_embed = false;
    for (const auto& output : model->outputs()) {
        const std::string name = output.get_any_name();
        if (name == "codes") {
            has_codes = true;
            const auto& ps = output.get_partial_shape();
            if (ps.rank().is_static() && ps.rank().get_length() == 2 && ps[1].is_static()) {
                const size_t out_groups = static_cast<size_t>(ps[1].get_length());
                if (out_groups == num_code_groups - 1) {
                    m_codes_include_first = false;
                } else {
                    OPENVINO_ASSERT(out_groups == num_code_groups,
                                    "Fused code predictor 'codes' output dim 1 must be ",
                                    num_code_groups,
                                    " or ",
                                    num_code_groups - 1,
                                    ", got ",
                                    out_groups);
                    m_codes_include_first = true;
                }
            }
        } else if (name == "next_embed") {
            has_next_embed = true;
        }
    }
    OPENVINO_ASSERT(has_codes && has_next_embed,
                    "Fused code predictor IR must declare outputs 'codes' and 'next_embed'");

    ov::AnyMap compile_props = properties;
    strip_speech_routing_properties(compile_props);
    if (device.find("GPU") != std::string::npos &&
        compile_props.find("ACTIVATIONS_SCALE_FACTOR") == compile_props.end()) {
        compile_props["ACTIVATIONS_SCALE_FACTOR"] = 8.0f;
    }

    ov::Core& core = ov::genai::utils::singleton_core();
    m_compiled = core.compile_model(model, device, compile_props);
    ov::genai::utils::print_compiled_model_properties(m_compiled, roles::CODE_PREDICTOR_FUSED);
    m_request = m_compiled.create_infer_request();

    m_single_past_hidden = ov::Tensor(ov::element::f32, ov::Shape{1, 1, m_hidden_size});
    m_single_first_code = ov::Tensor(ov::element::i64, ov::Shape{1, 1});
    m_zero_tts_pad = ov::Tensor(ov::element::f32, ov::Shape{1, 1, m_hidden_size});
    std::memset(m_zero_tts_pad.data(), 0, m_zero_tts_pad.get_byte_size());
    m_temperature_tensor = ov::Tensor(ov::element::f32, ov::Shape{1});
    m_temperature_tensor.data<float>()[0] = 0.0f;

    init_remote_next_embed_buffers();
    m_available = true;
    return true;
}

void Qwen3TTSFusedCodePredictor::init_remote_next_embed_buffers() {
    m_use_remote_next_embed = false;
    m_remote_context_device_name.clear();
    if (m_device.find("GPU") == std::string::npos) {
        return;
    }
    try {
        ov::RemoteContext ctx = m_compiled.get_context();
        const ov::Shape embed_shape{1, 1, m_hidden_size};
        for (size_t i = 0; i < m_remote_next_embeds.size(); ++i) {
            m_remote_next_embeds[i] = ctx.create_tensor(ov::element::f32, embed_shape);
        }
        m_remote_context_device_name = ctx.get_device_name();
        m_use_remote_next_embed = true;
    } catch (...) {
        // Device plugin does not expose a RemoteContext or USM/CL tensor creation; fall back to host tensors.
        m_remote_next_embeds = {};
        m_remote_context_device_name.clear();
        m_use_remote_next_embed = false;
    }
}

void Qwen3TTSFusedCodePredictor::prepare_sampling_inputs(size_t batch_size,
                                                         bool do_sample,
                                                         float temperature,
                                                         std::mt19937& rng) {
    if (m_has_temperature) {
        // In greedy mode, pair temperature = 1.0f with constant uniform_noise = 0.5f so in-graph
        // `logits / max(temperature, 1e-6)` never overflows FP16 on GPU while preserving exact argmax.
        const float eff_temp = (do_sample && temperature > 0.0f) ? temperature : 1.0f;
        m_temperature_tensor.data<float>()[0] = eff_temp;
        m_request.set_tensor("temperature", m_temperature_tensor);
    }
    if (m_has_uniform_noise) {
        const size_t residual_groups = m_num_code_groups - 1;
        ov::Tensor noise(ov::element::f32, ov::Shape{batch_size, residual_groups, m_sub_vocab_size});
        float* ptr = noise.data<float>();
        const size_t total = noise.get_size();
        if (do_sample && temperature > 0.0f) {
            std::uniform_real_distribution<float> dist(1e-6f, 1.0f - 1e-6f);
            for (size_t i = 0; i < total; ++i) {
                ptr[i] = dist(rng);
            }
        } else {
            // Deterministic constant noise when greedy (paired with temperature == 1.0f).
            std::fill_n(ptr, total, 0.5f);
        }
        m_request.set_tensor("uniform_noise", noise);
    }
}

Qwen3TTSFusedCodePredictor::SingleResult Qwen3TTSFusedCodePredictor::predict_single(
    const ov::Tensor& past_hidden,
    int64_t first_code,
    const ov::Tensor& additive_embed,
    bool additive_is_tts_pad,
    bool do_sample,
    float temperature,
    std::mt19937& rng,
    bool allow_device_zero_copy,
    std::unordered_map<std::string, double>& perf_ms,
    std::unordered_map<std::string, int64_t>& perf_calls) {
    OPENVINO_ASSERT(m_available, "Fused code predictor is not initialized");
    const auto ph_shape = past_hidden.get_shape();
    OPENVINO_ASSERT(ph_shape.size() == 3 && ph_shape[0] == 1 && ph_shape[2] == m_hidden_size,
                    "Fused code predictor past_hidden must have shape [1, T, ",
                    m_hidden_size,
                    "], got ",
                    shape_to_string(ph_shape));

    // Zero-copy bind when past_hidden is already contiguous [1, 1, H] (every decode step after prefill).
    if (ph_shape[1] == 1 && past_hidden.is_continuous()) {
        m_request.set_tensor("past_hidden", past_hidden);
    } else {
        const size_t last_row = ph_shape[1] - 1;
        std::memcpy(m_single_past_hidden.data<float>(),
                    past_hidden.data<const float>() + last_row * m_hidden_size,
                    m_hidden_size * sizeof(float));
        m_request.set_tensor("past_hidden", m_single_past_hidden);
    }

    m_single_first_code.data<int64_t>()[0] = first_code;
    m_request.set_tensor("first_code", m_single_first_code);

    const bool apply_additive_in_graph = m_has_tts_pad_embed && additive_is_tts_pad && static_cast<bool>(additive_embed);
    if (m_has_tts_pad_embed) {
        if (apply_additive_in_graph) {
            m_request.set_tensor("tts_pad_embed", additive_embed);
        } else {
            m_request.set_tensor("tts_pad_embed", m_zero_tts_pad);
        }
    }

    prepare_sampling_inputs(/*batch_size=*/1, do_sample, temperature, rng);

    const bool need_host_addition = static_cast<bool>(additive_embed) && !apply_additive_in_graph;
    const bool use_remote_out = allow_device_zero_copy && m_use_remote_next_embed && !need_host_addition;

    ov::Tensor host_next_embed;
    ov::Tensor bound_next_embed;
    if (use_remote_out) {
        bound_next_embed = m_remote_next_embeds[m_ping_pong_idx];
        m_ping_pong_idx = 1 - m_ping_pong_idx;
        m_request.set_tensor("next_embed", bound_next_embed);
    } else {
        host_next_embed = ov::Tensor(ov::element::f32, ov::Shape{1, 1, m_hidden_size});
        m_request.set_tensor("next_embed", host_next_embed);
    }

    run_and_time([&] { m_request.infer(); }, roles::CODE_PREDICTOR_FUSED, perf_ms, perf_calls);

    const ov::Tensor codes_tensor = m_request.get_tensor("codes");
    OPENVINO_ASSERT(codes_tensor.get_element_type() == ov::element::i64,
                    "Fused code predictor 'codes' output must be i64");
    const auto codes_shape = codes_tensor.get_shape();
    OPENVINO_ASSERT(codes_shape.size() == 2 && codes_shape[0] == 1,
                    "Fused code predictor 'codes' output must have shape [1, G], got ",
                    shape_to_string(codes_shape));
    const int64_t* codes_ptr = codes_tensor.data<const int64_t>();

    SingleResult result;
    result.codes.reserve(m_num_code_groups);
    if (m_codes_include_first && codes_shape[1] == m_num_code_groups) {
        result.codes.assign(codes_ptr, codes_ptr + m_num_code_groups);
    } else {
        result.codes.push_back(first_code);
        const size_t copy_n = std::min(codes_shape[1], m_num_code_groups - 1);
        result.codes.insert(result.codes.end(), codes_ptr, codes_ptr + copy_n);
    }

    if (use_remote_out) {
        result.next_embed = bound_next_embed;
    } else {
        if (need_host_addition) {
            OPENVINO_ASSERT(additive_embed.get_size() >= m_hidden_size,
                            "additive_embed size must be >= hidden_size");
            float* dst = host_next_embed.data<float>();
            const float* add_ptr = additive_embed.data<const float>();
            for (size_t h = 0; h < m_hidden_size; ++h) {
                dst[h] += add_ptr[h];
            }
        }
        result.next_embed = std::move(host_next_embed);
    }

    return result;
}

Qwen3TTSFusedCodePredictor::BatchResult Qwen3TTSFusedCodePredictor::predict_batch(
    const ov::Tensor& past_hidden_batch,
    const std::vector<int64_t>& first_codes,
    const ov::Tensor& shared_tts_pad_embed,
    const std::vector<const float*>& per_row_additive_embeds,
    bool do_sample,
    float temperature,
    std::mt19937& rng,
    std::unordered_map<std::string, double>& perf_ms,
    std::unordered_map<std::string, int64_t>& perf_calls) {
    OPENVINO_ASSERT(m_available, "Fused code predictor is not initialized");
    const size_t batch_size = first_codes.size();
    OPENVINO_ASSERT(batch_size > 0, "Fused code predictor predict_batch requires batch_size > 0");
    const auto ph_shape = past_hidden_batch.get_shape();
    OPENVINO_ASSERT(ph_shape.size() == 3 && ph_shape[0] == batch_size && ph_shape[1] == 1 &&
                        ph_shape[2] == m_hidden_size,
                    "Fused code predictor past_hidden_batch must have shape [B, 1, ",
                    m_hidden_size,
                    "], got ",
                    shape_to_string(ph_shape));

    ov::Tensor first_codes_tensor(ov::element::i64, ov::Shape{batch_size, 1});
    std::copy(first_codes.begin(), first_codes.end(), first_codes_tensor.data<int64_t>());

    m_request.set_tensor("past_hidden", past_hidden_batch);
    m_request.set_tensor("first_code", first_codes_tensor);

    const bool has_per_row_override = !per_row_additive_embeds.empty();
    OPENVINO_ASSERT(!has_per_row_override || per_row_additive_embeds.size() == batch_size,
                    "per_row_additive_embeds size must match batch_size");

    const bool apply_shared_in_graph =
        m_has_tts_pad_embed && !has_per_row_override && static_cast<bool>(shared_tts_pad_embed);
    if (m_has_tts_pad_embed) {
        m_request.set_tensor("tts_pad_embed", apply_shared_in_graph ? shared_tts_pad_embed : m_zero_tts_pad);
    }

    prepare_sampling_inputs(batch_size, do_sample, temperature, rng);

    ov::Tensor next_embeds(ov::element::f32, ov::Shape{batch_size, 1, m_hidden_size});
    m_request.set_tensor("next_embed", next_embeds);

    run_and_time([&] { m_request.infer(); }, roles::CODE_PREDICTOR_FUSED, perf_ms, perf_calls);

    const ov::Tensor codes_tensor = m_request.get_tensor("codes");
    OPENVINO_ASSERT(codes_tensor.get_element_type() == ov::element::i64,
                    "Fused code predictor 'codes' output must be i64");
    const auto codes_shape = codes_tensor.get_shape();
    OPENVINO_ASSERT(codes_shape.size() == 2 && codes_shape[0] == batch_size,
                    "Fused code predictor 'codes' output must have shape [B, G], got ",
                    shape_to_string(codes_shape));
    const size_t stride = codes_shape[1];
    const int64_t* codes_ptr = codes_tensor.data<const int64_t>();

    BatchResult result;
    result.codes.resize(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
        auto& row = result.codes[b];
        row.reserve(m_num_code_groups);
        const int64_t* row_ptr = codes_ptr + b * stride;
        if (m_codes_include_first && stride == m_num_code_groups) {
            row.assign(row_ptr, row_ptr + m_num_code_groups);
        } else {
            row.push_back(first_codes[b]);
            const size_t copy_n = std::min(stride, m_num_code_groups - 1);
            row.insert(row.end(), row_ptr, row_ptr + copy_n);
        }
    }

    if (has_per_row_override) {
        float* dst = next_embeds.data<float>();
        for (size_t b = 0; b < batch_size; ++b) {
            const float* add_ptr = per_row_additive_embeds[b];
            if (add_ptr != nullptr) {
                float* row_dst = dst + b * m_hidden_size;
                for (size_t h = 0; h < m_hidden_size; ++h) {
                    row_dst[h] += add_ptr[h];
                }
            }
        }
    } else if (!apply_shared_in_graph && static_cast<bool>(shared_tts_pad_embed)) {
        const float* pad_ptr = shared_tts_pad_embed.data<const float>();
        float* dst = next_embeds.data<float>();
        for (size_t b = 0; b < batch_size; ++b) {
            float* row_dst = dst + b * m_hidden_size;
            for (size_t h = 0; h < m_hidden_size; ++h) {
                row_dst[h] += pad_ptr[h];
            }
        }
    }

    result.next_embeds = std::move(next_embeds);
    return result;
}

}  // namespace qwen3_tts
}  // namespace genai
}  // namespace ov
