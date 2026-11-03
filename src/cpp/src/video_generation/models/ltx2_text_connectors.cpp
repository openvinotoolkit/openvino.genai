// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/ltx2_text_connectors.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

#include "json_utils.hpp"
#include "utils.hpp"
#include "video_generation/video_generation_utils.hpp"

using namespace ov::genai;

LTX2TextConnectors::LTX2TextConnectors(const std::filesystem::path& root_dir) {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");

    const std::filesystem::path config_path = root_dir / "config.json";
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);
    const nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;
    read_json_param(data, "connector_rope_base_seq_len", m_rope_base_seq_len);
    read_json_param(data, "rope_theta", m_rope_theta);
    read_json_param(data, "rope_double_precision", m_rope_double_precision);
    read_json_param(data, "rope_type", m_rope_type);
    for (const std::string name : {"video_connector", "audio_connector"}) {
        ConnectorRope rope{name};
        read_json_param(data, name + "_num_attention_heads", rope.num_attention_heads);
        read_json_param(data, name + "_attention_head_dim", rope.attention_head_dim);
        m_connector_ropes.push_back(rope);
    }
}

LTX2TextConnectors::LTX2TextConnectors(const std::filesystem::path& root_dir,
                                       const std::string& device,
                                       const ov::AnyMap& properties)
    : LTX2TextConnectors(root_dir) {
    compile(device, properties);
}

LTX2TextConnectors::LTX2TextConnectors(const LTX2TextConnectors&) = default;

LTX2TextConnectors LTX2TextConnectors::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "LTX2TextConnectors must have exactly one of m_model or m_request initialized");

    LTX2TextConnectors cloned = *this;

    if (m_model) {
        cloned.m_model = m_model->clone();
    } else {
        cloned.m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

LTX2TextConnectors& LTX2TextConnectors::reshape(const int batch_size) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    std::map<std::string, ov::PartialShape> name_to_shape;
    for (auto&& input : m_model->inputs()) {
        std::string input_name = input.get_any_name();
        name_to_shape[input_name] = input.get_partial_shape();
        name_to_shape[input_name][0] = batch_size;
    }

    m_model->reshape(name_to_shape);
    return *this;
}

std::pair<ov::Tensor, ov::Tensor> LTX2TextConnectors::compute_connector_rope(const ConnectorRope& rope,
                                                                             size_t batch_size,
                                                                             size_t seq_len) const {
    // diffusers v0.40.0, pipelines/ltx2/connectors.py: positions 0..L-1 as a fraction of
    // 'connector_rope_base_seq_len', frequencies theta ** linspace(0, 1, dim / 2) * pi / 2 (fp64 when
    // 'rope_double_precision'), angle (2 * position - 1) * frequency in fp32
    const size_t heads = rope.num_attention_heads;
    const size_t dim = heads * rope.attention_head_dim;
    const size_t steps = dim / 2;
    std::vector<float> freqs(steps);
    for (size_t k = 0; k < steps; ++k) {
        const double position = steps > 1 ? static_cast<double>(k) / static_cast<double>(steps - 1) : 0.0;
        if (m_rope_double_precision) {
            freqs[k] = static_cast<float>(std::pow(m_rope_theta, position) * video_generation_utils::PI / 2.0);
        } else {
            freqs[k] = std::pow(static_cast<float>(m_rope_theta), static_cast<float>(position)) *
                       static_cast<float>(video_generation_utils::PI) / 2.0f;
        }
    }
    auto angle = [&](size_t t, size_t k) {
        const float grid = static_cast<float>(t) / static_cast<float>(m_rope_base_seq_len);
        return (grid * 2.0f - 1.0f) * freqs[k];
    };

    ov::Tensor cos_tensor, sin_tensor;
    if (m_rope_type == "split") {
        // [B, H, L, D / 2 / H]; the frequency count is exactly D / 2, so there is no padding
        OPENVINO_ASSERT(steps % heads == 0, "Connector RoPE dim ", dim, " does not split over ", heads, " heads");
        const size_t head_dim = steps / heads;
        cos_tensor = ov::Tensor(ov::element::f32, {batch_size, heads, seq_len, head_dim});
        sin_tensor = ov::Tensor(ov::element::f32, {batch_size, heads, seq_len, head_dim});
        float* cos_data = cos_tensor.data<float>();
        float* sin_data = sin_tensor.data<float>();
        for (size_t h = 0; h < heads; ++h) {
            for (size_t t = 0; t < seq_len; ++t) {
                for (size_t d = 0; d < head_dim; ++d) {
                    const double a = angle(t, h * head_dim + d);
                    const size_t out = (h * seq_len + t) * head_dim + d;
                    cos_data[out] = static_cast<float>(std::cos(a));
                    sin_data[out] = static_cast<float>(std::sin(a));
                }
            }
        }
    } else {
        OPENVINO_ASSERT(m_rope_type == "interleaved", "Unsupported connector 'rope_type': ", m_rope_type);
        // [B, L, D], each frequency repeated twice, front-padded with cos = 1, sin = 0 when D is odd
        const size_t pad = dim % 2;
        cos_tensor = ov::Tensor(ov::element::f32, {batch_size, seq_len, dim});
        sin_tensor = ov::Tensor(ov::element::f32, {batch_size, seq_len, dim});
        float* cos_data = cos_tensor.data<float>();
        float* sin_data = sin_tensor.data<float>();
        for (size_t t = 0; t < seq_len; ++t) {
            float* cos_row = cos_data + t * dim;
            float* sin_row = sin_data + t * dim;
            std::fill_n(cos_row, pad, 1.0f);
            std::fill_n(sin_row, pad, 0.0f);
            for (size_t k = 0; k < steps; ++k) {
                const double a = angle(t, k);
                cos_row[pad + 2 * k] = cos_row[pad + 2 * k + 1] = static_cast<float>(std::cos(a));
                sin_row[pad + 2 * k] = sin_row[pad + 2 * k + 1] = static_cast<float>(std::sin(a));
            }
        }
    }

    // Every batch element sees the same positions
    const size_t batch_stride = cos_tensor.get_size() / batch_size;
    for (size_t b = 1; b < batch_size; ++b) {
        std::copy_n(cos_tensor.data<float>(), batch_stride, cos_tensor.data<float>() + b * batch_stride);
        std::copy_n(sin_tensor.data<float>(), batch_stride, sin_tensor.data<float>() + b * batch_stride);
    }
    return {cos_tensor, sin_tensor};
}

LTX2TextConnectors& LTX2TextConnectors::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    size_t rotary_inputs = 0;
    for (const auto& rope : m_connector_ropes) {
        rotary_inputs += utils::has_input(m_model, rope.name + "_rotary_cos");
        rotary_inputs += utils::has_input(m_model, rope.name + "_rotary_sin");
    }
    OPENVINO_ASSERT(rotary_inputs == 0 || rotary_inputs == 2 * m_connector_ropes.size(),
                    "LTX2 connectors require complete external rotary input pairs for both modalities, or none");
    m_external_rope = rotary_inputs != 0;
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "LTX2 text connectors model");
    m_request = compiled_model.create_infer_request();
    m_model.reset();

    return *this;
}

LTX2TextConnectors::Output LTX2TextConnectors::infer(const ov::Tensor& text_encoder_hidden_states,
                                                     const ov::Tensor& attention_mask) {
    OPENVINO_ASSERT(m_request, "Connectors model must be compiled first. Cannot infer non-compiled model");

    const auto mask_type = m_request.get_compiled_model().input("attention_mask").get_element_type();
    m_request.set_tensor("text_encoder_hidden_states", text_encoder_hidden_states);
    m_request.set_tensor("attention_mask", video_generation_utils::convert_tensor(attention_mask, mask_type));

    const ov::Shape& hidden_shape = text_encoder_hidden_states.get_shape();
    OPENVINO_ASSERT(hidden_shape.size() == 3,
                    "Text encoder hidden states must be rank-3 [B, L, C], got rank ",
                    hidden_shape.size());
    if (m_external_rope) {
        for (const auto& rope : m_connector_ropes) {
            auto [cos, sin] = compute_connector_rope(rope, hidden_shape[0], hidden_shape[1]);
            m_request.set_tensor(rope.name + "_rotary_cos", cos);
            m_request.set_tensor(rope.name + "_rotary_sin", sin);
        }
    }
    m_request.infer();

    auto copy_output = [this](const std::string& name) {
        const ov::Tensor output = m_request.get_tensor(name);
        ov::Tensor copied(output.get_element_type(), output.get_shape());
        output.copy_to(copied);
        return copied;
    };

    return {copy_output("video_text_embedding"),
            copy_output("audio_text_embedding"),
            copy_output("connector_attention_mask")};
}
