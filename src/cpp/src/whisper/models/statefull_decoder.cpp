// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "statefull_decoder.hpp"

#include "openvino/op/softmax.hpp"
#include "openvino/pass/manager.hpp"
#include "utils.hpp"
#include "whisper/alignment_heads.hpp"
#include "whisper/word_level_timestamps.hpp"

namespace {
void reshape_hidden_states_to_static(std::shared_ptr<ov::Model> model, const ov::PartialShape& lhstates_shape) {
    ov::PartialShape new_shape = model->input("encoder_hidden_states").get_partial_shape();
    OPENVINO_ASSERT(new_shape.size() > 1 && lhstates_shape.size() > 1);
    new_shape[1] = lhstates_shape[1];
    std::map<std::string, ov::PartialShape> name_to_shape{{"encoder_hidden_states", new_shape}};
    model->reshape(name_to_shape);
}
}  // namespace

namespace ov::genai {
WhisperStatefullDecoder::WhisperStatefullDecoder(const std::filesystem::path& models_path,
                                                 const std::string& device,
                                                 const ov::AnyMap& properties,
                                                 const ov::PartialShape& lhs_shape,
                                                 const bool decompose_cross_attention_spda)
    : m_decompose_cross_attention_spda_ops(decompose_cross_attention_spda) {
    ov::Core core = utils::singleton_core();

    auto model = core.read_model(models_path / "openvino_decoder_model.xml", {}, properties);

    if (m_decompose_cross_attention_spda_ops) {
        ov::genai::decompose_scaled_dot_product_attention_for_whisper(model);
        ov::genai::add_cross_attention_qk_scaled_scores_outputs_for_whisper(model);
    }

    m_has_cache_position = utils::has_input(model, "cache_position");

    ov::CompiledModel compiled_model;
    if (device == "NPU") {
        auto kv_pos = ov::genai::utils::get_kv_axes_pos(model);

        reshape_hidden_states_to_static(model, lhs_shape);

        utils::KVDesc kv_desc;
        std::tie(compiled_model, kv_desc) = utils::compile_decoder_for_npu(model, properties, kv_pos, true);
    } else {
        utils::apply_slice_before_matmul_transformation(model);

        compiled_model = core.compile_model(model, device, properties);
    }

    utils::print_compiled_model_properties(compiled_model, "whisper decoder model");
    m_request = compiled_model.create_infer_request();
}

void WhisperStatefullDecoder::start_async(const Tensor& encoder_hidden_state,
                                          const Tensor& input_ids,
                                          const Tensor& beam_idx) {
    const size_t batch_size = input_ids.get_shape().at(0);
    const size_t seq_len = input_ids.get_shape().at(1);

    _set_encoder_hidden_states_tensor(encoder_hidden_state, batch_size, m_request);

    if (m_has_cache_position) {
        _set_cache_position_tensor(seq_len);
    }
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("beam_idx", beam_idx);

    m_request.start_async();
};

void WhisperStatefullDecoder::_set_cache_position_tensor(const size_t seq_len) {
    ov::Tensor cache_position_tensor = m_request.get_tensor("cache_position");

    int64_t start_cache_position = 0;

    if (cache_position_tensor.get_size() != 0) {
        start_cache_position = cache_position_tensor.data<int64_t>()[cache_position_tensor.get_size() - 1] + 1;
    }

    cache_position_tensor.set_shape({seq_len});

    auto cache_data = cache_position_tensor.data<int64_t>();
    std::iota(cache_data, cache_data + seq_len, start_cache_position);
};

Tensor WhisperStatefullDecoder::wait() {
    m_request.wait();
    return m_request.get_tensor("logits");
}

void WhisperStatefullDecoder::reset_state() {
    m_request.reset_state();
    if (m_has_cache_position) {
        m_request.set_tensor("cache_position", create_host_tensor(ov::element::i64, {0}));
    }

    Shape encoder_hidden_states_shape{m_request.get_tensor("encoder_hidden_states").get_shape()};
    encoder_hidden_states_shape[0] = 0;
    m_request.set_tensor("encoder_hidden_states", create_host_tensor(ov::element::f32, encoder_hidden_states_shape));
};

ov::Tensor WhisperStatefullDecoder::create_host_tensor(const element::Type element_type, const Shape& shape) {
    try {
        return m_request.get_compiled_model().get_context().create_host_tensor(element_type, shape);
    } catch (std::exception& ex) {
        return ov::Tensor(element_type, shape);
    }
}

std::vector<Tensor> WhisperStatefullDecoder::get_alignments_heads_qks(
    const std::vector<std::pair<size_t, size_t>>& alignment_heads) {
    OPENVINO_ASSERT(m_decompose_cross_attention_spda_ops,
                    "Encoder attention heads are not decomposed. Cannot get encoder attention QKs.");

    return ov::genai::get_whisper_alignments_heads_qks(m_request, alignment_heads);
}

}  // namespace ov::genai
