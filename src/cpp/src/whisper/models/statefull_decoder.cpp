// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "statefull_decoder.hpp"

#include "openvino/op/softmax.hpp"
#include "openvino/pass/manager.hpp"
#include "utils.hpp"
#include "whisper/transformations/scaled_dot_product_attention_decomposition.hpp"

namespace {
void reshape_hidden_states_to_static(std::shared_ptr<ov::Model> model, const ov::PartialShape& lhstates_shape) {
    ov::PartialShape new_shape = model->input("encoder_hidden_states").get_partial_shape();
    OPENVINO_ASSERT(new_shape.size() > 1 && lhstates_shape.size() > 1);
    new_shape[1] = lhstates_shape[1];
    std::map<std::string, ov::PartialShape> name_to_shape{{"encoder_hidden_states", new_shape}};
    model->reshape(name_to_shape);
}

void decompose_scaled_dot_product_attention(std::shared_ptr<ov::Model> model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::genai::WhisperScaledDotProductAttentionDecomposition>();
    manager.run_passes(model);
}

/**
 * todo: mark encoder attention softmax nodes during decomposition to avoid such hacks
 */
void add_encoder_attention_qk_outputs(std::shared_ptr<ov::Model> model) {
    size_t idx = 0;
    for (auto& op : model->get_ordered_ops()) {
        if (op->get_type_info().name != std::string("Softmax")) {
            continue;
        }

        model->add_output(op->output(0)).add_names({"encoder_attn_qk_" + std::to_string(idx)});
        idx++;
    }
}

void add_qk_scaled_scores_outputs(std::shared_ptr<ov::Model> model) {
    // <layer id="278" name="Add_21002" type="Add" version="opset1">
    //     <data auto_broadcast="numpy" />
    //     <input>
    //         <port id="0" precision="FP32">
    //             <dim>-1</dim>
    //             <dim>6</dim>
    //             <dim>-1</dim>
    //             <dim>-1</dim>
    //         </port>
    //         <port id="1" precision="FP32" />
    //     </input>
    //     <output>
    //         <port id="2" precision="FP32" names="qk_scaled_scores">
    //             <dim>-1</dim>
    //             <dim>6</dim>
    //             <dim>-1</dim>
    //             <dim>-1</dim>
    //         </port>
    //     </output>
    // </layer>
    size_t idx = 0;
    for (auto& op : model->get_ordered_ops()) {
        if (op->get_type_info().name != std::string("Add")) {
            continue;
        }

        bool should_skip_op = true;

        for (const auto& output : op->outputs()) {
            for (const auto& name : output.get_names()) {
                if (name.find("qk_scaled_scores") != std::string::npos) {
                    should_skip_op = false;
                    break;
                }
            }

            // output found
            if (!should_skip_op) {
                break;
            }
        }

        if (should_skip_op) {
            continue;
        }

        model->add_output(op->output(0)).add_names({"qk_scaled_scores_" + std::to_string(idx)});
        idx++;
    }
}

}  // namespace

namespace ov::genai {
WhisperStatefullDecoder::WhisperStatefullDecoder(const std::filesystem::path& models_path,
                                                 const std::string& device,
                                                 const ov::AnyMap& properties,
                                                 const ov::PartialShape& lhs_shape,
                                                 const ov::genai::WhisperConfig& model_config,
                                                 const bool decompose_cross_attention_spda)
    : m_model_config(model_config),
      m_decompose_cross_attention_spda_ops(decompose_cross_attention_spda) {
    ov::Core core = utils::singleton_core();

    auto model = core.read_model(models_path / "openvino_decoder_model.xml", {}, properties);

    if (m_decompose_cross_attention_spda_ops) {
        auto start_time = std::chrono::steady_clock::now();
        decompose_scaled_dot_product_attention(model);

        add_encoder_attention_qk_outputs(model);
        add_qk_scaled_scores_outputs(model);
    }

    m_has_cache_position = utils::has_input(model, "cache_position");

    ov::CompiledModel compiled_model;
    // todo: check if applicable for NPU
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
    if (!m_decompose_cross_attention_spda_ops) {
        OPENVINO_THROW("Encoder attention heads are not decomposed. Cannot get encoder QKs.");
    }

    // [layers] * [batch, num_heads, seq_len, frame_len] -> [layers] * [batch, seq_len, frame_len]
    std::vector<ov::Tensor> alignment_qks;
    for (const auto& [layer_idx, head_idx] : alignment_heads) {
        const Tensor alignment_tensor = m_request.get_tensor("qk_scaled_scores_" + std::to_string(layer_idx));

        // [batch, num_heads, seq_len, frame_len]
        const ov::Shape& alignment_shape = alignment_tensor.get_shape();

        // [batch, seq_len, frame_len]
        ov::Tensor head_tensor{ov::element::f32, {alignment_shape[0], alignment_shape[2], alignment_shape[3]}};
        auto* alignment_data = alignment_tensor.data<float>();
        auto* head_data = head_tensor.data<float>();
        const size_t batch_size = alignment_shape[0];
        const size_t num_heads = alignment_shape[1];
        const size_t seq_len = alignment_shape[2];
        const size_t frame_len = alignment_shape[3];

        for (size_t batch = 0; batch < batch_size; ++batch) {
            const size_t batch_offset = batch * num_heads * seq_len * frame_len;
            const size_t head_offset = head_idx * seq_len * frame_len;
            const size_t head_batch_offset = batch * seq_len * frame_len;

            std::memcpy(head_data + head_batch_offset,
                        alignment_data + batch_offset + head_offset,
                        seq_len * frame_len * sizeof(float));
        }

        alignment_qks.push_back(head_tensor);
    }

    return alignment_qks;
}

}  // namespace ov::genai
