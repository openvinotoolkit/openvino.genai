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
 * It's not reliable in generale case to extract encoder attn just by node Softmax type
 * Whisper only has Softmax for encoder attention weights output as only encoder attention sdpa block were decomposed
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

}  // namespace

namespace ov::genai {
WhisperStatefullDecoder::WhisperStatefullDecoder(const std::filesystem::path& models_path,
                                                 const std::string& device,
                                                 const ov::AnyMap& properties,
                                                 const ov::PartialShape& lhs_shape,
                                                 const ov::genai::WhisperConfig& model_config,
                                                 const bool enable_encoder_attention_qk_accumulation)
    : m_model_config(model_config),
      m_encoder_attention_qk_accumulation_enabled(enable_encoder_attention_qk_accumulation) {
    ov::Core core = utils::singleton_core();

    auto model = core.read_model(models_path / "openvino_decoder_model.xml", {}, properties);

    if (m_encoder_attention_qk_accumulation_enabled) {
        auto start_time = std::chrono::steady_clock::now();
        decompose_scaled_dot_product_attention(model);
        add_encoder_attention_qk_outputs(model);
        std::cout << "[WhisperStatefullDecoder] SDPA decomposition took: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                           start_time)
                         .count()
                  << " ms" << std::endl;
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
    if (m_encoder_attention_qk_accumulation_enabled) {
        _accumulate_encoder_qks();
    }
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

std::vector<Tensor> WhisperStatefullDecoder::get_encoder_qks() const {
    if (!m_encoder_attention_qk_accumulation_enabled) {
        OPENVINO_THROW("Encoder attention QK accumulation is not enabled");
    }
    return m_encoder_qks;
}

// todo: accumulate only alignment heads QKs
void WhisperStatefullDecoder::_accumulate_encoder_qks() {
    const size_t decoder_layers = m_model_config.decoder_layers;
    for (size_t layer = 0; layer < decoder_layers; layer++) {
        // [batch, head_num, seq_len, frame_len]
        const Tensor encoder_qk_tensor = m_request.get_tensor("encoder_attn_qk_" + std::to_string(layer));

        if (m_encoder_qks.size() <= layer) {
            Tensor copy{encoder_qk_tensor.get_element_type(), encoder_qk_tensor.get_shape()};
            encoder_qk_tensor.copy_to(copy);
            m_encoder_qks.push_back(copy);
        } else {
            const Tensor& accumulated_tensor = m_encoder_qks.at(layer);

            const Shape accumulated_shape = accumulated_tensor.get_shape();

            const Shape& iteration_shape = encoder_qk_tensor.get_shape();
            OPENVINO_ASSERT(accumulated_shape[0] == iteration_shape[0],
                            "Batch size mismatch during encoder QK accumulation");
            OPENVINO_ASSERT(accumulated_shape[1] == iteration_shape[1],
                            "Head num mismatch during encoder QK accumulation");
            OPENVINO_ASSERT(accumulated_shape[3] == iteration_shape[3],
                            "Frame len mismatch during encoder QK accumulation");

            Shape new_shape{accumulated_shape};
            // set new seq_len
            new_shape[2] += iteration_shape[2];

            // copy to new tensor
            // todo: create host tensor
            Tensor new_accumulated_tensor{ov::element::f32, new_shape};
            auto new_data = new_accumulated_tensor.data<float>();
            auto accumulated_data = accumulated_tensor.data<float>();
            auto iteration_data = encoder_qk_tensor.data<float>();

            const size_t batch_size = accumulated_shape[0];
            const size_t head_num = accumulated_shape[1];
            const size_t frame_len = accumulated_shape[3];

            // accumulated shape [batch_size, head_num, acc_seq_len, frame_len]
            // iteration shape   [batch_size, head_num, iter_seq_len, frame_len]
            // new shape         [batch_size, head_num, acc_seq_len + iter_seq_len, frame_len]

            for (size_t batch = 0; batch < batch_size; batch++) {
                const size_t accumulated_batch_offset = batch * head_num * accumulated_shape[2] * frame_len;
                const size_t iteration_batch_offset = batch * head_num * iteration_shape[2] * frame_len;
                const size_t new_batch_offset = batch * head_num * new_shape[2] * frame_len;

                for (size_t head = 0; head < head_num; head++) {
                    const size_t accumulated_offset =
                        accumulated_batch_offset + head * accumulated_shape[2] * frame_len;
                    const size_t iteration_offset = iteration_batch_offset + head * iteration_shape[2] * frame_len;
                    const size_t new_offset = new_batch_offset + head * new_shape[2] * frame_len;

                    // copy accumulated
                    std::memcpy(new_data + new_offset,
                                accumulated_data + accumulated_offset,
                                accumulated_shape[2] * frame_len * sizeof(float));

                    // copy iteration
                    std::memcpy(new_data + new_offset + accumulated_shape[2] * frame_len,
                                iteration_data + iteration_offset,
                                iteration_shape[2] * frame_len * sizeof(float));
                }
            }

            // Replace the old accumulated tensor with the new one
            m_encoder_qks[layer] = new_accumulated_tensor;
        }
    }
}

}  // namespace ov::genai
