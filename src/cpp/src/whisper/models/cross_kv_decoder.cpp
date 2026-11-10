// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cross_kv_decoder.hpp"

#include <cstring>
#include <numeric>
#include <utility>

#include "utils.hpp"
#include "whisper/alignment_heads.hpp"
#include "whisper/transformations/cross_kv_inputs.hpp"
#include "whisper/word_level_timestamps.hpp"

namespace {

size_t row_bytes(const ov::Tensor& tensor) {
    return tensor.get_byte_size() / tensor.get_shape().at(0);
}

}  // namespace

namespace ov::genai {

std::shared_ptr<WhisperCrossKVDecoder> WhisperCrossKVDecoder::create(const std::filesystem::path& models_path,
                                                                     const std::string& device,
                                                                     const ov::AnyMap& properties,
                                                                     const ov::PartialShape& lhs_shape,
                                                                     const bool decompose_cross_attention_sdpa_ops) {
    if (lhs_shape.rank().is_dynamic() || lhs_shape.size() != 3 || lhs_shape[1].is_dynamic() ||
        lhs_shape[2].is_dynamic()) {
        return nullptr;
    }

    ov::Core core = utils::singleton_core();
    auto model = core.read_model(models_path / "openvino_decoder_model.xml", {}, std::as_const(properties));

    auto split = split_whisper_cross_kv(model);
    if (!split.has_value()) {
        return nullptr;
    }
    const ov::element::Type cross_kv_type = split->projector->output(0).get_element_type();
    if (cross_kv_type != ov::element::f32) {
        return nullptr;
    }

    split->projector->reshape(
        {{"encoder_hidden_states", ov::PartialShape{1, lhs_shape[1].get_length(), lhs_shape[2].get_length()}}});
    const ov::PartialShape& first_output_shape = split->projector->output(0).get_partial_shape();
    for (const auto& output : split->projector->outputs()) {
        if (output.get_element_type() != cross_kv_type || output.get_partial_shape().is_dynamic() ||
            output.get_partial_shape() != first_output_shape) {
            return nullptr;
        }
    }
    const ov::Shape cross_kv_row_shape = first_output_shape.to_shape();

    auto& decoder = split->decoder;
    const bool has_cache_position = utils::has_input(decoder, "cache_position");
    if (decompose_cross_attention_sdpa_ops) {
        decompose_scaled_dot_product_attention_for_whisper(decoder);
        add_cross_attention_qk_scaled_scores_outputs_for_whisper(decoder);
    }
    utils::apply_slice_before_matmul_transformation(decoder);

    ov::CompiledModel compiled_decoder = core.compile_model(decoder, device, properties);
    utils::print_compiled_model_properties(compiled_decoder, "whisper decoder model");
    ov::CompiledModel compiled_projector = core.compile_model(split->projector, device, properties);

    return std::shared_ptr<WhisperCrossKVDecoder>(new WhisperCrossKVDecoder(compiled_decoder.create_infer_request(),
                                                                            compiled_projector.create_infer_request(),
                                                                            std::move(split->names),
                                                                            cross_kv_type,
                                                                            cross_kv_row_shape,
                                                                            has_cache_position,
                                                                            decompose_cross_attention_sdpa_ops));
}

WhisperCrossKVDecoder::WhisperCrossKVDecoder(ov::InferRequest request,
                                             ov::InferRequest projector,
                                             std::vector<std::string> cross_kv_names,
                                             const ov::element::Type& cross_kv_type,
                                             const ov::Shape& cross_kv_row_shape,
                                             const bool has_cache_position,
                                             const bool decompose_cross_attention_sdpa_ops)
    : m_request(std::move(request)),
      m_projector(std::move(projector)),
      m_cross_kv_names(std::move(cross_kv_names)),
      m_cross_kv_type(cross_kv_type),
      m_cross_kv_row_shape(cross_kv_row_shape),
      m_has_cache_position(has_cache_position),
      m_decompose_cross_attention_sdpa_ops(decompose_cross_attention_sdpa_ops) {
    const auto& projector_input = m_projector.get_compiled_model().input("encoder_hidden_states");
    m_idle_projector_input = ov::Tensor(projector_input.get_element_type(), projector_input.get_shape());
    m_cross_kv.resize(m_cross_kv_names.size());
}

void WhisperCrossKVDecoder::compute_cross_kv(const Tensor& encoder_hidden_state) {
    OPENVINO_ASSERT(!encoder_hidden_state.is<ov::RemoteTensor>(),
                    "Whisper cross-attention K/V projection expects encoder hidden states in host memory.");
    const ov::Shape encoder_shape = encoder_hidden_state.get_shape();
    const ov::Shape encoder_row_shape = m_idle_projector_input.get_shape();
    OPENVINO_ASSERT(encoder_shape.size() == 3 && encoder_shape[0] > 0 && encoder_shape[1] == encoder_row_shape[1] &&
                        encoder_shape[2] == encoder_row_shape[2],
                    "Unexpected Whisper encoder hidden state shape ",
                    encoder_shape,
                    ".");
    const size_t rows = encoder_shape[0];

    ov::Shape storage_shape = m_cross_kv_row_shape;
    storage_shape[0] = rows;
    for (auto& tensor : m_cross_kv) {
        if (rows > m_cross_kv_capacity) {
            tensor = ov::Tensor(m_cross_kv_type, storage_shape);
        } else {
            tensor.set_shape(storage_shape);
        }
    }
    m_cross_kv_capacity = std::max(m_cross_kv_capacity, rows);

    // Encoder rows are only read by the projector.
    auto* encoder_data = static_cast<char*>(const_cast<void*>(encoder_hidden_state.data()));
    const size_t encoder_row_bytes = row_bytes(encoder_hidden_state);
    for (size_t row = 0; row < rows; ++row) {
        m_projector.set_tensor("encoder_hidden_states",
                               ov::Tensor(encoder_hidden_state.get_element_type(),
                                          encoder_row_shape,
                                          encoder_data + row * encoder_row_bytes));
        for (size_t i = 0; i < m_cross_kv.size(); ++i) {
            auto* storage = static_cast<char*>(m_cross_kv[i].data());
            m_projector.set_tensor(
                m_cross_kv_names[i],
                ov::Tensor(m_cross_kv_type, m_cross_kv_row_shape, storage + row * row_bytes(m_cross_kv[i])));
        }
        m_projector.infer();
    }
    m_projector.set_tensor("encoder_hidden_states", m_idle_projector_input);

    m_cross_kv_rows = rows;
    m_cross_kv_shared = rows == 1;
    bind_cross_kv();
    m_cross_kv_valid = true;
}

void WhisperCrossKVDecoder::bind_cross_kv() {
    for (size_t i = 0; i < m_cross_kv.size(); ++i) {
        m_request.set_tensor(m_cross_kv_names[i], m_cross_kv[i]);
    }
}

void WhisperCrossKVDecoder::start_async(const Tensor& encoder_hidden_state,
                                        const Tensor& input_ids,
                                        const Tensor& beam_idx) {
    OPENVINO_ASSERT(!m_infer_in_flight,
                    "Whisper decoder cannot start an inference before the previous one is waited for.");
    if (!m_cross_kv_valid) {
        compute_cross_kv(encoder_hidden_state);
    }

    validate_beam_idx(input_ids.get_shape().at(0), beam_idx);

    if (m_has_cache_position) {
        set_cache_position_tensor(input_ids.get_shape().at(1));
    }
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("beam_idx", beam_idx);

    m_request.start_async();
    m_infer_in_flight = true;
}

void WhisperCrossKVDecoder::validate_beam_idx(const size_t rows, const Tensor& beam_idx) const {
    OPENVINO_ASSERT(beam_idx.get_element_type() == ov::element::i32 && beam_idx.get_shape() == ov::Shape{rows},
                    "Whisper decoder expects an i32 beam_idx of shape [",
                    rows,
                    "] for ",
                    rows,
                    " input rows, got ",
                    beam_idx.get_element_type(),
                    " ",
                    beam_idx.get_shape(),
                    ".");
    if (m_cross_kv_shared) {
        return;
    }

    OPENVINO_ASSERT(rows == m_cross_kv_rows,
                    "Whisper decoder got ",
                    rows,
                    " input rows for ",
                    m_cross_kv_rows,
                    " per-row cross-attention rows.");
    const int32_t* beams = beam_idx.data<const int32_t>();
    for (size_t row = 0; row < rows; ++row) {
        const int32_t expected = m_retained_rows ? (*m_retained_rows)[row] : static_cast<int32_t>(row);
        OPENVINO_ASSERT(beams[row] == expected,
                        "Whisper decoder beam_idx[",
                        row,
                        "] is ",
                        beams[row],
                        " but must be ",
                        expected,
                        m_retained_rows ? ", the source row passed to retain_rows(), so that self-attention state "
                                          "follows the retained cross-attention rows."
                                        : ": decoder rows with per-row cross-attention K/V keep their order unless "
                                          "they are reordered by retain_rows().");
    }
}

void WhisperCrossKVDecoder::retain_rows(const std::vector<int32_t>& source_rows) {
    OPENVINO_ASSERT(m_cross_kv_valid, "Whisper decoder rows can be retained only after cross-attention K/V exist.");
    OPENVINO_ASSERT(!m_infer_in_flight, "Whisper decoder rows cannot be retained while an inference is in flight.");
    OPENVINO_ASSERT(!m_cross_kv_shared,
                    "Whisper cross-attention K/V projected from one encoder row are shared by all decoder rows and "
                    "cannot be retained per row.");
    OPENVINO_ASSERT(!m_retained_rows,
                    "Whisper decoder rows were already retained. Run a decoder step with the retained rows as "
                    "beam_idx before retaining rows again.");
    const size_t new_rows = source_rows.size();
    OPENVINO_ASSERT(new_rows > 0 && new_rows <= m_cross_kv_rows,
                    "Cannot retain ",
                    new_rows,
                    " of ",
                    m_cross_kv_rows,
                    " Whisper decoder rows.");

    std::vector<bool> used(m_cross_kv_rows, false);
    for (size_t row = 0; row < new_rows; ++row) {
        const int32_t source = source_rows[row];
        OPENVINO_ASSERT(source >= 0 && static_cast<size_t>(source) < m_cross_kv_rows,
                        "Whisper decoder row ",
                        row,
                        " cannot be filled from row ",
                        source,
                        " of ",
                        m_cross_kv_rows,
                        " rows.");
        OPENVINO_ASSERT(static_cast<size_t>(source) == row || static_cast<size_t>(source) >= new_rows,
                        "Whisper decoder row ",
                        row,
                        " can be filled only from itself or from a row at or beyond the new width ",
                        new_rows,
                        ", got row ",
                        source,
                        ".");
        OPENVINO_ASSERT(!used[source], "Whisper decoder row ", source, " is retained more than once.");
        used[source] = true;
    }

    for (size_t row = 0; row < new_rows; ++row) {
        const size_t source = static_cast<size_t>(source_rows[row]);
        if (source == row) {
            continue;
        }
        for (auto& tensor : m_cross_kv) {
            auto* data = static_cast<char*>(tensor.data());
            const size_t bytes = row_bytes(tensor);
            std::memcpy(data + row * bytes, data + source * bytes, bytes);
        }
    }

    ov::Shape storage_shape = m_cross_kv_row_shape;
    storage_shape[0] = new_rows;
    for (auto& tensor : m_cross_kv) {
        tensor.set_shape(storage_shape);
    }
    m_cross_kv_rows = new_rows;
    bind_cross_kv();
    m_retained_rows = source_rows;
}

void WhisperCrossKVDecoder::set_cache_position_tensor(const size_t seq_len) {
    ov::Tensor cache_position_tensor = m_request.get_tensor("cache_position");

    int64_t start_cache_position = 0;

    if (cache_position_tensor.get_size() != 0) {
        start_cache_position = cache_position_tensor.data<int64_t>()[cache_position_tensor.get_size() - 1] + 1;
    }

    cache_position_tensor.set_shape({seq_len});

    auto cache_data = cache_position_tensor.data<int64_t>();
    std::iota(cache_data, cache_data + seq_len, start_cache_position);
}

Tensor WhisperCrossKVDecoder::wait() {
    const bool completes_step = std::exchange(m_infer_in_flight, false);
    try {
        m_request.wait();
    } catch (...) {
        // A failed step leaves self-attention state undefined, so later steps must start from a fresh state.
        try {
            reset_state();
        } catch (...) {
        }
        throw;
    }
    if (completes_step) {
        m_retained_rows.reset();
    }
    return m_request.get_tensor("logits");
}

void WhisperCrossKVDecoder::reset_state() {
    if (m_infer_in_flight) {
        wait();
    }
    m_request.reset_state();
    if (m_has_cache_position) {
        m_request.set_tensor("cache_position", create_host_tensor(ov::element::i64, {0}));
    }
    m_cross_kv_valid = false;
    m_cross_kv_shared = false;
    m_retained_rows.reset();
    m_infer_in_flight = false;
}

ov::Tensor WhisperCrossKVDecoder::create_host_tensor(const element::Type element_type, const Shape& shape) {
    try {
        return m_request.get_compiled_model().get_context().create_host_tensor(element_type, shape);
    } catch (std::exception&) {
        return ov::Tensor(element_type, shape);
    }
}

std::vector<Tensor> WhisperCrossKVDecoder::get_alignments_heads_qks(
    const std::vector<std::pair<size_t, size_t>>& alignment_heads) {
    OPENVINO_ASSERT(m_decompose_cross_attention_sdpa_ops,
                    "Encoder attention heads are not decomposed. Cannot get encoder attention QKs.");

    return get_whisper_alignments_heads_qks(m_request, alignment_heads);
}

}  // namespace ov::genai
