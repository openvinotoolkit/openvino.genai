// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/t5_encoder_model.hpp"

#include <fstream>

#include "json_utils.hpp"
#include "lora/helper.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

std::filesystem::path get_tokenizer_path_by_text_encoder(const std::filesystem::path& text_encoder_path);

T5EncoderModel::T5EncoderModel(const std::filesystem::path& root_dir) :
    m_tokenizer(get_tokenizer_path_by_text_encoder(root_dir)) {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

T5EncoderModel::T5EncoderModel(const std::filesystem::path& root_dir,
                             const std::string& device,
                             const ov::AnyMap& properties) :
    T5EncoderModel(root_dir) {
    compile(device, properties);
}

T5EncoderModel::T5EncoderModel(const std::string& model,
                               const Tensor& weights,
                               const Tokenizer& tokenizer) :
    m_tokenizer(tokenizer) {
    m_model = utils::singleton_core().read_model(model, weights);
}

T5EncoderModel::T5EncoderModel(const std::string& model,
                               const Tensor& weights,
                               const Tokenizer& tokenizer,
                               const std::string& device,
                               const ov::AnyMap& properties) :
    T5EncoderModel(model, weights, tokenizer) {
    compile(device, properties);
}

T5EncoderModel::T5EncoderModel(const T5EncoderModel&) = default;

std::shared_ptr<T5EncoderModel> T5EncoderModel::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request), "T5EncoderModel must have exactly one of m_model or m_request initialized");

    std::shared_ptr<T5EncoderModel> cloned = std::make_shared<T5EncoderModel>(*this);

    if (m_model) {
        cloned->m_model = m_model->clone();
    } else {
        cloned->m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

T5EncoderModel& T5EncoderModel::reshape(int batch_size, int max_sequence_length) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    ov::PartialShape input_shape = m_model->input(0).get_partial_shape();
    input_shape[0] = batch_size;
    input_shape[1] = max_sequence_length;
    std::map<size_t, ov::PartialShape> idx_to_shape{{0, input_shape}};
    m_model->reshape(idx_to_shape);

    return *this;
}

T5EncoderModel& T5EncoderModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *extract_adapters_from_properties(properties));
    ov::genai::utils::print_compiled_model_properties(compiled_model, "T5 encoder model");
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

ov::Tensor T5EncoderModel::infer(const std::string& pos_prompt, const std::string& neg_prompt, bool do_classifier_free_guidance, int max_sequence_length) {
    OPENVINO_ASSERT(m_request, "T5 encoder model must be compiled first. Cannot infer non-compiled model");

    const int32_t pad_token_id = m_tokenizer.get_pad_token_id();

    auto perform_tokenization = [&](const std::string& prompt, ov::Tensor input_ids) {
        ov::Tensor input_ids_token = m_tokenizer.encode(prompt).input_ids;
        size_t min_length = std::min(input_ids.get_size(), input_ids_token.get_size());

        if (input_ids.get_element_type() == ov::element::i32) {
            std::fill_n(input_ids.data<int32_t>(), input_ids.get_size(), pad_token_id);
            std::copy_n(input_ids_token.data<int64_t>(), min_length, input_ids.data<int32_t>());
        } else {
            std::fill_n(input_ids.data<int64_t>(), input_ids.get_size(), pad_token_id);
            std::copy_n(input_ids_token.data<int64_t>(), min_length, input_ids.data<int64_t>());
        }
    };

    ov::Tensor input_ids = m_request.get_input_tensor();

    // reshape in case of dynamic model
    ov::Shape input_ids_shape = input_ids.get_shape();

    OPENVINO_ASSERT(input_ids_shape[1] == 0 || max_sequence_length == input_ids_shape[1],
        "In case of T5EncoderModel was reshaped before, reshape's max_sequence_length ", input_ids_shape[1], " must be equal to ",
        "infer's max_sequence_length ", max_sequence_length);

    if (input_ids_shape[0] == 0 || input_ids_shape[1] == 0) {
        size_t batch_size = do_classifier_free_guidance ? 2 : 1;
        input_ids.set_shape({batch_size, static_cast<size_t>(max_sequence_length)});
    }

    size_t current_batch_idx = 0;
    if (do_classifier_free_guidance) {
        perform_tokenization(neg_prompt,
                             ov::Tensor(input_ids, {current_batch_idx    , 0},
                                                   {current_batch_idx + 1, input_ids.get_shape()[1]}));
        ++current_batch_idx;
    } else {
        // Negative prompt is ignored when --guidanceScale < 1.0
    }

    // perform_tokenization(pos_prompt, input_ids);
    perform_tokenization(pos_prompt,
                         ov::Tensor(input_ids, {current_batch_idx    , 0},
                                               {current_batch_idx + 1, input_ids.get_shape()[1]}));

    // text embeddings
    m_request.infer();

    return m_request.get_output_tensor(0);
}

ov::Tensor T5EncoderModel::get_output_tensor(const size_t idx) {
    return m_request.get_output_tensor(idx);
}

} // namespace genai
} // namespace ov
