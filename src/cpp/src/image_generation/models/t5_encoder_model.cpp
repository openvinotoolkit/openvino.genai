// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/t5_encoder_model.hpp"

#include <fstream>

#include "json_utils.hpp"
#include "lora_helper.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

std::filesystem::path get_tokenizer_path_by_text_encoder(const std::filesystem::path& text_encoder_path);

T5EncoderModel::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);
}

T5EncoderModel::T5EncoderModel(const std::filesystem::path& root_dir) :
    m_tokenizer(get_tokenizer_path_by_text_encoder(root_dir)),
    m_config(root_dir / "config.json") {
    ov::Core core = utils::singleton_core();
    m_model = core.read_model((root_dir / "openvino_model.xml").string());
}

T5EncoderModel::T5EncoderModel(const std::filesystem::path& root_dir,
                             const std::string& device,
                             const ov::AnyMap& properties) :
    T5EncoderModel(root_dir) {
    compile(device, properties);
}

T5EncoderModel::T5EncoderModel(const T5EncoderModel&) = default;

const T5EncoderModel::Config& T5EncoderModel::get_config() const {
    return m_config;
}

void T5EncoderModel::set_max_sequence_length(size_t max_sequence_length) {
    if (max_sequence_length != -1)
        m_config.max_sequence_length = max_sequence_length;
}

T5EncoderModel& T5EncoderModel::reshape(int batch_size) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    ov::PartialShape input_shape = m_model->input(0).get_partial_shape();
    input_shape[0] = batch_size;
    input_shape[1] = m_config.max_sequence_length;
    std::map<size_t, ov::PartialShape> idx_to_shape{{0, input_shape}};
    m_model->reshape(idx_to_shape);

    return *this;
}

T5EncoderModel& T5EncoderModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::Core core = utils::singleton_core();
    ov::CompiledModel compiled_model;
    compiled_model = core.compile_model(m_model, device, properties);
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

ov::Tensor T5EncoderModel::infer(const std::string& pos_prompt) {
    OPENVINO_ASSERT(m_request, "T5 encoder model must be compiled first. Cannot infer non-compiled model");

    const int32_t pad_token_id = m_tokenizer.get_pad_token_id();

    auto perform_tokenization = [&](const std::string& prompt, ov::Tensor input_ids) {
        ov::Tensor input_ids_token = m_tokenizer.encode(prompt).input_ids;
        size_t min_size = std::min(input_ids.get_size(), input_ids_token.get_size());

        std::fill_n(input_ids.data<int32_t>(), input_ids.get_size(), pad_token_id);
        std::copy_n(input_ids_token.data<std::int64_t>(), min_size, input_ids.data<std::int32_t>());
    };

    ov::Tensor input_ids = m_request.get_input_tensor();

    // reshape in case of dynamic model
    if (input_ids.get_shape()[0] == 0 || input_ids.get_shape()[1] == 0) {
        input_ids.set_shape({1, m_config.max_sequence_length});
    }

    perform_tokenization(pos_prompt, input_ids);

    // text embeddings
    m_request.set_tensor("input_ids", input_ids);
    m_request.infer();

    return m_request.get_output_tensor(0);
}

ov::Tensor T5EncoderModel::get_output_tensor(const size_t idx) {
    return m_request.get_output_tensor(idx);
}

} // namespace genai
} // namespace ov
