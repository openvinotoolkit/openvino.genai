// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "video_generation/models/ltx2_text_connectors.hpp"

#include "utils.hpp"
#include "video_generation/video_generation_utils.hpp"

using namespace ov::genai;

LTX2TextConnectors::LTX2TextConnectors(const std::filesystem::path& root_dir) {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

LTX2TextConnectors::LTX2TextConnectors(const std::filesystem::path& root_dir,
                                       const std::string& device,
                                       const ov::AnyMap& properties)
    : LTX2TextConnectors(root_dir) {
    compile(device, properties);
}

LTX2TextConnectors::LTX2TextConnectors(const LTX2TextConnectors&) = default;

std::shared_ptr<LTX2TextConnectors> LTX2TextConnectors::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "LTX2TextConnectors must have exactly one of m_model or m_request initialized");

    std::shared_ptr<LTX2TextConnectors> cloned = std::make_shared<LTX2TextConnectors>(*this);

    if (m_model) {
        cloned->m_model = m_model->clone();
    } else {
        cloned->m_request = m_request.get_compiled_model().create_infer_request();
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

LTX2TextConnectors& LTX2TextConnectors::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
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
