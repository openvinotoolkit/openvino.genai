// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "encoder.hpp"

#include "utils.hpp"

namespace ov::genai {

FunASREncoder::FunASREncoder(const std::filesystem::path& models_path,
                             const std::string& device,
                             const ov::AnyMap& properties) {
    ov::Core core = utils::singleton_core();
    ov::CompiledModel model = core.compile_model(models_path / "openvino_encoder_model.xml", device, properties);
    utils::print_compiled_model_properties(model, "fun-asr encoder model");
    m_request = model.create_infer_request();
}

ov::Tensor FunASREncoder::encode(const ov::Tensor& features) {
    m_request.set_tensor("input_features", features);
    m_request.infer();
    return m_request.get_tensor("last_hidden_state");
}

}  // namespace ov::genai
