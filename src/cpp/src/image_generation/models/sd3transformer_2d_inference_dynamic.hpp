// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "image_generation/models/sd3transformer_2d_inference.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

class SD3Transformer2DModel::InferenceDynamic : public SD3Transformer2DModel::Inference {
public:
    virtual std::shared_ptr<Inference> clone() override {
        OPENVINO_ASSERT(static_cast<bool>(m_request), "SD3Transformer2DModel must have m_request initialized");
        InferenceDynamic cloned(*this);
        cloned.m_request = m_request.get_compiled_model().create_infer_request();
        return std::make_shared<InferenceDynamic>(cloned);
    }

    virtual void compile(std::shared_ptr<ov::Model> model,
                         const std::string& device,
                         const ov::AnyMap& properties) override {
        ov::CompiledModel compiled_model = utils::singleton_core().compile_model(model, device, properties);
        ov::genai::utils::print_compiled_model_properties(compiled_model, "SD3 Transformer 2D model");
        m_request = compiled_model.create_infer_request();
    }

    virtual void set_adapters(AdapterController& m_adapter_controller, const AdapterConfig& adapters) override {
        OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
        m_adapter_controller.apply(m_request, adapters);
    }

    virtual void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) override {
        OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
        m_request.set_tensor(tensor_name, encoder_hidden_states);
    }

    virtual ov::Tensor infer(ov::Tensor latent_model_input, ov::Tensor timestep) override {
        OPENVINO_ASSERT(m_request, "Transformer model must be compiled first. Cannot infer non-compiled model");

        m_request.set_tensor("hidden_states", latent_model_input);
        m_request.set_tensor("timestep", timestep);
        m_request.infer();

        return m_request.get_output_tensor();
    }

private:
    ov::InferRequest m_request;
};

}  // namespace genai
}  // namespace ov
