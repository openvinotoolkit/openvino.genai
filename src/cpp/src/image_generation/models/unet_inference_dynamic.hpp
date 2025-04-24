// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "image_generation/models/unet_inference.hpp"
#include "lora_helper.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

class UNet2DConditionModel::UNetInferenceDynamic : public UNet2DConditionModel::UNetInference {
public:
    virtual void compile(std::shared_ptr<ov::Model> model, const std::string& device, const ov::AnyMap& properties) override {
        ov::CompiledModel compiled_model = utils::singleton_core().compile_model(model, device, properties);
        ov::genai::utils::print_compiled_model_properties(compiled_model, "UNet 2D Condition dynamic model");

        for (size_t i = 0; i < 4; i++)
            m_requests.emplace_back(compiled_model.create_infer_request());
    }

    virtual void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states, size_t request_idx = 0) override {
        OPENVINO_ASSERT(m_requests.size(), "UNet model must be compiled first");
        m_requests[request_idx].set_tensor(tensor_name, encoder_hidden_states);
    }

    virtual void set_adapters(AdapterController &adapter_controller, const AdapterConfig& adapters, size_t request_idx = 0) override {
        OPENVINO_ASSERT(m_requests.size(), "UNet model must be compiled first");
        adapter_controller.apply(m_requests[request_idx], adapters);
    }

    virtual ov::Tensor infer(ov::Tensor sample, ov::Tensor timestep, size_t request_idx = 0) override {
        OPENVINO_ASSERT(m_requests.size(), "UNet model must be compiled first. Cannot infer non-compiled model");

        m_requests[request_idx].set_tensor("sample", sample);
        m_requests[request_idx].set_tensor("timestep", timestep);

        m_requests[request_idx].infer();

        return m_requests[request_idx].get_output_tensor();
    }

private:
    std::vector<ov::InferRequest> m_requests;
};

}  // namespace genai
}  // namespace ov