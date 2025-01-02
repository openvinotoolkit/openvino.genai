// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "image_generation/models/unet_inference.hpp"
#include "lora_helper.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {


class UNet2DConditionModel::UNetInferenceDynamic : public UNet2DConditionModel::UNetInference {

public:

    virtual void compile(std::shared_ptr<ov::Model> model, const std::string& device, const ov::AnyMap& properties) override
    {
        ov::Core core = utils::singleton_core();

        compiled_model = std::make_shared<ov::CompiledModel>(core.compile_model(model, device, properties));
        ov::genai::utils::print_compiled_model_properties(*compiled_model, "UNet 2D Condition dynamic model");
        std::cout << "unet dynamic compile" << std::endl;
        m_request = compiled_model->create_infer_request();
    }

    virtual void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) override
    {
        std::cout << "unet dynamic set_hidden_states" << std::endl;
        OPENVINO_ASSERT(m_request, "UNet model must be compiled first");
        m_request.set_tensor(tensor_name, encoder_hidden_states);
    }

    virtual void set_adapters(AdapterController &adapter_controller, const AdapterConfig& adapters) override
    {
        OPENVINO_ASSERT(m_request, "UNet model must be compiled first");
        adapter_controller.apply(m_request, adapters);
    }

    virtual ov::Tensor infer(ov::Tensor sample, ov::Tensor timestep) override
    {
        std::cout << "unet dynamic infer" << std::endl;
        OPENVINO_ASSERT(m_request, "UNet model must be compiled first. Cannot infer non-compiled model");
        std::cout << sample.get_shape().to_string() << std::endl; 
        m_request.set_tensor("sample", sample);
        std::cout << timestep.get_shape().to_string() << std::endl; 
        m_request.set_tensor("timestep", timestep);
        ov::CompiledModel test =  m_request.get_compiled_model();
        std::cout << "compiled model ok" << test.inputs().at(0).get_any_name() << std::endl; 
        m_request.infer();
        std::cout << "unet dynamic inference complated" << std::endl; 

        return m_request.get_output_tensor();
    }

    UNetInferenceDynamic(const UNetInferenceDynamic & origin_model){
        OPENVINO_ASSERT(origin_model.compiled_model, "Source model must be compiled first");
        compiled_model = origin_model.compiled_model;
        m_request = compiled_model->create_infer_request();
    }
    UNetInferenceDynamic() = default;

private:

    ov::InferRequest m_request;
    std::shared_ptr<ov::CompiledModel> compiled_model;
};


}  // namespace genai
}  // namespace ov