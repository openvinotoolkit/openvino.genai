// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <memory>

#include "image_generation/models/unet_inference.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

class UNet2DConditionModel::UNetInferenceDynamic : public UNet2DConditionModel::UNetInference {
public:
    virtual std::shared_ptr<UNetInference> clone() override {
        OPENVINO_ASSERT(static_cast<bool>(m_request), "UNet2DConditionModel must have m_request initialized");
        UNetInferenceDynamic cloned(*this);
        cloned.m_request = m_request.get_compiled_model().create_infer_request();
        return std::make_shared<UNetInferenceDynamic>(cloned);
    }

    virtual void compile(std::shared_ptr<ov::Model> model, const std::string& device, const ov::AnyMap& properties) override {
        ov::CompiledModel compiled_model = utils::singleton_core().compile_model(model, device, properties);
        ov::genai::utils::print_compiled_model_properties(compiled_model, "UNet 2D Condition dynamic model");
        m_request = compiled_model.create_infer_request();
    }

    virtual void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) override {
        OPENVINO_ASSERT(m_request, "UNet model must be compiled first");
        m_request.set_tensor(tensor_name, encoder_hidden_states);
    }

    virtual void set_adapters(AdapterController &adapter_controller, const AdapterConfig& adapters) override {
        OPENVINO_ASSERT(m_request, "UNet model must be compiled first");
        adapter_controller.apply(m_request, adapters);
    }

    virtual ov::Tensor infer(ov::Tensor sample, ov::Tensor timestep) override {
        OPENVINO_ASSERT(m_request, "UNet model must be compiled first. Cannot infer non-compiled model");

        m_request.set_tensor("sample", sample);
        m_request.set_tensor("timestep", timestep);

        m_request.infer();

        return m_request.get_output_tensor();
    }

    virtual void export_model(const std::filesystem::path& blob_path) override {
        OPENVINO_ASSERT(m_request, "UNet model must be compiled first");
        auto compiled_model = m_request.get_compiled_model();
        utils::export_model(compiled_model, blob_path / "openvino_model.blob");
    }

    virtual void import_model(const std::filesystem::path& blob_path,
                              const std::string& device,
                              const ov::AnyMap& properties) override {
        auto compiled_model = utils::import_model(blob_path / "openvino_model.blob", device, properties);
        ov::genai::utils::print_compiled_model_properties(compiled_model, "UNet 2D Condition dynamic model");
        m_request = compiled_model.create_infer_request();
    }

private:
    ov::InferRequest m_request;
};

}  // namespace genai
}  // namespace ov