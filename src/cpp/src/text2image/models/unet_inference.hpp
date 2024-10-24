// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/text2image/unet2d_condition_model.hpp"

namespace ov {
namespace genai {

class UNet2DConditionModel::UNetInference {

public:
    virtual void compile(std::shared_ptr<ov::Model> model, const std::string& device, const ov::AnyMap& properties) = 0;
    virtual void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) = 0;
    virtual void set_adapters(AdapterController& adapter_controller, const AdapterConfig& adapters) = 0;
    virtual ov::Tensor infer(ov::Tensor sample, ov::Tensor timestep) = 0;
};

}  // namespace genai
}  // namespace ov