// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/image_generation/unet2d_condition_model.hpp"

namespace ov {
namespace genai {

//REMOVE THIS
static inline void logBasicModelInfo(const std::shared_ptr<ov::Model>& model) {
    std::cout << "Model name: " << model->get_friendly_name() << std::endl;

    // Dump information about model inputs/outputs
    ov::OutputVector inputs = model->inputs();
    ov::OutputVector outputs = model->outputs();

    std::cout << "\tInputs: " << std::endl;
    for (const ov::Output<ov::Node>& input : inputs) {
        const std::string name = input.get_any_name();
        const ov::element::Type type = input.get_element_type();
        const ov::PartialShape shape = input.get_partial_shape();
        const ov::Layout layout = ov::layout::get_layout(input);

        std::cout << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string() << std::endl;
    }

    std::cout << "\tOutputs: " << std::endl;
    for (const ov::Output<ov::Node>& output : outputs) {
        const std::string name = output.get_any_name();
        const ov::element::Type type = output.get_element_type();
        const ov::PartialShape shape = output.get_partial_shape();
        const ov::Layout layout = ov::layout::get_layout(output);

        std::cout << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string() << std::endl;
    }

    return;
}

class UNet2DConditionModel::UNetInference {

public:
    virtual void compile(std::shared_ptr<ov::Model> model, const std::string& device, const ov::AnyMap& properties) = 0;
    virtual void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) = 0;
    virtual void set_adapters(AdapterController& adapter_controller, const AdapterConfig& adapters) = 0;
    virtual ov::Tensor infer(ov::Tensor sample, ov::Tensor timestep) = 0;

    // utility function to resize model given optional dimensions.
    static void reshape(std::shared_ptr<ov::Model> model,
                        std::optional<int> batch_size = {},
                        std::optional<int> height = {},
                        std::optional<int> width = {},
                        std::optional<int> tokenizer_model_max_length = {})
    {
        std::map<std::string, ov::PartialShape> name_to_shape;
        for (auto&& input : model->inputs()) {
            std::string input_name = input.get_any_name();
            name_to_shape[input_name] = input.get_partial_shape();
            if (input_name == "timestep") {
                name_to_shape[input_name][0] = 1;
            } else if (input_name == "sample") {
                if (batch_size) {
                    name_to_shape[input_name][0] = *batch_size;
                }

                if (height) {
                    name_to_shape[input_name][2] = *height;
                }

                if (width) {
                    name_to_shape[input_name][3] = *width;
                }
            } else if (input_name == "time_ids" || input_name == "text_embeds") {
                if (batch_size) {
                    name_to_shape[input_name][0] = *batch_size;
                }
            } else if (input_name == "encoder_hidden_states") {
                if (batch_size) {
                    name_to_shape[input_name][0] = *batch_size;
                }

                if (tokenizer_model_max_length) {
                    name_to_shape[input_name][1] = *tokenizer_model_max_length;
                }
            } else if (input_name == "timestep_cond") {
                if (batch_size) {
                    name_to_shape[input_name][0] = *batch_size;
                }
            }
        }

        model->reshape(name_to_shape);
    }
};

}  // namespace genai
}  // namespace ov