// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include "openvino/core/model.hpp"
#include "openvino/genai/image_generation/sd3_transformer_2d_model.hpp"

namespace ov {
namespace genai {

class SD3Transformer2DModel::Inference {
public:
    virtual std::shared_ptr<Inference> clone() = 0;
    virtual void compile(std::shared_ptr<ov::Model> model, const std::string& device, const ov::AnyMap& properties) = 0;
    virtual void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) = 0;
    virtual void set_adapters(AdapterController& m_adapter_controller, const AdapterConfig& adapters) = 0;
    virtual ov::Tensor infer(ov::Tensor latent_model_input, ov::Tensor timestep) = 0;

    // utility function to resize model given optional dimensions.
    static void reshape(std::shared_ptr<ov::Model> model,
                        std::optional<int> batch_size = {},
                        std::optional<int> height = {},
                        std::optional<int> width = {},
                        std::optional<int> tokenizer_model_max_length = {}) {
        std::map<std::string, ov::PartialShape> name_to_shape;
        for (auto&& input : model->inputs()) {
            std::string input_name = input.get_any_name();
            name_to_shape[input_name] = input.get_partial_shape();
            if (input_name == "timestep") {
                name_to_shape[input_name][0] = 1;
            } else if (input_name == "hidden_states") {
                if (batch_size) {
                    name_to_shape[input_name][0] = *batch_size;
                }

                if (height) {
                    name_to_shape[input_name][2] = *height;
                }

                if (width) {
                    name_to_shape[input_name][3] = *width;
                }
            }  else if (input_name == "encoder_hidden_states") {
                if (batch_size) {
                    name_to_shape[input_name][0] = *batch_size;
                }

                if (tokenizer_model_max_length) {
                    name_to_shape[input_name][1] = *tokenizer_model_max_length;
                }
            } else if (input_name == "pooled_projections") {
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