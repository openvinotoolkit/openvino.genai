// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/generation_config.hpp"

#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {

class DiagonalGaussianDistribution {
public:
    explicit DiagonalGaussianDistribution(ov::Tensor parameters) {
        ov::Shape shape = parameters.get_shape();
        OPENVINO_ASSERT(shape[0] == 1, "Batch size must be 1");
        shape[1] /= 2;

        m_mean = ov::Tensor(parameters.get_element_type(), shape, parameters.data());
        m_std = ov::Tensor(m_mean.get_element_type(), shape);
        ov::Tensor logvar(parameters.get_element_type(), shape, m_mean.data<float>() + m_mean.get_size());

        float * logvar_data = logvar.data<float>();
        float * std_data = m_std.data<float>();

        for (size_t i = 0; i < logvar.get_size(); ++i) {
            logvar_data[i] = std::min(std::max(logvar_data[i], -30.0f), 20.0f);
            std_data[i] = std::exp(0.5 * logvar_data[i]);
        }
    }

    ov::Tensor sample(std::shared_ptr<Generator> generator) const {
        ov::Tensor rand_tensor =  generator->randn_tensor(m_mean.get_shape());

        float * rand_tensor_data = rand_tensor.data<float>();
        const float * mean_data = m_mean.data<float>();
        const float * std_data = m_std.data<float>();

        for (size_t i = 0; i < rand_tensor.get_size(); ++i) {
            rand_tensor_data[i] = mean_data[i] + std_data[i] * rand_tensor_data[i];
        }

        return rand_tensor;
    }

private:
    ov::Tensor m_mean, m_std;
};

}  // namespace genai
}  // namespace ov
