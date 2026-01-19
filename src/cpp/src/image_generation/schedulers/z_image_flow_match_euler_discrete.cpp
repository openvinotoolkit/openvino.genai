// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "z_image_flow_match_euler_discrete.hpp"
#include "image_generation/numpy_utils.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov::genai {

void ZImageFlowMatchEulerDiscreteScheduler::set_timesteps(size_t image_seq_len, size_t num_inference_steps, float strength) {
    m_timesteps.clear();
    m_sigmas.clear();

    m_num_inference_steps = num_inference_steps;
    m_strength = strength;

    m_sigmas = numpy_utils::linspace<float>(m_sigma_max, m_sigma_min, m_num_inference_steps, true);

    float shift = m_config.shift;

    // fill sigma
    float mu = calculate_shift(image_seq_len);
    if (m_config.use_dynamic_shifting) {
        float exp_mu = std::exp(mu);
        for (size_t i = 0; i < m_sigmas.size(); ++i) {
            m_sigmas[i] = exp_mu / (exp_mu + (1 / m_sigmas[i] - 1));
        }
    } else {
        for (size_t i = 0; i < m_sigmas.size(); ++i) {
            m_sigmas[i] = shift * m_sigmas[i] / (1 + (shift - 1) * m_sigmas[i]);
        }
    }

    // fill timesteps
    for (size_t i = 0; i < m_sigmas.size(); ++i) {
        m_timesteps.push_back(m_sigmas[i] * m_config.num_train_timesteps);
    }
    m_sigmas.push_back(0);
    m_step_index = -1, m_begin_index = -1;
}

void ZImageFlowMatchEulerDiscreteScheduler::set_sigma_min(float sigma_min) {
    m_sigma_min = sigma_min;
}

std::map<std::string, ov::Tensor> ZImageFlowMatchEulerDiscreteScheduler::step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step, std::shared_ptr<Generator> generator) {
    // noise_pred - model_output
    // latents - sample
    // inference_step

    if (m_step_index == -1)
        init_step_index();

    float sigma_diff = m_sigmas[m_step_index + 1] - m_sigmas[m_step_index];

    ov::Tensor sigma_diff_tensor(ov::element::f32, ov::Shape{1});
    sigma_diff_tensor.data<float>()[0] = sigma_diff;

    m_step_infer_request.set_tensor("latents", latents);
    m_step_infer_request.set_tensor("noise", noise_pred);
    m_step_infer_request.set_tensor("sigma_diff", sigma_diff_tensor);

    m_step_infer_request.infer();

    m_step_index++;

    return {{"latent", m_step_infer_request.get_output_tensor()}};
}

void ZImageFlowMatchEulerDiscreteScheduler::init_step_process(const std::string &device) {
    auto input_latents = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 16, -1, -1});
    input_latents->output(0).get_tensor().set_names({"latents"});
    auto input_noise = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{16, -1, -1, -1});
    input_noise->output(0).get_tensor().set_names({"noise"});
    auto input_sigma_diff = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1});
    input_sigma_diff->output(0).get_tensor().set_names({"sigma_diff"});
    auto new_shape = ov::opset8::Constant::create(ov::element::i64, ov::Shape{4}, {-1, 16, 0, 0});
    auto reshaped_noise = std::make_shared<ov::opset8::Reshape>(input_noise, new_shape, true);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(reshaped_noise, input_sigma_diff);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(input_latents, multiply);
    auto model = std::make_shared<ov::Model>(ov::OutputVector {subtract}, ov::ParameterVector{input_latents, input_noise, input_sigma_diff});
    model->validate_nodes_and_infer_types();
    auto compiled_model = ov::genai::utils::singleton_core().compile_model(model, device);
    m_step_infer_request = compiled_model.create_infer_request();
}

}