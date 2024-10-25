// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "lora_helper.hpp"
#include "text2image/models/unet_inference.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

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

// Static Batch-Size 1 variant of UNetInference
class UNet2DConditionModel::UNetInferenceStaticBS1 : public UNet2DConditionModel::UNetInference {
public:
    virtual void compile(std::shared_ptr<ov::Model> model,
                         const std::string& device,
                         const ov::AnyMap& properties) override {
        std::cout << "model passed into compile:" << std::endl;
        logBasicModelInfo(model);

        auto sample_node = model->input("sample");
        const ov::PartialShape sample_shape = sample_node.get_partial_shape();

        if (sample_shape.is_dynamic())
        {
            throw std::runtime_error(
                "UNetInferenceStaticBS1::compile: sample shape is dynamic. This implementation only support static.");
        }

        m_nativeBatchSize = sample_shape.get_shape()[0];
        m_requests.resize(m_nativeBatchSize);

        //reshape to batch-1
        UNetInference::reshape(model, 1);

        std::cout << "model we are compiling:" << std::endl;
        logBasicModelInfo(model);

        ov::Core core = utils::singleton_core();
        ov::CompiledModel compiled_model = core.compile_model(model, device, properties);

        for (int i = 0; i < m_nativeBatchSize; i++ )
        {
            m_requests[i] = compiled_model.create_infer_request();
        }
    }

    virtual void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) override {
        OPENVINO_ASSERT(m_nativeBatchSize && m_nativeBatchSize == m_requests.size(),
                        "UNet model must be compiled first");

        size_t encoder_hidden_states_bs = encoder_hidden_states.get_shape()[0];
        if (encoder_hidden_states_bs != m_nativeBatchSize)
        {
            throw std::runtime_error("UNetInferenceStaticBS1::set_hidden_states: native batch size is " + std::to_string(m_nativeBatchSize) 
                + ", but encoder_hidden_states has batch size of " + std::to_string(encoder_hidden_states_bs));
        }

        //TODO: We should use stride, and perhaps there's a cleaner way to do this..
        float* pHiddenStates = encoder_hidden_states.data<float>();
        for (int i = 0; i < m_nativeBatchSize; i++)
        {
            auto hidden_states_bs1 = m_requests[i].get_tensor(tensor_name);

            //wrap exiting tensor location as a batch1 tensor
            ov::Tensor bs1_wrapper(hidden_states_bs1.get_element_type(), hidden_states_bs1.get_shape(), pHiddenStates);

            //copy it to infer request.
            bs1_wrapper.copy_to(hidden_states_bs1);

            pHiddenStates += bs1_wrapper.get_size();
        }
    }

    virtual void set_adapters(AdapterController& adapter_controller, const AdapterConfig& adapters) override {
        OPENVINO_ASSERT(m_nativeBatchSize && m_nativeBatchSize == m_requests.size(),
                        "UNet model must be compiled first");
        for (int i = 0; i < m_nativeBatchSize; i++) {
            adapter_controller.apply(m_requests[i], adapters);
        }
        
    }

    virtual ov::Tensor infer(ov::Tensor sample, ov::Tensor timestep) override {
        OPENVINO_ASSERT(m_nativeBatchSize && m_nativeBatchSize == m_requests.size(),
                        "UNet model must be compiled first");

        float* pSample = sample.data<float>();
        for (int i = 0; i < m_nativeBatchSize; i++) {

            m_requests[i].set_tensor("timestep", timestep);

            auto sample_bs1 = m_requests[i].get_tensor("sample");

            // wrap exiting tensor location as a batch1 tensor
            ov::Tensor bs1_wrapper(sample_bs1.get_element_type(), sample_bs1.get_shape(), pSample);

            bs1_wrapper.copy_to(sample_bs1);

            pSample += bs1_wrapper.get_size();

            // kick off infer for this request.
            m_requests[i].start_async();
        }

        auto out_sample = ov::Tensor(sample.get_element_type(), sample.get_shape());

        float* pOutSample = out_sample.data<float>();
        for (int i = 0; i < m_nativeBatchSize; i++) {

            // wait for infer to complete.
            m_requests[i].wait();

            auto out_sample_bs1 = m_requests[i].get_tensor("out_sample");
            ov::Tensor bs1_wrapper(out_sample_bs1.get_element_type(), out_sample_bs1.get_shape(), pOutSample);
            out_sample_bs1.copy_to(bs1_wrapper);

            pOutSample += bs1_wrapper.get_size();
        }

        return out_sample;
    }

private:
    std::vector<ov::InferRequest> m_requests;
    size_t m_nativeBatchSize = 0;
};

}  // namespace genai
}  // namespace ov