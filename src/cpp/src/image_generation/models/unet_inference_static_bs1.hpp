// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "lora_helper.hpp"
#include "image_generation/models/unet_inference.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

// Static Batch-Size 1 variant of UNetInference
class UNet2DConditionModel::UNetInferenceStaticBS1 : public UNet2DConditionModel::UNetInference {
public:
    virtual void compile(std::shared_ptr<ov::Model> model,
                         const std::string& device,
                         const ov::AnyMap& properties) override {

        // All shapes for input/output tensors should be static. 
        // Double check this and throw runtime error if it's not the case.
        for (auto& input : model->inputs()) {
            if (input.get_partial_shape().is_dynamic()) {
                throw std::runtime_error(
                    "UNetInferenceStaticBS1::compile: input tensor " + input.get_any_name() +
                    " shape is dynamic. Tensors must be reshaped to be static before compile is invoked.");
            }
        }

        for (auto& output : model->outputs()) {
            if (output.get_partial_shape().is_dynamic()) {
                throw std::runtime_error(
                    "UNetInferenceStaticBS1::compile: output tensor " + output.get_any_name() +
                    " shape is dynamic. Tensors must be reshaped to be static before compile is invoked.");
            }
        }

        // we'll create a separate infer request for each batch.
        m_nativeBatchSize = model->input("sample").get_shape()[0];
        m_requests.resize(m_nativeBatchSize);

        //reshape to batch-1
        UNetInference::reshape(model, 1);

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

        char* pHiddenStates = (char *)encoder_hidden_states.data();
        size_t hidden_states_batch_stride_bytes = encoder_hidden_states.get_strides()[0];

        for (int i = 0; i < m_nativeBatchSize; i++)
        {
            auto hidden_states_bs1 = m_requests[i].get_tensor(tensor_name);

            // wrap current pHiddenStates location as batch-1 tensor.
            ov::Tensor bs1_wrapper(hidden_states_bs1.get_element_type(),
                                   hidden_states_bs1.get_shape(),
                                   pHiddenStates,
                                   encoder_hidden_states.get_strides());

            // copy it to infer request batch-1 tensor
            bs1_wrapper.copy_to(hidden_states_bs1);

            // increment pHiddenStates to start location of next batch (using stride)
            pHiddenStates += hidden_states_batch_stride_bytes;
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

        char* pSample = (char *)sample.data();
        size_t sample_batch_stride_bytes = sample.get_strides()[0];

        for (int i = 0; i < m_nativeBatchSize; i++) {
            m_requests[i].set_tensor("timestep", timestep);

            auto sample_bs1 = m_requests[i].get_tensor("sample");

            // wrap current pSample location as batch-1 tensor.
            ov::Tensor bs1_wrapper(sample_bs1.get_element_type(), sample_bs1.get_shape(), pSample, sample.get_strides());

            // copy it to infer request batch-1 tensor
            bs1_wrapper.copy_to(sample_bs1);

            //increment pSample to start location of next batch (using stride)
            pSample += sample_batch_stride_bytes;

            // kick off infer for this request.
            m_requests[i].start_async();
        }

        auto out_sample = ov::Tensor(sample.get_element_type(), sample.get_shape());

        char* pOutSample = (char *)out_sample.data();
        size_t out_sample_batch_stride_bytes = out_sample.get_strides()[0];
        for (int i = 0; i < m_nativeBatchSize; i++) {

            // wait for infer to complete.
            m_requests[i].wait();

            auto out_sample_bs1 = m_requests[i].get_tensor("out_sample");
            ov::Tensor bs1_wrapper(out_sample_bs1.get_element_type(), out_sample_bs1.get_shape(), pOutSample, out_sample.get_strides());
            out_sample_bs1.copy_to(bs1_wrapper);

            pOutSample += out_sample_batch_stride_bytes;
        }

        return out_sample;
    }

private:
    std::vector<ov::InferRequest> m_requests;
    size_t m_nativeBatchSize = 0;
};

}  // namespace genai
}  // namespace ov