// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "image_generation/models/sd3transformer_2d_inference.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

// Static Batch-Size 1 variant of SD3Transformer2DModel::Inference
class SD3Transformer2DModel::InferenceStaticBS1 : public SD3Transformer2DModel::Inference {

    InferenceStaticBS1(const InferenceStaticBS1&) = delete;
    InferenceStaticBS1(InferenceStaticBS1&&) = delete;
    InferenceStaticBS1(InferenceStaticBS1& other) = delete;

public:
    InferenceStaticBS1() : Inference(), m_native_batch_size(0) {}

    virtual std::shared_ptr<Inference> clone() override {
        OPENVINO_ASSERT(m_requests.size(), "SD3Transformer2DModel must have m_requests initialized");
        auto clone = std::make_shared<InferenceStaticBS1>();
        clone->m_native_batch_size = m_native_batch_size;
        clone->m_requests.reserve(m_requests.size());
        for (auto& request : m_requests) {
            clone->m_requests.push_back(request.get_compiled_model().create_infer_request());
        }
        return clone;
    }

    virtual void compile(std::shared_ptr<ov::Model> model,
                         const std::string& device,
                         const ov::AnyMap& properties) override {
        // All shapes for input/output tensors should be static.
        // Double check this and throw runtime error if it's not the case.
        for (auto& input : model->inputs()) {
            OPENVINO_ASSERT(!input.get_partial_shape().is_dynamic(),
                            "SD3Transformer2DModel::InferenceStaticBS1::compile: input tensor " + input.get_any_name() +
                                " shape is dynamic. Tensors must be reshaped to be static before compile is invoked.");
        }

        for (auto& output : model->outputs()) {
            OPENVINO_ASSERT(!output.get_partial_shape().is_dynamic(),
                            "SD3Transformer2DModel::InferenceStaticBS1::compile: output tensor " + output.get_any_name() +
                                " shape is dynamic. Tensors must be reshaped to be static before compile is invoked.");
        }

        // we'll create a separate infer request for each batch.
        m_native_batch_size = model->input("hidden_states").get_shape()[0];
        m_requests.resize(m_native_batch_size);

        // reshape to batch-1
        Inference::reshape(model, 1);

        ov::Core core = utils::singleton_core();
        ov::CompiledModel compiled_model = core.compile_model(model, device, properties);
        ov::genai::utils::print_compiled_model_properties(compiled_model, "SD3 Transformer 2D batch-1 model");

        for (int i = 0; i < m_native_batch_size; i++) {
            m_requests[i] = compiled_model.create_infer_request();
        }
    }

    virtual void set_adapters(AdapterController& m_adapter_controller, const AdapterConfig& adapters) override {
        for (auto& m_request : m_requests) {
            OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
            m_adapter_controller.apply(m_request, adapters);
        }
    }

    virtual void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) override {
        OPENVINO_ASSERT(m_native_batch_size && m_native_batch_size == m_requests.size(),
                        "Transformer model must be compiled first");

        size_t encoder_hidden_states_bs = encoder_hidden_states.get_shape()[0];

        OPENVINO_ASSERT(
            encoder_hidden_states_bs == m_native_batch_size,
            ("SD3Transformer2DModel::InferenceStaticBS1::set_hidden_states: native batch size is " + std::to_string(m_native_batch_size) +
             ", but encoder_hidden_states has batch size of " + std::to_string(encoder_hidden_states_bs)));

        char* pHiddenStates = (char*)encoder_hidden_states.data();
        size_t hidden_states_batch_stride_bytes = encoder_hidden_states.get_strides()[0];

        for (int i = 0; i < m_native_batch_size; i++) {
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

    virtual ov::Tensor infer(ov::Tensor sample, ov::Tensor timestep) override {
        OPENVINO_ASSERT(m_native_batch_size && m_native_batch_size == m_requests.size(),
                        "Transformer model must be compiled first");

        OPENVINO_ASSERT(sample.get_shape()[0] == m_native_batch_size, "sample batch size must match native batch size");

        char* pSample = (char*)sample.data();
        size_t sample_batch_stride_bytes = sample.get_strides()[0];

        auto out_sample = ov::Tensor(sample.get_element_type(), sample.get_shape());
        char* pOutSample = (char*)out_sample.data();
        size_t out_sample_batch_stride_bytes = out_sample.get_strides()[0];

        auto bs1_sample_shape = sample.get_shape();
        bs1_sample_shape[0] = 1;

        for (int i = 0; i < m_native_batch_size; i++) {
            m_requests[i].set_tensor("timestep", timestep);

            // wrap a portion of sample tensor as a batch-1 tensor, as set this as input tensor.
            {
                ov::Tensor bs1_wrapper(sample.get_element_type(), bs1_sample_shape, pSample, sample.get_strides());
                m_requests[i].set_tensor("hidden_states", bs1_wrapper);
            }

            // wrap a portion of out_sample tensor as a batch-1 tensor, as set this as output tensor.
            {
                ov::Tensor bs1_wrapper(sample.get_element_type(),
                                       bs1_sample_shape,
                                       pOutSample,
                                       out_sample.get_strides());
                m_requests[i].set_tensor("out_sample", bs1_wrapper);
            }

            // increment pSample & pOutSample to start location of next batch (using stride)
            pSample += sample_batch_stride_bytes;
            pOutSample += out_sample_batch_stride_bytes;

            // kick off infer for this request.
            m_requests[i].start_async();
        }

        for (int i = 0; i < m_native_batch_size; i++) {
            // wait for infer to complete.
            m_requests[i].wait();
        }

        return out_sample;
    }

private:
    std::vector<ov::InferRequest> m_requests;
    size_t m_native_batch_size = 0;
};

}  // namespace genai
}  // namespace ov