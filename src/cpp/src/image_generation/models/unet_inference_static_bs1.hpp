// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "lora/helper.hpp"
#include "image_generation/models/unet_inference.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

// Static Batch-Size 1 variant of UNetInference
class UNet2DConditionModel::UNetInferenceStaticBS1 : public UNet2DConditionModel::UNetInference {

    UNetInferenceStaticBS1(const UNetInferenceStaticBS1&) = delete;
    UNetInferenceStaticBS1(UNetInferenceStaticBS1&&) = delete;
    UNetInferenceStaticBS1(UNetInferenceStaticBS1& other) = delete;

public:
    UNetInferenceStaticBS1() : UNetInference(), m_native_batch_size(0) {}

    virtual std::shared_ptr<UNetInference> clone() override {
        OPENVINO_ASSERT(m_requests.size(), "UNet2DConditionModel must have m_requests initialized");
        auto clone = std::make_shared<UNetInferenceStaticBS1>();
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
                            "UNetInferenceStaticBS1::compile: input tensor " + input.get_any_name() +
                                " shape is dynamic. Tensors must be reshaped to be static before compile is invoked.");
        }

        for (auto& output : model->outputs()) {
            OPENVINO_ASSERT(!output.get_partial_shape().is_dynamic(),
                            "UNetInferenceStaticBS1::compile: output tensor " + output.get_any_name() +
                            " shape is dynamic. Tensors must be reshaped to be static before compile is invoked.");
        }

        // we'll create a separate infer request for each batch.
        m_native_batch_size = model->input("sample").get_shape()[0];
        m_requests.resize(m_native_batch_size);

        //reshape to batch-1
        UNetInference::reshape(model, 1);

        ov::Core core = utils::singleton_core();
        ov::CompiledModel compiled_model = core.compile_model(model, device, properties);
        ov::genai::utils::print_compiled_model_properties(compiled_model, "UNet 2D Condition batch-1 model");

        for (int i = 0; i < m_native_batch_size; i++) {
            m_requests[i] = compiled_model.create_infer_request();
        }
    }

    virtual void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) override {
        OPENVINO_ASSERT(m_native_batch_size && m_native_batch_size == m_requests.size(),
                        "UNet model must be compiled first");

        size_t encoder_hidden_states_bs = encoder_hidden_states.get_shape()[0];

        OPENVINO_ASSERT(
            encoder_hidden_states_bs == m_native_batch_size,
            ("UNetInferenceStaticBS1::set_hidden_states: native batch size is "
            + std::to_string(m_native_batch_size) +
             ", but encoder_hidden_states has batch size of " + std::to_string(encoder_hidden_states_bs)));

        char* pHiddenStates = (char *)encoder_hidden_states.data();
        size_t hidden_states_batch_stride_bytes = encoder_hidden_states.get_strides()[0];

        for (int i = 0; i < m_native_batch_size; i++)
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
        OPENVINO_ASSERT(m_native_batch_size && m_native_batch_size == m_requests.size(),
                        "UNet model must be compiled first");
        for (int i = 0; i < m_native_batch_size; i++) {
            adapter_controller.apply(m_requests[i], adapters);
        }
    }

    virtual ov::Tensor infer(ov::Tensor sample, ov::Tensor timestep) override {
        OPENVINO_ASSERT(m_native_batch_size && m_native_batch_size == m_requests.size(),
                        "UNet model must be compiled first");

        OPENVINO_ASSERT(sample.get_shape()[0] == m_native_batch_size,
                        "sample batch size must match native batch size");

        char* pSample = (char*)sample.data();
        size_t sample_batch_stride_bytes = sample.get_strides()[0];

        auto out_sample = ov::Tensor(sample.get_element_type(), sample.get_shape());
        char* pOutSample = (char*)out_sample.data();
        size_t out_sample_batch_stride_bytes = out_sample.get_strides()[0];

        auto bs1_sample_shape = sample.get_shape();
        bs1_sample_shape[0] = 1;

        for (int i = 0; i < m_native_batch_size; i++) {
            m_requests[i].set_tensor("timestep", timestep);

            //wrap a portion of sample tensor as a batch-1 tensor, as set this as input tensor.
            {
                ov::Tensor bs1_wrapper(sample.get_element_type(), bs1_sample_shape, pSample, sample.get_strides());
                m_requests[i].set_tensor("sample", bs1_wrapper);
            }

            // wrap a portion of out_sample tensor as a batch-1 tensor, as set this as output tensor.
            {
                ov::Tensor bs1_wrapper(sample.get_element_type(), bs1_sample_shape, pOutSample, out_sample.get_strides());
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

    virtual void export_model(const std::filesystem::path& blob_path) override {
        OPENVINO_ASSERT(m_native_batch_size && m_native_batch_size == m_requests.size(),
                        "UNet model must be compiled first");
        OPENVINO_ASSERT(m_requests.size() > 0, "UNet model must have at least one infer request");
        auto compiled_model = m_requests[0].get_compiled_model();
        utils::export_model(compiled_model, blob_path / "openvino_model.blob");
    }

    virtual void import_model(const std::filesystem::path& blob_path, const std::string& device, const ov::AnyMap& properties) override {
        auto compiled_model = utils::import_model(blob_path / "openvino_model.blob", device, properties);

        // we'll create a separate infer request for each batch.
        // todo: preserve original requested batch size when exporting the model
        // current implementation imports model with batch = 1 and creates a single infer request.
        m_native_batch_size = compiled_model.input("sample").get_shape()[0];
        m_requests.resize(m_native_batch_size);

        ov::genai::utils::print_compiled_model_properties(compiled_model, "UNet 2D Condition batch-1 model");

        for (int i = 0; i < m_native_batch_size; i++) {
            m_requests[i] = compiled_model.create_infer_request();
        }
    }

private:
    std::vector<ov::InferRequest> m_requests;
    size_t m_native_batch_size = 0;
};

}  // namespace genai
}  // namespace ov