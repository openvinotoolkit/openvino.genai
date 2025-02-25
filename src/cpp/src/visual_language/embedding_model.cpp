// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <memory>

#include "openvino/runtime/core.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/constant.hpp"

#include "utils.hpp"

#include "embedding_model.hpp"

namespace {

std::tuple<ov::InferRequest, ov::Tensor, ov::Tensor> init(ov::CompiledModel& compiled) {
    ov::InferRequest ireq = compiled.create_infer_request();
    ov::Tensor cpu_tensor = ireq.get_output_tensor();
    ov::RemoteContext context;
    try {
        context = compiled.get_context();
    } catch (const ov::Exception&) {
        return {std::move(ireq), cpu_tensor, cpu_tensor};
    }
    ov::RemoteTensor remote = context.create_tensor(ov::element::f32, cpu_tensor.get_shape());
    return {std::move(ireq), std::move(cpu_tensor), std::move(remote)};
}

}  // namespace

namespace ov {
namespace genai {

EmbeddingsModel::EmbeddingsModel(const std::filesystem::path& model_dir,
                                 const float scale_emb,
                                 const std::string& device,
                                 const ov::AnyMap& properties) {
    ov::Core core = utils::singleton_core();
    std::shared_ptr<ov::Model> m_model = core.read_model(model_dir / "openvino_text_embeddings_model.xml", {}, properties);
    // apply embedding postprocessing step by merging them into the model
    merge_postprocess(m_model, scale_emb);

    ov::CompiledModel compiled_model = core.compile_model(m_model, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "text embeddings model");
    std::tie(m_request, m_cpu_tensor, m_remote_tensor) = init(compiled_model);
}

EmbeddingsModel::EmbeddingsModel(const std::string& model,
                                 const ov::Tensor& weights,
                                 const float scale_emb,
                                 const std::string& device,
                                 const ov::AnyMap& properties) {
    ov::Core core = utils::singleton_core();
    std::shared_ptr<ov::Model> m_model = core.read_model(model, weights);
    // apply embedding postprocessing step by merging them into the model
    merge_postprocess(m_model, scale_emb);

    ov::CompiledModel compiled_model = core.compile_model(m_model, device, properties);
    std::tie(m_request, m_cpu_tensor, m_remote_tensor) = init(compiled_model);
}

ov::Tensor EmbeddingsModel::infer(const ov::Tensor& input_idx, bool return_remote_tensor) {
    OPENVINO_ASSERT(m_request, "Text embeddings decoder model must be compiled first. Cannot infer non-compiled model");

    m_request.set_input_tensor(input_idx);
    if (return_remote_tensor) {
        m_request.set_output_tensor(m_remote_tensor);
    } else {
        m_request.set_output_tensor(m_cpu_tensor);
    }
    m_request.infer();
    ov::Tensor out = m_request.get_output_tensor();
    OPENVINO_ASSERT(input_idx.get_shape().at(1) == out.get_shape().at(1));
    return out;
}

void EmbeddingsModel::merge_postprocess(std::shared_ptr<ov::Model> model, float scale_emb) const {
    ov::preprocess::PrePostProcessor ppp(model);

    ppp.output().postprocess().custom([scale_emb](const ov::Output<ov::Node>& node) {
        auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, scale_emb);
        return std::make_shared<ov::op::v1::Multiply>(node, constant);
    });

    ppp.build();
}

} // namespace genai
} // namespace ov