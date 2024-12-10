// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <memory>

#include "openvino/runtime/core.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/constant.hpp"

#include "utils.hpp"

#include "embedding_model.hpp"

namespace ov {
namespace genai {

EmbeddingsModel::EmbeddingsModel(const std::filesystem::path& model_dir,
                                 const float scale_emb,
                                 const std::string& device,
                                 const ov::AnyMap& properties) {
    ov::Core core = utils::singleton_core();
    std::shared_ptr<ov::Model> m_model = core.read_model((model_dir / "openvino_text_embeddings_model.xml").string());
    // apply embedding postprocessing step by merging them into the model
    merge_postprocess(m_model, scale_emb);

    ov::CompiledModel compiled_model = core.compile_model(m_model, device, properties);
    m_request = compiled_model.create_infer_request();
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
    m_request = compiled_model.create_infer_request();
}

ov::Tensor EmbeddingsModel::infer(ov::Tensor input_idx) {
    OPENVINO_ASSERT(m_request, "Text embeddings decoder model must be compiled first. Cannot infer non-compiled model");

    m_request.set_input_tensor(input_idx);
    m_request.infer();
    return m_request.get_output_tensor();
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