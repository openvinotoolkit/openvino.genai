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

EmbeddingsModel::EmbeddingsModel(const std::filesystem::path& model_dir, const VLMConfig& vlm_config) :
                                 m_vlm_config{vlm_config} {
    ov::Core core = utils::singleton_core();
    m_model = core.read_model((model_dir / "openvino_text_embeddings_model.xml").string());
    // apply embedding postprocessing step by merging them into the model
    merge_postprocess();
}

EmbeddingsModel::EmbeddingsModel(const std::filesystem::path& root_dir,
                                const VLMConfig& vlm_config,
                                const std::string& device,
                                const ov::AnyMap& properties)
    : EmbeddingsModel(root_dir, vlm_config) {
    compile(device, properties);
}

EmbeddingsModel::EmbeddingsModel(const EmbeddingsModel&) = default;

EmbeddingsModel& EmbeddingsModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::Core core = utils::singleton_core();
    ov::CompiledModel compiled_model = core.compile_model(m_model, device, properties);
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

ov::Tensor EmbeddingsModel::infer(ov::Tensor input_idx) {
    OPENVINO_ASSERT(m_request, "Text embeddings decoder model must be compiled first. Cannot infer non-compiled model");

    m_request.set_input_tensor(input_idx);
    m_request.infer();
    return m_request.get_output_tensor();
}

void EmbeddingsModel::merge_postprocess() const {
    ov::preprocess::PrePostProcessor ppp(m_model);

    auto scale_emb = m_vlm_config.scale_emb;
    ppp.output().postprocess().custom([scale_emb](const ov::Output<ov::Node>& node) {
        auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, scale_emb);
        return std::make_shared<ov::op::v1::Multiply>(node, constant);
    });

    ppp.build();
}

} // namespace genai
} // namespace ov