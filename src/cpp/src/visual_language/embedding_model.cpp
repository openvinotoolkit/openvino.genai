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

std::unique_ptr<ov::genai::CircularBufferQueue<ov::genai::EmbeddingsRequest>> init(ov::CompiledModel& compiled) {
    auto embeddings_requests_queue = std::make_unique<ov::genai::CircularBufferQueue<ov::genai::EmbeddingsRequest>>(
        compiled.get_property(ov::optimal_number_of_infer_requests),
        [&compiled]() -> ov::genai::EmbeddingsRequest {
            ov::genai::EmbeddingsRequest req;
            req.ireq = compiled.create_infer_request();
            req.cpu_tensor = req.ireq.get_output_tensor();
            ov::RemoteContext context;
            try {
                context = compiled.get_context();
            } catch (const ov::Exception&) {
                req.remote_tensor = req.cpu_tensor;
                return req;
            }
            req.remote_tensor = context.create_tensor(ov::element::f32, req.cpu_tensor.get_shape());
            return req;
        });
    return embeddings_requests_queue;
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

    auto device_config = properties;
    ov::genai::utils::disable_cpu_acceleration_in_AUTO(device, device_config, "text embeddings model");
    ov::CompiledModel compiled_model = core.compile_model(m_model, device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "text embeddings model");
    m_embeddings_requests_queue = init(compiled_model);
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

    auto device_config = properties;
    ov::genai::utils::disable_cpu_acceleration_in_AUTO(device, device_config, "text embeddings model");
    ov::CompiledModel compiled_model = core.compile_model(m_model, device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "text embeddings model");
    m_embeddings_requests_queue = init(compiled_model);
}

ov::Tensor EmbeddingsModel::infer(const ov::Tensor& input_idx, bool return_remote_tensor) {
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(this->m_embeddings_requests_queue.get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    OPENVINO_ASSERT(req.ireq, "Text embeddings decoder model must be compiled first. Cannot infer non-compiled model");
    req.ireq.set_input_tensor(input_idx);
    if (return_remote_tensor) {
        req.ireq.set_output_tensor(req.remote_tensor);
    } else {
        req.ireq.set_output_tensor(req.cpu_tensor);
    }
    req.ireq.start_async();
    req.ireq.wait();
    return req.ireq.get_output_tensor();
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