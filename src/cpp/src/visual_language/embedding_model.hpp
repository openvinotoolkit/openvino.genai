// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>
#include <string>
#include <memory>

#include "openvino/core/any.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"

#include "visual_language/vlm_config.hpp"

#include <openvino/openvino.hpp>
#include "visual_language/processor_config.hpp"
#include "circular_buffer_queue.hpp"

namespace ov {
namespace genai {

struct EmbeddingsRequest {
    ov::InferRequest ireq;
    ov::Tensor cpu_tensor;
    ov::Tensor remote_tensor;
};

class EmbeddingsModel {
public:
    using Ptr = std::shared_ptr<EmbeddingsModel>;

    EmbeddingsModel(const std::filesystem::path& model_dir,
                    const float scale_emb,
                    const std::string& device,
                    const ov::AnyMap& properties);

    EmbeddingsModel(const std::string& model,
                    const ov::Tensor& weights,
                    const float scale_emb,
                    const std::string& device,
                    const ov::AnyMap& properties);

    EmbeddingsModel() = default;

    static Ptr create(const std::filesystem::path& model_dir,
                      const float scale_emb,
                      const std::string& device,
                      const ov::AnyMap& properties) {
        return std::make_shared<EmbeddingsModel>(model_dir, scale_emb, device, properties);
    }

    static Ptr create(const std::string& model,
                      const ov::Tensor& weights,
                      const float scale_emb,
                      const std::string& device,
                      const ov::AnyMap& properties) {
        return std::make_shared<EmbeddingsModel>(model, weights, scale_emb, device, properties);
    }

    // We have getter for the request queue, so we can reserve request outside of infer scope
    // Tensor produced by infer is stored in the request and used further in the pipeline, so we can't free it right after infer call
    std::unique_ptr<CircularBufferQueue<EmbeddingsRequest>>& get_request_queue();
    ov::Tensor infer(EmbeddingsRequest& req, const ov::Tensor& input_idx, bool return_remote_tensor=false);
private:
    void merge_postprocess(std::shared_ptr<ov::Model> model, float scale_emb) const;

    std::unique_ptr<CircularBufferQueue<EmbeddingsRequest>> m_embeddings_requests_queue;
};

} // namespace genai
} // namespace ov