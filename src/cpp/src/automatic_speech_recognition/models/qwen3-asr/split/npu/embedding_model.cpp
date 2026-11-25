// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "embedding_model.hpp"

#include "properties.hpp"
#include "utils.hpp"

namespace ov::genai {
namespace {

ov::AnyMap token_lookup_properties(const ov::AnyMap& properties) {
    auto result = qwen3_asr_input_model_properties(utils::get_model_properties(properties, "text_embeddings", "NPU"));
    utils::set_config_default(result, "NPU_USE_NPUW", std::string{"YES"});
    OPENVINO_ASSERT(utils::is_npuw_enabled(result), "Qwen3-ASR NPU text embeddings require NPU_USE_NPUW=YES.");
    utils::set_config_default(result, "NPUW_TOKEN_LOOKUP", std::string{"YES"});
    OPENVINO_ASSERT(utils::is_property_enabled(result, "NPUW_TOKEN_LOOKUP"),
                    "Qwen3-ASR NPU text embeddings require NPUW_TOKEN_LOOKUP=YES.");
    utils::set_config_default(result, "NPUW_DEVICES", std::string{"NPU"});
    return result;
}

class Qwen3ASRNPUEmbeddingsModel final : public EmbeddingsModel {
public:
    Qwen3ASRNPUEmbeddingsModel(const std::filesystem::path& model_dir, float scale_emb, const ov::AnyMap& properties)
        : EmbeddingsModel(model_dir, scale_emb, "NPU", token_lookup_properties(properties)) {}

    Qwen3ASRNPUEmbeddingsModel(const std::string& model,
                               const ov::Tensor& weights,
                               float scale_emb,
                               const ov::AnyMap& properties)
        : EmbeddingsModel(model, weights, scale_emb, "NPU", token_lookup_properties(properties)) {}

    ov::Tensor infer(EmbeddingsRequest& request, const ov::Tensor& input_ids, bool return_remote_tensor) override {
        OPENVINO_ASSERT(!return_remote_tensor, "Qwen3-ASR NPU text embeddings do not support remote output tensors.");
        return EmbeddingsModel::infer(request, input_ids, false);
    }
};

}  // namespace

EmbeddingsModel::Ptr create_qwen3_asr_embeddings(const std::filesystem::path& model_dir,
                                                 float scale_emb,
                                                 const std::string& device,
                                                 const ov::AnyMap& properties) {
    if (qwen3_asr_input_device(device, properties) != "NPU") {
        return EmbeddingsModel::create(model_dir, scale_emb, device, properties);
    }
    return std::make_shared<Qwen3ASRNPUEmbeddingsModel>(model_dir, scale_emb, properties);
}

EmbeddingsModel::Ptr create_qwen3_asr_embeddings(const std::string& model,
                                                 const ov::Tensor& weights,
                                                 float scale_emb,
                                                 const std::string& device,
                                                 const ov::AnyMap& properties) {
    if (qwen3_asr_input_device(device, properties) != "NPU") {
        return EmbeddingsModel::create(model, weights, scale_emb, device, properties);
    }
    return std::make_shared<Qwen3ASRNPUEmbeddingsModel>(model, weights, scale_emb, properties);
}

}  // namespace ov::genai
