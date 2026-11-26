// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "embedding_model.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "properties.hpp"
#include "utils.hpp"

namespace ov::genai {
namespace {

constexpr size_t EMBEDDING_CHUNK_SIZE = 64;

bool depends_only_on_constants(const std::shared_ptr<ov::Node>& root) {
    std::vector<std::shared_ptr<ov::Node>> pending{root};
    std::unordered_set<const ov::Node*> visited;
    while (!pending.empty()) {
        const auto node = pending.back();
        pending.pop_back();
        if (!visited.insert(node.get()).second) {
            continue;
        }
        if (node->get_input_size() == 0 && !ov::is_type<ov::op::v0::Constant>(node)) {
            return false;
        }
        for (const auto& input : node->input_values()) {
            pending.push_back(input.get_node_shared_ptr());
        }
    }
    return true;
}

class Qwen3ASRNPUEmbeddingsModel final : public EmbeddingsModel {
public:
    Qwen3ASRNPUEmbeddingsModel(const std::shared_ptr<ov::Model>& model,
                               float scale_emb,
                               const std::string& device,
                               const ov::AnyMap& properties) {
        OPENVINO_ASSERT(model && model->get_parameters().size() == 1 && model->get_results().size() == 1 &&
                            model->get_variables().empty(),
                        "Qwen3-ASR NPU text embeddings require a stateless, single-input lookup model.");
        const auto& input = model->get_parameters().front();
        OPENVINO_ASSERT(input->get_element_type() == ov::element::i64 && input->get_partial_shape().rank() == 2,
                        "Qwen3-ASR NPU text embeddings require a rank-2 int64 token input.");
        const auto gather = ov::as_type_ptr<ov::op::util::GatherBase>(
            model->get_results().front()->input_value(0).get_node_shared_ptr());
        OPENVINO_ASSERT(gather && gather->get_axis() == 0 && gather->get_batch_dims() == 0,
                        "Qwen3-ASR NPU text embeddings require a token-wise vocabulary lookup.");
        auto indices = gather->input_value(1).get_node_shared_ptr();
        while (const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(indices)) {
            indices = convert->input_value(0).get_node_shared_ptr();
        }
        const auto vocabulary_shape = gather->input_value(0).get_partial_shape();
        OPENVINO_ASSERT(indices == input && depends_only_on_constants(gather->input_value(0).get_node_shared_ptr()) &&
                            vocabulary_shape.rank() == 2 && vocabulary_shape.is_static() &&
                            model->output().get_element_type() == ov::element::f32,
                        "Qwen3-ASR NPU text embeddings require a constant rank-2 vocabulary and float32 output.");
        m_vocab_size = static_cast<size_t>(vocabulary_shape[0].get_length());
        m_hidden_size = static_cast<size_t>(vocabulary_shape[1].get_length());
        OPENVINO_ASSERT(m_vocab_size > 0 && m_hidden_size > 0,
                        "Qwen3-ASR NPU text embeddings require a nonempty vocabulary.");

        model->reshape({{model->input().get_any_name(), ov::PartialShape{1, EMBEDDING_CHUNK_SIZE}}});
        merge_postprocess(model, scale_emb);
        auto model_properties = qwen3_asr_npu_input_model_properties(properties, "text_embeddings", device);
        OPENVINO_ASSERT(utils::is_npuw_enabled(model_properties),
                        "Qwen3-ASR NPU text embeddings require NPU_USE_NPUW=YES.");
        auto compiled = utils::singleton_core().compile_model(model, "NPU", model_properties);
        utils::print_compiled_model_properties(compiled, "text embeddings model");
        m_requests = std::make_unique<CircularBufferQueue<EmbeddingsRequest>>(
            compiled.get_property(ov::optimal_number_of_infer_requests),
            [compiled]() mutable -> EmbeddingsRequest {
                EmbeddingsRequest request;
                request.ireq = compiled.create_infer_request();
                return request;
            });
    }

    std::unique_ptr<CircularBufferQueue<EmbeddingsRequest>>& get_request_queue() override {
        return m_requests;
    }

    ov::Tensor infer(EmbeddingsRequest& request, const ov::Tensor& input_ids, bool return_remote_tensor) override {
        OPENVINO_ASSERT(!return_remote_tensor, "Qwen3-ASR NPU text embeddings do not support remote output tensors.");
        const auto& shape = input_ids.get_shape();
        OPENVINO_ASSERT(
            input_ids.get_element_type() == ov::element::i64 && shape.size() == 2 && shape[0] == 1 && shape[1] > 0,
            "Qwen3-ASR NPU text embeddings require nonempty int64 token IDs with batch size 1.");
        ov::Tensor output{ov::element::f32, {1, shape[1], m_hidden_size}};
        const auto* source = static_cast<const uint8_t*>(input_ids.data());
        const auto token_stride = input_ids.get_strides().at(1);
        auto* destination = output.data<float>();
        auto chunk_input = request.ireq.get_input_tensor();
        for (size_t offset = 0; offset < shape[1]; offset += EMBEDDING_CHUNK_SIZE) {
            const size_t count = std::min(EMBEDDING_CHUNK_SIZE, shape[1] - offset);
            auto* chunk_ids = chunk_input.data<int64_t>();
            std::fill_n(chunk_ids, EMBEDDING_CHUNK_SIZE, int64_t{0});
            for (size_t index = 0; index < count; ++index) {
                int64_t token_id;
                std::memcpy(&token_id, source + (offset + index) * token_stride, sizeof(token_id));
                OPENVINO_ASSERT(token_id >= 0 && static_cast<size_t>(token_id) < m_vocab_size,
                                "Qwen3-ASR text embedding token ID is outside the vocabulary.");
                chunk_ids[index] = token_id;
            }
            request.ireq.infer();
            const auto chunk_output = request.ireq.get_output_tensor();
            OPENVINO_ASSERT(chunk_output.get_element_type() == ov::element::f32 &&
                                chunk_output.get_shape() == ov::Shape({1, EMBEDDING_CHUNK_SIZE, m_hidden_size}) &&
                                chunk_output.is_continuous(),
                            "Qwen3-ASR NPU text embeddings returned an unexpected output shape, type, or stride.");
            std::memcpy(destination + offset * m_hidden_size,
                        chunk_output.data<const float>(),
                        count * m_hidden_size * sizeof(float));
        }
        return output;
    }

private:
    size_t m_vocab_size = 0;
    size_t m_hidden_size = 0;
    std::unique_ptr<CircularBufferQueue<EmbeddingsRequest>> m_requests;
};

}  // namespace

EmbeddingsModel::Ptr create_qwen3_asr_embeddings(const std::filesystem::path& model_dir,
                                                 float scale_emb,
                                                 const std::string& device,
                                                 const ov::AnyMap& properties) {
    if (qwen3_asr_input_device(device, properties, "text_embeddings") != "NPU") {
        return EmbeddingsModel::create(model_dir, scale_emb, device, properties);
    }
    const auto model_properties = qwen3_asr_input_model_properties(properties, "text_embeddings", device);
    auto model =
        utils::singleton_core().read_model(model_dir / "openvino_text_embeddings_model.xml", {}, model_properties);
    return std::make_shared<Qwen3ASRNPUEmbeddingsModel>(model, scale_emb, device, properties);
}

EmbeddingsModel::Ptr create_qwen3_asr_embeddings(const std::string& model,
                                                 const ov::Tensor& weights,
                                                 float scale_emb,
                                                 const std::string& device,
                                                 const ov::AnyMap& properties) {
    if (qwen3_asr_input_device(device, properties, "text_embeddings") != "NPU") {
        return EmbeddingsModel::create(model, weights, scale_emb, device, properties);
    }
    auto ov_model = utils::singleton_core().read_model(model, weights);
    return std::make_shared<Qwen3ASRNPUEmbeddingsModel>(ov_model, scale_emb, device, properties);
}

}  // namespace ov::genai
