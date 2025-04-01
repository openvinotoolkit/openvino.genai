// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag_components/embedding_pipeline.hpp"

#include "debug_utils.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

using namespace ov::op;

class TextEmbeddingPipeline::TextEmbeddingPipelineImpl {
public:
    TextEmbeddingPipelineImpl(const std::filesystem::path& models_path,
                              const std::string& device,
                              const std::optional<Config>& config,
                              const ov::AnyMap& properties = {})
        : m_tokenizer{models_path} {
        if (config.has_value()) {
            m_config = *config;
        }

        ov::Core core = utils::singleton_core();

        auto model = core.read_model(models_path / "openvino_model.xml");

        model = apply_postprocessing(model, m_config);

        ov::CompiledModel compiled_model = core.compile_model(model, device, properties);

        ov::genai::utils::print_compiled_model_properties(compiled_model, "embedding model");
        m_request = compiled_model.create_infer_request();
    };

    std::vector<EmbeddingResult> embed_documents(const std::vector<std::string>& texts) {
        std::vector<std::string> copy{texts};
        const auto tokenized_inputs = m_tokenizer.encode(copy);

        m_request.set_tensor("input_ids", tokenized_inputs.input_ids);
        m_request.set_tensor("attention_mask", tokenized_inputs.attention_mask);
        m_request.infer();

        // [batch_size, hidden_size]
        const Tensor last_hidden_state = m_request.get_tensor("last_hidden_state");

        return to_embedding_result(last_hidden_state);
    };

    EmbeddingResult embed_query(const std::string& text) {
        return embed_documents({text})[0];
    };

private:
    Tokenizer m_tokenizer;
    InferRequest m_request;
    Config m_config;

    std::vector<EmbeddingResult> to_embedding_result(const Tensor last_hidden_state) {
        const auto last_hidden_state_data = last_hidden_state.data<float>();

        std::vector<EmbeddingResult> result;
        const auto shape = last_hidden_state.get_shape();

        const size_t batch_size = shape[0];
        const size_t hidden_size = shape[1];

        for (size_t batch = 0; batch < batch_size; batch++) {
            const auto batch_offset = batch * hidden_size;
            const auto batch_data = last_hidden_state_data + batch_offset;
            const std::vector<float> batch_result(batch_data, batch_data + hidden_size);
            result.push_back(batch_result);
        }

        return result;
    }

    std::shared_ptr<Model> apply_postprocessing(std::shared_ptr<Model> model, const Config& config) {
        ov::preprocess::PrePostProcessor processor(model);

        processor.output().postprocess().custom([this, model, &config](const ov::Output<ov::Node>& node) {
            if (config.pooling_type == PoolingType::CLS) {
                return get_cls_pooling_op(node);
            } else if (config.pooling_type == PoolingType::MEAN) {
                return get_mean_pooling_op(model, node);
            }

            OPENVINO_THROW("Pooling type is not supported");
        });

        if (config.normalize) {
            processor.output().postprocess().custom([](const ov::Output<ov::Node>& node) {
                auto axis_const = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector{1});
                return std::make_shared<v0::NormalizeL2>(node, axis_const, 1e-12, EpsMode::ADD);
            });
        }

        return processor.build();
    }

    /**
     * CLS pooling slices first element from seq_length dimension
     * [batch_size, seq_length, hidden_size] -> [batch_size, seq_length[0], hidden_size]
     * [10, 5, 768] -> [10, 1, 768]
     */
    std::shared_ptr<ov::op::Op> get_cls_pooling_op(const ov::Output<ov::Node>& last_hidden_state_node) {
        auto start = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
        auto stop = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        auto step = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});

        auto slice = std::make_shared<ov::op::v8::Slice>(last_hidden_state_node, start, stop, step, axis);

        // [10, 1, 768] -> [10, 768]
        auto squeeze_axis =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        return std::make_shared<ov::op::v15::Squeeze>(slice, squeeze_axis);
    }

    std::shared_ptr<ov::op::Op> get_mean_pooling_op(std::shared_ptr<Model> model,
                                                    const ov::Output<ov::Node>& last_hiddent_state_node) {
        const auto hidden_state_size = last_hiddent_state_node.get_partial_shape()[2].get_length();

        // shape: [batch_size, seq_length]
        auto attention_mask = model->input("attention_mask").get_node()->outputs()[0];

        auto unsqueze_axis =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

        // shape: [batch_size, seq_length, 1]
        auto unsqueze = std::make_shared<ov::op::v0::Unsqueeze>(attention_mask, unsqueze_axis);

        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(attention_mask);

        auto hidden_state_size_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                              ov::Shape{1},
                                                                              std::vector<int64_t>{hidden_state_size});

        // value: [batch_size, seq_length, hidden_state_size]
        auto broadcast_shape =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{shape_of, hidden_state_size_const}, 0);

        // shape: [batch_size, seq_length, hidden_state_size]
        auto input_mask_expanded = std::make_shared<ov::op::v3::Broadcast>(unsqueze, broadcast_shape);
        auto input_mask_expanded_convert =
            std::make_shared<ov::op::v0::Convert>(input_mask_expanded, last_hiddent_state_node.get_element_type());

        // shape: [batch_size, seq_length, hidden_state_size]
        auto attention_mask_multiply =
            std::make_shared<ov::op::v1::Multiply>(last_hiddent_state_node, input_mask_expanded_convert->outputs()[0]);

        auto mean_axis =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});

        // shape: [batch_size, 1, hidden_state_size]
        return std::make_shared<ov::op::v1::ReduceMean>(attention_mask_multiply, mean_axis);
    }
};

TextEmbeddingPipeline::TextEmbeddingPipeline(const std::filesystem::path& models_path,
                                             const std::string& device,
                                             const std::optional<Config>& config,
                                             const ov::AnyMap& properties) {
    m_impl = std::make_unique<TextEmbeddingPipelineImpl>(models_path, device, config, properties);
};

std::vector<EmbeddingResult> TextEmbeddingPipeline::embed_documents(const std::vector<std::string>& texts) {
    return m_impl->embed_documents(texts);
}

EmbeddingResult TextEmbeddingPipeline::embed_query(const std::string& text) {
    return m_impl->embed_query(text);
}

TextEmbeddingPipeline::~TextEmbeddingPipeline() = default;

}  // namespace genai
}  // namespace ov
