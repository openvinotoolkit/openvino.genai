// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "text_embedding_utils.hpp"

#include "logger.hpp"
#include "openvino/core/except.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace {

using namespace ov::genai;
using namespace ov;

void set_node_name(const std::shared_ptr<ov::Node>& node, const std::string& name) {
    node->set_friendly_name(name);
    node->get_output_tensor(0).set_names({name});
}

/**
 * CLS pooling slices first element from seq_length dimension
 * [batch_size, seq_length, hidden_size] -> [batch_size, seq_length[0], hidden_size]
 * [10, 5, 768] -> [10, 768]
 */
std::shared_ptr<op::Op> get_cls_pooling_op(const ov::Output<ov::Node>& last_hidden_state_node) {
    auto start = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto stop = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto step = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});

    auto slice = std::make_shared<op::v8::Slice>(last_hidden_state_node, start, stop, step, axis);

    auto squeeze_axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    return std::make_shared<op::v15::Squeeze>(slice, squeeze_axis);
}

std::shared_ptr<op::Op> get_mean_pooling_op(const ov::Output<ov::Node>& last_hidden_state_node,
                                            const ov::Output<ov::Node>& attention_mask) {
    auto shape_of = std::make_shared<op::v3::ShapeOf>(last_hidden_state_node);

    auto unsqueze_axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

    auto unsqueze = std::make_shared<op::v0::Unsqueeze>(attention_mask, unsqueze_axis);

    auto input_mask_expanded = std::make_shared<op::v3::Broadcast>(unsqueze, shape_of);

    auto input_mask_expanded_convert =
        std::make_shared<op::v0::Convert>(input_mask_expanded, last_hidden_state_node.get_element_type());

    auto last_hidden_node_with_applied_attention_mask =
        std::make_shared<op::v1::Multiply>(last_hidden_state_node, input_mask_expanded_convert->outputs()[0]);

    auto axis_1 = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto sum_hidden_state = std::make_shared<op::v1::ReduceSum>(last_hidden_node_with_applied_attention_mask, axis_1);

    // f32 overflow possible
    // ReduceMean might help with overflow but its precision diverges from LlamaIndex
    auto sum_expanded_mask = std::make_shared<op::v1::ReduceSum>(input_mask_expanded_convert, axis_1);

    auto nearest_to_zero =
        std::make_shared<op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{1e-12});
    auto max_expanded_mask = std::make_shared<op::v1::Maximum>(sum_expanded_mask, nearest_to_zero);

    // shape: [batch_size, hidden_state_size]
    return std::make_shared<op::v1::Divide>(sum_hidden_state, max_expanded_mask);
}

std::shared_ptr<op::Op> get_last_token_pooling_op(const ov::Output<ov::Node>& last_hidden_state_node,
                                                  const ov::Output<ov::Node>& attention_mask,
                                                  const TextEmbeddingPipeline::Config& config) {
    const auto left_padding = config.padding_side.has_value() && config.padding_side.value() == "left";

    // shortcut for left padding. We can slice last token directly
    if (left_padding) {
        auto start = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
        auto stop = std::make_shared<op::v0::Constant>(ov::element::i64,
                                                       ov::Shape{1},
                                                       std::vector<int64_t>{std::numeric_limits<int64_t>::max()});
        auto step = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        auto axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});

        auto slice = std::make_shared<op::v8::Slice>(last_hidden_state_node, start, stop, step, axis);

        auto squeeze_axis = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        return std::make_shared<op::v15::Squeeze>(slice, squeeze_axis);
    }

    auto axis_1 = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto reduce_sum = std::make_shared<op::v1::ReduceSum>(attention_mask, axis_1);
    auto subtract_1 = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto subtract = std::make_shared<op::v1::Subtract>(reduce_sum, subtract_1);

    return std::make_shared<op::v8::Gather>(last_hidden_state_node, subtract, axis_1, 1);
}

std::shared_ptr<op::Op> create_post_ops(const ov::Output<ov::Node>& input,
                                        const ov::Output<ov::Node>& attention_mask,
                                        const TextEmbeddingPipeline::Config& config) {
    if (config.pooling_type == TextEmbeddingPipeline::PoolingType::CLS) {
        return get_cls_pooling_op(input);
    } else if (config.pooling_type == TextEmbeddingPipeline::PoolingType::MEAN) {
        return get_mean_pooling_op(input, attention_mask);
    } else if (config.pooling_type == TextEmbeddingPipeline::PoolingType::LAST_TOKEN) {
        return get_last_token_pooling_op(input, attention_mask, config);
    }

    OPENVINO_THROW("Pooling type is not supported");
}

std::shared_ptr<op::Op> create_normalize_ops(const ov::Output<ov::Node>& input,
                                             const TextEmbeddingPipeline::Config& config) {
    if (config.normalize) {
        auto axis_const = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector{1});
        return std::make_shared<op::v0::NormalizeL2>(input, axis_const, 1e-12, op::EpsMode::MAX);
    }
    return std::dynamic_pointer_cast<op::Op>(input.get_node_shared_ptr());
}

}  // namespace

namespace ov {
namespace genai {
namespace utils {

std::shared_ptr<Model> apply_postprocessing(std::shared_ptr<Model> model, const TextEmbeddingPipeline::Config& config) {
    ov::preprocess::PrePostProcessor processor(model);

    processor.output().postprocess().custom([model, &config](const ov::Output<ov::Node>& node) {
        auto attention_mask = model->input("attention_mask").get_node()->outputs()[0];
        return create_post_ops(node, attention_mask, config);
    });

    if (config.normalize) {
        processor.output().postprocess().custom([&config](const ov::Output<ov::Node>& node) {
            return create_normalize_ops(node, config);
        });
    }

    return processor.build();
}

std::shared_ptr<ov::Model> create_post_model(std::shared_ptr<ov::Model> model,
                                             const TextEmbeddingPipeline::Config& config) {
    auto output_node = model->outputs()[0];
    auto output_shape = output_node.get_partial_shape();
    auto input_param = std::make_shared<ov::op::v0::Parameter>(output_node.get_element_type(),
                                                               ov::PartialShape{1, -1, output_shape[2]});
    set_node_name(input_param, "embedding_hidden_state");

    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    set_node_name(attention_mask, "attention_mask");

    auto post_output = create_post_ops(input_param, attention_mask, config);
    auto post_normalize_output = create_normalize_ops(post_output, config);
    OPENVINO_ASSERT(post_normalize_output != nullptr);

    auto result_node = std::make_shared<ov::op::v0::Result>(post_normalize_output);
    set_node_name(result_node, "last_hidden_state");
    auto post_model =
        std::make_shared<ov::Model>(ov::OutputVector{result_node}, ov::ParameterVector{input_param, attention_mask});
    post_model->set_friendly_name(model->get_friendly_name() + "_post_process");
    post_model->validate_nodes_and_infer_types();
    return post_model;
}

void reshape_model(std::shared_ptr<Model>& model,
                   const TextEmbeddingPipeline::Config& config,
                   std::optional<size_t> max_position_embeddings) {
    ov::PartialShape target_shape{ov::Dimension::dynamic(), ov::Dimension::dynamic()};

    if (config.batch_size.has_value()) {
        target_shape[0] = ov::Dimension(*config.batch_size);
    }

    if (config.max_length.has_value()) {
        if (max_position_embeddings.has_value() && *config.max_length > max_position_embeddings.value()) {
            std::stringstream message;
            message << "max_length is set to " << *config.max_length
                    << " which is greater than models max_position_embeddings (" << max_position_embeddings.value()
                    << ")."
                    << " Some models may fail with such configuration."
                    << " Remove max_position_embeddings from config.json to silence this warning.";
            GENAI_WARN(message.str());
        }

        if (config.pad_to_max_length.has_value() && config.pad_to_max_length.value()) {
            target_shape[1] = ov::Dimension(*config.max_length);
        } else {
            target_shape[1] = ov::Dimension{1, static_cast<int64_t>(config.max_length.value())};
        }
    }

    std::map<std::string, ov::PartialShape> input_name_to_shape;
    input_name_to_shape["input_ids"] = target_shape;
    input_name_to_shape["attention_mask"] = target_shape;

    if (has_token_type_ids_input(model->inputs())) {
        input_name_to_shape["token_type_ids"] = target_shape;
    }

    model->reshape(input_name_to_shape);
}

}  // namespace utils
}  // namespace genai
}  // namespace ov
