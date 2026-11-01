// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_clip_adapter.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tensor_iterator.hpp"

namespace ov::genai {
std::shared_ptr<ov::Model> adapt_gguf_clip_encoder(const std::shared_ptr<ov::Model>& model) {
    auto body = model->clone();
    OPENVINO_ASSERT(body->get_parameters().size() == 1 && body->get_results().size() == 1,
                    "GGUF LLaVA and InternVL require one pixel input and one vision output");
    const auto pixels = body->get_parameters().front();
    auto shape = pixels->get_partial_shape();
    OPENVINO_ASSERT(pixels->output(0).get_names().count("vision.pixel_values") && shape.rank().is_static() &&
                        shape.rank().get_length() == 4 && shape[0] == 1 && shape[1] == 3,
                    "GGUF LLaVA and InternVL require RGB NCHW pixels with a singleton native batch");
    const auto features = body->get_results().front()->input_value(0);
    const auto output_shape = features.get_partial_shape();
    OPENVINO_ASSERT(features.get_names().count("vision.embeddings") && output_shape.rank().is_static() &&
                        output_shape.rank().get_length() == 4 && output_shape[0] == 1 && output_shape[1] == 1,
                    "GGUF LLaVA and InternVL require native [1, 1, tokens, width] embeddings");
    auto squeezed =
        std::make_shared<ov::op::v0::Squeeze>(features, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    body->remove_result(body->get_results().front());
    body->add_results({std::make_shared<ov::op::v0::Result>(squeezed)});
    body->validate_nodes_and_infer_types();

    shape[0] = ov::Dimension::dynamic();
    auto input = std::make_shared<ov::op::v0::Parameter>(pixels->get_element_type(), shape);
    input->set_friendly_name("pixel_values");
    input->output(0).set_names({"pixel_values"});
    auto iterator = std::make_shared<ov::op::v0::TensorIterator>();
    iterator->set_body(body);
    iterator->set_sliced_input(pixels, input, 0, 1, 1, -1, 0);
    auto output = iterator->get_concatenated_slices(squeezed, 0, 1, 1, -1, 0);
    output.set_names({"last_hidden_state"});
    iterator->validate_and_infer_types();
    auto result = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input});
    result->get_rt_info() = model->get_rt_info();
    return result;
}
}  // namespace ov::genai
