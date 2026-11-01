// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_clip_adapter.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/opsets/opset15.hpp"

namespace ov::genai {
namespace {
std::shared_ptr<ov::Model> batch_encoder(const std::shared_ptr<ov::Model>& body,
                                         const std::shared_ptr<ov::op::v0::Parameter>& pixels,
                                         const ov::Output<ov::Node>& features) {
    auto shape = pixels->get_partial_shape();
    shape[0] = ov::Dimension::dynamic();
    auto input = std::make_shared<ov::op::v0::Parameter>(pixels->get_element_type(), shape);
    input->set_friendly_name("pixel_values");
    input->output(0).set_names({"pixel_values"});
    auto iterator = std::make_shared<ov::op::v0::TensorIterator>();
    iterator->set_body(body);
    iterator->set_sliced_input(pixels, input, 0, 1, 1, -1, 0);
    auto output = iterator->get_concatenated_slices(features, 0, 1, 1, -1, 0);
    output.set_names({"last_hidden_state"});
    iterator->validate_and_infer_types();
    auto result = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input});
    result->get_rt_info() = body->get_rt_info();
    return result;
}
}  // namespace

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

    return batch_encoder(body, pixels, squeezed);
}

std::shared_ptr<ov::Model> adapt_gguf_packed_vision_encoder(const std::shared_ptr<ov::Model>& model,
                                                            size_t patch_size,
                                                            size_t temporal_patch_size) {
    using namespace ov::opset15;
    OPENVINO_ASSERT(patch_size > 0 && (temporal_patch_size == 1 || temporal_patch_size == 2),
                    "GGUF packed vision requires a positive patch size and one or two temporal patches");
    auto body = model->clone();
    const auto parameters = body->get_parameters();
    OPENVINO_ASSERT((parameters.size() == 2 || parameters.size() == 3) && body->get_results().size() == 1,
                    "GGUF packed vision requires pixel inputs, positions and one embedding output");
    const auto axes = [](std::initializer_list<int64_t> values) {
        return Constant::create(ov::element::i64, {values.size()}, values);
    };
    auto pixels = std::make_shared<Parameter>(
        ov::element::f32,
        ov::PartialShape{1, -1, -1, static_cast<int64_t>(3 * temporal_patch_size * patch_size * patch_size)});
    auto shape = std::make_shared<ShapeOf>(pixels);
    auto height = std::make_shared<Gather>(shape, axes({1}), axes({0}));
    auto width = std::make_shared<Gather>(shape, axes({2}), axes({0}));
    auto unpacked_shape = std::make_shared<Concat>(ov::OutputVector{axes({1}),
                                                                    height,
                                                                    width,
                                                                    axes({3,
                                                                          static_cast<int64_t>(temporal_patch_size),
                                                                          static_cast<int64_t>(patch_size),
                                                                          static_cast<int64_t>(patch_size)})},
                                                   0);
    auto unpacked = std::make_shared<Reshape>(pixels, unpacked_shape, false);
    auto image_shape = std::make_shared<Concat>(
        ov::OutputVector{axes({1, 3}),
                         std::make_shared<Multiply>(height, axes({static_cast<int64_t>(patch_size)})),
                         std::make_shared<Multiply>(width, axes({static_cast<int64_t>(patch_size)}))},
        0);
    auto count = std::make_shared<Multiply>(height, width);
    auto indices = std::make_shared<Range>(std::make_shared<Squeeze>(axes({0})),
                                           std::make_shared<Squeeze>(count),
                                           std::make_shared<Squeeze>(axes({1})),
                                           ov::element::i64);
    auto rows = std::make_shared<Divide>(indices, width, true);
    auto columns = std::make_shared<FloorMod>(indices, width);
    auto zeros = std::make_shared<Broadcast>(axes({0}), count);
    auto positions = std::make_shared<Reshape>(
        std::make_shared<Convert>(std::make_shared<Concat>(ov::OutputVector{rows, columns, zeros, zeros}, 0),
                                  ov::element::i32),
        axes({1, 1, 1, -1}),
        false);
    bool first = false, position_input = false;
    for (const auto& parameter : parameters) {
        const auto& names = parameter->output(0).get_names();
        ov::Output<ov::Node> replacement;
        if (names.count("vision.positions")) {
            position_input = true;
            replacement = positions;
        } else {
            const bool second = names.count("vision.pixel_values_1");
            OPENVINO_ASSERT(names.count("vision.pixel_values") || (second && temporal_patch_size == 2),
                            "GGUF packed vision has an unsupported native input");
            first |= !second;
            auto frame = std::make_shared<Squeeze>(
                std::make_shared<Slice>(unpacked, axes({second ? 1 : 0}), axes({second ? 2 : 1}), axes({1}), axes({4})),
                axes({4}));
            replacement = std::make_shared<Reshape>(std::make_shared<Transpose>(frame, axes({0, 3, 1, 4, 2, 5})),
                                                    image_shape,
                                                    false);
        }
        ov::replace_node(parameter, replacement.get_node_shared_ptr());
        body->remove_parameter(parameter);
    }
    OPENVINO_ASSERT(first && position_input, "GGUF packed vision requires pixels and spatial positions");
    body->add_parameters({pixels});
    const auto features = body->get_results().front()->input_value(0);
    const auto output_shape = features.get_partial_shape();
    OPENVINO_ASSERT(features.get_names().count("vision.embeddings") && output_shape.rank().is_static() &&
                        output_shape.rank().get_length() == 4 && output_shape[0] == 1 && output_shape[1] == 1,
                    "GGUF packed vision requires native [1, 1, tokens, width] embeddings");
    auto squeezed = std::make_shared<Squeeze>(features, axes({0}));
    body->remove_result(body->get_results().front());
    body->add_results({std::make_shared<Result>(squeezed)});
    body->validate_nodes_and_infer_types();
    return batch_encoder(body, pixels, squeezed);
}
}  // namespace ov::genai
