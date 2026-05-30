// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "speculative_decoding/eagle3_model_transforms.hpp"

namespace {

std::shared_ptr<ov::Model> make_annotated_hidden_state_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4});
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 2, 4}, 1.0f);

    auto layer0 = std::make_shared<ov::op::v1::Add>(input, bias);
    layer0->output(0).set_names({"ov.hidden_states.decoder_layer_0"});
    auto layer1 = std::make_shared<ov::op::v1::Add>(layer0, bias);
    layer1->output(0).set_names({"ov.hidden_states.decoder_layer_1"});
    auto layer2 = std::make_shared<ov::op::v1::Add>(layer1, bias);
    layer2->output(0).set_names({"ov.hidden_states.decoder_layer_2"});
    auto layer3 = std::make_shared<ov::op::v1::Add>(layer2, bias);
    layer3->output(0).set_names({"ov.hidden_states.decoder_layer_3"});
    auto layer4 = std::make_shared<ov::op::v1::Add>(layer3, bias);
    layer4->output(0).set_names({"ov.hidden_states.decoder_layer_4"});

    auto result = std::make_shared<ov::op::v0::Result>(layer4);
    result->output(0).set_names({"logits"});
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    model->set_rt_info(
        std::string(R"({"version":1,"layers":{"0":"ov.hidden_states.decoder_layer_0","1":"ov.hidden_states.decoder_layer_1","2":"ov.hidden_states.decoder_layer_2","3":"ov.hidden_states.decoder_layer_3","4":"ov.hidden_states.decoder_layer_4"}})"),
        "hidden_states_decoder_layers");
    return model;
}

size_t count_outputs_with_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    size_t count = 0;
    for (const auto& output : model->outputs()) {
        if (output.get_names().count(name) != 0) {
            ++count;
        }
    }
    return count;
}

}  // namespace

TEST(Eagle3ModelTransforms, AddsSelectedAnnotatedHiddenStatesAsOutput) {
    auto model = make_annotated_hidden_state_model();

    ov::genai::utils::eagle3::transform_hidden_state(model, {0, 1, 2});

    ASSERT_EQ(count_outputs_with_name(model, "last_hidden_state"), 1);
    const auto hidden_state = model->output("last_hidden_state");
    ASSERT_EQ(hidden_state.get_partial_shape(), ov::PartialShape({1, 2, 12}));
    ASSERT_EQ(count_outputs_with_name(model, "ov.hidden_states.decoder_layer_0"), 0);
}

TEST(Eagle3ModelTransforms, ThrowsForMissingAnnotatedHiddenStateLayer) {
    auto model = make_annotated_hidden_state_model();

    EXPECT_THROW(ov::genai::utils::eagle3::transform_hidden_state(model, {0, 1, 99}), ov::Exception);
}
