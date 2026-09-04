// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace {

template <typename T>
size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::as_type_ptr<T>(op))
            count++;
    }
    return count;
}

std::shared_ptr<ov::op::v0::Parameter> find_parameter(const std::shared_ptr<ov::Model>& model,
                                                      const std::string& name) {
    for (const auto& param : model->get_parameters()) {
        if (param->get_friendly_name() == name || param->output(0).get_names().count(name) != 0)
            return param;
    }
    return nullptr;
}

// Build: Parameter("visual_pos_masks")[mask_shape] -> NonZero -> Result
std::shared_ptr<ov::Model> make_mask_nonzero_model(const ov::PartialShape& mask_shape,
                                                   const std::string& friendly_name = "visual_pos_masks") {
    auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, mask_shape);
    mask->set_friendly_name(friendly_name);
    mask->output(0).set_names({friendly_name});
    auto non_zero = std::make_shared<ov::op::v3::NonZero>(mask);
    auto result = std::make_shared<ov::op::v0::Result>(non_zero);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{mask});
}

}  // namespace

// ===================== fix_deepstack_visual_pos_masks_layout_for_paged_attention =====================

// Happy path: visual_pos_masks -> NonZero, matching the expected pattern exactly.
TEST(DeepstackVisualPosMasksLayout, InsertsTransposeWhenPatternMatches) {
    auto model = make_mask_nonzero_model(ov::PartialShape{1, -1});
    ASSERT_EQ(count_ops<ov::op::v1::Transpose>(model), 0u);

    ov::genai::utils::fix_deepstack_visual_pos_masks_layout_for_paged_attention(model);

    ASSERT_EQ(count_ops<ov::op::v1::Transpose>(model), 1u);
    auto mask = find_parameter(model, "visual_pos_masks");
    ASSERT_NE(mask, nullptr);

    // The Parameter's only consumer must now be the inserted Transpose ...
    auto targets = mask->output(0).get_target_inputs();
    ASSERT_EQ(targets.size(), 1u);
    auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(targets.begin()->get_node()->shared_from_this());
    ASSERT_NE(transpose, nullptr);

    // ... and the original NonZero must now consume the Transpose's output, not the Parameter directly.
    auto non_zero = ov::as_type_ptr<ov::op::v3::NonZero>(
        transpose->output(0).get_target_inputs().begin()->get_node()->shared_from_this());
    ASSERT_NE(non_zero, nullptr);
    ASSERT_EQ(non_zero->input_value(0).get_node_shared_ptr(), transpose);
}

// Matched by tensor name even when the friendly name has been mangled/renamed by other passes.
TEST(DeepstackVisualPosMasksLayout, MatchesByTensorNameWhenFriendlyNameDiffers) {
    auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::PartialShape{1, -1});
    mask->set_friendly_name("Parameter_123");
    mask->output(0).set_names({"visual_pos_masks"});
    auto non_zero = std::make_shared<ov::op::v3::NonZero>(mask);
    auto result = std::make_shared<ov::op::v0::Result>(non_zero);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{mask});

    ov::genai::utils::fix_deepstack_visual_pos_masks_layout_for_paged_attention(model);

    ASSERT_EQ(count_ops<ov::op::v1::Transpose>(model), 1u);
}

// Multiple NonZero consumers of the same mask (e.g. one feeding GatherND, another feeding
// ScatterNDUpdate) must all be redirected to the same inserted Transpose.
TEST(DeepstackVisualPosMasksLayout, RedirectsAllNonZeroConsumers) {
    auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::PartialShape{1, -1});
    mask->set_friendly_name("visual_pos_masks");
    mask->output(0).set_names({"visual_pos_masks"});
    auto non_zero_1 = std::make_shared<ov::op::v3::NonZero>(mask);
    auto non_zero_2 = std::make_shared<ov::op::v3::NonZero>(mask);
    auto result_1 = std::make_shared<ov::op::v0::Result>(non_zero_1);
    auto result_2 = std::make_shared<ov::op::v0::Result>(non_zero_2);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result_1, result_2}, ov::ParameterVector{mask});

    ov::genai::utils::fix_deepstack_visual_pos_masks_layout_for_paged_attention(model);

    // Exactly one Transpose must be shared by both NonZero consumers, not one per consumer.
    ASSERT_EQ(count_ops<ov::op::v1::Transpose>(model), 1u);
    ASSERT_EQ(non_zero_1->input_value(0).get_node_shared_ptr(), non_zero_2->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(ov::is_type<ov::op::v1::Transpose>(non_zero_1->input_value(0).get_node_shared_ptr()));
}

// No "visual_pos_masks" input at all - the common case for every non-DeepStack model - must be a no-op.
TEST(DeepstackVisualPosMasksLayout, NoOpWhenParameterMissing) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 16});
    input->set_friendly_name("hidden_states");
    auto result = std::make_shared<ov::op::v0::Result>(input);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});

    ov::genai::utils::fix_deepstack_visual_pos_masks_layout_for_paged_attention(model);

    ASSERT_EQ(count_ops<ov::op::v1::Transpose>(model), 0u);
}

// visual_pos_masks present but not consumed by NonZero at all - conservative skip, not a crash.
TEST(DeepstackVisualPosMasksLayout, NoOpWhenConsumerIsNotNonZero) {
    auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1});
    mask->set_friendly_name("visual_pos_masks");
    mask->output(0).set_names({"visual_pos_masks"});
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 0.0f);
    auto add = std::make_shared<ov::op::v1::Add>(mask, bias);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{mask});

    ov::genai::utils::fix_deepstack_visual_pos_masks_layout_for_paged_attention(model);

    ASSERT_EQ(count_ops<ov::op::v1::Transpose>(model), 0u);
    ASSERT_EQ(add->input_value(0).get_node_shared_ptr(), mask);
}

// visual_pos_masks present with a mix of NonZero and non-NonZero consumers - conservative skip
// of the whole parameter, since we can't safely retarget only some of its consumers.
TEST(DeepstackVisualPosMasksLayout, NoOpWhenOnlySomeConsumersAreNonZero) {
    auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1});
    mask->set_friendly_name("visual_pos_masks");
    mask->output(0).set_names({"visual_pos_masks"});
    auto non_zero = std::make_shared<ov::op::v3::NonZero>(mask);
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 0.0f);
    auto add = std::make_shared<ov::op::v1::Add>(mask, bias);
    auto result_1 = std::make_shared<ov::op::v0::Result>(non_zero);
    auto result_2 = std::make_shared<ov::op::v0::Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result_1, result_2}, ov::ParameterVector{mask});

    ov::genai::utils::fix_deepstack_visual_pos_masks_layout_for_paged_attention(model);

    ASSERT_EQ(count_ops<ov::op::v1::Transpose>(model), 0u);
    ASSERT_EQ(non_zero->input_value(0).get_node_shared_ptr(), mask);
    ASSERT_EQ(add->input_value(0).get_node_shared_ptr(), mask);
}

// visual_pos_masks with an unexpected (non-2D) rank - conservative skip.
TEST(DeepstackVisualPosMasksLayout, NoOpWhenMaskRankIsNot2D) {
    auto model = make_mask_nonzero_model(ov::PartialShape{-1});

    ov::genai::utils::fix_deepstack_visual_pos_masks_layout_for_paged_attention(model);

    ASSERT_EQ(count_ops<ov::op::v1::Transpose>(model), 0u);
}

// visual_pos_masks with a dynamic rank - conservative skip.
TEST(DeepstackVisualPosMasksLayout, NoOpWhenMaskRankIsDynamic) {
    auto model = make_mask_nonzero_model(ov::PartialShape::dynamic());

    ov::genai::utils::fix_deepstack_visual_pos_masks_layout_for_paged_attention(model);

    ASSERT_EQ(count_ops<ov::op::v1::Transpose>(model), 0u);
}
