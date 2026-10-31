// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the VLSDPA enablement helpers used by the Qwen vision encoders:
//   request_vl_sdpa_transformations() - tags a model with the "QWenVL" hint the GPU plugin keys off.
//   check_vl_sdpa_transformations()   - reports whether a compiled model exposes the packed
//                                       "cu_seq_lens" input expected by its callers.
// Both are device-independent, so the assertions run on CPU.

#include <gtest/gtest.h>

#include <string>

#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"

#include "visual_language/vl_sdpa_transformations.hpp"

namespace {

// Build a trivial Parameter -> Relu -> Result model whose single input carries `input_name`.
std::shared_ptr<ov::Model> make_named_input_model(const std::string& input_name) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 4});
    param->set_friendly_name(input_name);
    param->output(0).get_tensor().set_names({input_name});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto result = std::make_shared<ov::op::v0::Result>(relu);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

}  // namespace

using namespace ov::genai;

// request_vl_sdpa_transformations must tag the model with the "QWenVL" model_type_hint.
TEST(VlSdpaTransformations, RequestSetsModelTypeHint) {
    auto model = make_named_input_model("attention_mask");
    ASSERT_FALSE(model->has_rt_info("model_type_hint"));

    utils::request_vl_sdpa_transformations(model);

    ASSERT_TRUE(model->has_rt_info("model_type_hint"));
    EXPECT_EQ(model->get_rt_info<std::string>("model_type_hint"), "QWenVL");
}

// check must report true when a compiled model exposes the packed "cu_seq_lens" input.
TEST(VlSdpaTransformations, CheckDetectsCuSeqLens) {
    ov::Core core;
    auto compiled = core.compile_model(make_named_input_model("cu_seq_lens"), "CPU");
    EXPECT_TRUE(utils::check_vl_sdpa_transformations(compiled));
}

// A window-only input must not enable callers that unconditionally set "cu_seq_lens".
TEST(VlSdpaTransformations, CheckRejectsCuWindowSeqlensOnly) {
    ov::Core core;
    auto compiled = core.compile_model(make_named_input_model("cu_window_seqlens"), "CPU");
    EXPECT_FALSE(utils::check_vl_sdpa_transformations(compiled));
}

// Callers that feed a specific packed input must be able to distinguish the two names.
TEST(VlSdpaTransformations, HasVlSdpaInputMatchesExactName) {
    ov::Core core;
    auto cu_seq_lens = core.compile_model(make_named_input_model("cu_seq_lens"), "CPU");
    auto cu_window_seqlens = core.compile_model(make_named_input_model("cu_window_seqlens"), "CPU");

    EXPECT_TRUE(utils::has_vl_sdpa_input(cu_seq_lens, "cu_seq_lens"));
    EXPECT_FALSE(utils::has_vl_sdpa_input(cu_seq_lens, "cu_window_seqlens"));
    EXPECT_FALSE(utils::has_vl_sdpa_input(cu_window_seqlens, "cu_seq_lens"));
    EXPECT_TRUE(utils::has_vl_sdpa_input(cu_window_seqlens, "cu_window_seqlens"));
}

// A model without the packed input (the dense attention_mask path) must report false.
TEST(VlSdpaTransformations, CheckAbsentReturnsFalse) {
    ov::Core core;
    auto compiled = core.compile_model(make_named_input_model("attention_mask"), "CPU");
    EXPECT_FALSE(utils::check_vl_sdpa_transformations(compiled));
}
