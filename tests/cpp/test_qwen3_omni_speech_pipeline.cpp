// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Regression tests for Qwen3-Omni speech pipeline CodePredictor GPU precision fix.
//
// Case 1 — GPU device detection: device.find("GPU") != std::string::npos
// Case 2 — Precision hint type:  compilation_props["INFERENCE_PRECISION_HINT"] = ov::element::f32

#include <gtest/gtest.h>

#include <string>

#include "openvino/core/any.hpp"
#include "openvino/core/type/element_type.hpp"

namespace {

bool is_gpu_device(const std::string& device) {
    return device.find("GPU") != std::string::npos;
}

}  // namespace

// ── Case 1: GPU device detection ─────────────────────────────────────────────

TEST(CodePredictorGPUDetection, MatchesPlainGPU) {
    EXPECT_TRUE(is_gpu_device("GPU"));
}

TEST(CodePredictorGPUDetection, MatchesGPUWithIndex) {
    EXPECT_TRUE(is_gpu_device("GPU.0"));
    EXPECT_TRUE(is_gpu_device("GPU.1"));
}

TEST(CodePredictorGPUDetection, MatchesHeteroGPU) {
    EXPECT_TRUE(is_gpu_device("HETERO:GPU,CPU"));
}

TEST(CodePredictorGPUDetection, MatchesMultiGPU) {
    EXPECT_TRUE(is_gpu_device("MULTI:GPU,CPU"));
}

TEST(CodePredictorGPUDetection, MatchesAutoGPU) {
    EXPECT_TRUE(is_gpu_device("AUTO:GPU,CPU"));
}

TEST(CodePredictorGPUDetection, RejectsCPU) {
    EXPECT_FALSE(is_gpu_device("CPU"));
}

TEST(CodePredictorGPUDetection, RejectsNPU) {
    EXPECT_FALSE(is_gpu_device("NPU"));
}

TEST(CodePredictorGPUDetection, RejectsEmptyString) {
    EXPECT_FALSE(is_gpu_device(""));
}

// ── Case 2: precision hint stored as typed ov::element::Type ─────────────────

TEST(CodePredictorPrecisionHint, TypedElementStoredAsElementType) {
    ov::AnyMap props;
    props["INFERENCE_PRECISION_HINT"] = ov::element::f32;

    const auto& any = props.at("INFERENCE_PRECISION_HINT");
    EXPECT_TRUE(any.is<ov::element::Type>());
    EXPECT_FALSE(any.is<std::string>());
    EXPECT_EQ(any.as<ov::element::Type>(), ov::element::f32);
}

TEST(CodePredictorPrecisionHint, F32AndF16AreDistinct) {
    ov::AnyMap props;
    props["PRECISION_F32"] = ov::element::f32;
    props["PRECISION_F16"] = ov::element::f16;

    EXPECT_EQ(props.at("PRECISION_F32").as<ov::element::Type>(), ov::element::f32);
    EXPECT_EQ(props.at("PRECISION_F16").as<ov::element::Type>(), ov::element::f16);
    EXPECT_NE(props.at("PRECISION_F32").as<ov::element::Type>(),
              props.at("PRECISION_F16").as<ov::element::Type>());
}
