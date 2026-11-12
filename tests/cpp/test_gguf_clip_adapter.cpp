// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_utils/gguf_clip_adapter.hpp"
#include "gtest/gtest.h"
#include "model_desc.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/openvino.hpp"

TEST(GGUFClipAdapter, RejectsDraftModelBeforeLoading) {
    for (const std::string backend : {"SDPA", "PA"}) {
        ov::AnyMap properties{{"draft_model", ov::genai::ModelDesc{}},
                              {"ATTENTION_BACKEND", backend},
                              {"mmproj_path", std::string("unused.gguf")}};
        try {
            ov::genai::VLMPipeline pipeline("unused.gguf", "CPU", properties);
            FAIL() << "Expected draft_model rejection";
        } catch (const ov::Exception& error) {
            EXPECT_NE(std::string(error.what()).find("draft_model is not supported for GGUF VLMPipeline"),
                      std::string::npos);
        }
    }
}

TEST(GGUFClipAdapter, ImageSlotsPreserveTextAndTruncatedImageRows) {
    const std::vector<bool> slots{false, true, true, true, true, false};
    ov::Tensor first(ov::element::f32, {1, 2, 2}), second(ov::element::f32, {1, 2, 2});
    for (size_t i = 0; i < 4; ++i) {
        first.data<float>()[i] = float(i + 1);
        second.data<float>()[i] = float(i + 5);
    }
    const std::vector<float> expected{.25f, .25f, 1, 2, 3, 4, 5, 6, 7, 8, .25f, .25f};
    for (size_t length : {size_t(6), size_t(4), size_t(2), size_t(1)}) {
        ov::Tensor text(ov::element::f32, {1, length, 2});
        std::fill_n(text.data<float>(), text.get_size(), .25f);
        auto merged = ov::genai::merge_gguf_llava_embeddings(slots, text, {first, second});
        for (size_t i = 0; i < text.get_size(); ++i) {
            EXPECT_FLOAT_EQ(merged.data<float>()[i], expected[expected.size() - text.get_size() + i]);
            EXPECT_FLOAT_EQ(text.data<float>()[i], .25f);
        }
    }
}
