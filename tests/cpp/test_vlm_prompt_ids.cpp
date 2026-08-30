// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "sampling/logit_processor.hpp"
#include "sequence_group.hpp"
#include "visual_language/vlm_utils.hpp"

namespace {

std::vector<int64_t> tensor_to_ids(const ov::Tensor& tensor) {
    return {tensor.data<const int64_t>(), tensor.data<const int64_t>() + tensor.get_size()};
}

TEST(VLMPromptIdsTest, ExtractsFullStatefulPromptInsteadOfPaddingCurrentTokens) {
    const std::vector<int64_t> cache_ids{10, 20, 30, 40};

    const ov::Tensor prompt_ids = ov::genai::vlm_utils::extract_prompt_ids(cache_ids, 0, cache_ids.size());

    EXPECT_EQ(tensor_to_ids(prompt_ids), cache_ids);
}

TEST(VLMPromptIdsTest, EmbeddingsPromptReachesLogitProcessor) {
    const std::vector<int64_t> cache_ids{10, 20, 7};
    const ov::Tensor prompt_ids = ov::genai::vlm_utils::extract_prompt_ids(cache_ids, 0, cache_ids.size());
    ov::Tensor input_embeds(ov::element::f32, {1, cache_ids.size(), 4});

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 1;
    config.repetition_penalty = 2.0f;
    ov::genai::SequenceGroup sequence_group(0,
                                            input_embeds,
                                            config,
                                            std::nullopt,
                                            std::nullopt,
                                            std::nullopt,
                                            std::nullopt,
                                            prompt_ids);

    ASSERT_EQ(sequence_group.get_prompt_ids(), cache_ids);
    std::vector<float> raw_logits(32, 2.0f);
    ov::genai::Logits logits(raw_logits.data(), raw_logits.size());
    ov::genai::LogitProcessor processor(config, sequence_group.get_prompt_ids());
    processor.apply(logits);

    EXPECT_FLOAT_EQ(raw_logits[7], 1.0f);
    EXPECT_FLOAT_EQ(raw_logits[8], 2.0f);
}

TEST(VLMPromptIdsTest, ExtractsOnePromptPerContinuousBatchingRequest) {
    const std::vector<int64_t> cache_after_first_prompt{101, 102, 103};
    const auto first_prompt = ov::genai::vlm_utils::extract_prompt_ids(cache_after_first_prompt, 0, 3);
    EXPECT_EQ(tensor_to_ids(first_prompt), (std::vector<int64_t>{101, 102, 103}));

    const std::vector<int64_t> cache_after_second_prompt{101, 102, 103, 201, -2, 202};
    const auto second_prompt = ov::genai::vlm_utils::extract_prompt_ids(cache_after_second_prompt, 3, 3);
    EXPECT_EQ(tensor_to_ids(second_prompt), (std::vector<int64_t>{201, -2, 202}));
}

TEST(VLMPromptIdsTest, RejectsTokenAndEmbeddingLengthMismatch) {
    const std::vector<int64_t> cache_ids{10, 20, 30};

    EXPECT_THROW(ov::genai::vlm_utils::extract_prompt_ids(cache_ids, 0, 4), ov::Exception);
}

}  // namespace
