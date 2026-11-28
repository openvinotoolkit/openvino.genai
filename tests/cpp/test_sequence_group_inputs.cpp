// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "sequence_group.hpp"

namespace {

using ExtraInputs = std::unordered_map<std::string, ov::Tensor>;

constexpr size_t kPromptLen = 3;

ov::Tensor make_tensor(ov::element::Type type, ov::Shape shape) {
    ov::Tensor tensor(type, shape);
    if (tensor.get_byte_size() > 0) {
        std::memset(tensor.data(), 0, tensor.get_byte_size());
    }
    return tensor;
}

ov::Tensor make_input_embeds() {
    return make_tensor(ov::element::f32, {1, kPromptLen, 4});
}

ov::Tensor make_prompt_ids() {
    return make_tensor(ov::element::i64, {1, kPromptLen});
}

ExtraInputs make_extra_inputs() {
    return {{"deepstack_visual_embeds", make_tensor(ov::element::f32, {2, kPromptLen, 4})},
            {"visual_pos_masks", make_tensor(ov::element::boolean, {1, kPromptLen})},
            {"per_layer_inputs", make_tensor(ov::element::f32, {1, kPromptLen, 2, 4})},
            {"token_type_ids", make_tensor(ov::element::i64, {1, kPromptLen})},
            {"ngram_embeddings", make_tensor(ov::element::f32, {1, kPromptLen, 2, 4})},
            {"ngram_history", make_tensor(ov::element::i64, {1, 2})}};
}

std::string error_of(const std::function<void()>& call) {
    try {
        call();
    } catch (const ov::Exception& error) {
        return error.what();
    }
    return {};
}

std::shared_ptr<ov::genai::SequenceGroup> make_group(const ov::Tensor& input_ids,
                                                     const std::optional<ov::Tensor>& prompt_ids,
                                                     const std::optional<ExtraInputs>& lm_extra_inputs) {
    return std::make_shared<ov::genai::SequenceGroup>(0,
                                                      input_ids,
                                                      ov::genai::GenerationConfig{},
                                                      lm_extra_inputs,
                                                      std::nullopt,
                                                      std::nullopt,
                                                      prompt_ids);
}

}  // namespace

TEST(SequenceGroupInputs, WellFormedInputsPass) {
    const auto extra_inputs = make_extra_inputs();
    EXPECT_NO_THROW(ov::genai::SequenceGroup::validate_inputs(make_input_embeds(), make_prompt_ids(), extra_inputs));
    EXPECT_NO_THROW(make_group(make_input_embeds(), make_prompt_ids(), extra_inputs));
}

TEST(SequenceGroupInputs, ValidationRejectsWhatTheConstructorRejects) {
    struct Case {
        std::string name;
        ov::Tensor input_ids;
        std::optional<ov::Tensor> prompt_ids;
        std::optional<ExtraInputs> lm_extra_inputs;
        std::string error;
    };
    const auto with_extra_input = [](const std::string& input_name, const ov::Tensor& tensor) {
        auto extra_inputs = make_extra_inputs();
        extra_inputs[input_name] = tensor;
        return extra_inputs;
    };
    const std::vector<Case> cases = {
        {"empty prompt", make_tensor(ov::element::i64, {1, 0}), std::nullopt, std::nullopt,
         "Prompt length cannot be 0"},
        {"i32 token IDs", make_tensor(ov::element::i32, {1, kPromptLen}), std::nullopt, std::nullopt,
         "input_ids must hold i64 token IDs or f32 input embeddings"},
        {"rank-2 input embeddings", make_tensor(ov::element::f32, {1, kPromptLen}), std::nullopt, std::nullopt,
         "input_ids must hold i64 token IDs or f32 input embeddings"},
        {"i32 prompt IDs", make_input_embeds(), make_tensor(ov::element::i32, {1, kPromptLen}), std::nullopt,
         "prompt_ids must have element type i64"},
        {"u8 visual_pos_masks", make_input_embeds(), make_prompt_ids(),
         with_extra_input("visual_pos_masks", make_tensor(ov::element::u8, {1, kPromptLen})),
         "visual_pos_masks must have element type boolean"},
        {"f16 per_layer_inputs", make_input_embeds(), make_prompt_ids(),
         with_extra_input("per_layer_inputs", make_tensor(ov::element::f16, {1, kPromptLen, 2, 4})),
         "per_layer_inputs must have element type f32"},
        {"rank-3 per_layer_inputs", make_input_embeds(), make_prompt_ids(),
         with_extra_input("per_layer_inputs", make_tensor(ov::element::f32, {1, kPromptLen, 8})),
         "per_layer_inputs must have shape"},
        {"i32 token_type_ids", make_input_embeds(), make_prompt_ids(),
         with_extra_input("token_type_ids", make_tensor(ov::element::i32, {1, kPromptLen})),
         "token_type_ids must have element type i64"},
        {"ngram_embeddings of another length", make_input_embeds(), make_prompt_ids(),
         with_extra_input("ngram_embeddings", make_tensor(ov::element::f32, {1, kPromptLen + 1, 2, 4})),
         "ngram_embeddings must have shape"},
        {"unset deepstack_visual_embeds", make_input_embeds(), make_prompt_ids(),
         with_extra_input("deepstack_visual_embeds", ov::Tensor()),
         "Extra input deepstack_visual_embeds is not set"},
        {"unknown extra input", make_input_embeds(), make_prompt_ids(),
         with_extra_input("unknown", make_tensor(ov::element::f32, {1})),
         "Unsupported extra input for LLM: unknown"},
    };
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        const auto validation_error = error_of([&] {
            ov::genai::SequenceGroup::validate_inputs(test_case.input_ids,
                                                      test_case.prompt_ids,
                                                      test_case.lm_extra_inputs);
        });
        EXPECT_NE(validation_error.find(test_case.error), std::string::npos) << validation_error;
        const auto construction_error = error_of([&] {
            make_group(test_case.input_ids, test_case.prompt_ids, test_case.lm_extra_inputs);
        });
        EXPECT_NE(construction_error.find(test_case.error), std::string::npos) << construction_error;
    }
}

TEST(SequenceGroupInputs, TokenIdsIgnorePromptIdsAndExtraInputs) {
    const auto input_ids = make_tensor(ov::element::i64, {1, kPromptLen});
    const auto prompt_ids = make_tensor(ov::element::i32, {1, kPromptLen});
    const ExtraInputs extra_inputs = {{"unknown", make_tensor(ov::element::f32, {1})}};
    EXPECT_NO_THROW(ov::genai::SequenceGroup::validate_inputs(input_ids, prompt_ids, extra_inputs));
    EXPECT_NO_THROW(make_group(input_ids, prompt_ids, extra_inputs));
}
