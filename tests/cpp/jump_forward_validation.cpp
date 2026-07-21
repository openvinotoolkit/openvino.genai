// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <utility>

#include "sampling/structured_output/jump_forward_validation.hpp"

namespace {

using ov::genai::GenerationConfig;
using ov::genai::StructuredOutputConfig;
using ov::genai::jump_forward_validation::validate_continuous_config;

GenerationConfig make_jump_forward_config(const std::string& backend = "xgrammar") {
    GenerationConfig config;
    StructuredOutputConfig structured_output;
    structured_output.backend = backend;
    structured_output.enable_jump_forward = true;
    config.structured_output_config = std::move(structured_output);
    return config;
}

void expect_rejected(GenerationConfig config, const std::function<void(GenerationConfig&)>& make_unsupported) {
    make_unsupported(config);
    EXPECT_THROW(validate_continuous_config(config, true, true), ov::Exception);
}

TEST(JumpForwardValidation, AcceptsSupportedContinuousBatchingConfig) {
    EXPECT_NO_THROW(validate_continuous_config(GenerationConfig{}, true, true));
}

TEST(JumpForwardValidation, RejectsNonTokenInput) {
    EXPECT_THROW(validate_continuous_config(GenerationConfig{}, false, true), ov::Exception);
}

TEST(JumpForwardValidation, RejectsLogprobs) {
    expect_rejected({}, [](GenerationConfig& config) {
        config.logprobs = 1;
    });
}

TEST(JumpForwardValidation, RejectsMultipleReturnSequences) {
    expect_rejected({}, [](GenerationConfig& config) {
        config.num_return_sequences = 2;
    });
}

TEST(JumpForwardValidation, RejectsBeamSearch) {
    expect_rejected({}, [](GenerationConfig& config) {
        config.num_beams = 2;
    });
}

TEST(JumpForwardValidation, RejectsPromptLookup) {
    expect_rejected({}, [](GenerationConfig& config) {
        config.num_assistant_tokens = 2;
        config.max_ngram_size = 3;
    });
}

TEST(JumpForwardValidation, RejectsAssistingGeneration) {
    expect_rejected({}, [](GenerationConfig& config) {
        config.assistant_confidence_threshold = 0.5f;
    });
}

TEST(JumpForwardValidation, RejectsSpeculativePipeline) {
    EXPECT_THROW(validate_continuous_config(GenerationConfig{}, true, true, true), ov::Exception);
}

TEST(JumpForwardValidation, RejectsTreeSearch) {
    expect_rejected({}, [](GenerationConfig& config) {
        config.tree_depth = 2;
    });
}

TEST(JumpForwardValidation, RejectsStopStrings) {
    expect_rejected({}, [](GenerationConfig& config) {
        config.stop_strings.insert("stop");
    });
}

TEST(JumpForwardValidation, RejectsBackendWithoutCapability) {
    EXPECT_THROW(validate_continuous_config(GenerationConfig{}, true, false), ov::Exception);
}

TEST(JumpForwardValidation, DefaultOffDoesNotRequireBackendCapability) {
    EXPECT_NO_THROW(ov::genai::jump_forward_validation::validate_continuous(
        GenerationConfig{},
        true));
}

TEST(JumpForwardValidation, EnabledConfigUsesBackendCapability) {
    EXPECT_NO_THROW(ov::genai::jump_forward_validation::validate_continuous(
        make_jump_forward_config(),
        true));
    EXPECT_THROW(ov::genai::jump_forward_validation::validate_continuous(
                     make_jump_forward_config("backend_without_jump_forward"),
                     true),
                 ov::Exception);
}

TEST(JumpForwardValidation, RejectsUnsupportedPipeline) {
    EXPECT_THROW(ov::genai::jump_forward_validation::validate_unsupported_pipeline(
                     make_jump_forward_config(),
                     "stateful"),
                 ov::Exception);
}

}  // namespace
