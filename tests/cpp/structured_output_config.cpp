// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "openvino/genai/generation_config.hpp"

using ov::genai::StructuredOutputConfig;

TEST(StructuredOutputConfigTest, JSONSchemaSerializesXGrammarOptions) {
    StructuredOutputConfig::JSONSchema schema("{\"type\":\"object\"}", "qwen_xml", true);

    EXPECT_EQ(schema.to_json(),
              "{\"type\": \"json_schema\", \"json_schema\": {\"type\":\"object\"}, "
              "\"style\": \"qwen_xml\", \"any_order\": true}");
}

TEST(StructuredOutputConfigTest, AnyTextAndTriggeredTagsSerializeExcludes) {
    StructuredOutputConfig::AnyText any_text({"</think>", "</answer>"});
    EXPECT_EQ(any_text.to_json(), "{\"type\": \"any_text\", \"excludes\": [\"</think>\", \"</answer>\"]}");

    StructuredOutputConfig::TriggeredTags triggered_tags(
        {"<tool>"},
        {StructuredOutputConfig::Tag("<tool>", StructuredOutputConfig::ConstString("ok"), "</tool>")},
        true,
        false,
        {"<blocked>"});

    EXPECT_EQ(triggered_tags.to_json(),
              "{\"type\": \"triggered_tags\", \"triggers\": [\"<tool>\"], \"tags\": ["
              "{\"type\": \"tag\", \"begin\": \"<tool>\", \"content\": "
              "{\"type\": \"const_string\", \"value\": \"ok\"}, \"end\": \"</tool>\"}"
              "], \"at_least_one\": true, \"stop_after_first\": false, \"excludes\": [\"<blocked>\"]}");
}

TEST(StructuredOutputConfigTest, RepetitionHelpersSerializeXGrammarTypes) {
    StructuredOutputConfig::Optional optional(StructuredOutputConfig::ConstString("x"));
    StructuredOutputConfig::Plus plus(StructuredOutputConfig::ConstString("x"));
    StructuredOutputConfig::Star star(StructuredOutputConfig::ConstString("x"));
    StructuredOutputConfig::Repeat repeat(StructuredOutputConfig::ConstString("x"), 2, 4);

    EXPECT_EQ(optional.to_json(),
              "{\"type\": \"optional\", \"content\": {\"type\": \"const_string\", \"value\": \"x\"}}");
    EXPECT_EQ(plus.to_json(),
              "{\"type\": \"plus\", \"content\": {\"type\": \"const_string\", \"value\": \"x\"}}");
    EXPECT_EQ(star.to_json(),
              "{\"type\": \"star\", \"content\": {\"type\": \"const_string\", \"value\": \"x\"}}");
    EXPECT_EQ(repeat.to_json(),
              "{\"type\": \"repeat\", \"min\": 2, \"max\": 4, \"content\": "
              "{\"type\": \"const_string\", \"value\": \"x\"}}");
}
