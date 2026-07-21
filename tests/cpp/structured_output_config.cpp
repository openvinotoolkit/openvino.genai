// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <stdexcept>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/json_container.hpp"

using ov::genai::JsonContainer;
using ov::genai::StructuredOutputConfig;

namespace {

JsonContainer weather_tools() {
    return JsonContainer::from_json_string(R"([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "parameters": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}}
                },
                "strict": false
            }
        }
    ])");
}

ov::genai::ModelStructuralTagOptions options(bool reasoning = true,
                                             bool any_order = false,
                                             bool exclude_special_tokens = true) {
    return {reasoning, any_order, exclude_special_tokens};
}

std::string structural_tag_json(const StructuredOutputConfig& config) {
    EXPECT_TRUE(config.structural_tags_config.has_value());
    const auto* tag = std::get_if<StructuredOutputConfig::StructuralTag>(&*config.structural_tags_config);
    EXPECT_NE(tag, nullptr);
    return std::visit([](const auto& value) {
        return StructuredOutputConfig::structural_tag_to_json(value);
    }, *tag);
}

}  // namespace

TEST(StructuredOutputConfigTest, JumpForwardDefaultsToFalseAndCanBeEnabled) {
    StructuredOutputConfig default_config;
    EXPECT_FALSE(default_config.enable_jump_forward);

    StructuredOutputConfig enabled_config({{"regex", std::string("a")}, {"enable_jump_forward", true}});
    EXPECT_TRUE(enabled_config.enable_jump_forward);
}

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

TEST(StructuredOutputConfigTest, FromModelFormatCoversRegisteredFormats) {
    const std::vector<std::string> formats = {
        "llama", "kimi", "deepseek_r1", "deepseek_v3_1", "qwen_3_5", "qwen_3_coder",
        "qwen_3", "harmony", "deepseek_v3_2", "minimax", "glm_4_7", "deepseek_v4"};

    for (const auto& format : formats) {
        const auto config = StructuredOutputConfig::from_model_format(
            format, JsonContainer::array(), JsonContainer("auto"), options());
        EXPECT_TRUE(config.structural_tags_config.has_value()) << format;
        EXPECT_FALSE(structural_tag_json(config).empty()) << format;
    }

    EXPECT_THROW(StructuredOutputConfig::from_model_format("gemma_4", JsonContainer::array(), JsonContainer("auto"), options()),
                 ov::Exception);
}

TEST(StructuredOutputConfigTest, FromModelFormatUnknownFormatListsSupportedFormats) {
    try {
        (void)StructuredOutputConfig::from_model_format("unknown", JsonContainer::array(), JsonContainer("auto"), options());
        FAIL() << "Expected exception";
    } catch (const ov::Exception& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("Unknown format type: unknown"), std::string::npos);
        EXPECT_NE(message.find("llama"), std::string::npos);
        EXPECT_NE(message.find("deepseek_v4"), std::string::npos);
        EXPECT_EQ(message.find("gemma_4"), std::string::npos);
    }
}

TEST(StructuredOutputConfigTest, FromModelFormatValidatesToolsUpFront) {
    EXPECT_THROW(StructuredOutputConfig::from_model_format("llama", JsonContainer::object(), JsonContainer("auto"), options()),
                 ov::Exception);

    const auto missing_function_object = JsonContainer::from_json_string(R"([{"type": "function"}])");
    EXPECT_THROW(StructuredOutputConfig::from_model_format("llama", missing_function_object, JsonContainer("auto"), options()),
                 ov::Exception);

    const auto invalid_function_tool = JsonContainer::from_json_string(R"([
        {"type": "function", "function": {"parameters": {"type": "object"}}}
    ])");
    EXPECT_THROW(StructuredOutputConfig::from_model_format("llama", invalid_function_tool, JsonContainer("auto"), options()),
                 ov::Exception);

    const auto invalid_parameters = JsonContainer::from_json_string(R"([
        {"type": "function", "function": {"name": "bad", "parameters": true}}
    ])");
    EXPECT_THROW(StructuredOutputConfig::from_model_format("llama", invalid_parameters, JsonContainer("auto"), options()),
                 ov::Exception);
}

TEST(StructuredOutputConfigTest, FromModelFormatNormalizesToolChoice) {
    EXPECT_THROW(StructuredOutputConfig::from_model_format(
                     "llama", JsonContainer::array(), JsonContainer("required"), options()),
                 ov::Exception);

    const auto forced = StructuredOutputConfig::from_model_format(
        "llama",
        weather_tools(),
        JsonContainer::from_json_string(R"({"type": "function", "function": {"name": "get_weather"}})"),
        options(false, true));
    const std::string forced_json = structural_tag_json(forced);
    EXPECT_NE(forced_json.find("\"begin\": \"{\\\"name\\\": \\\"get_weather"), std::string::npos);
    EXPECT_NE(forced_json.find("\"any_order\": true"), std::string::npos);
    EXPECT_EQ(forced_json.find("get_time"), std::string::npos);

    const auto none = StructuredOutputConfig::from_model_format(
        "llama", weather_tools(), JsonContainer("none"), options());
    EXPECT_EQ(structural_tag_json(none).find("get_weather"), std::string::npos);

    const auto allowed_required = StructuredOutputConfig::from_model_format(
        "llama",
        weather_tools(),
        JsonContainer::from_json_string(R"({
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "required",
                "tools": [{"type": "function", "function": {"name": "get_time"}}]
            }
        })"),
        options(false));
    const std::string allowed_json = structural_tag_json(allowed_required);
    EXPECT_NE(allowed_json.find("get_time"), std::string::npos);
    EXPECT_NE(allowed_json.find("\"json_schema\": true"), std::string::npos);
    EXPECT_EQ(allowed_json.find("get_weather"), std::string::npos);
}

TEST(StructuredOutputConfigTest, FromModelFormatSupportsHarmonyBuiltinTools) {
    const auto tools = JsonContainer::from_json_string(R"([
        {
            "type": "web_search_preview",
            "name": "browser.search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    ])");

    const auto config = StructuredOutputConfig::from_model_format(
        "harmony",
        tools,
        JsonContainer::from_json_string(R"({"type": "web_search_preview"})"),
        options(true));
    const std::string json = structural_tag_json(config);
    EXPECT_NE(json.find("browser.search"), std::string::npos);
    EXPECT_NE(json.find("\"end\": \"<|call|>\""), std::string::npos);

    EXPECT_THROW(StructuredOutputConfig::from_model_format(
                     "harmony", tools, JsonContainer::from_json_string(R"({"type": "file_search"})"), options()),
                 ov::Exception);
}
