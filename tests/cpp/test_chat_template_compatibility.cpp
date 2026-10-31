// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "tokenizer/tokenizer_impl.hpp"
#include "utils.hpp"

namespace {

ov::genai::ChatHistory chat_history() {
    return {{{"role", "user"}, {"content", "hello"}}};
}

std::unique_ptr<ov::genai::Tokenizer::TokenizerImpl> template_tokenizer(const std::string& chat_template) {
    auto tokenizer = std::make_unique<ov::genai::Tokenizer::TokenizerImpl>();
    tokenizer->m_bos_token = "<bos>";
    tokenizer->m_eos_token = "<eos>";
    tokenizer->m_pad_token = "<pad>";
    tokenizer->set_chat_template(chat_template);
    return tokenizer;
}

}  // namespace

TEST(TokenizerChatTemplate, UnsupportedUndefinedTestIsRetried) {
    auto tokenizer =
        template_tokenizer("{% if missing is undefined %}{{ bos_token }}{{ messages[0].content }}{% endif %}");
    EXPECT_EQ(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt), "<bos>hello");
    EXPECT_EQ(tokenizer->get_chat_template(),
              "{% if missing is undefined %}{{ bos_token }}{{ messages[0].content }}{% endif %}");
}

TEST(TokenizerChatTemplate, AdjacentMultilineLiteralsAreRetried) {
    auto tokenizer = template_tokenizer("{{ \"first\"\n    \"second\" }}{{ messages[0].content }}");
    EXPECT_EQ(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt),
              "firstsecondhello");
}

TEST(TokenizerChatTemplate, FallbackUsesExplicitContextAndTools) {
    auto tokenizer = template_tokenizer("{{ bos_token }}{{ eos_token }}{{ pad_token }}");
    auto history = chat_history();
    history.set_extra_context({{"bos_token", "history"}, {"label", "old"}});
    history.set_tools(ov::genai::JsonContainer::from_json_string(R"([{"name":"history"}])"));
    const auto tools = ov::genai::JsonContainer::from_json_string(R"([{"name":"explicit"}])");
    const ov::genai::JsonContainer context{{"bos_token", "override"}, {"label", "new"}};
    EXPECT_EQ(tokenizer->apply_chat_template(
                  history,
                  true,
                  "{% if missing is undefined %}{{ bos_token }}{{ eos_token }}{{ pad_token }}"
                  "{{ label }}{{ tools[0].name }}{{ add_generation_prompt }}{{ \"one\"\n\"two\" }}{% endif %}",
                  tools,
                  context),
              "override<eos><pad>newexplicitTrueonetwo");
}

TEST(TokenizerChatTemplate, SuccessfulRenderingPreservesLiteralText) {
    auto tokenizer = template_tokenizer("{{ \"missing is undefined\" }}|{{ messages[0].content }}");
    EXPECT_EQ(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt),
              "missing is undefined|hello");
}

TEST(TokenizerChatTemplate, FallbackPreservesUndefinedTextInsideAndOutsideExpressions) {
    auto tokenizer = template_tokenizer("x is undefined|{% if missing is undefined %}{{ \"x is undefined\" }}"
                                        "{{ \"one is undefined\"\n\"two\" }}{% endif %}");
    EXPECT_EQ(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt),
              "x is undefined|x is undefinedone is undefinedtwo");
}

TEST(TokenizerChatTemplate, FallbackPreservesComments) {
    auto tokenizer = template_tokenizer("{# {{ x is undefined }} #}{% if missing is undefined %}ok{% endif %}");
    EXPECT_EQ(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt), "ok");
}

TEST(TokenizerChatTemplate, UndefinedRewritePreservesRawBlocks) {
    EXPECT_EQ(ov::genai::utils::replace_is_undefined_tests(
                  "{% raw %}{{ x is undefined }}{% endraw %}{% if missing is undefined %}ok{% endif %}"),
              "{% raw %}{{ x is undefined }}{% endraw %}{% if missing is not defined %}ok{% endif %}");
}

TEST(TokenizerChatTemplate, FallbackPreservesMultilineQuotesOutsideExpressions) {
    auto tokenizer = template_tokenizer("\"line one\"\n\"line two\"|{% if missing is undefined %}ok{% endif %}");
    EXPECT_EQ(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt),
              "\"line one\"\n\"line two\"|ok");
}

TEST(TokenizerChatTemplate, LiteralJoinPreservesRawBlocks) {
    EXPECT_EQ(ov::genai::utils::join_multiline_string_literals(
                  "{% raw %}\"line one\"\n\"line two\"{% endraw %}{{ \"joined\"\n\"literal\" }}"),
              "{% raw %}\"line one\"\n\"line two\"{% endraw %}{{ \"joinedliteral\" }}");
}

TEST(TokenizerChatTemplate, UndefinedRewriteMatchesOnlyCompleteTestName) {
    EXPECT_EQ(ov::genai::utils::replace_is_undefined_tests(
                  "{{ x is undefined_value }}{{ (x is undefined) }}{{ message.content is undefined }}"),
              "{{ x is undefined_value }}{{ (x is not defined) }}{{ message.content is not defined }}");
}

TEST(TokenizerChatTemplate, UndefinedFallbackPreservesIndexedOperands) {
    for (const std::string operand : {"messages[0].missing", "messages[0][\"missing\"]", "messages[0]['missing']"}) {
        auto tokenizer = template_tokenizer("{% if " + operand + " is undefined %}missing{% endif %}");
        EXPECT_EQ(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt), "missing");
    }
}

TEST(TokenizerChatTemplate, UndefinedFallbackDoesNotTreatPresentValuesAsMissing) {
    auto tokenizer = template_tokenizer(
        "{% if messages[0].content is undefined %}missing{% else %}{{ messages[0].content }}{% endif %}");
    EXPECT_EQ(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt), "hello");
}

TEST(TokenizerChatTemplate, InvalidTemplateStillReportsAnError) {
    auto tokenizer = template_tokenizer("{% if missing is undefined %}{{ unknown_function() }}{% endif %}");
    EXPECT_THROW(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt), ov::Exception);
}

TEST(TokenizerChatTemplate, UndefinedFallbackPreservesNestedDictionaries) {
    auto tokenizer = template_tokenizer("{{ {'outer': {}} is undefined }}");
    EXPECT_EQ(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt), "False");
}

TEST(TokenizerChatTemplate, FallbackReportsBothRenderingErrors) {
    auto tokenizer = template_tokenizer("{% if missing is undefined %}{{ unknown_function() }}{% endif %}");
    try {
        tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt);
        FAIL() << "Expected rendering to fail";
    } catch (const ov::Exception& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("Minja's error:"), std::string::npos);
        EXPECT_NE(message.find("Minja's retry error:"), std::string::npos);
        EXPECT_NE(message.find("unknown_function"), std::string::npos);
    }
}
