// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>

#include "tokenizer/tokenizer_impl.hpp"
#include "utils.hpp"

namespace {

std::unique_ptr<ov::genai::Tokenizer::TokenizerImpl> template_tokenizer(const std::string& chat_template) {
    auto tokenizer = std::make_unique<ov::genai::Tokenizer::TokenizerImpl>();
    tokenizer->m_bos_token = "<bos>";
    tokenizer->m_eos_token = "<eos>";
    tokenizer->m_pad_token = "<pad>";
    tokenizer->set_chat_template(chat_template);
    return tokenizer;
}

std::string render(ov::genai::Tokenizer::TokenizerImpl& tokenizer, const std::string& chat_template = "",
                   const std::optional<ov::genai::JsonContainer>& context = std::nullopt) {
    return tokenizer.apply_chat_template({{{"role", "user"}, {"content", "hello"}}}, false,
                                         chat_template, std::nullopt, context, {});
}

}  // namespace

TEST(TokenizerChatTemplate, PreparationPreservesOriginalTemplate) {
    const std::string chat_template = "{% if missing is undefined %}{{ 'first'\n'second' }}{% endif %}";
    auto tokenizer = template_tokenizer(chat_template);
    EXPECT_EQ(render(*tokenizer), "firstsecond");
    EXPECT_EQ(tokenizer->get_original_chat_template(), chat_template);
    EXPECT_EQ(tokenizer->get_chat_template(),
              "{% if missing is not defined %}{{ ('first' ~ 'second') }}{% endif %}");
    EXPECT_EQ(render(*tokenizer), "firstsecond");
}

TEST(TokenizerChatTemplate, ExplicitTemplateUsesSamePreparationAndContext) {
    auto tokenizer = template_tokenizer("{{ messages[0].content }}");
    const auto tools = ov::genai::JsonContainer::from_json_string(R"([{"name":"explicit"}])");
    const ov::genai::JsonContainer context{{"bos_token", "override"}, {"label", "new"}};
    auto history = ov::genai::ChatHistory{{{"role", "user"}, {"content", "hello"}}};
    history.set_extra_context({{"bos_token", "history"}, {"label", "old"}});
    history.set_tools(ov::genai::JsonContainer::from_json_string(R"([{"name":"history"}])"));
    EXPECT_EQ(tokenizer->apply_chat_template(history, true,
                  "{% if missing is undefined %}{{ bos_token }}{{ eos_token }}{{ pad_token }}"
                  "{{ label }}{{ tools[0].name }}{{ add_generation_prompt }}{{ 'one'\n\"two\" }}{% endif %}",
                  tools, context, {}), "override<eos><pad>newexplicitTrueonetwo");
    EXPECT_EQ(tokenizer->get_original_chat_template(), "{{ messages[0].content }}");
    EXPECT_EQ(render(*tokenizer), "hello");
}

TEST(TokenizerChatTemplate, PreparationPreservesLiteralTextCommentsAndRawBlocks) {
    const std::string chat_template =
        "{% raw %}{{ x is undefined }}{{ 'a'\n'b' }}{% endraw %}{# {{ x is undefined }} #}"
        "\"a\"\n\"b\"|{{ 'x is undefined' }}{{ x is undefined_value }}";
    EXPECT_EQ(ov::genai::utils::patch_chat_template(chat_template), chat_template);
    auto tokenizer = template_tokenizer("\"a\"\n\"b\"|{{ 'x is undefined' }}");
    EXPECT_EQ(render(*tokenizer), "\"a\"\n\"b\"|x is undefined");
}

TEST(TokenizerChatTemplate, UndefinedTestsPreserveIndexedAndNestedOperands) {
    for (const std::string operand : {"missing", "messages[0].missing", "messages[0][\"missing\"]",
                                     "messages[0]['missing']", "({'outer': {}}).missing"}) {
        auto tokenizer = template_tokenizer("{% if " + operand + " is undefined %}missing{% endif %}");
        EXPECT_EQ(render(*tokenizer), "missing") << operand;
    }
    auto tokenizer = template_tokenizer("{% if messages[0].content is undefined %}missing{% else %}present{% endif %}");
    EXPECT_EQ(render(*tokenizer), "present");
}

TEST(TokenizerChatTemplate, UndefinedTestsRetainMinjaDefinedSemantics) {
    auto tokenizer = template_tokenizer("{% if x is undefined %}missing{% else %}present{% endif %}");
    EXPECT_EQ(render(*tokenizer), "missing");
    EXPECT_EQ(render(*tokenizer, "", ov::genai::JsonContainer::from_json_string(R"({"x":null})")), "missing");
    EXPECT_EQ(render(*tokenizer, "", ov::genai::JsonContainer{{"x", false}}), "present");
}

TEST(TokenizerChatTemplate, AdjacentLiteralsPreserveQuotesEscapesAndFilterPrecedence) {
    for (const std::string expression : {"'first' 'second'", "\"first\"\n'second'", "'first'\n\"second\"",
                                         "\"first\"\n\n\"second\"", "'first' \r\n\t\r\n 'second'",
                                         "'fi' 'rst'\n'second'"}) {
        auto tokenizer = template_tokenizer("{{ " + expression + " | upper }}");
        EXPECT_EQ(render(*tokenizer), "FIRSTSECOND") << expression;
    }
    auto tokenizer = template_tokenizer(R"({{ "a\"b\\"
 'c\'d\\'
 "e" }})");
    EXPECT_EQ(render(*tokenizer), "a\"b\\c'd\\e");
}

TEST(TokenizerChatTemplate, InvalidTemplatesAndEmptyOutputKeepExistingErrors) {
    auto tokenizer = template_tokenizer("{% if missing is undefined %}{{ unknown_function() }}{% endif %}");
    EXPECT_THROW(render(*tokenizer), std::exception);
    tokenizer->set_chat_template("{% if messages[0].content is undefined %}missing{% endif %}");
    try {
        render(*tokenizer);
        FAIL() << "Expected empty output to be rejected";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("Applied chat template resulted in an empty string"), std::string::npos);
    }
}

TEST(TokenizerChatTemplate, LongQuotedLiteralsAreScannedWithoutRegexRecursion) {
    const std::string text(64 * 1024, 'x');
    for (const char quote : {'\'', '"'}) {
        const std::string literal = std::string(1, quote) + text + quote;
        EXPECT_EQ(ov::genai::utils::patch_chat_template("{% if missing is undefined %}{{ " + literal + " }}{% endif %}"),
                  "{% if missing is not defined %}{{ " + literal + " }}{% endif %}");
        EXPECT_EQ(ov::genai::utils::patch_chat_template("{{ " + literal + "\n" + literal + " }}"),
                  "{{ (" + literal + " ~ " + literal + ") }}");
    }
}
