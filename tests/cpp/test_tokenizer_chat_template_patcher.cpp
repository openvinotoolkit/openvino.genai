// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "gtest/gtest.h"
#include "tokenizer/tokenizer_impl.hpp"

namespace {

using ov::genai::ChatHistory;
using TokenizerImpl = ov::genai::Tokenizer::TokenizerImpl;

std::string replace_all(std::string text, const std::string& from, const std::string& to) {
    for (size_t pos = text.find(from); pos != std::string::npos; pos = text.find(from, pos + to.size())) {
        text.replace(pos, from.size(), to);
    }
    return text;
}

std::string patch_raw(const std::string& chat_template) {
    return replace_all(chat_template, "RAW", "{{ messages[0]['content'] }}");
}

std::string append_marker(const std::string& chat_template) {
    return chat_template + "|patched";
}

}  // namespace

TEST(TokenizerChatTemplatePatcher, SetChatTemplateIsPatchedAndOriginalIsKept) {
    TokenizerImpl impl;
    impl.set_chat_template_patcher(patch_raw);
    impl.set_chat_template("RAW");

    EXPECT_EQ(impl.get_original_chat_template(), "RAW");
    EXPECT_EQ(impl.get_chat_template(), "{{ messages[0]['content'] }}");
}

TEST(TokenizerChatTemplatePatcher, RegistrationPatchesExistingTemplateAndKeepsOriginal) {
    TokenizerImpl impl;
    impl.set_chat_template("RAW");
    ASSERT_EQ(impl.get_chat_template(), "RAW");

    impl.set_chat_template_patcher(patch_raw);

    EXPECT_EQ(impl.get_original_chat_template(), "RAW");
    EXPECT_EQ(impl.get_chat_template(), "{{ messages[0]['content'] }}");
}

TEST(TokenizerChatTemplatePatcher, RemapIsAppliedBeforePatch) {
    const auto& [known, fallback] = chat_template_fallback_map[0];
    TokenizerImpl impl;
    impl.set_chat_template(known);
    impl.set_chat_template_patcher(append_marker);

    EXPECT_EQ(impl.get_original_chat_template(), known);
    EXPECT_EQ(impl.get_chat_template(), fallback + "|patched");
}

TEST(TokenizerChatTemplatePatcher, ApplyChatTemplateUsesPatchedTemplate) {
    TokenizerImpl impl;
    impl.set_chat_template_patcher(patch_raw);
    impl.set_chat_template("RAW");
    const ChatHistory history({{{"role", "user"}, {"content", "hi"}}});

    EXPECT_EQ(impl.apply_chat_template(history, false, {}, std::nullopt, std::nullopt, {}), "hi");
    // A per-call template goes through the same transforms without being stored.
    impl.set_chat_template("unused {{ messages[0]['content'] }}");
    EXPECT_EQ(impl.apply_chat_template(history, false, "RAW", std::nullopt, std::nullopt, {}), "hi");
    EXPECT_EQ(impl.get_chat_template(), "unused {{ messages[0]['content'] }}");
}

TEST(TokenizerChatTemplatePatcher, ClearingModelPatcherKeepsCommonPreparation) {
    TokenizerImpl impl;
    impl.set_chat_template_patcher(append_marker);
    impl.set_chat_template_patcher({});
    impl.set_chat_template("{% if missing is undefined %}{{ messages[0]['content'] }}{% endif %}");
    const ChatHistory history({{{"role", "user"}, {"content", "hi"}}});

    EXPECT_EQ(impl.apply_chat_template(history, false, {}, std::nullopt, std::nullopt, {}), "hi");
}

TEST(TokenizerChatTemplatePatcher, CommonPreparationAndModelPatchWorkTogether) {
    const std::string chat_template =
        "{% if missing is undefined %}[{{ messages[0]['content'] | trim }}]{{ 'first'\n\"second\" }}{% endif %}";
    const ChatHistory history({{{"role", "user"}, {"content", "  hi  "}}});
    TokenizerImpl impl;
    impl.set_chat_template(chat_template);
    impl.set_chat_template_patcher([](const std::string& text) {
        return replace_all(text, " | trim", "");
    });

    EXPECT_EQ(impl.get_original_chat_template(), chat_template);
    EXPECT_EQ(impl.apply_chat_template(history, false, {}, std::nullopt, std::nullopt, {}), "[  hi  ]firstsecond");

    const std::string stored_template = impl.get_chat_template();
    EXPECT_EQ(impl.apply_chat_template(history, false, "override:" + chat_template, std::nullopt, std::nullopt, {}),
              "override:[  hi  ]firstsecond");
    EXPECT_EQ(impl.get_chat_template(), stored_template);
    EXPECT_EQ(impl.get_original_chat_template(), chat_template);
}
