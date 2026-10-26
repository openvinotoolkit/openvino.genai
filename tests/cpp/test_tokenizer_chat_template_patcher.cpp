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
    impl.set_chat_template_patcher(append_marker);

    impl.set_chat_template(known);
    EXPECT_EQ(impl.get_original_chat_template(), known);
    EXPECT_EQ(impl.get_chat_template(), fallback + "|patched");

    // Same order when the patcher is registered after the template was set.
    TokenizerImpl late_impl;
    late_impl.set_chat_template(known);
    late_impl.set_chat_template_patcher(append_marker);
    EXPECT_EQ(late_impl.get_chat_template(), fallback + "|patched");
}

TEST(TokenizerChatTemplatePatcher, ApplyChatTemplateUsesPatchedTemplate) {
    TokenizerImpl impl;
    impl.set_chat_template_patcher(patch_raw);
    impl.set_chat_template("RAW");
    const ChatHistory history({{{"role", "user"}, {"content", "hi"}}});

    EXPECT_EQ(impl.apply_chat_template(history, false, {}, std::nullopt, std::nullopt), "hi");
    // A per-call template goes through the same transforms without being stored.
    impl.set_chat_template("unused {{ messages[0]['content'] }}");
    EXPECT_EQ(impl.apply_chat_template(history, false, "RAW", std::nullopt, std::nullopt), "hi");
    EXPECT_EQ(impl.get_chat_template(), "unused {{ messages[0]['content'] }}");
}

TEST(TokenizerChatTemplatePatcher, NoPatcherKeepsTemplateUnchanged) {
    TokenizerImpl impl;
    impl.set_chat_template("RAW");

    EXPECT_EQ(impl.get_original_chat_template(), "RAW");
    EXPECT_EQ(impl.get_chat_template(), "RAW");
}
