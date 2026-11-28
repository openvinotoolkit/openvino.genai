// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstring>

#include "gtest/gtest.h"
#include "openvino/genai/gguf.hpp"

namespace {
ov::genai::GGUFTokenizerParameters sentencepiece_config() {
    std::vector<std::string> vocab{"<unk>",
                                   "<bos>",
                                   "<eos>",
                                   "<pad>",
                                   "<start_of_turn>",
                                   "<end_of_turn>",
                                   "<start_of_image>",
                                   "<end_of_image>",
                                   "▁",
                                   "a",
                                   "b"};
    std::vector<int32_t> types{2, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1};
    for (int i = 0; i < 256; ++i) {
        char byte[7];
        std::snprintf(byte, sizeof(byte), "<0x%02X>", i);
        vocab.emplace_back(byte);
        types.push_back(6);
    }
    vocab.emplace_back("<é>");
    types.push_back(4);
    ov::Tensor type_tensor(ov::element::i32, {types.size()});
    std::memcpy(type_tensor.data(), types.data(), type_tensor.get_byte_size());
    ov::Tensor scores(ov::element::f32, {types.size()});
    std::fill_n(scores.data<float>(), scores.get_size(), -1.f);
    const auto id = [](uint32_t value) {
        ov::Tensor tensor(ov::element::u32, {});
        tensor.data<uint32_t>()[0] = value;
        return tensor;
    };
    ov::Tensor no_prefix(ov::element::boolean, {}), add_bos(ov::element::boolean, {});
    no_prefix.data<bool>()[0] = false;
    add_bos.data<bool>()[0] = true;
    return ov::genai::GGUFTokenizerParameters(
        {{"model", std::string("llama")},
         {"tokens", vocab},
         {"token_type", type_tensor},
         {"scores", scores},
         {"unknown_token_id", id(0)},
         {"bos_token_id", id(1)},
         {"eos_token_id", id(2)},
         {"padding_token_id", id(3)},
         {"add_space_prefix", no_prefix},
         {"add_bos_token", add_bos},
         {"chat_template", std::string("{{ bos_token }}<start_of_turn>{{ messages[0]['content'] }}<end_of_turn>")}});
}

constexpr char qwen35_chat_template[] =
    "{%- if enable_thinking is undefined or enable_thinking is true %}think{%- else %}direct{%- endif %}"
    ":{{ messages[0]['content'] }}";

ov::genai::GGUFTokenizerParameters byte_level_bpe_config(const std::string& pre) {
    std::vector<std::string> vocab{"<|endoftext|>", "a", "b"};
    std::vector<int32_t> types{3, 1, 1};
    ov::Tensor type_tensor(ov::element::i32, {types.size()});
    std::memcpy(type_tensor.data(), types.data(), type_tensor.get_byte_size());
    ov::Tensor eos(ov::element::u32, {});
    eos.data<uint32_t>()[0] = 0;
    return ov::genai::GGUFTokenizerParameters({{"model", std::string("gpt2")},
                                               {"pre", pre},
                                               {"tokens", vocab},
                                               {"token_type", type_tensor},
                                               {"merges", std::vector<std::string>{}},
                                               {"eos_token_id", eos},
                                               {"chat_template", std::string(qwen35_chat_template)}});
}

// Unigram vocab: <unk>=0, <s>=1, </s>=2, byte pieces 3..258, then @p pieces from id 259.
ov::genai::GGUFTokenizerParameters unigram_config(const std::string& model,
                                                  const std::vector<std::string>& pieces,
                                                  const std::vector<float>& piece_scores) {
    std::vector<std::string> vocab{"<unk>", "<s>", "</s>"};
    std::vector<int32_t> types{2, 3, 3};
    std::vector<float> scores{0.f, 0.f, 0.f};
    for (int i = 0; i < 256; ++i) {
        char byte[7];
        std::snprintf(byte, sizeof(byte), "<0x%02X>", i);
        vocab.emplace_back(byte);
        types.push_back(6);
        scores.push_back(0.f);
    }
    vocab.insert(vocab.end(), pieces.begin(), pieces.end());
    types.insert(types.end(), pieces.size(), 1);
    scores.insert(scores.end(), piece_scores.begin(), piece_scores.end());
    ov::Tensor type_tensor(ov::element::i32, {types.size()});
    std::memcpy(type_tensor.data(), types.data(), type_tensor.get_byte_size());
    ov::Tensor score_tensor(ov::element::f32, {scores.size()});
    std::memcpy(score_tensor.data(), scores.data(), score_tensor.get_byte_size());
    const auto id = [](uint32_t value) {
        ov::Tensor tensor(ov::element::u32, {});
        tensor.data<uint32_t>()[0] = value;
        return tensor;
    };
    return ov::genai::GGUFTokenizerParameters({{"model", model},
                                               {"tokens", vocab},
                                               {"token_type", type_tensor},
                                               {"scores", score_tensor},
                                               {"unknown_token_id", id(0)},
                                               {"bos_token_id", id(1)},
                                               {"eos_token_id", id(2)}});
}

ov::Tensor flag(bool value) {
    ov::Tensor tensor(ov::element::boolean, {});
    tensor.data<bool>()[0] = value;
    return tensor;
}

std::vector<int64_t> encode_ids(ov::genai::Tokenizer& tokenizer, const std::string& text, bool add_special = false) {
    const auto ids = tokenizer.encode(text, ov::genai::add_special_tokens(add_special)).input_ids;
    return {ids.data<int64_t>(), ids.data<int64_t>() + ids.get_size()};
}
}  // namespace

TEST(GGUFTokenizer, SentencePieceControlTokensUseTheirVocabularyIds) {
    ov::genai::GGUFTokenizer tokenizer(sentencepiece_config());
    auto encoded = tokenizer.encode("<start_of_turn>a<end_of_turn><start_of_image><pad><end_of_image><é>",
                                    ov::genai::add_special_tokens(false));
    const std::vector<int64_t> expected{4, 9, 5, 6, 3, 7, 267};
    ASSERT_EQ(encoded.input_ids.get_size(), expected.size());
    EXPECT_EQ(std::vector<int64_t>(encoded.input_ids.data<int64_t>(),
                                   encoded.input_ids.data<int64_t>() + encoded.input_ids.get_size()),
              expected);
    auto with_bos = tokenizer.encode("a", ov::genai::add_special_tokens(true));
    ASSERT_EQ(with_bos.input_ids.get_size(), 2);
    EXPECT_EQ(with_bos.input_ids.data<int64_t>()[0], 1);
    EXPECT_EQ(tokenizer.get_bos_token(), "<bos>");
    EXPECT_EQ(tokenizer.get_eos_token(), "<eos>");
    EXPECT_EQ(tokenizer.apply_chat_template({{{"role", "user"}, {"content", "a"}}}, false),
              "<bos><start_of_turn>a<end_of_turn>");
}

// llama.cpp "t5" (UGM): unigram segmentation with byte fallback; no BOS and an EOS by default.
TEST(GGUFTokenizer, T5UsesUnigramSegmentation) {
    // ids 259..264
    auto config = unigram_config("t5", {"▁", "a", "b", "▁a", "ab", "▁ab"}, {-3.f, -3.f, -3.f, -1.f, -1.2f, -2.5f});
    config.config["add_space_prefix"] = flag(true);
    ov::genai::GGUFTokenizer tokenizer(config);
    // ▁ab (-2.5) beats ▁a+b (-4) and ▁+ab (-4.2); '\n' falls back to byte <0x0A>.
    EXPECT_EQ(encode_ids(tokenizer, "ab ab\n"), (std::vector<int64_t>{264, 264, 3 + 0x0A}));
    // remove_extra_whitespaces is false unless the GGUF sets it.
    EXPECT_EQ(encode_ids(tokenizer, "a  b"), (std::vector<int64_t>{262, 259, 259, 261}));
    EXPECT_EQ(encode_ids(tokenizer, "a", true), (std::vector<int64_t>{262, 2}));
    EXPECT_EQ(tokenizer.decode(std::vector<int64_t>{264, 264, 3 + 0x0A}), "ab ab\n");

    config.config["remove_extra_whitespaces"] = flag(true);
    ov::genai::GGUFTokenizer collapsing(config);
    EXPECT_EQ(encode_ids(collapsing, "a  b"), (std::vector<int64_t>{262, 259, 261}));
}

// PLaMo pieces spell spaces literally: whitespace is neither escaped to U+2581 nor collapsed.
TEST(GGUFTokenizer, Plamo2KeepsLiteralSpaces) {
    // ids 259..263
    ov::genai::GGUFTokenizer tokenizer(
        unigram_config("plamo2", {"a", "b", " ", "ab", " ab"}, {-3.f, -3.f, -3.f, -1.f, -1.f}));
    EXPECT_EQ(encode_ids(tokenizer, "ab ab"), (std::vector<int64_t>{262, 263}));
    EXPECT_EQ(encode_ids(tokenizer, "ab", true), (std::vector<int64_t>{262}));
    EXPECT_EQ(tokenizer.decode(std::vector<int64_t>{262, 263, 261, 261}), "ab ab  ");
}

// PLaMo-3 isolates runs of 2+ spaces and 4+ repeated characters before the unigram search.
TEST(GGUFTokenizer, Plamo3PreSegmentsWhitespaceRuns) {
    // ids 259..263; without pre-segmentation "a    b" would end in " b" (-0.1).
    const std::vector<std::string> pieces{"a", "b", " ", "  ", " b"};
    const std::vector<float> scores{-1.f, -3.f, -2.5f, -2.f, -0.1f};
    ov::genai::GGUFTokenizer plamo3(unigram_config("plamo3", pieces, scores));
    EXPECT_EQ(encode_ids(plamo3, "a    b"), (std::vector<int64_t>{259, 262, 262, 260}));
    EXPECT_EQ(encode_ids(plamo3, "a b"), (std::vector<int64_t>{259, 263}));
    ov::genai::GGUFTokenizer plamo2(unigram_config("plamo2", pieces, scores));
    const auto unsegmented = encode_ids(plamo2, "a    b");
    ASSERT_FALSE(unsegmented.empty());
    EXPECT_EQ(unsegmented.back(), 263);
}

TEST(GGUFTokenizer, MetadataRejectsModelsWithoutTokenizerInfo) {
    EXPECT_THROW(ov::genai::GGUFTokenizerParameters::from_model(nullptr), ov::Exception);
}

class GGUFBpeTokenizer : public ::testing::TestWithParam<std::string> {};

TEST_P(GGUFBpeTokenizer, UserDefinedTokensRemainVisibleWhenSkippingControlTokens) {
    auto config = byte_level_bpe_config("qwen35");
    config.config["model"] = GetParam();
    config.config["tokens"] =
        std::vector<std::string>{"<|endoftext|>", "a", "b", "<think>", "</think>", "<|tool_call|>", "<é>"};
    const std::vector<int32_t> types{3, 1, 1, 4, 4, 3, 4};
    ov::Tensor type_tensor(ov::element::i32, {types.size()});
    std::memcpy(type_tensor.data(), types.data(), type_tensor.get_byte_size());
    config.config["token_type"] = type_tensor;

    ov::genai::GGUFTokenizer tokenizer(config);
    auto encoded = tokenizer.encode("a<think></think><é><|tool_call|>b", ov::genai::add_special_tokens(false));
    const std::vector<int64_t> ids{1, 3, 4, 6, 5, 2};
    ASSERT_EQ(encoded.input_ids.get_size(), ids.size());
    EXPECT_EQ(std::vector<int64_t>(encoded.input_ids.data<int64_t>(),
                                   encoded.input_ids.data<int64_t>() + encoded.input_ids.get_size()),
              ids);
    EXPECT_EQ(tokenizer.decode(ids), "a<think></think><é>b");
    EXPECT_EQ(tokenizer.decode(ids, ov::genai::skip_special_tokens(true)), "a<think></think><é>b");
    EXPECT_EQ(tokenizer.decode(ids, ov::genai::skip_special_tokens(false)), "a<think></think><é><|tool_call|>b");
}

TEST_P(GGUFBpeTokenizer, RejectsTokenTypeSizeMismatch) {
    auto config = byte_level_bpe_config("qwen35");
    config.config["model"] = GetParam();
    ov::Tensor type_tensor(ov::element::i32, {2});
    std::fill_n(type_tensor.data<int32_t>(), 2, 1);
    config.config["token_type"] = type_tensor;
    EXPECT_THROW(ov::genai::GGUFTokenizer{config}, ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(GGUFTokenizer, GGUFBpeTokenizer, ::testing::Values("gpt2", "gemma4"));

TEST(GGUFTokenizer, Gemma4PreservesMetaspaceInAtomicTokens) {
    auto config = byte_level_bpe_config("qwen35");
    config.config["model"] = std::string("gemma4");
    config.config["tokens"] =
        std::vector<std::string>{"<|endoftext|>", "a", "b", "▁", "▁▁", "<▁>", "<ctl▁>", "▁a"};
    const std::vector<int32_t> types{3, 1, 1, 1, 1, 4, 3, 1};
    ov::Tensor type_tensor(ov::element::i32, {types.size()});
    std::memcpy(type_tensor.data(), types.data(), type_tensor.get_byte_size());
    config.config["token_type"] = type_tensor;

    ov::genai::GGUFTokenizer tokenizer(config);
    auto encoded = tokenizer.encode("a<▁>b", ov::genai::add_special_tokens(false));
    const std::vector<int64_t> expected_ids{1, 5, 2};
    ASSERT_EQ(encoded.input_ids.get_size(), expected_ids.size());
    EXPECT_EQ(std::vector<int64_t>(encoded.input_ids.data<int64_t>(),
                                   encoded.input_ids.data<int64_t>() + encoded.input_ids.get_size()),
              expected_ids);

    const std::vector<int64_t> ids{7, 5, 3, 6, 4, 2};
    EXPECT_EQ(tokenizer.decode(ids), " a<▁>   b");
    EXPECT_EQ(tokenizer.decode(ids, ov::genai::skip_special_tokens(true)), " a<▁>   b");
    EXPECT_EQ(tokenizer.decode(ids, ov::genai::skip_special_tokens(false)), " a<▁> <ctl▁>  b");
}

TEST(GGUFTokenizer, Qwen35ChatTemplateUndefinedTestsArePatched) {
    ov::genai::GGUFTokenizer tokenizer(byte_level_bpe_config("qwen35"));
    ov::genai::ChatHistory history({{{"role", "user"}, {"content", "a"}}});
    EXPECT_EQ(tokenizer.get_original_chat_template(), qwen35_chat_template);
    EXPECT_EQ(tokenizer.apply_chat_template(history, false), "think:a");
    history.set_extra_context(ov::genai::JsonContainer({{"enable_thinking", false}}));
    EXPECT_EQ(tokenizer.apply_chat_template(history, false), "direct:a");

    tokenizer.set_chat_template("{% if custom is undefined %}set{% endif %}");
    EXPECT_EQ(tokenizer.apply_chat_template(history, false), "set");
    EXPECT_EQ(tokenizer.apply_chat_template(history, false, "{% if other is undefined %}call{% endif %}"), "call");
}

TEST(GGUFTokenizer, NonQwen35ChatTemplateIsNotPatched) {
    ov::genai::GGUFTokenizer tokenizer(byte_level_bpe_config("qwen2"));
    EXPECT_EQ(tokenizer.get_chat_template(), qwen35_chat_template);
}
