// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstring>

#include "gtest/gtest.h"
#include "gguf_utils/gguf_multimodal.hpp"
#include "gguf_utils/gguf_tokenizer.hpp"
#include "openvino/genai/gguf.hpp"
#include "openvino/op/parameter.hpp"
#include "sampling/sampler.hpp"

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

TEST(GGUFTokenizer, MetadataRejectsModelsWithoutTokenizerInfo) {
    EXPECT_THROW(ov::genai::GGUFTokenizerParameters::from_model(nullptr), ov::Exception);
}

TEST(GGUFTokenizer, FimEndOfGenerationAliasesMatchLlamaCPU) {
    for (const std::string token : {"<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]",
                                   "<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>",
                                   "<|file_sep|>"}) {
        SCOPED_TRACE(token);
        const ov::AnyMap metadata{{"tokens", std::vector<std::string>{"<eos>", token}}};
        EXPECT_EQ(ov::genai::gguf_stop_token_ids(metadata), (std::set<int64_t>{0, 1}));
    }
    const ov::AnyMap metadata{
        {"tokens", std::vector<std::string>{"<eos>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"}}};
    EXPECT_EQ(ov::genai::gguf_stop_token_ids(metadata), (std::set<int64_t>{0}));
}

TEST(GGUFTokenizer, FimMetadataOverridesVocabularyAliases) {
    ov::AnyMap metadata{{"tokens", std::vector<std::string>{"<eos>", "<|fim_pad|>", "<|repo_name|>",
                                                           "<|file_sep|>", "custom-pad", "custom-repo", "custom-sep"}}};
    for (const auto& [key, id] : std::vector<std::pair<std::string, uint32_t>>{
             {"fim_pad_token_id", 4}, {"fim_rep_token_id", 5}, {"fim_sep_token_id", 6}}) {
        ov::Tensor value(ov::element::u32, {});
        value.data<uint32_t>()[0] = id;
        metadata[key] = value;
    }
    EXPECT_EQ(ov::genai::gguf_stop_token_ids(metadata), (std::set<int64_t>{0, 4, 5, 6}));
    metadata["fim_pad_token_id"] = 99;
    metadata["fim_rep_token_id"] = -1;
    EXPECT_EQ(ov::genai::gguf_stop_token_ids(metadata), (std::set<int64_t>{0, 1, 2, 6}));
}

TEST(GGUFTokenizer, FimStopsTerminateGenerationUnlessEosIsIgnored) {
    const std::vector<std::string> tokens{"<eos>", "x", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>",
                                           "<|fim_prefix|>", "custom-pad", "custom-repo", "custom-sep"};
    for (const bool use_metadata : {false, true}) {
        ov::AnyMap metadata{{"tokens", tokens}};
        if (use_metadata) {
            metadata["fim_pad_token_id"] = 6;
            metadata["fim_rep_token_id"] = 7;
            metadata["fim_sep_token_id"] = 8;
        }
        for (const int64_t token : use_metadata ? std::vector<int64_t>{6, 7, 8, 5}
                                                : std::vector<int64_t>{2, 3, 4, 5}) {
            for (const bool ignore_eos : {false, true}) {
                SCOPED_TRACE(::testing::Message() << use_metadata << " " << token << " " << ignore_eos);
                ov::genai::GenerationConfig config;
                config.max_new_tokens = 3;
                config.eos_token_id = 0;
                config.stop_token_ids = ov::genai::gguf_stop_token_ids(metadata);
                config.ignore_eos = ignore_eos;
                const std::vector<int64_t> prompt{1};
                ov::Tensor input(ov::element::i64, {1, prompt.size()}, prompt.data());
                auto group = std::make_shared<ov::genai::SequenceGroup>(0, input, config);
                const auto sequence = group->get_sequences().front();
                group->schedule_tokens(group->get_num_available_tokens_for_batching());
                ov::Tensor logits(ov::element::f32, {1, 1, tokens.size()});
                std::fill_n(logits.data<float>(), logits.get_size(), -10.0f);
                logits.data<float>()[token] = 10.0f;
                ov::genai::Sampler sampler;
                sampler.sample({group}, logits, false);
                EXPECT_EQ(sequence->has_finished(), !ignore_eos && token != 5);
                if (sequence->has_finished())
                    EXPECT_EQ(sequence->get_finish_reason(), ov::genai::GenerationFinishReason::STOP);
            }
        }
    }
}

TEST(GGUFTokenizer, TruncationIsOptInWithoutBosOrEos) {
    using namespace ov::genai;
    auto parameters = sentencepiece_config();
    parameters.config.at("add_bos_token").as<ov::Tensor>().data<bool>()[0] = false;
    GGUFTokenizer tokenizer(parameters);
    const std::string text(9000, 'a');
    EXPECT_EQ(tokenizer.encode(text).input_ids.get_size(), 9000);
    EXPECT_EQ(tokenizer.encode(text, add_special_tokens(false)).input_ids.get_size(), 9000);
    EXPECT_EQ(tokenizer.encode(text, truncation(true), max_length(16)).input_ids.get_size(), 16);
}

TEST(GGUFMultimodal, ProjectorWidthsMustMatchLanguage) {
    using namespace ov::genai;
    const auto embedding = [](int64_t width) {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, width});
        return std::make_shared<ov::Model>(ov::OutputVector{input}, ov::ParameterVector{input});
    };
    auto language = embedding(4);
    language->get_parameters()[0]->output(0).set_names({"inputs_embeds"});
    VLMModels models;
    models.models["text_embeddings"] = embedding(4);
    models.models["vision_embeddings"] = embedding(4);
    EXPECT_NO_THROW(validate_gguf_multimodal_dimensions(language, models));
    for (int64_t width : {2, 8, -1}) {
        models.models["vision_embeddings"] = embedding(width);
        EXPECT_THROW(validate_gguf_multimodal_dimensions(language, models), ov::Exception);
    }
    models.models["vision_embeddings_merger"] = embedding(4);
    EXPECT_NO_THROW(validate_gguf_multimodal_dimensions(language, models));
    models.models["vision_embeddings_merger"] = embedding(2);
    EXPECT_THROW(validate_gguf_multimodal_dimensions(language, models), ov::Exception);
    models.models["vision_embeddings_merger"] = embedding(4);
    models.models["text_embeddings"] = embedding(2);
    EXPECT_THROW(validate_gguf_multimodal_dimensions(language, models), ov::Exception);
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
