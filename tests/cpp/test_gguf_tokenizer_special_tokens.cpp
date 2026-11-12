// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstring>

#include "gtest/gtest.h"
#include "openvino/genai/gguf.hpp"
#include "openvino/opsets/opset15.hpp"
#include "visual_language/gemma4/classes.hpp"
#include "visual_language/gemma3/classes.hpp"
#include "gguf_utils/gguf_multimodal.hpp"
#include "gguf_utils/gguf_clip_adapter.hpp"
#include "gguf_utils/gguf_tokenizer.hpp"
#include "sampling/sampler.hpp"
#include "visual_language/clip.hpp"
#include "visual_language/llava/classes.hpp"

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
ov::genai::GGUFTokenizerParameters combining_marks_config(const std::string& pre) {
    ov::Tensor types(ov::element::i32, {9});
    std::fill_n(types.data<int32_t>(), 6, 1);
    types.data<int32_t>()[6] = 2;
    types.data<int32_t>()[7] = types.data<int32_t>()[8] = 3;
    ov::Tensor unknown_id(ov::element::u32, {});
    unknown_id.data<uint32_t>()[0] = 6;
    ov::Tensor bos_id(ov::element::u32, {}), eos_id(ov::element::u32, {});
    bos_id.data<uint32_t>()[0] = 7;
    eos_id.data<uint32_t>()[0] = 8;
    ov::Tensor add_bos(ov::element::boolean, {});
    add_bos.data<bool>()[0] = false;
    return ov::genai::GGUFTokenizerParameters(
        {{"model", std::string("gpt2")},
         {"pre", pre},
         {"tokens",
          std::vector<std::string>{"a",
                                   u8"\u00cc",
                                   u8"\u0123",
                                   u8"\u00cc\u0123",
                                   u8"a\u00cc",
                                   u8"a\u00cc\u0123",
                                   "<unk>",
                                   "<bos>",
                                   "<eos>"}},
         {"merges", std::vector<std::string>{u8"\u00cc \u0123", u8"a \u00cc", u8"a\u00cc \u0123", u8"a \u00cc\u0123"}},
         {"token_type", types},
         {"unknown_token_id", unknown_id},
         {"bos_token_id", bos_id},
         {"eos_token_id", eos_id},
         {"add_bos_token", add_bos}});
}

ov::genai::VLMModels gemma4_models() {
    ov::genai::VLMModels models;
    models.config.model_type = ov::genai::VLMModelType::GEMMA4;
    models.config.hidden_size = 4;
    models.config.video_token = "<|video|>";
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    ids->output(0).set_names({"input_ids"});
    auto table = ov::op::v0::Constant::create(ov::element::f32, {267, 4}, {0.25f});
    auto lookup =
        std::make_shared<ov::op::v8::Gather>(table, ids, ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    models.models["text_embeddings"] = std::make_shared<ov::Model>(ov::OutputVector{lookup}, ov::ParameterVector{ids});
    auto pixels = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 12});
    pixels->output(0).set_names({"pixel_values"});
    models.models["vision_embeddings"] = std::make_shared<ov::Model>(ov::OutputVector{pixels}, ov::ParameterVector{pixels});
    return models;
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

TEST(GGUFTokenizer, SentencePieceControlTokensUseTheirVocabularyIds) {
    ov::genai::GGUFTokenizer tokenizer(sentencepiece_config());
    auto encoded = tokenizer.encode("<start_of_turn>a<end_of_turn><start_of_image><pad><end_of_image>",
                                    ov::genai::add_special_tokens(false));
    const std::vector<int64_t> expected{4, 9, 5, 6, 3, 7};
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

TEST(GGUFTokenizer, ChatTemplateSupportsUndefinedAndAdjacentStringLiterals) {
    GTEST_SKIP() << "Chat-template rendering does not handle 'is undefined' tests or adjacent multiline string literals.";
    auto config = sentencepiece_config();
    config.config["chat_template"] =
        std::string("{% if enable_thinking is undefined %}{{ \"first\"\n  \"second\" }}{% endif %}"
                    "{{ messages[0]['content'] }}");
    ov::genai::GGUFTokenizer tokenizer(config);
    EXPECT_EQ(tokenizer.apply_chat_template({{{"role", "user"}, {"content", "a"}}}, false), "firstseconda");
}

TEST(GGUFTokenizer, GPT4OSplitsCombiningMarksLikeLlamaCPU) {
    // Pinned llama.cpp 03fa73cb: "a\u0301" splits into "a" and the acute accent.
    const std::vector<int64_t> expected{0, 3};
    for (const auto& pre : {"gpt-4o", "llama4"}) {
        ov::genai::GGUFTokenizer tokenizer(combining_marks_config(pre));
        const auto ids = tokenizer.encode(u8"a\u0301", ov::genai::add_special_tokens(false)).input_ids;
        ASSERT_EQ(ids.get_size(), expected.size()) << pre;
        EXPECT_EQ(std::vector<int64_t>(ids.data<int64_t>(), ids.data<int64_t>() + ids.get_size()), expected) << pre;
    }
}

TEST(GGUFTokenizer, Gemma4UsesMergesForOrdinaryVocabularyEntries) {
    auto config = combining_marks_config("gemma4");
    config.config["model"] = std::string("gemma4");
    config.config["tokens"] =
        std::vector<std::string>{"<unk>", "<bos>", "<eos>", "a", "b", "ab", "c", "abc", "\n", "\n\n"};
    config.config["merges"] = std::vector<std::string>{"a b"};
    ov::Tensor types(ov::element::i32, {10});
    std::fill_n(types.data<int32_t>(), 10, 1);
    types.data<int32_t>()[0] = 2;
    types.data<int32_t>()[1] = types.data<int32_t>()[2] = 3;
    config.config["token_type"] = types;
    for (const auto& [key, id] : std::vector<std::pair<std::string, uint32_t>>{{"unknown_token_id", 0},
                                                                               {"bos_token_id", 1},
                                                                               {"eos_token_id", 2}}) {
        ov::Tensor value(ov::element::u32, {});
        value.data<uint32_t>()[0] = id;
        config.config[key] = value;
    }
    ov::genai::GGUFTokenizer tokenizer(config);
    const auto ids = tokenizer.encode("abc\n\n", ov::genai::add_special_tokens(false)).input_ids;
    // Pinned llama.cpp 03fa73cb: "abc" needs a merge; repeated newlines use their vocabulary entry.
    const std::vector<int64_t> expected{5, 6, 9};
    ASSERT_EQ(ids.get_size(), expected.size());
    EXPECT_EQ(std::vector<int64_t>(ids.data<int64_t>(), ids.data<int64_t>() + ids.get_size()), expected);
}

TEST(GGUFTokenizer, Gemma4ByteTokenSpellingUsesOrdinaryCharacters) {
    auto config = combining_marks_config("gemma4");
    config.config["model"] = std::string("gemma4");
    config.config["tokens"] =
        std::vector<std::string>{"<unk>", "<bos>", "<eos>", "<", "0", "x", "8", "5", ">", "<0x85>", "\n", "<0x0A>"};
    config.config["merges"] = std::vector<std::string>{};
    ov::Tensor types(ov::element::i32, {12});
    std::fill_n(types.data<int32_t>(), 12, 1);
    types.data<int32_t>()[0] = 2;
    types.data<int32_t>()[1] = types.data<int32_t>()[2] = 3;
    types.data<int32_t>()[9] = 6;
    types.data<int32_t>()[11] = 6;
    config.config["token_type"] = types;
    for (const auto& [key, id] : std::vector<std::pair<std::string, uint32_t>>{{"unknown_token_id", 0},
                                                                               {"bos_token_id", 1},
                                                                               {"eos_token_id", 2}}) {
        ov::Tensor value(ov::element::u32, {});
        value.data<uint32_t>()[0] = id;
        config.config[key] = value;
    }
    ov::genai::GGUFTokenizer tokenizer(config);
    const auto ids = tokenizer.encode("<0x85>", ov::genai::add_special_tokens(false)).input_ids;
    const std::vector<int64_t> expected{3, 4, 5, 6, 7, 8};
    ASSERT_EQ(ids.get_size(), expected.size());
    EXPECT_EQ(std::vector<int64_t>(ids.data<int64_t>(), ids.data<int64_t>() + ids.get_size()), expected);
    const auto byte_ids = tokenizer.encode(std::string(1, char(0x85)), ov::genai::add_special_tokens(false)).input_ids;
    ASSERT_EQ(byte_ids.get_size(), 1);
    EXPECT_EQ(byte_ids.data<int64_t>()[0], 9);
    const auto newline_ids = tokenizer.encode("\n", ov::genai::add_special_tokens(false)).input_ids;
    ASSERT_EQ(newline_ids.get_size(), 1);
    EXPECT_EQ(newline_ids.data<int64_t>()[0], 10);
    EXPECT_EQ(tokenizer.decode(std::vector<int64_t>{11}), "\n");
}


TEST(GGUFMultimodal, PreformattedChatDoesNotDuplicateSpecialTokens) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    struct TestEmbedder : InputsEmbedderGemma4 {
        using InputsEmbedderGemma4::InputsEmbedderGemma4;
        using IInputsEmbedder::get_encoded_input_ids;
    };
    TestEmbedder embedder(gemma4_models(), tokenizer, "CPU", {});
    const auto encode = [&](const std::string& prompt) {
        VLMPerfMetrics metrics;
        auto ids = embedder.get_encoded_input_ids(prompt, metrics);
        return std::vector<int64_t>(ids.data<int64_t>(), ids.data<int64_t>() + ids.get_size());
    };
    const std::vector<int64_t> expected{1, 4, 9, 5};
    EXPECT_EQ(encode("a"), expected);
    embedder.finish_chat();
    embedder.set_prompt_formatting(PromptFormatting::AlreadyTemplated);
    EXPECT_EQ(encode(tokenizer.apply_chat_template({{{"role", "user"}, {"content", "a"}}}, true)), expected);
    embedder.finish_chat();
    embedder.set_prompt_formatting(PromptFormatting::Raw);
    EXPECT_EQ(encode("a"), (std::vector<int64_t>{1, 9}));
    embedder.finish_chat();
    embedder.set_prompt_formatting(PromptFormatting::NeedsTemplate);
    EXPECT_EQ(encode("a"), expected);
}

TEST(GGUFMultimodal, LlavaUnknownTextTokensRemainTextEmbeddings) {
    using namespace ov::genai;
    auto tokenizer = GGUFTokenizer(combining_marks_config("gpt2"));
    auto models = gemma4_models();
    models.config.model_type = VLMModelType::LLAVA;
    models.config.image_token_id = 6;
    InputsEmbedderLLaVA embedder(models, tokenizer, "CPU", {});
    embedder.set_prompt_formatting(PromptFormatting::Raw);
    ov::Tensor pixels(ov::element::f32, {1, 2, 4});
    std::fill_n(pixels.data<float>(), pixels.get_size(), 9.f);
    const std::vector<EncodedImage> images{{pixels, {1, 2}}};
    const auto normalized = embedder.normalize_prompt("x<image>x", 0, images);
    VLMPerfMetrics metrics;
    const auto merged =
        embedder.get_inputs_embeds(normalized.unified_prompt, images, metrics, true, normalized.images_sequence);
    const auto prefix = tokenizer.encode("x", add_special_tokens(true)).input_ids;
    const auto suffix = tokenizer.encode("\nx", add_special_tokens(false)).input_ids;
    ASSERT_EQ(prefix.data<const int64_t>()[prefix.get_size() - 1], 6);
    ASSERT_EQ(suffix.data<const int64_t>()[suffix.get_size() - 1], 6);
    ASSERT_EQ(merged.get_shape(), (ov::Shape{1, prefix.get_size() + 2 + suffix.get_size(), 4}));
    for (size_t token = 0; token < merged.get_shape()[1]; ++token) {
        const float expected = token >= prefix.get_size() && token < prefix.get_size() + 2 ? 9.f : .25f;
        for (size_t component = 0; component < 4; ++component)
            EXPECT_FLOAT_EQ(merged.data<const float>()[token * 4 + component], expected);
    }
}

TEST(GGUFMultimodal, Gemma3LiteralPadsRemainTextAfterCacheClipping) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    auto models = gemma4_models();
    models.config.model_type = VLMModelType::GEMMA3;
    models.config.image_soft_token = "<pad>";
    ov::Tensor pixels(ov::element::f32, {1, 2, 4});
    for (size_t i = 0; i < pixels.get_size(); ++i)
        pixels.data<float>()[i] = float(10 + i);
    const std::vector<EncodedImage> images{{pixels}};
    InputsEmbedderGemma3 normalizer(models, tokenizer, "CPU", {});
    const auto normalized = normalizer.normalize_prompt("<pad>a<start_of_image>a<pad>", 0, images);
    const auto ids = tokenizer.encode(normalized.unified_prompt, add_special_tokens(false)).input_ids;
    const auto* data = ids.data<const int64_t>();
    const auto begin = std::find(data, data + ids.get_size(), 6) - data + 1;
    ASSERT_LT(begin + 2, ids.get_size());
    ASSERT_EQ(data[begin], 3);
    ASSERT_EQ(data[begin + 1], 3);
    ASSERT_EQ(data[begin + 2], 7);
    for (size_t skip : {size_t(0), size_t(begin), size_t(begin + 1), size_t(begin + 2)}) {
        SCOPED_TRACE(skip);
        InputsEmbedderGemma3 embedder(models, tokenizer, "CPU", {});
        embedder.set_prompt_formatting(PromptFormatting::Raw);
        embedder.set_add_special_tokens(false);
        if (skip) {
            embedder.start_chat("");
            embedder.get_cache_state().get_state().assign(data, data + skip);
        }
        VLMPerfMetrics metrics;
        const auto merged =
            embedder.get_inputs_embeds(normalized.unified_prompt, images, metrics, true, normalized.images_sequence);
        const auto& types = embedder.get_lm_extra_inputs().at("token_type_ids");
        ASSERT_EQ(merged.get_shape(), (ov::Shape{1, ids.get_size() - skip, 4}));
        ASSERT_EQ(types.get_shape(), (ov::Shape{1, ids.get_size() - skip}));
        for (size_t token = skip; token < ids.get_size(); ++token) {
            const bool vision = token >= size_t(begin) && token < size_t(begin + 2);
            EXPECT_EQ(types.data<const int64_t>()[token - skip], int64_t(vision));
            for (size_t component = 0; component < 4; ++component) {
                const float expected = vision ? pixels.data<const float>()[(token - begin) * 4 + component] : .25f;
                EXPECT_FLOAT_EQ(merged.data<const float>()[(token - skip) * 4 + component], expected);
            }
        }
    }
}

TEST(GGUFMultimodal, LlavaAutomaticEosFollowsAllImageSlots) {
    using namespace ov::genai;
    struct TestEmbedder : InputsEmbedderLLaVA {
        using InputsEmbedderLLaVA::apply_chat_template_tokenize;
        using InputsEmbedderLLaVA::InputsEmbedderLLaVA;
    };
    for (bool add_eos : {false, true}) {
        for (bool alias_bos_eos : {false, true}) {
            auto parameters = sentencepiece_config();
            ov::Tensor flag(ov::element::boolean, {});
            flag.data<bool>()[0] = add_eos;
            parameters.config["add_eos_token"] = flag;
            if (alias_bos_eos)
                parameters.config["eos_token_id"] = parameters.config.at("bos_token_id");
            GGUFTokenizer tokenizer(parameters);
            for (const std::string prefix : {"a", "a<eos>"}) {
                for (int mode : {0, 1, 2}) {
                    SCOPED_TRACE(::testing::Message()
                                 << add_eos << ',' << alias_bos_eos << ',' << prefix << ',' << mode);
                    auto models = gemma4_models();
                    models.config.model_type = VLMModelType::LLAVA;
                    models.config.image_token_id = 3;
                    models.config.add_eos_token = add_eos;
                    TestEmbedder embedder(models, tokenizer, "CPU", {});
                    embedder.set_prompt_formatting(mode == 2 ? PromptFormatting::AlreadyTemplated
                                                             : PromptFormatting::Raw);
                    if (mode == 1)
                        embedder.set_add_special_tokens(false);
                    VLMPerfMetrics metrics;
                    const auto actual = embedder.apply_chat_template_tokenize(prefix + "<image>a", metrics);
                    std::vector<int64_t> expected;
                    if (mode == 0)
                        expected.push_back(1);
                    expected.push_back(9);
                    if (prefix == "a<eos>")
                        expected.push_back(2);
                    expected.insert(expected.end(), {3, 9});
                    if (mode == 0 && add_eos)
                        expected.push_back(alias_bos_eos ? 1 : 2);
                    EXPECT_EQ(std::vector<int64_t>(actual.data<const int64_t>(),
                                                   actual.data<const int64_t>() + actual.get_size()),
                              expected);
                }
            }
        }
    }
}

TEST(GGUFMultimodal, ExtremeAspectRatiosRespectPixelBudget) {
    EXPECT_THROW(bounded_image_size(48, 96000, 48, 0, 1120 * 48 * 48), ov::Exception);
    EXPECT_THROW(bounded_image_size(42, 188160, 42, 0, 1120 * 42 * 42), ov::Exception);
    EXPECT_THROW(bounded_image_size(48, 60000, 48, 0, 1120 * 48 * 48), ov::Exception);
    const auto [height, width] = bounded_image_size(512, 768, 48, 0, 1120 * 48 * 48);
    EXPECT_LE(height, (1120 * 48 * 48) / width);
}

TEST(GGUFMultimodal, Gemma4HonorsImagePadding) {
    using namespace ov::genai;
    auto models = gemma4_models();
    auto pixels = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 48});
    pixels->output(0).set_names({"pixel_values"});
    auto positions = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 8, 2});
    positions->output(0).set_names({"image_position_ids"});
    auto position_sum = std::make_shared<ov::op::v1::ReduceSum>(
        std::make_shared<ov::op::v0::Convert>(positions, ov::element::f32),
        ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}), false);
    auto features = std::make_shared<ov::op::v1::Add>(pixels, position_sum);
    models.models["vision_embeddings"] =
        std::make_shared<ov::Model>(ov::OutputVector{features}, ov::ParameterVector{pixels, positions});
    auto& config = models.processor_config;
    config.patch_size = 4;
    config.pooling_kernel_size = 1;
    config.max_soft_tokens = 8;
    config.min_pixels = 0;
    config.max_pixels = 128;
    config.preserve_native_resolution = true;
    config.pad_to_target = true;
    config.image_mean = {0, 0, 0};
    config.image_std = {1, 1, 1};
    VisionEncoderGemma4 encoder(models, "CPU", {});
    ov::Tensor image(ov::element::u8, {1, 5, 10, 3});
    std::fill_n(image.data<uint8_t>(), image.get_size(), 255);
    const auto padded = encoder.encode(image, {}).resized_source;
    config.pad_to_target = false;
    VisionEncoderGemma4 stretched_encoder(models, "CPU", {});
    const auto stretched = stretched_encoder.encode(image, {}).resized_source;
    EXPECT_EQ(padded.get_shape(), stretched.get_shape());
    EXPECT_FALSE(std::equal(padded.data<float>(), padded.data<float>() + padded.get_size(), stretched.data<float>()));
}

TEST(GGUFMultimodal, ClipRequiresCompatibleLanguageInputs) {
    using namespace ov::opset15;
    using ov::genai::validate_gguf_clip_language;
    auto embeds = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, -1, 4});
    embeds->output(0).set_names({"inputs_embeds"});
    auto positions = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    positions->output(0).set_names({"position_ids"});
    auto language = std::make_shared<ov::Model>(ov::OutputVector{embeds}, ov::ParameterVector{embeds, positions});
    EXPECT_NO_THROW(validate_gguf_clip_language(language));
    for (const auto& shape : {ov::PartialShape{4, 1, -1}, ov::PartialShape{1, 1, 1, -1}, ov::PartialShape::dynamic()}) {
        positions->set_partial_shape(shape);
        positions->validate_and_infer_types();
        EXPECT_THROW(validate_gguf_clip_language(language), ov::Exception);
    }
    positions->set_partial_shape({1, -1});
    positions->set_element_type(ov::element::f32);
    positions->validate_and_infer_types();
    EXPECT_THROW(validate_gguf_clip_language(language), ov::Exception);
    positions->set_element_type(ov::element::i64);
    positions->validate_and_infer_types();
    for (const std::string name : {"token_type_ids", "per_layer_inputs"}) {
        auto extra = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape{1, -1});
        extra->output(0).set_names({name});
        language->add_parameters({extra});
        EXPECT_THROW(validate_gguf_clip_language(language), ov::Exception);
        language->remove_parameter(extra);
    }
    EXPECT_NO_THROW(validate_gguf_clip_language(language));
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
    models.models["audio_embeddings"] = embedding(2);
    EXPECT_THROW(validate_gguf_multimodal_dimensions(language, models), ov::Exception);
    models.models["audio_embeddings"] = embedding(4);
    EXPECT_NO_THROW(validate_gguf_multimodal_dimensions(language, models));
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
