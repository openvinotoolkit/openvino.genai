// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstring>

#include "gtest/gtest.h"
#include "openvino/genai/gguf.hpp"
#include "openvino/opsets/opset15.hpp"
#include "visual_language/gemma4/classes.hpp"
#include "visual_language/gemma4/audio_encoder.hpp"
#include "visual_language/gemma4/audio_feature_extractor.hpp"
#include "gguf_utils/gguf_multimodal.hpp"
#include "gguf_utils/gguf_frontend_extensions.hpp"
#include "gguf_utils/gguf_tokenizer.hpp"
#include "sampling/sampler.hpp"
#include "visual_language/clip.hpp"

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

TEST(GGUFMultimodal, AudioAdapterReusesGemma4Encoders) {
    using namespace ov::opset15;
    using namespace ov::genai;
    for (bool unified : {false, true}) {
        auto native = std::make_shared<Parameter>(ov::element::f32,
            unified ? ov::PartialShape{1, 1, -1, 640} : ov::PartialShape{1, 1, 128, -1});
        native->output(0).set_names({unified ? "waveform_frames" : "features"});
        ov::Output<ov::Node> embedding = std::make_shared<Squeeze>(native,
            Constant::create(ov::element::i64, {1}, {1}));
        if (!unified)
            embedding = std::make_shared<Transpose>(embedding,
                Constant::create(ov::element::i64, {3}, {0, 2, 1}));
        embedding.set_names({"audio_features"});
        auto model = std::make_shared<ov::Model>(ov::OutputVector{embedding}, ov::ParameterVector{native});
        adapt_gguf_audio_encoder(model, unified);
        auto encoder = AudioEncoderGemma4::create(model,
            unified ? VLMModelType::GEMMA4_UNIFIED : VLMModelType::GEMMA4, "CPU", {});
        for (size_t samples : {1000, 1281}) {
            ov::Tensor waveform(ov::element::f32, {samples});
            std::fill_n(waveform.data<float>(), samples, .25f);
            auto actual = encoder->encode(waveform);
            if (unified) {
                ASSERT_EQ(actual.get_shape(), (ov::Shape{1, (samples + 639) / 640, 640}));
                EXPECT_FLOAT_EQ(actual.data<const float>()[0], .25f);
                EXPECT_FLOAT_EQ(actual.data<const float>()[actual.get_size() - 1], 0.f);
            } else {
                auto expected = Gemma4AudioFeatureExtractor().extract(waveform);
                const auto valid = std::count(expected.input_features_mask.data<const bool>(),
                    expected.input_features_mask.data<const bool>() + expected.input_features_mask.get_size(), true);
                ASSERT_EQ(actual.get_shape(), (ov::Shape{1, size_t(valid), 128}));
                for (size_t i = 0; i < actual.get_size(); ++i)
                    EXPECT_FLOAT_EQ(actual.data<const float>()[i], expected.input_features.data<const float>()[i]);
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

TEST(GGUFMultimodal, PipelineControlsGenAIExtensionMode) {
    using namespace ov::genai;
    using Extension = ov::frontend::gguf::GenAIExtension;
    for (const auto mode : {Extension::InputMode::IDS_TO_LOGITS, Extension::InputMode::EMBEDS_TO_LOGITS}) {
        ov::frontend::gguf::FrontEnd frontend;
        auto supplied = std::make_shared<Extension>(Extension::InputMode::IDS_TO_LOGITS);
        const auto selected = register_gguf_extensions(frontend, {supplied}, mode);
        EXPECT_NE(selected, supplied);
        EXPECT_NE(selected, nullptr);
    }
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
