// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

#include "gguf_utils/gguf_lfm2_audio.hpp"
#include "gguf_utils/gguf_multimodal.hpp"
#include "gtest/gtest.h"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/opsets/opset15.hpp"
#include "visual_language/clip.hpp"
#include "visual_language/embedding_model.hpp"
#include "visual_language/gemma3/classes.hpp"
#include "visual_language/gemma4/classes.hpp"
#include "visual_language/lfm2/classes.hpp"
#include "visual_language/muse_glimmer/classes.hpp"
#include "visual_language/vlm_chat_context.hpp"
#include "visual_language/vlm_utils.hpp"
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
    for (const auto* token : {"<|audio|>", "<|audio>", "<audio|>", "<|image|>", "<|video|>", "<|audio_start|>"}) {
        vocab.emplace_back(token);
        types.push_back(3);
    }
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
constexpr size_t vocab_size = 273;  // tokens in sentencepiece_config()

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
         {"tokens", std::vector<std::string>{"a", "Ì", "ģ", "Ìģ", "aÌ", "aÌģ", "<unk>", "<bos>", "<eos>"}},
         {"merges", std::vector<std::string>{"Ì ģ", "a Ì", "aÌ ģ", "a Ìģ"}},
         {"token_type", types},
         {"unknown_token_id", unknown_id},
         {"bos_token_id", bos_id},
         {"eos_token_id", eos_id},
         {"add_bos_token", add_bos}});
}

// Text embeddings are all 0.25; the vision encoder is never run.
ov::genai::VLMModels gemma4_models() {
    ov::genai::VLMModels models;
    models.config.model_type = ov::genai::VLMModelType::GEMMA4;
    models.config.hidden_size = 4;
    models.config.video_token = "<|video|>";
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    ids->output(0).set_names({"input_ids"});
    auto table = ov::op::v0::Constant::create(ov::element::f32, {vocab_size, 4}, {0.25f});
    auto lookup =
        std::make_shared<ov::op::v8::Gather>(table, ids, ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    models.models["text_embeddings"] = std::make_shared<ov::Model>(ov::OutputVector{lookup}, ov::ParameterVector{ids});
    auto pixels = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 12});
    pixels->output(0).set_names({"pixel_values"});
    models.models["vision_embeddings"] = std::make_shared<ov::Model>(ov::OutputVector{pixels}, ov::ParameterVector{pixels});
    return models;
}

// Two audio tokens whose features all equal the first sample, so tests can trace their placement.
std::vector<ov::Tensor> first_sample_audio_features(const ov::Tensor& audio) {
    ov::Tensor features(ov::element::f32, {1, 2, 4});
    std::fill_n(features.data<float>(), features.get_size(), audio.data<const float>()[0]);
    return {features};
}
}  // namespace

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

TEST(GGUFTokenizer, ChatTemplateSupportsUndefinedAndAdjacentStringLiterals) {
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
        const auto ids = tokenizer.encode("a\u0301", ov::genai::add_special_tokens(false)).input_ids;
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
    embedder.set_apply_chat_template_status(false, true);
    EXPECT_EQ(encode(tokenizer.apply_chat_template({{{"role", "user"}, {"content", "a"}}}, true)), expected);
    embedder.finish_chat();
    embedder.set_apply_chat_template_status(false);
    EXPECT_EQ(encode("a"), (std::vector<int64_t>{1, 9}));
    embedder.finish_chat();
    embedder.set_apply_chat_template_status(true);
    EXPECT_EQ(encode("a"), expected);
}

TEST(GGUFMultimodal, GemmaAudioPlacementFollowsTheAudioSequence) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    InputsEmbedderGemma4 embedder(gemma4_models(), tokenizer, "CPU", {});
    embedder.set_apply_chat_template_status(false);
    embedder.set_audio_encoder(first_sample_audio_features);
    const auto sample = [](float value) {
        ov::Tensor audio(ov::element::f32, {1});
        audio.data<float>()[0] = value;
        return audio;
    };
    const auto encoded = embedder.encode_audios({sample(7.f), ov::Tensor(ov::element::f32, {0}), sample(9.f)});
    ASSERT_EQ(encoded.size(), 3);
    EXPECT_EQ(encoded[0].num_audio_tokens, 2);
    EXPECT_EQ(encoded[0].audio_features.get_shape(), (ov::Shape{2, 4}));
    EXPECT_EQ(encoded[1].num_audio_tokens, 0);
    // The first value of each feature run, in prompt order.
    const auto placed = [&](const NormalizedPrompt& prompt, size_t base) {
        auto sequence = prompt.audios_sequence;
        vlm_utils::rebase_media_sequence(sequence, base);
        VLMPerfMetrics metrics;
        const auto features =
            embedder.get_inputs_embeds(prompt.unified_prompt, {}, {}, encoded, metrics, true, {}, {}, sequence, {});
        std::vector<float> runs;
        const auto* data = features.data<const float>();
        for (size_t i = 0; i < features.get_size(); i += 4)
            if (data[i] != 0.25f && (runs.empty() || runs.back() != data[i]))
                runs.push_back(data[i]);
        return runs;
    };
    const auto native = embedder.normalize_prompt("a<|audio|>b<|audio|><|audio|>", 0, 0, 0, {}, {}, encoded);
    EXPECT_EQ(native.audios_sequence, (std::vector<size_t>{0, 1, 2}));
    EXPECT_EQ(native.unified_prompt, "a<|audio><|audio|><|audio|><audio|>b<|audio><audio|><|audio><|audio|><|audio|><audio|>");
    EXPECT_EQ(placed(native, 0), (std::vector<float>{7.f, 9.f}));
    // A later chat turn with history audios 0-1: reversed universal tags bind by index.
    const auto universal = embedder.normalize_prompt("<ov_genai_audio_4>x<ov_genai_audio_2>", 0, 0, 2, {}, {}, encoded);
    EXPECT_EQ(universal.audios_sequence, (std::vector<size_t>{4, 2}));
    EXPECT_EQ(placed(universal, 2), (std::vector<float>{9.f, 7.f}));
    // Audio-free prompts are untouched, and a tag without its audio is rejected.
    EXPECT_EQ(embedder.normalize_prompt("ab", 0, 0, 0, {}, {}, {}).unified_prompt, "ab");
    EXPECT_ANY_THROW(embedder.normalize_prompt("<ov_genai_audio_0>", 0, 0, 0, {}, {}, {}));
    EXPECT_TRUE(embedder.encode_audios({}).empty());
}

TEST(GGUFMultimodal, GemmaPerLayerInputsUseTheirLookupModel) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    constexpr size_t layers = 2, width = 3;
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    // Row i of the per-layer table holds i, so each value identifies the looked-up token.
    std::vector<float> rows(vocab_size * layers * width);
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = float(i / (layers * width));
    auto per_layer_table = ov::op::v0::Constant::create(ov::element::f32, {vocab_size, layers * width}, rows);
    auto per_layer_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    per_layer_ids->output(0).set_names({"input_ids"});
    auto per_layer = std::make_shared<ov::op::v1::Reshape>(
        std::make_shared<ov::op::v8::Gather>(per_layer_table, per_layer_ids, axis),
        ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, layers, width}),
        true);
    per_layer->output(0).set_names({"per_layer_inputs"});
    auto models = gemma4_models();
    models.config.hidden_size_per_layer_input = width;
    models.models["text_embeddings_per_layer"] =
        std::make_shared<ov::Model>(ov::OutputVector{per_layer}, ov::ParameterVector{per_layer_ids});
    InputsEmbedderGemma4 embedder(models, tokenizer, "CPU", {});
    embedder.set_apply_chat_template_status(false);
    embedder.set_audio_encoder([](const ov::Tensor&) {
        return std::vector<ov::Tensor>{ov::Tensor(ov::element::f32, {1, 2, 4})};
    });
    const auto audios = embedder.encode_audios({ov::Tensor(ov::element::f32, {1})});
    const auto normalized = embedder.normalize_prompt("a<|audio|>b", 0, 0, 0, {}, {}, audios);
    const auto& prompt = normalized.unified_prompt;
    VLMPerfMetrics metrics;
    const auto inputs_embeds =
        embedder.get_inputs_embeds(prompt, {}, {}, audios, metrics, true, {}, {}, normalized.audios_sequence, {});
    const auto token_ids = tokenizer.encode(prompt).input_ids;
    const auto& per_layer_inputs = embedder.get_lm_extra_inputs().at("per_layer_inputs");
    ASSERT_EQ(per_layer_inputs.get_shape(), (ov::Shape{1, inputs_embeds.get_shape()[1], layers, width}));
    ASSERT_EQ(token_ids.get_size(), inputs_embeds.get_shape()[1]);
    for (size_t t = 0; t < token_ids.get_size(); ++t) {
        for (size_t i = 0; i < layers * width; ++i)
            EXPECT_EQ(per_layer_inputs.data<const float>()[t * layers * width + i],
                      float(token_ids.data<const int64_t>()[t]));
    }
    // Generated tokens use the same lookup through the continuous-batching callback.
    ov::Tensor generated(ov::element::i64, {1, 1});
    generated.data<int64_t>()[0] = 9;
    const auto decoded = embedder.get_per_layer_embeddings_callback()(generated);
    ASSERT_EQ(decoded.get_shape(), (ov::Shape{1, 1, layers, width}));
    EXPECT_EQ(decoded.data<const float>()[0], 9.f);
}

TEST(GGUFMultimodal, ModernAudioHistorySwitchEditAndRollback) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    InputsEmbedder embedder(gemma4_models(), tokenizer, "CPU", {});
    embedder.set_apply_chat_template_status(false);
    embedder.set_audio_encoder(first_sample_audio_features);
    auto registry = std::make_shared<VisionRegistry>();
    const auto audio = [](float value) {
        ov::Tensor tensor(ov::element::f32, {1});
        tensor.data<float>()[0] = value;
        return tensor;
    };
    // Embeds the whole normalized history with the audios the chat context resolved for it.
    const auto count_features = [&](const VLMChatContext::ProcessedChatData& data, float value) {
        std::string prompt;
        for (size_t i = 0; i < data.normalized_history.size(); ++i)
            prompt += data.normalized_history[i]["content"].get_string();
        VLMPerfMetrics metrics;
        const auto features = embedder.get_inputs_embeds(
            prompt, {}, {}, data.encoded_audios, metrics, true, {}, {}, data.audio_sequence, {});
        return std::count(features.data<const float>(), features.data<const float>() + features.get_size(), value);
    };
    ChatHistory first({{{"role", "user"}, {"content", "a<|audio|>b"}}});
    auto first_data = VLMChatContext(first, registry, embedder).process({}, {}, {}, {audio(7.f)});
    EXPECT_EQ(count_features(first_data, 7.f), 8);
    ChatHistory second({{{"role", "user"}, {"content", "<|audio|>"}}});
    auto second_data = VLMChatContext(second, registry, embedder).process({}, {}, {}, {audio(9.f)});
    EXPECT_EQ(count_features(second_data, 9.f), 8);
    first.push_back({{"role", "assistant"}, {"content", "a"}});
    first.push_back({{"role", "user"}, {"content", "b"}});
    first_data = VLMChatContext(first, registry, embedder).process({});
    EXPECT_EQ(count_features(first_data, 7.f), 8);
    EXPECT_EQ(count_features(first_data, 9.f), 0);
    first.push_back({{"role", "user"}, {"content", "<|audio|>"}});
    VLMChatContext cancelled(first, registry, embedder);
    auto cancelled_data = cancelled.process({}, {}, {}, {audio(11.f)});
    EXPECT_EQ(count_features(cancelled_data, 11.f), 8);
    cancelled.rollback();
    first.pop_back();
    first_data = VLMChatContext(first, registry, embedder).process({});
    EXPECT_EQ(count_features(first_data, 7.f), 8);
    EXPECT_EQ(count_features(first_data, 11.f), 0);
    first[0]["content"] = "b<|audio|>a";
    first.pop_back();
    first.pop_back();
    first_data = VLMChatContext(first, registry, embedder).process({}, {}, {}, {audio(13.f)});
    EXPECT_EQ(count_features(first_data, 13.f), 8);
    EXPECT_EQ(count_features(first_data, 7.f), 0);
    first[0]["content"] = "a<|audio|>";
    VLMChatContext edited(first, registry, embedder);
    edited.process({}, {}, {}, {audio(17.f)});
    edited.rollback();
    EXPECT_TRUE(ChatHistoryInternalState::get_or_create(first, registry)->get_messages_metadata().empty());
}

TEST(GGUFMultimodal, ReusingTextEmbeddingsDoesNotAccumulateScaling) {
    using namespace ov::genai;
    auto model = gemma4_models().at("text_embeddings");
    const auto original_output = model->output().get_node_shared_ptr();
    ov::Tensor ids(ov::element::i64, {1, 1});
    ids.data<int64_t>()[0] = 0;
    for (float scale : {2.f, 3.f}) {
        EmbeddingsModel embeddings(model, scale, "CPU", {});
        CircularBufferQueueElementGuard<EmbeddingsRequest> guard(embeddings.get_request_queue().get());
        auto& request = guard.get();
        const auto output = embeddings.infer(request, ids);
        EXPECT_FLOAT_EQ(output.data<float>()[0], .25f * scale);
        EXPECT_EQ(model->output().get_node_shared_ptr(), original_output);
    }
}

TEST(GGUFMultimodal, Gemma3PositionOffsetAppliesDuringPrefillAndDecode) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    auto models = gemma4_models();
    models.config.model_type = VLMModelType::GEMMA3;
    models.config.image_soft_token = "<pad>";
    for (size_t offset : {0u, 1u}) {
        models.config.position_ids_offset = offset;
        InputsEmbedderGemma3 embedder(models, tokenizer, "CPU", {});
        const auto prefill = embedder.get_position_ids(3, 0).first;
        const auto decode = embedder.get_generation_phase_position_ids(1, 3, 0).first;
        EXPECT_EQ(prefill.data<int64_t>()[0], offset);
        EXPECT_EQ(prefill.data<int64_t>()[2], 2 + offset);
        EXPECT_EQ(decode.data<int64_t>()[0], 3 + offset);
    }
}

TEST(GGUFMultimodal, BoundedResizeKeepsNativeAreaBetweenTokenLimits) {
    EXPECT_EQ(bounded_image_size(488, 640, 48, 70 * 48 * 48, 1120 * 48 * 48), (std::pair<size_t, size_t>{480, 624}));
    const auto small = bounded_image_size(1, 100, 48, 70 * 48 * 48, 1120 * 48 * 48);
    EXPECT_GT(small.first, 0);
    EXPECT_GT(small.second, 0);
    EXPECT_EQ(small.first % 48, 0);
    EXPECT_EQ(small.second % 48, 0);
    EXPECT_ANY_THROW(bounded_image_size(0, 100, 48, 1, 100));
}

TEST(GGUFMultimodal, Gemma3ImageSeparatorFollowsModelConfig) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    auto models = gemma4_models();
    models.config.model_type = VLMModelType::GEMMA3;
    models.config.image_soft_token = "<pad>";
    EncodedImage image{ov::Tensor(ov::element::f32, {1, 2, 4})};
    for (const std::string separator : {std::string{}, std::string{"\n\n"}}) {
        models.config.image_separator = separator;
        InputsEmbedderGemma3 embedder(models, tokenizer, "CPU", {});
        const auto prompt = embedder.normalize_prompt("<ov_genai_image_0>a", 0, {image});
        EXPECT_EQ(prompt.unified_prompt, separator + "<start_of_image><pad><pad><end_of_image>" + separator + "a");
    }
}

TEST(GGUFMultimodal, MuseVideoKeepsFramesWhenSamplingIsDisabled) {
    using namespace ov::genai;
    auto models = gemma4_models();
    models.config.model_type = VLMModelType::MUSE_GLIMMER;
    models.processor_config.patch_size = 2;
    models.processor_config.merge_size = 2;
    models.processor_config.temporal_patch_size = 1;
    models.video_processor_config.patch_size = 2;
    models.video_processor_config.merge_size = 2;
    models.video_processor_config.temporal_patch_size = 1;
    models.video_processor_config.do_sample_frames = false;
    auto pixels = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 12});
    pixels->output(0).set_names({"pixel_values"});
    auto grid = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3});
    grid->output(0).set_names({"image_grid_thw"});
    models.models["vision_embeddings"] =
        std::make_shared<ov::Model>(ov::OutputVector{pixels}, ov::ParameterVector{pixels, grid});
    InputsEmbedderMuseGlimmer embedder(models, GGUFTokenizer(sentencepiece_config()), "CPU", {});
    ov::Tensor video(ov::element::u8, {3, 4, 4, 3});
    std::memset(video.data(), 0, video.get_byte_size());
    VideoMetadata metadata;
    metadata.fps = 2.f;
    const auto encoded = embedder.encode_videos({video}, {metadata});
    ASSERT_EQ(encoded.size(), 1);
    EXPECT_EQ(encoded[0].frame_num, 3);
    EXPECT_EQ(encoded[0].metadata.frames_indices, (std::vector<size_t>{0, 1, 2}));
    EXPECT_FLOAT_EQ(encoded[0].metadata.fps, 2.f);
    const auto assumed_fps = embedder.encode_videos({video});
    EXPECT_EQ(assumed_fps[0].frame_num, 3);
    EXPECT_GT(assumed_fps[0].metadata.fps, 0.f);
}

TEST(GGUFMultimodal, ImageOnlyProcessorConfigIsLoadedForBothModalities) {
    using namespace ov::genai;
    struct ConfigEncoder : VisionEncoder {
        explicit ConfigEncoder(const std::filesystem::path& path) : VisionEncoder(path, ConfigOnlyTag{}) {}
        EncodedImage encode(const ov::Tensor&, const ov::AnyMap&) override {
            return {};
        }
    };
    ConfigEncoder encoder(std::filesystem::path(__FILE__).parent_path() / "data" / "image_only_processor");
    const auto image = encoder.get_processor_config();
    const auto video = encoder.get_video_processor_config();
    EXPECT_EQ(image.size_height, 32);
    EXPECT_EQ(image.size_width, 48);
    EXPECT_EQ(video.size_height, image.size_height);
    EXPECT_EQ(video.size_width, image.size_width);
    EXPECT_EQ(video.patch_size, image.patch_size);
}

TEST(GGUFMultimodal, LFM2AudioPlacementPreservesAdjacentAndReversedClips) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    auto models = gemma4_models();
    models.config.model_type = VLMModelType::LFM2_AUDIO;
    models.models.erase("vision_embeddings");
    InputsEmbedderLFM2Audio embedder(models, tokenizer, "CPU", {});
    embedder.set_apply_chat_template_status(false);
    embedder.set_audio_encoder(first_sample_audio_features);
    const auto sample = [](float value) {
        ov::Tensor audio(ov::element::f32, {1});
        audio.data<float>()[0] = value;
        return audio;
    };
    const auto encoded = embedder.encode_audios({sample(7.f), sample(9.f)});
    const auto placed = [&](const NormalizedPrompt& prompt, size_t base) {
        auto sequence = prompt.audios_sequence;
        vlm_utils::rebase_media_sequence(sequence, base);
        VLMPerfMetrics metrics;
        const auto features =
            embedder.get_inputs_embeds(prompt.unified_prompt, {}, {}, encoded, metrics, true, {}, {}, sequence, {});
        std::vector<float> values;
        for (size_t i = 0; i < features.get_size(); i += 4)
            if (features.data<const float>()[i] != 0.25f)
                values.push_back(features.data<const float>()[i]);
        return values;
    };
    const auto native = embedder.normalize_prompt("a<audio><audio>b", 0, 0, 0, {}, {}, encoded);
    EXPECT_EQ(native.audios_sequence, (std::vector<size_t>{0, 1}));
    EXPECT_EQ(native.unified_prompt, "a<|audio_start|><|audio_start|><|audio_start|><|audio_start|>b");
    EXPECT_EQ(placed(native, 0), (std::vector<float>{7.f, 7.f, 9.f, 9.f}));
    const auto reversed = embedder.normalize_prompt("<ov_genai_audio_3><ov_genai_audio_2>", 0, 0, 2, {}, {}, encoded);
    EXPECT_EQ(reversed.audios_sequence, (std::vector<size_t>{3, 2}));
    EXPECT_EQ(placed(reversed, 2), (std::vector<float>{9.f, 9.f, 7.f, 7.f}));
    EXPECT_EQ(embedder.normalize_prompt("ab", 0, 0, 0, {}, {}, {}).unified_prompt, "ab");
    EXPECT_ANY_THROW(embedder.normalize_prompt("<ov_genai_audio_0>", 0, 0, 0, {}, {}, {}));
    EXPECT_ANY_THROW(embedder.normalize_prompt("<|audio_start|><audio><audio>", 0, 0, 0, {}, {}, encoded));
    EXPECT_ANY_THROW(embedder.encode_audios({ov::Tensor(ov::element::f32, {0})}));
    EXPECT_ANY_THROW(embedder.encode_images({ov::Tensor(ov::element::u8, {1, 2, 2, 3})}));
    EXPECT_TRUE(embedder.encode_audios({}).empty());
}

TEST(GGUFMultimodal, LFM2AudioPreprocessingMatchesLiquidReference) {
    using namespace ov::opset15;
    using namespace ov::genai;
    const auto directory = std::filesystem::path(__FILE__).parent_path() / "data" / "lfm2_audio";
    auto features = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 1, 128, -1});
    features->output(0).set_names({"features"});
    auto positions = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 512});
    positions->output(0).set_names({"pos_emb"});
    auto transposed = std::make_shared<Transpose>(features, Constant::create(ov::element::i64, {4}, {0, 1, 3, 2}));
    auto output = std::make_shared<Reshape>(transposed, Constant::create(ov::element::i64, {3}, {1, -1, 128}), false);
    auto dependency = std::make_shared<Multiply>(
        std::make_shared<ReduceSum>(positions, Constant::create(ov::element::i64, {4}, {0, 1, 2, 3}), false),
        Constant::create(ov::element::f32, {}, {0.f}));
    auto result = std::make_shared<Add>(output, dependency);
    result->output(0).set_names({"audio_features"});
    auto models = gemma4_models();
    models.models["audio_embeddings"] =
        std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{features, positions});
    models.audio_encoder = AudioEncoderConfig{AudioEncoderType::LFM2, 128};
    models.config.model_type = VLMModelType::LFM2_AUDIO;
    models.config.hidden_size = 128;
    models.models.erase("vision_embeddings");
    auto ids = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    ids->output(0).set_names({"input_ids"});
    auto text = std::make_shared<Gather>(Constant::create(ov::element::f32, {vocab_size, 128}, {0.25f}),
                                         ids,
                                         Constant::create(ov::element::i64, {}, {0}));
    models.models["text_embeddings"] = std::make_shared<ov::Model>(ov::OutputVector{text}, ov::ParameterVector{ids});
    InputsEmbedder embedder(models,
                            GGUFTokenizer(sentencepiece_config()),
                            "CPU",
                            {{ov::hint::inference_precision.name(), ov::element::f32}});
    for (size_t count : {160, 1281, 16000}) {
        const auto load = [&](const std::string& name, void* data, size_t bytes) {
            std::ifstream stream(directory / name, std::ios::binary);
            ASSERT_TRUE(stream.is_open()) << name;
            ASSERT_TRUE(bool(stream.read(static_cast<char*>(data), bytes))) << name;
            EXPECT_EQ(stream.peek(), std::char_traits<char>::eof());
        };
        ov::Tensor waveform(ov::element::f32, {count});
        load("waveform-" + std::to_string(count) + ".bin", waveform.data(), waveform.get_byte_size());
        const auto frames = count / 160 + 1;
        std::vector<float> reference(128 * frames);
        load("features-" + std::to_string(count) + ".bin", reference.data(), reference.size() * sizeof(float));
        const auto encoded = embedder.encode_audios({waveform});
        ASSERT_EQ(encoded.size(), 1);
        ASSERT_EQ(encoded[0].audio_features.get_shape(), (ov::Shape{frames, 128}));
        double error = 0., energy = 0.;
        for (size_t t = 0; t < frames; ++t)
            for (size_t m = 0; m < 128; ++m) {
                const auto expected = reference[m * frames + t];
                const auto actual = encoded[0].audio_features.data<const float>()[t * 128 + m];
                error += (actual - expected) * (actual - expected);
                energy += expected * expected;
                EXPECT_NEAR(actual, expected, 3e-4f) << "samples=" << count << " frame=" << t << " mel=" << m;
            }
        EXPECT_LT(error / std::max(energy, 1.), 1e-8) << "samples=" << count;
    }
    EXPECT_ANY_THROW(embedder.encode_audios({ov::Tensor(ov::element::f32, {159})}));
    EXPECT_ANY_THROW(embedder.encode_audios({ov::Tensor(ov::element::f32, {2, 160})}));
}

TEST(GGUFMultimodal, LFM2VisionTiledMarkersAndFeaturesFollowPromptOrder) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    auto models = gemma4_models();
    models.config.model_type = VLMModelType::LFM2_VL;
    InputsEmbedderLFM2Vision embedder(models, tokenizer, "CPU", {});
    embedder.set_apply_chat_template_status(false);
    const auto image = [](float value, size_t tokens) {
        EncodedImage result;
        result.resized_source = ov::Tensor(ov::element::f32, {1, tokens, 4});
        std::fill_n(result.resized_source.data<float>(), result.resized_source.get_size(), value);
        result.num_image_tokens = tokens;
        return result;
    };
    auto tiled = image(7.f, 6);
    tiled.slices_grid = {1, 2};
    tiled.crop_sizes = {{2, 1}, {2, 1}, {2, 1}};
    auto overview = image(9.f, 1);
    overview.crop_sizes = {{1, 1}};
    const std::vector<EncodedImage> images{tiled, overview};
    const auto normalized = embedder.normalize_prompt("<ov_genai_image_3>a<ov_genai_image_2>", 2, images);
    EXPECT_EQ(normalized.images_sequence, (std::vector<size_t>{3, 2}));
    EXPECT_EQ(normalized.unified_prompt,
              "<|image_start|><|image|><|image_end|>a<|image_start|>"
              "<|img_row_1_col_1|><|image|><|image|><|img_row_1_col_2|><|image|><|image|>"
              "<|img_thumbnail|><|image|><|image|><|image_end|>");
    VLMPerfMetrics metrics;
    auto sequence = normalized.images_sequence;
    vlm_utils::rebase_media_sequence(sequence, 2);
    const auto embeds = embedder.get_inputs_embeds(normalized.unified_prompt, images, metrics, true, sequence);
    std::vector<float> values;
    for (size_t i = 0; i < embeds.get_size(); i += 4)
        if (embeds.data<const float>()[i] != .25f)
            values.push_back(embeds.data<const float>()[i]);
    EXPECT_EQ(values, (std::vector<float>{9.f, 7.f, 7.f, 7.f, 7.f, 7.f, 7.f}));
}

TEST(GGUFTokenizerSpecialTokens, LFM2WaveformReconstructionMatchesLiquidReference) {
    using namespace ov::opset15;
    const auto directory = std::filesystem::path(__FILE__).parent_path() / "data" / "lfm2_audio";
    const auto load = [&](const std::string& name, void* data, size_t bytes) {
        std::ifstream stream(directory / name, std::ios::binary);
        ASSERT_TRUE(stream.is_open()) << name;
        ASSERT_TRUE(bool(stream.read(static_cast<char*>(data), bytes))) << name;
        EXPECT_EQ(stream.peek(), std::char_traits<char>::eof());
    };
    std::vector<float> window(1280);
    load("istft-window.bin", window.data(), window.size() * sizeof(float));
    auto input = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{-1, 641, 2});
    auto output = std::make_shared<IRDFT>(input,
                                          Constant::create(ov::element::i64, {1}, {1}),
                                          Constant::create(ov::element::i64, {1}, {1280}));
    auto request = ov::Core{}
                       .compile_model(std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input}),
                                      "CPU",
                                      ov::hint::inference_precision(ov::element::f32))
                       .create_infer_request();
    for (size_t frames : {1, 18, 37, 18}) {
        SCOPED_TRACE(frames);
        ov::Tensor spectral(ov::element::f32, {1, frames, 1282});
        load("spectral-" + std::to_string(frames) + ".bin", spectral.data(), spectral.get_byte_size());
        std::vector<float> reference(frames * 320);
        load("istft-" + std::to_string(frames) + ".bin", reference.data(), reference.size() * sizeof(float));
        auto actual = ov::genai::lfm2_spectrum_to_waveform(spectral, window, request);
        ASSERT_EQ(actual.get_shape(), (ov::Shape{1, frames * 320}));
        double error = 0., energy = 0.;
        for (size_t i = 0; i < reference.size(); ++i) {
            const auto delta = double(actual.data<const float>()[i]) - reference[i];
            error += delta * delta;
            energy += double(reference[i]) * reference[i];
        }
        EXPECT_LT(error / energy, 1e-8);
    }
}

TEST(GGUFMultimodal, LFM2VisionNormalizesPixelsAndPadsTiledCanvas) {
    using namespace ov::genai;
    using namespace ov::opset15;
    const auto models_for_pixels = [] {
        auto models = gemma4_models();
        models.config.model_type = VLMModelType::LFM2_VL;
        models.processor_config.patch_size = 1;
        models.processor_config.merge_size = 1;
        models.processor_config.min_pixels = 1;
        models.processor_config.max_pixels = 16;
        models.processor_config.tile_size = 4;
        models.processor_config.min_patches = 2;
        models.processor_config.max_patches = 2;
        models.processor_config.image_mean = {0.5f, 0.5f, 0.5f};
        models.processor_config.image_std = {0.5f, 0.5f, 0.5f};
        auto pixels = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 3, -1, -1});
        pixels->output(0).set_names({"pixel_values"});
        auto indices = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{1, 1, 1, -1});
        indices->output(0).set_names({"merge_indices"});
        auto flat = std::make_shared<Reshape>(pixels, Constant::create(ov::element::i64, {3}, {1, 3, -1}), false);
        auto output = std::make_shared<Transpose>(flat, Constant::create(ov::element::i64, {3}, {0, 2, 1}));
        auto dependency = std::make_shared<Multiply>(
            std::make_shared<Convert>(
                std::make_shared<ReduceSum>(indices, Constant::create(ov::element::i64, {4}, {0, 1, 2, 3}), false),
                ov::element::f32),
            Constant::create(ov::element::f32, {}, {0.f}));
        auto result = std::make_shared<Add>(output, dependency);
        models.models["vision_embeddings"] =
            std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{pixels, indices});
        return models;
    };
    VisionEncoderLFM2 encoder(models_for_pixels(), "CPU", {{ov::hint::inference_precision.name(), ov::element::f32}});
    ov::Tensor image(ov::element::u8, {1, 2, 3, 3});
    for (size_t i = 0; i < 6; ++i) {
        image.data<uint8_t>()[i * 3] = 255;
        image.data<uint8_t>()[i * 3 + 1] = 0;
        image.data<uint8_t>()[i * 3 + 2] = 128;
    }
    const auto encoded = encoder.encode(image);
    ASSERT_EQ(encoded.num_image_tokens, 6);
    ASSERT_EQ(encoded.resized_source.get_shape(), (ov::Shape{1, 6, 3}));
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(encoded.resized_source.data<const float>()[i * 3], 1.f);
        EXPECT_FLOAT_EQ(encoded.resized_source.data<const float>()[i * 3 + 1], -1.f);
        EXPECT_NEAR(encoded.resized_source.data<const float>()[i * 3 + 2], 1.f / 255.f, 1e-7f);
    }
    ov::Tensor tiled(ov::element::u8, {1, 4, 12, 3});
    std::fill_n(tiled.data<uint8_t>(), tiled.get_size(), 255);
    auto output = encoder.encode(tiled);
    ASSERT_TRUE(output.slices_grid.has_slices());
    ASSERT_EQ(output.crop_sizes.size(), 3);
    ASSERT_EQ(output.crop_sizes[0].height, 16);
    ASSERT_EQ(output.crop_sizes[1].height, 16);
    // The 12x4 source is resized to 8x3 and padded to the 8x4 canvas.
    for (size_t tile = 0; tile < 2; ++tile)
        for (size_t y = 0; y < 4; ++y)
            for (size_t x = 0; x < 4; ++x)
                for (size_t c = 0; c < 3; ++c)
                    EXPECT_FLOAT_EQ(output.resized_source.data<const float>()[((tile * 16 + y * 4 + x) * 3) + c],
                                    y < 3 ? 1.f : -1.f);
}

TEST(GGUFTokenizer, MetadataRejectsModelsWithoutTokenizerInfo) {
    EXPECT_THROW(ov::genai::GGUFTokenizerParameters::from_model(nullptr), ov::Exception);
}
