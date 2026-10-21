// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/serialize.hpp"
#include "tokenizer/tokenizer_impl.hpp"
#include "visual_language/muse_glimmer/classes.hpp"
#include "visual_language/vlm_chat_context.hpp"

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

std::pair<std::string, ov::Tensor> serialize_model(const std::shared_ptr<ov::Model>& model) {
    std::ostringstream xml, weights;
    ov::pass::Serialize(xml, weights).run_on_model(model);
    const auto weights_data = weights.str();
    ov::Tensor weights_tensor(ov::element::u8, {weights_data.size()});
    if (!weights_data.empty())
        std::memcpy(weights_tensor.data(), weights_data.data(), weights_data.size());
    return {xml.str(), std::move(weights_tensor)};
}

ov::genai::ModelsMap video_models() {
    namespace op = ov::opset13;
    auto pixels = std::make_shared<op::Parameter>(ov::element::f32, ov::PartialShape{1, 6});
    pixels->output(0).get_tensor().set_names({"pixel_values"});
    auto grid = std::make_shared<op::Parameter>(ov::element::i64, ov::PartialShape{1, 3});
    grid->output(0).get_tensor().set_names({"image_grid_thw"});
    auto grid_sum = std::make_shared<op::ReduceSum>(grid, op::Constant::create(ov::element::i64, {2}, {0, 1}));
    auto features = std::make_shared<op::Add>(pixels, std::make_shared<op::Convert>(grid_sum, ov::element::f32));
    auto vision = std::make_shared<ov::Model>(ov::OutputVector{features}, ov::ParameterVector{pixels, grid});

    auto input_ids = std::make_shared<op::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    auto embeddings = std::make_shared<op::Unsqueeze>(std::make_shared<op::Convert>(input_ids, ov::element::f32),
                                                      op::Constant::create(ov::element::i64, {1}, {2}));
    auto text = std::make_shared<ov::Model>(ov::OutputVector{embeddings}, ov::ParameterVector{input_ids});
    return {{"vision_embeddings", serialize_model(vision)}, {"text_embeddings", serialize_model(text)}};
}

class MuseGlimmerVideo : public testing::Test {
protected:
    static void SetUpTestSuite() {
        models = video_models();
        config_dir =
            std::filesystem::temp_directory_path() /
            ("genai-muse-video-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        ASSERT_TRUE(std::filesystem::create_directory(config_dir));
        std::ofstream(config_dir / "config.json") << R"({"model_type":"muse_glimmer"})";
    }

    static void TearDownTestSuite() {
        const auto config_file = config_dir / "processor_config.json";
        const auto model_config_file = config_dir / "config.json";
        const auto entries =
            std::vector<std::filesystem::directory_entry>(std::filesystem::directory_iterator(config_dir),
                                                          std::filesystem::directory_iterator{});
        ASSERT_EQ(entries.size(), 2);
        for (const auto& entry : entries) {
            ASSERT_TRUE(entry.path() == config_file || entry.path() == model_config_file);
            ASSERT_TRUE(entry.is_regular_file());
        }
        EXPECT_TRUE(std::filesystem::remove(config_file));
        EXPECT_TRUE(std::filesystem::remove(model_config_file));
        EXPECT_TRUE(std::filesystem::remove(config_dir));
        models.clear();
    }

    std::unique_ptr<ov::genai::InputsEmbedderMuseGlimmer> embedder(bool sample_frames,
                                                                   float processor_fps = 2.0f,
                                                                   size_t num_frames = 2) {
        std::ofstream stream(config_dir / "processor_config.json");
        stream << nlohmann::json{
            {"image_processor", {{"patch_size", 1}, {"merge_size", 1}, {"temporal_patch_size", 2}}},
            {"video_processor",
             {{"patch_size", 1},
              {"merge_size", 1},
              {"temporal_patch_size", 2},
              {"max_video_frame_tokens", 1},
              {"do_sample_frames", sample_frames},
              {"fps", processor_fps},
              {"num_frames", num_frames}}}};
        stream.close();
        ov::genai::VLMConfig config;
        config.model_type = ov::genai::VLMModelType::MUSE_GLIMMER;
        return std::make_unique<ov::genai::InputsEmbedderMuseGlimmer>(config,
                                                                      models,
                                                                      ov::genai::Tokenizer{},
                                                                      config_dir,
                                                                      "CPU",
                                                                      ov::AnyMap{});
    }

    ov::Tensor video(size_t num_frames = 8) {
        ov::Tensor frames(ov::element::u8, {num_frames, 1, 1, 3});
        for (size_t frame = 0; frame < num_frames; ++frame)
            std::fill_n(frames.data<uint8_t>() + frame * 3, 3, static_cast<uint8_t>(frame * 20));
        return frames;
    }

    void expect_encoded_frames(const ov::genai::EncodedVideo& encoded, const std::vector<size_t>& indices) {
        ASSERT_EQ(encoded.metadata.frames_indices, indices);
        EXPECT_EQ(encoded.frame_num, indices.size() / 2);
        EXPECT_EQ(encoded.num_video_tokens, indices.size() / 2);
        ASSERT_EQ(encoded.video_features.get_shape(), (ov::Shape{1, indices.size() / 2, 6}));
        const auto* features = encoded.video_features.data<float>();
        for (size_t idx = 0; idx < indices.size(); ++idx)
            for (size_t channel = 0; channel < 3; ++channel)
                EXPECT_NEAR(features[idx * 3 + channel], 3.0f + static_cast<float>(indices[idx] * 20) / 255.0f, 1e-5f);
    }

    inline static std::filesystem::path config_dir;
    inline static ov::genai::ModelsMap models;
};

}  // namespace

TEST(VLMChatTemplate, UnsupportedUndefinedTestIsRetried) {
    auto tokenizer =
        template_tokenizer("{% if missing is undefined %}{{ bos_token }}{{ messages[0].content }}{% endif %}");
    EXPECT_EQ(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt), "<bos>hello");
    EXPECT_EQ(tokenizer->get_chat_template(),
              "{% if missing is undefined %}{{ bos_token }}{{ messages[0].content }}{% endif %}");
}

TEST(VLMChatTemplate, AdjacentMultilineLiteralsAreRetried) {
    auto tokenizer = template_tokenizer("{{ \"first\"\n    \"second\" }}{{ messages[0].content }}");
    EXPECT_EQ(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt),
              "firstsecondhello");
}

TEST(VLMChatTemplate, FallbackUsesExplicitContextAndTools) {
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

TEST(VLMChatTemplate, SuccessfulRenderingPreservesLiteralText) {
    auto tokenizer = template_tokenizer("{{ \"missing is undefined\" }}|{{ messages[0].content }}");
    EXPECT_EQ(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt),
              "missing is undefined|hello");
}

TEST(VLMChatTemplate, FallbackPreservesUndefinedTextInsideAndOutsideExpressions) {
    auto tokenizer = template_tokenizer("x is undefined|{% if missing is undefined %}{{ \"x is undefined\" }}"
                                        "{{ \"one is undefined\"\n\"two\" }}{% endif %}");
    EXPECT_EQ(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt),
              "x is undefined|x is undefinedone is undefinedtwo");
}

TEST(VLMChatTemplate, FallbackPreservesComments) {
    auto tokenizer = template_tokenizer("{# {{ x is undefined }} #}{% if missing is undefined %}ok{% endif %}");
    EXPECT_EQ(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt), "ok");
}

TEST(VLMChatTemplate, UndefinedRewritePreservesRawBlocks) {
    EXPECT_EQ(ov::genai::utils::replace_is_undefined_tests(
                  "{% raw %}{{ x is undefined }}{% endraw %}{% if missing is undefined %}ok{% endif %}"),
              "{% raw %}{{ x is undefined }}{% endraw %}{% if missing is not defined %}ok{% endif %}");
}

TEST(VLMChatTemplate, FallbackPreservesMultilineQuotesOutsideExpressions) {
    auto tokenizer = template_tokenizer("\"line one\"\n\"line two\"|{% if missing is undefined %}ok{% endif %}");
    EXPECT_EQ(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt),
              "\"line one\"\n\"line two\"|ok");
}

TEST(VLMChatTemplate, LiteralJoinPreservesRawBlocks) {
    EXPECT_EQ(ov::genai::utils::join_multiline_string_literals(
                  "{% raw %}\"line one\"\n\"line two\"{% endraw %}{{ \"joined\"\n\"literal\" }}"),
              "{% raw %}\"line one\"\n\"line two\"{% endraw %}{{ \"joinedliteral\" }}");
}

TEST(VLMChatTemplate, UndefinedRewriteMatchesOnlyCompleteTestName) {
    EXPECT_EQ(ov::genai::utils::replace_is_undefined_tests(
                  "{{ x is undefined_value }}{{ (x is undefined) }}{{ message.content is undefined }}"),
              "{{ x is undefined_value }}{{ (x is not defined) }}{{ message.content is not defined }}");
}

TEST(VLMChatTemplate, UndefinedFallbackPreservesIndexedOperands) {
    for (const std::string operand : {"messages[0].missing", "messages[0][\"missing\"]", "messages[0]['missing']"}) {
        auto tokenizer = template_tokenizer("{% if " + operand + " is undefined %}missing{% endif %}");
        EXPECT_EQ(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt), "missing");
    }
}

TEST(VLMChatTemplate, UndefinedFallbackDoesNotTreatPresentValuesAsMissing) {
    auto tokenizer = template_tokenizer(
        "{% if messages[0].content is undefined %}missing{% else %}{{ messages[0].content }}{% endif %}");
    EXPECT_EQ(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt), "hello");
}

TEST(VLMChatTemplate, InvalidTemplateStillReportsAnError) {
    auto tokenizer = template_tokenizer("{% if missing is undefined %}{{ unknown_function() }}{% endif %}");
    EXPECT_THROW(tokenizer->apply_chat_template(chat_history(), false, "", std::nullopt, std::nullopt), ov::Exception);
}

TEST_F(MuseGlimmerVideo, SamplingEnabledKeepsExistingSelection) {
    auto encoder = embedder(true);
    ov::genai::VideoMetadata metadata;
    metadata.fps = 8.0f;
    const auto encoded = encoder->encode_videos({video()}, {metadata});
    ASSERT_EQ(encoded.size(), 1);
    expect_encoded_frames(encoded.front(), {0, 7});
    EXPECT_FLOAT_EQ(encoded.front().metadata.fps, 8.0f);
}

TEST_F(MuseGlimmerVideo, SamplingDisabledPreservesAllFrames) {
    auto encoder = embedder(false);
    ov::genai::VideoMetadata metadata;
    metadata.fps = 8.0f;
    const auto encoded = encoder->encode_videos({video()}, {metadata});
    ASSERT_EQ(encoded.size(), 1);
    expect_encoded_frames(encoded.front(), {0, 1, 2, 3, 4, 5, 6, 7});
    EXPECT_FLOAT_EQ(encoded.front().metadata.fps, 8.0f);
}

TEST_F(MuseGlimmerVideo, SamplingDisabledPreservesPreselectedFrames) {
    auto encoder = embedder(false, 0.0f, 0);
    ov::genai::VideoMetadata metadata;
    metadata.fps = 6.0f;
    metadata.frames_indices = {1, 2, 4, 6};
    const auto encoded = encoder->encode_videos({video()}, {metadata});
    ASSERT_EQ(encoded.size(), 1);
    expect_encoded_frames(encoded.front(), metadata.frames_indices);
    EXPECT_FLOAT_EQ(encoded.front().metadata.fps, 6.0f);
}

TEST_F(MuseGlimmerVideo, SamplingDisabledUsesProcessorFpsFallback) {
    auto encoder = embedder(false);
    const auto encoded = encoder->encode_videos({video()});
    ASSERT_EQ(encoded.size(), 1);
    expect_encoded_frames(encoded.front(), {0, 1, 2, 3, 4, 5, 6, 7});
    EXPECT_FLOAT_EQ(encoded.front().metadata.fps, 2.0f);
}

TEST_F(MuseGlimmerVideo, SamplingDisabledUsesDefaultFpsFallback) {
    auto encoder = embedder(false, 0.0f, 0);
    const auto encoded = encoder->encode_videos({video()});
    ASSERT_EQ(encoded.size(), 1);
    expect_encoded_frames(encoded.front(), {0, 1, 2, 3, 4, 5, 6, 7});
    EXPECT_FLOAT_EQ(encoded.front().metadata.fps, 24.0f);
}

TEST_F(MuseGlimmerVideo, SamplingDisabledRejectsIncompleteTemporalGroups) {
    auto encoder = embedder(false);
    EXPECT_THROW(encoder->encode_videos({video(3)}), ov::Exception);
    ov::genai::VideoMetadata metadata;
    metadata.frames_indices = {0, 1, 2};
    EXPECT_THROW(encoder->encode_videos({video()}, {metadata}), ov::Exception);
}

TEST_F(MuseGlimmerVideo, EditedHistoryRollbackRetainsOnlyMatchingPrefix) {
    auto configured = embedder(false);
    ov::genai::InputsEmbedder inputs(models, ov::genai::Tokenizer{}, config_dir, "CPU", {});
    auto registry = std::make_shared<ov::genai::VisionRegistry>();
    ov::genai::ChatHistory history{{{"role", "system"}, {"content", "unchanged"}},
                                   {{"role", "user"}, {"content", "<|video|> original"}}};
    {
        ov::genai::VLMChatContext initial(history, registry, inputs);
        initial.process({}, {video()});
    }
    auto state = ov::genai::ChatHistoryInternalState::get_or_create(history, registry);
    ASSERT_EQ(state->get_messages_metadata().size(), 2);
    ASSERT_EQ(state->get_base_video_index(), 1);
    ASSERT_EQ(registry->size(), 1);
    const auto original_id = state->get_video_vision_id(0);
    history.get_messages()[1]["content"] = "<|video|> edited";
    auto replacement = video();
    std::fill_n(replacement.data<uint8_t>(), replacement.get_size(), 100);
    ov::genai::VLMChatContext edited(history, registry, inputs);
    const auto processed = edited.process({}, {replacement});
    EXPECT_TRUE(processed.needs_kv_cache_reset);
    const auto replacement_id = state->get_video_vision_id(0);
    EXPECT_FALSE(registry->contains(original_id));
    edited.rollback();
    ASSERT_EQ(state->get_messages_metadata().size(), 1);
    EXPECT_EQ(state->get_messages_metadata().front().original_message, history.first());
    EXPECT_EQ(state->get_base_video_index(), 0);
    EXPECT_EQ(registry->size(), 0);
    EXPECT_FALSE(registry->contains(replacement_id));
}
