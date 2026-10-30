// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/serialize.hpp"
#include "visual_language/muse_glimmer/classes.hpp"
#include "visual_language/vlm_chat_context.hpp"

namespace {

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

TEST_F(MuseGlimmerVideo, SamplingDisabledRejectsMissingSourceFps) {
    auto encoder = embedder(false);
    EXPECT_THROW(encoder->encode_videos({video()}), ov::Exception);
}

TEST_F(MuseGlimmerVideo, SamplingDisabledRejectsInvalidSourceFps) {
    auto encoder = embedder(false, 0.0f, 0);
    for (float fps : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
        ov::genai::VideoMetadata metadata;
        metadata.fps = fps;
        EXPECT_THROW(encoder->encode_videos({video()}, {metadata}), ov::Exception);
    }
}

TEST_F(MuseGlimmerVideo, SamplingDisabledRejectsIncompleteTemporalGroups) {
    auto encoder = embedder(false);
    ov::genai::VideoMetadata metadata;
    metadata.fps = 8.0f;
    EXPECT_THROW(encoder->encode_videos({video(3)}, {metadata}), ov::Exception);
    metadata.frames_indices = {0, 1, 2};
    EXPECT_THROW(encoder->encode_videos({video()}, {metadata}), ov::Exception);
}

TEST_F(MuseGlimmerVideo, EditedHistoryRollbackRetainsOnlyMatchingPrefix) {
    auto configured = embedder(false);
    ov::genai::InputsEmbedder inputs(models, ov::genai::Tokenizer{}, config_dir, "CPU", {});
    auto registry = std::make_shared<ov::genai::EmbeddingsRegistry>();
    ov::genai::ChatHistory history{{{"role", "system"}, {"content", "unchanged"}},
                                   {{"role", "user"}, {"content", "<|video|> original"}}};
    {
        ov::genai::VLMChatContext initial(history, registry, inputs);
        ov::genai::VideoMetadata metadata;
        metadata.fps = 8.0f;
        initial.process({}, {video()}, {metadata});
    }
    auto state = ov::genai::ChatHistoryInternalState::get_or_create(history, registry);
    ASSERT_EQ(state->get_messages_metadata().size(), 2);
    ASSERT_EQ(state->get_base_video_index(), 1);
    ASSERT_EQ(registry->size(), 1);
    const auto original_id = state->get_video_embedding_id(0);
    history.get_messages()[1]["content"] = "<|video|> edited";
    auto replacement = video();
    std::fill_n(replacement.data<uint8_t>(), replacement.get_size(), 100);
    ov::genai::VLMChatContext edited(history, registry, inputs);
    ov::genai::VideoMetadata metadata;
    metadata.fps = 8.0f;
    const auto processed = edited.process({}, {replacement}, {metadata});
    EXPECT_TRUE(processed.needs_kv_cache_reset);
    const auto replacement_id = state->get_video_embedding_id(0);
    EXPECT_FALSE(registry->contains(original_id));
    edited.rollback();
    ASSERT_EQ(state->get_messages_metadata().size(), 1);
    EXPECT_EQ(state->get_messages_metadata().front().original_message, history.first());
    EXPECT_EQ(state->get_base_video_index(), 0);
    EXPECT_EQ(registry->size(), 0);
    EXPECT_FALSE(registry->contains(replacement_id));
}
