// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

#include "openvino/genai/c/text2speech_pipeline.h"

namespace {

class Text2SpeechPipelineCTest : public ::testing::Test {
protected:
    void SetUp() override {
        // We'll use a mock approach or check for a small model if available
        // For unit tests in CI, usually there's a tiny model
        model_path = "tiny-random-SpeechT5ForTextToSpeech";
        if (!std::filesystem::exists(model_path)) {
            GTEST_SKIP() << "Model path " << model_path << " does not exist, skipping test.";
        }
    }

    std::string model_path;
};

TEST_F(Text2SpeechPipelineCTest, create_and_free) {
    ov_genai_text2speech_pipeline* pipe = nullptr;
    ov_status_e status = ov_genai_text2speech_pipeline_create(model_path.c_str(), "CPU", 0, &pipe);
    ASSERT_EQ(status, 0);
    ASSERT_NE(pipe, nullptr);
    ov_genai_text2speech_pipeline_free(pipe);
}

TEST_F(Text2SpeechPipelineCTest, generation_config) {
    ov_genai_text2speech_pipeline* pipe = nullptr;
    ASSERT_EQ(ov_genai_text2speech_pipeline_create(model_path.c_str(), "CPU", 0, &pipe), 0);
    
    ov_genai_speech_generation_config* config = nullptr;
    ASSERT_EQ(ov_genai_text2speech_pipeline_get_generation_config(pipe, &config), 0);
    ASSERT_NE(config, nullptr);

    float val = 0.0f;
    ASSERT_EQ(ov_genai_speech_generation_config_get_threshold(config, &val), 0);
    ASSERT_EQ(ov_genai_speech_generation_config_set_threshold(config, 0.5f), 0);
    ASSERT_EQ(ov_genai_speech_generation_config_get_threshold(config, &val), 0);
    EXPECT_FLOAT_EQ(val, 0.5f);

    ASSERT_EQ(ov_genai_text2speech_pipeline_set_generation_config(pipe, config), 0);

    ov_genai_speech_generation_config_free(config);
    ov_genai_text2speech_pipeline_free(pipe);
}

TEST_F(Text2SpeechPipelineCTest, generate) {
    ov_genai_text2speech_pipeline* pipe = nullptr;
    ASSERT_EQ(ov_genai_text2speech_pipeline_create(model_path.c_str(), "CPU", 0, &pipe), 0);

    const char* texts[] = {"Hello world"};
    ov_genai_text2speech_decoded_results* results = nullptr;
    // Test without speaker embedding (SpeechT5 uses default)
    ASSERT_EQ(ov_genai_text2speech_pipeline_generate(pipe, texts, 1, nullptr, 0, &results), 0);
    ASSERT_NE(results, nullptr);

    size_t count = 0;
    ASSERT_EQ(ov_genai_text2speech_decoded_results_get_speeches_count(results, &count), 0);
    EXPECT_EQ(count, 1);

    ov_tensor_t* speech = nullptr;
    ASSERT_EQ(ov_genai_text2speech_decoded_results_get_speech_at(results, 0, &speech), 0);
    ASSERT_NE(speech, nullptr);

    ov_shape_t shape;
    ASSERT_EQ(ov_tensor_get_shape(speech, &shape), 0);
    EXPECT_GT(shape.rank, 0);
    ov_shape_free(&shape);

    ov_tensor_free(speech);
    ov_genai_text2speech_decoded_results_free(results);
    ov_genai_text2speech_pipeline_free(pipe);
}

} // namespace
