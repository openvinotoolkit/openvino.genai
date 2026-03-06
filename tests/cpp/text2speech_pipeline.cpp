// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "openvino/genai/c/text2speech_pipeline.h"
#include "openvino/genai/c/speech_generation_config.h"
#include "openvino/genai/c/speech_generation_perf_metrics.h"

#include <filesystem>
#include <string>

class Text2SpeechPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create speech generation config
        ASSERT_EQ(speech_generation_config_create(&config), 0);
        
        // Set default values
        ASSERT_EQ(speech_generation_config_set_minlenratio(config, 0.1f), 0);
        ASSERT_EQ(speech_generation_config_set_maxlenratio(config, 10.0f), 0);
        ASSERT_EQ(speech_generation_config_set_threshold(config, 0.5f), 0);
    }

    void TearDown() override {
        if (config) {
            EXPECT_EQ(speech_generation_config_destroy(config), 0);
        }
    }

    speech_generation_config_handle_t config = nullptr;
};

TEST_F(Text2SpeechPipelineTest, ConfigurationTest) {
    float value;
    
    // Test minlenratio
    ASSERT_EQ(speech_generation_config_get_minlenratio(config, &value), 0);
    EXPECT_FLOAT_EQ(value, 0.1f);

    // Test maxlenratio
    ASSERT_EQ(speech_generation_config_get_maxlenratio(config, &value), 0);
    EXPECT_FLOAT_EQ(value, 10.0f);

    // Test threshold
    ASSERT_EQ(speech_generation_config_get_threshold(config, &value), 0);
    EXPECT_FLOAT_EQ(value, 0.5f);
}

TEST_F(Text2SpeechPipelineTest, PipelineCreationTest) {
    text2speech_pipeline_handle_t pipeline = nullptr;
    
    // Test invalid arguments
    EXPECT_NE(text2speech_pipeline_create(nullptr, "models", "CPU"), 0);
    EXPECT_NE(text2speech_pipeline_create(&pipeline, nullptr, "CPU"), 0);
    EXPECT_NE(text2speech_pipeline_create(&pipeline, "models", nullptr), 0);
}

TEST_F(Text2SpeechPipelineTest, GenerationConfigTest) {
    text2speech_pipeline_handle_t pipeline = nullptr;
    const char* test_path = "path/to/test/models";
    
    // Create pipeline
    ASSERT_EQ(text2speech_pipeline_create(&pipeline, test_path, "CPU"), 0);
    ASSERT_NE(pipeline, nullptr);

    // Test get/set generation config
    speech_generation_config_handle_t retrieved_config = nullptr;
    ASSERT_EQ(text2speech_pipeline_get_generation_config(pipeline, &retrieved_config), 0);
    ASSERT_NE(retrieved_config, nullptr);

    ASSERT_EQ(text2speech_pipeline_set_generation_config(pipeline, config), 0);

    // Cleanup
    EXPECT_EQ(speech_generation_config_destroy(retrieved_config), 0);
    EXPECT_EQ(text2speech_pipeline_destroy(pipeline), 0);
}

TEST_F(Text2SpeechPipelineTest, PerformanceMetricsTest) {
    speech_generation_perf_metrics_t* metrics = nullptr;
    
    // Create metrics
    ASSERT_EQ(speech_generation_perf_metrics_create(&metrics), 0);
    ASSERT_NE(metrics, nullptr);

    // Set test values
    metrics->num_generated_samples = 1000;
    metrics->base.generate_duration = 2.5f;
    metrics->base.throughput = 400.0f;

    // Test getters
    int num_samples;
    float duration, throughput;
    
    ASSERT_EQ(speech_generation_perf_metrics_get_num_generated_samples(metrics, &num_samples), 0);
    EXPECT_EQ(num_samples, 1000);

    ASSERT_EQ(speech_generation_perf_metrics_get_generate_duration(metrics, &duration), 0);
    EXPECT_FLOAT_EQ(duration, 2.5f);

    ASSERT_EQ(speech_generation_perf_metrics_get_throughput(metrics, &throughput), 0);
    EXPECT_FLOAT_EQ(throughput, 400.0f);

    // Cleanup
    EXPECT_EQ(speech_generation_perf_metrics_destroy(metrics), 0);
}

TEST_F(Text2SpeechPipelineTest, SingleTextGenerationTest) {
    text2speech_pipeline_handle_t pipeline = nullptr;
    const char* test_path = "path/to/test/models";
    
    // Create pipeline
    ASSERT_EQ(text2speech_pipeline_create(&pipeline, test_path, "CPU"), 0);
    ASSERT_NE(pipeline, nullptr);

    // Test single text generation
    const char* input_text = "Hello, world!";
    text2speech_decoded_results_t* results = nullptr;

    // Test with no speaker embedding
    ASSERT_EQ(text2speech_pipeline_generate(pipeline, input_text, nullptr, 0, &results), 0);
    ASSERT_NE(results, nullptr);
    EXPECT_GT(results->num_speeches, 0);
    EXPECT_NE(results->speeches, nullptr);

    // Verify speech data
    for (size_t i = 0; i < results->num_speeches; ++i) {
        EXPECT_GT(results->speeches[i].num_samples, 0);
        EXPECT_NE(results->speeches[i].samples, nullptr);
        EXPECT_EQ(results->speeches[i].sample_rate, 16000);
    }

    // Test error handling
    EXPECT_NE(text2speech_pipeline_generate(nullptr, input_text, nullptr, 0, &results), 0);
    EXPECT_NE(text2speech_pipeline_generate(pipeline, nullptr, nullptr, 0, &results), 0);
    EXPECT_NE(text2speech_pipeline_generate(pipeline, input_text, nullptr, 0, nullptr), 0);

    // Cleanup
    EXPECT_EQ(text2speech_decoded_results_destroy(results), 0);
    EXPECT_EQ(text2speech_pipeline_destroy(pipeline), 0);
}

TEST_F(Text2SpeechPipelineTest, BatchGenerationTest) {
    text2speech_pipeline_handle_t pipeline = nullptr;
    const char* test_path = "path/to/test/models";
    
    // Create pipeline
    ASSERT_EQ(text2speech_pipeline_create(&pipeline, test_path, "CPU"), 0);
    ASSERT_NE(pipeline, nullptr);

    // Test batch text generation
    const char* input_texts[] = {
        "First text",
        "Second text",
        "Third text"
    };
    const size_t num_texts = 3;
    text2speech_decoded_results_t* results = nullptr;

    // Test batch generation
    ASSERT_EQ(text2speech_pipeline_generate_batch(pipeline, input_texts, num_texts, nullptr, 0, &results), 0);
    ASSERT_NE(results, nullptr);
    EXPECT_EQ(results->num_speeches, num_texts);

    // Verify each generated speech
    for (size_t i = 0; i < results->num_speeches; ++i) {
        EXPECT_GT(results->speeches[i].num_samples, 0);
        EXPECT_NE(results->speeches[i].samples, nullptr);
        EXPECT_EQ(results->speeches[i].sample_rate, 16000);
    }

    // Test error handling
    EXPECT_NE(text2speech_pipeline_generate_batch(nullptr, input_texts, num_texts, nullptr, 0, &results), 0);
    EXPECT_NE(text2speech_pipeline_generate_batch(pipeline, nullptr, num_texts, nullptr, 0, &results), 0);
    EXPECT_NE(text2speech_pipeline_generate_batch(pipeline, input_texts, 0, nullptr, 0, &results), 0);
    EXPECT_NE(text2speech_pipeline_generate_batch(pipeline, input_texts, num_texts, nullptr, 0, nullptr), 0);

    // Cleanup
    EXPECT_EQ(text2speech_decoded_results_destroy(results), 0);
    EXPECT_EQ(text2speech_pipeline_destroy(pipeline), 0);
}

TEST_F(Text2SpeechPipelineTest, SpeakerEmbeddingTest) {
    text2speech_pipeline_handle_t pipeline = nullptr;
    const char* test_path = "path/to/test/models";
    
    // Create pipeline
    ASSERT_EQ(text2speech_pipeline_create(&pipeline, test_path, "CPU"), 0);
    ASSERT_NE(pipeline, nullptr);

    // Create test speaker embedding
    const size_t embedding_size = 512;  // Example size
    std::vector<float> speaker_embedding(embedding_size, 0.5f);
    const char* input_text = "Test with speaker embedding";
    text2speech_decoded_results_t* results = nullptr;

    // Test generation with speaker embedding
    ASSERT_EQ(text2speech_pipeline_generate(pipeline, 
                                          input_text,
                                          speaker_embedding.data(),
                                          embedding_size,
                                          &results), 0);
    ASSERT_NE(results, nullptr);
    EXPECT_GT(results->num_speeches, 0);

    // Test error handling for speaker embedding
    EXPECT_NE(text2speech_pipeline_generate(pipeline,
                                          input_text,
                                          nullptr,
                                          embedding_size,
                                          &results), 0);
    EXPECT_NE(text2speech_pipeline_generate(pipeline,
                                          input_text,
                                          speaker_embedding.data(),
                                          0,
                                          &results), 0);

    // Cleanup
    EXPECT_EQ(text2speech_decoded_results_destroy(results), 0);
    EXPECT_EQ(text2speech_pipeline_destroy(pipeline), 0);
}

TEST_F(Text2SpeechPipelineTest, MemoryManagementTest) {
    // Test memory management for results
    text2speech_decoded_results_t* results = new text2speech_decoded_results_t();
    results->num_speeches = 2;
    results->speeches = new speech_data_t[2];

    // Initialize test data
    for (size_t i = 0; i < 2; ++i) {
        results->speeches[i].num_samples = 100;
        results->speeches[i].samples = new float[100];
        results->speeches[i].sample_rate = 16000;
    }

    // Test cleanup
    EXPECT_EQ(text2speech_decoded_results_destroy(results), 0);
    EXPECT_EQ(text2speech_decoded_results_destroy(nullptr), 1);  // Should handle null safely

    // Test pipeline cleanup
    text2speech_pipeline_handle_t pipeline = nullptr;
    ASSERT_EQ(text2speech_pipeline_create(&pipeline, "test_path", "CPU"), 0);
    EXPECT_EQ(text2speech_pipeline_destroy(pipeline), 0);
    EXPECT_EQ(text2speech_pipeline_destroy(nullptr), 1);  // Should handle null safely
}
