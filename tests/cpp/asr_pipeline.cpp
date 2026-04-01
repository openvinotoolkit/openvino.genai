// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <vector>

#include "openvino/genai/speech_recognition/asr_pipeline.hpp"

namespace fs = std::filesystem;
using ov::genai::ASRPipeline;
using ov::genai::ASRGenerationConfig;

// ── Helpers ─────────────────────────────────────────────────────────────

namespace {

// Portable pi constant (M_PI is not guaranteed on all platforms/compilers)
constexpr float kPi = 3.14159265358979323846f;

// Resolve model paths from the environment or well-known CI/container locations.
// If neither env var nor CI paths exist, we skip the test rather than
// embedding developer-local assumptions.

fs::path whisper_model_dir() {
    if (const char* p = std::getenv("WHISPER_MODEL_DIR"))
        return p;
    // Fallback only to known CI/container locations
    if (fs::exists("/app/optimum-intel/whisper-base-ov"))
        return "/app/optimum-intel/whisper-base-ov";
    if (fs::exists("/home/optimum-intel/whisper-base-ov"))
        return "/home/optimum-intel/whisper-base-ov";
    // Return empty path - test will GTEST_SKIP() if model not found
    return {};
}

fs::path paraformer_model_dir() {
    if (const char* p = std::getenv("PARAFORMER_MODEL_DIR"))
        return p;
    // Fallback only to known CI/container locations
    if (fs::exists("/app/optimum-intel/paraformer-zh/ov_models"))
        return "/app/optimum-intel/paraformer-zh/ov_models";
    if (fs::exists("/home/optimum-intel/paraformer-zh/ov_models"))
        return "/home/optimum-intel/paraformer-zh/ov_models";
    // Return empty path - test will GTEST_SKIP() if model not found
    return {};
}

bool whisper_model_exists() {
    auto dir = whisper_model_dir();
    return !dir.empty() && fs::exists(dir / "openvino_encoder_model.xml");
}

bool paraformer_model_exists() {
    auto dir = paraformer_model_dir();
    return !dir.empty() && fs::exists(dir / "openvino_model.xml");
}

// Generate a simple sine-wave of the given length (16 kHz sample rate).
std::vector<float> make_sine_wave(size_t num_samples = 16000,
                                  float freq = 440.0f,
                                  float sample_rate = 16000.0f) {
    std::vector<float> pcm(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        pcm[i] = 0.5f * std::sin(2.0f * kPi * freq * static_cast<float>(i) / sample_rate);
    }
    return pcm;
}

}  // anonymous namespace

// ── Model-type detection tests ──────────────────────────────────────────

class ASRPipelineWhisperTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto dir = whisper_model_dir();
        if (dir.empty()) {
            GTEST_SKIP() << "WHISPER_MODEL_DIR env var not set and no model found in CI paths";
        }
        if (!whisper_model_exists()) {
            GTEST_SKIP() << "Whisper model not found at " << dir;
        }
    }
};

class ASRPipelineParaformerTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto dir = paraformer_model_dir();
        if (dir.empty()) {
            GTEST_SKIP() << "PARAFORMER_MODEL_DIR env var not set and no model found in CI paths";
        }
        if (!paraformer_model_exists()) {
            GTEST_SKIP() << "Paraformer model not found at " << dir;
        }
    }
};

// ── Whisper tests ───────────────────────────────────────────────────────

TEST_F(ASRPipelineWhisperTest, ConstructAndDetectWhisper) {
    ASRPipeline pipe(whisper_model_dir(), "CPU");
    EXPECT_TRUE(pipe.is_whisper());
    EXPECT_FALSE(pipe.is_paraformer());
    EXPECT_EQ(pipe.get_model_type(), ASRPipeline::ModelType::WHISPER);
}

TEST_F(ASRPipelineWhisperTest, GetGenerationConfig) {
    ASRPipeline pipe(whisper_model_dir(), "CPU");
    auto config = pipe.get_generation_config();
    // Whisper generation config should have reasonable defaults
    EXPECT_GT(config.max_new_tokens, 0);
}

TEST_F(ASRPipelineWhisperTest, SetGenerationConfig) {
    ASRPipeline pipe(whisper_model_dir(), "CPU");
    auto config = pipe.get_generation_config();
    config.max_new_tokens = 100;
    pipe.set_generation_config(config);
    auto updated = pipe.get_generation_config();
    EXPECT_EQ(updated.max_new_tokens, 100);
}

TEST_F(ASRPipelineWhisperTest, GetTokenizer) {
    ASRPipeline pipe(whisper_model_dir(), "CPU");
    auto tokenizer = pipe.get_tokenizer();
    // Tokenizer should be functional - try encoding a simple string
    auto encoded = tokenizer.encode("hello");
    EXPECT_GT(encoded.input_ids.get_size(), 0);
}

TEST_F(ASRPipelineWhisperTest, GenerateSineWave) {
    ASRPipeline pipe(whisper_model_dir(), "CPU");
    ov::genai::RawSpeechInput pcm = make_sine_wave(16000);  // 1 second

    auto result = pipe.generate(pcm);
    // Result should have at least one output text
    ASSERT_FALSE(result.texts.empty());
    // Output strings should be non-null (can be empty for sine wave)
    EXPECT_TRUE(result.texts.size() >= 1);
}

TEST_F(ASRPipelineWhisperTest, GenerateWithConfig) {
    ASRPipeline pipe(whisper_model_dir(), "CPU");
    ov::genai::RawSpeechInput pcm = make_sine_wave(16000);

    ASRGenerationConfig config = pipe.get_generation_config();
    config.max_new_tokens = 10;

    auto result = pipe.generate(pcm, config);
    ASSERT_FALSE(result.texts.empty());
}

TEST_F(ASRPipelineWhisperTest, GenerateWithAnyMap) {
    ASRPipeline pipe(whisper_model_dir(), "CPU");
    ov::genai::RawSpeechInput pcm = make_sine_wave(16000);

    ov::AnyMap config_map;
    config_map["max_new_tokens"] = static_cast<size_t>(10);
TEST_F(ASRPipelineWhisperTest, GenerateLongerAudio) {
    ASRPipeline pipe(whisper_model_dir(), "CPU");
    // 5 seconds of silence/sine
    ov::genai::RawSpeechInput pcm = make_sine_wave(80000);

    auto result = pipe.generate(pcm);
    ASSERT_FALSE(result.texts.empty());
}

// ── Paraformer tests ────────────────────────────────────────────────────

TEST_F(ASRPipelineParaformerTest, ConstructAndDetectParaformer) {
    ASRPipeline pipe(paraformer_model_dir(), "CPU");
    EXPECT_TRUE(pipe.is_paraformer());
    EXPECT_FALSE(pipe.is_whisper());
    EXPECT_EQ(pipe.get_model_type(), ASRPipeline::ModelType::PARAFORMER);
}

TEST_F(ASRPipelineParaformerTest, GetGenerationConfig) {
    ASRPipeline pipe(paraformer_model_dir(), "CPU");
    auto config = pipe.get_generation_config();
    // Should return a valid config
    EXPECT_GE(config.max_new_tokens, 0);
}

TEST_F(ASRPipelineParaformerTest, SetGenerationConfig) {
    ASRPipeline pipe(paraformer_model_dir(), "CPU");
    auto config = pipe.get_generation_config();
    config.max_new_tokens = 50;
    pipe.set_generation_config(config);
    auto updated = pipe.get_generation_config();
    EXPECT_EQ(updated.max_new_tokens, 50);
}

TEST_F(ASRPipelineParaformerTest, GenerateSineWave) {
    ASRPipeline pipe(paraformer_model_dir(), "CPU");
    ov::genai::RawSpeechInput pcm = make_sine_wave(16000);  // 1 second

    auto result = pipe.generate(pcm);
    // Result should have at least one output text
    ASSERT_FALSE(result.texts.empty());
    // Output strings should be non-null (can be empty for sine wave)
    EXPECT_TRUE(result.texts.size() >= 1);
}

TEST_F(ASRPipelineParaformerTest, GenerateWithConfig) {
    ASRPipeline pipe(paraformer_model_dir(), "CPU");
    ov::genai::RawSpeechInput pcm = make_sine_wave(16000);

    ASRGenerationConfig config = pipe.get_generation_config();
    config.max_new_tokens = 10;

    auto result = pipe.generate(pcm, config);
    ASSERT_FALSE(result.texts.empty());
}

TEST_F(ASRPipelineParaformerTest, GenerateWithAnyMap) {
    ASRPipeline pipe(paraformer_model_dir(), "CPU");
    ov::genai::RawSpeechInput pcm = make_sine_wave(16000);

    ov::AnyMap config_map;
    config_map["max_new_tokens"] = static_cast<size_t>(10);
TEST_F(ASRPipelineParaformerTest, GenerateLongerAudio) {
    ASRPipeline pipe(paraformer_model_dir(), "CPU");
    // 5 seconds of sine wave
    ov::genai::RawSpeechInput pcm = make_sine_wave(80000);

    auto result = pipe.generate(pcm);
    ASSERT_FALSE(result.texts.empty());
}

TEST_F(ASRPipelineParaformerTest, GenerateMultipleCalls) {
    // Test that multiple generate calls return consistent results
    ASRPipeline pipe(paraformer_model_dir(), "CPU");
    ov::genai::RawSpeechInput pcm = make_sine_wave(16000);

    auto result1 = pipe.generate(pcm);
    auto result2 = pipe.generate(pcm);
    
    ASSERT_FALSE(result1.texts.empty());
    ASSERT_FALSE(result2.texts.empty());
    // Results should be identical for same input (deterministic)
    EXPECT_EQ(result1.texts[0], result2.texts[0]);
}

// ── Shared error tests ──────────────────────────────────────────────────

TEST(ASRPipelineError, InvalidDirectory) {
    // Constructing with a nonexistent directory should throw.
    // Use cross-platform temp path construction instead of POSIX-specific paths.
    fs::path nonexistent = fs::temp_directory_path() / "nonexistent_asr_model_dir_1234567890";
    // Ensure the path does not exist
    if (fs::exists(nonexistent)) {
        fs::remove_all(nonexistent);
    }
    EXPECT_THROW(ASRPipeline(nonexistent, "CPU"), ov::Exception);
}

TEST(ASRPipelineError, EmptyDirectory) {
    // Create a temporary empty directory
    fs::path tmp = fs::temp_directory_path() / "asr_pipeline_test_empty";
    fs::create_directories(tmp);
    EXPECT_THROW(ASRPipeline(tmp, "CPU"), ov::Exception);
    fs::remove_all(tmp);
}
