// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"
#include "openvino/genai/c/asr_pipeline.h"
#include "openvino/genai/c/whisper_pipeline.h"
#include "openvino/genai/whisper_pipeline.hpp"

extern "C" {
#include "../../samples/c/whisper_speech_recognition/whisper_utils.h"
}

namespace {
template <typename T, void (*Free)(T*)>
using Handle = std::unique_ptr<T, decltype(Free)>;

std::string read_result(const ov_genai_asr_decoded_results* result) {
    size_t size = 0;
    EXPECT_EQ(OK, ov_genai_asr_decoded_results_get_string(result, nullptr, &size));
    std::vector<char> text(size);
    EXPECT_EQ(OK, ov_genai_asr_decoded_results_get_string(result, text.data(), &size));
    return text.data();
}

class ASRCAPIE2E : public testing::TestWithParam<const char*> {};

TEST_P(ASRCAPIE2E, MatchesCppAndRetainsResultsAcrossCalls) {
    const char* model = std::getenv(GetParam());
    const char* wav = std::getenv("ASR_TEST_WAV");
    if (!model || !wav) {
        GTEST_SKIP() << "Set " << GetParam() << " and ASR_TEST_WAV to run real-model comparisons";
    }
    const bool whisper = std::string(GetParam()) == "ASR_TEST_WHISPER_MODEL";
    float* audio_raw = nullptr;
    size_t length = 0;
    float rate = 0.f;
    ASSERT_EQ(0, load_wav_file(wav, &audio_raw, &length, &rate));
    std::unique_ptr<float, decltype(&std::free)> audio(audio_raw, &std::free);
    ASSERT_FLOAT_EQ(16000.f, rate);
    std::vector<float> speech(audio.get(), audio.get() + length);

    ov_genai_asr_pipeline* pipeline_raw = nullptr;
    ASSERT_EQ(OK,
              ov_genai_asr_pipeline_create(model, "CPU", 2, &pipeline_raw, "word_timestamps", whisper ? "YES" : "NO"));
    Handle<ov_genai_asr_pipeline, ov_genai_asr_pipeline_free> pipeline(pipeline_raw, ov_genai_asr_pipeline_free);
    ov::genai::ASRPipeline cpp(model, "CPU", ov::genai::word_timestamps(whisper));

    // NULL config must select model defaults, including automatic language selection.
    ov_genai_asr_decoded_results* defaults_raw = nullptr;
    ASSERT_EQ(OK, ov_genai_asr_pipeline_generate(pipeline.get(), speech.data(), speech.size(), nullptr, &defaults_raw));
    Handle<ov_genai_asr_decoded_results, ov_genai_asr_decoded_results_free> defaults(defaults_raw,
                                                                                     ov_genai_asr_decoded_results_free);
    const auto cpp_defaults = cpp.generate(speech);
    const auto retained_text = read_result(defaults.get());
    EXPECT_EQ(static_cast<std::string>(cpp_defaults), retained_text);

    ov_genai_asr_generation_config* config_raw = nullptr;
    ASSERT_EQ(OK, ov_genai_asr_pipeline_get_generation_config(pipeline.get(), &config_raw));
    Handle<ov_genai_asr_generation_config, ov_genai_asr_generation_config_free> config(
        config_raw,
        ov_genai_asr_generation_config_free);
    const char* language = whisper ? "<|en|>" : "English";
    ASSERT_EQ(OK, ov_genai_asr_generation_config_set_language(config.get(), language));
    ASSERT_EQ(OK, ov_genai_asr_generation_config_set_return_timestamps(config.get(), whisper));
    ASSERT_EQ(OK, ov_genai_asr_generation_config_set_word_timestamps(config.get(), whisper));
    auto cpp_config = cpp.get_generation_config();
    cpp_config.language = language;
    cpp_config.return_timestamps = whisper;
    cpp_config.word_timestamps = whisper;
    const auto expected = cpp.generate(speech, cpp_config);
    ov_genai_asr_decoded_results* result_raw = nullptr;
    ASSERT_EQ(OK,
              ov_genai_asr_pipeline_generate(pipeline.get(), speech.data(), speech.size(), config.get(), &result_raw));
    Handle<ov_genai_asr_decoded_results, ov_genai_asr_decoded_results_free> result(result_raw,
                                                                                   ov_genai_asr_decoded_results_free);
    EXPECT_EQ(static_cast<std::string>(expected), read_result(result.get()));
    EXPECT_EQ(retained_text, read_result(defaults.get()));
    size_t count = 0;
    ASSERT_EQ(OK, ov_genai_asr_decoded_results_get_texts_count(result.get(), &count));
    ASSERT_EQ(expected.texts.size(), count);
    for (size_t i = 0; i < count; ++i) {
        float score = 0.f;
        ASSERT_EQ(OK, ov_genai_asr_decoded_results_get_score_at(result.get(), i, &score));
        EXPECT_FLOAT_EQ(expected.scores[i], score);
        size_t size = 0;
        ASSERT_EQ(OK, ov_genai_asr_decoded_results_get_language_at(result.get(), i, nullptr, &size));
        std::vector<char> language_text(size);
        ASSERT_EQ(OK, ov_genai_asr_decoded_results_get_language_at(result.get(), i, language_text.data(), &size));
        EXPECT_EQ(expected.languages[i], language_text.data());
    }
    for (bool words : {false, true}) {
        const auto& timestamps = words ? expected.words : expected.chunks;
        bool has = false;
        ASSERT_EQ(OK,
                  words ? ov_genai_asr_decoded_results_has_words(result.get(), &has)
                        : ov_genai_asr_decoded_results_has_chunks(result.get(), &has));
        EXPECT_EQ(timestamps.has_value(), has);
        if (!has)
            continue;
        for (size_t i = 0; i < timestamps->size(); ++i) {
            ASSERT_EQ(OK,
                      words ? ov_genai_asr_decoded_results_get_words_count(result.get(), i, &count)
                            : ov_genai_asr_decoded_results_get_chunks_count(result.get(), i, &count));
            ASSERT_EQ(timestamps->at(i).size(), count);
            for (size_t j = 0; j < count; ++j) {
                ov_genai_asr_decoded_result_chunk* chunk_raw = nullptr;
                ASSERT_EQ(OK,
                          words ? ov_genai_asr_decoded_results_get_word_at(result.get(), i, j, &chunk_raw)
                                : ov_genai_asr_decoded_results_get_chunk_at(result.get(), i, j, &chunk_raw));
                Handle<ov_genai_asr_decoded_result_chunk, ov_genai_asr_decoded_result_chunk_free> chunk(
                    chunk_raw,
                    ov_genai_asr_decoded_result_chunk_free);
                float start = 0.f, end = 0.f;
                ASSERT_EQ(OK, ov_genai_asr_decoded_result_chunk_get_start_ts(chunk.get(), &start));
                ASSERT_EQ(OK, ov_genai_asr_decoded_result_chunk_get_end_ts(chunk.get(), &end));
                EXPECT_FLOAT_EQ(timestamps->at(i)[j].start_ts, start);
                EXPECT_FLOAT_EQ(timestamps->at(i)[j].end_ts, end);
                size_t size = 0;
                ASSERT_EQ(OK, ov_genai_asr_decoded_result_chunk_get_text(chunk.get(), nullptr, &size));
                std::vector<char> text(size);
                ASSERT_EQ(OK, ov_genai_asr_decoded_result_chunk_get_text(chunk.get(), text.data(), &size));
                EXPECT_EQ(timestamps->at(i)[j].text, text.data());
            }
        }
    }
    ov_genai_perf_metrics* metrics_raw = nullptr;
    ASSERT_EQ(OK, ov_genai_asr_decoded_results_get_perf_metrics(result.get(), &metrics_raw));
    Handle<ov_genai_perf_metrics, ov_genai_asr_decoded_results_perf_metrics_free> metrics(
        metrics_raw,
        ov_genai_asr_decoded_results_perf_metrics_free);
    float load_time = 0.f;
    EXPECT_EQ(OK, ov_genai_perf_metrics_get_load_time(metrics.get(), &load_time));
    EXPECT_GT(load_time, 0.f);
    size_t tokens = 0;
    ASSERT_EQ(OK, ov_genai_perf_metrics_get_num_generation_tokens(metrics.get(), &tokens));
    auto expected_metrics = expected.perf_metrics;
    EXPECT_EQ(expected_metrics.get_num_generated_tokens(), tokens);
    ASSERT_EQ(OK, ov_genai_perf_metrics_get_num_input_tokens(metrics.get(), &tokens));
    EXPECT_EQ(expected_metrics.get_num_input_tokens(), tokens);
    ASSERT_EQ(OK, ov_genai_asr_pipeline_set_generation_config(pipeline.get(), config.get()));
    ov_genai_asr_decoded_results* stored_raw = nullptr;
    ASSERT_EQ(OK, ov_genai_asr_pipeline_generate(pipeline.get(), speech.data(), speech.size(), nullptr, &stored_raw));
    Handle<ov_genai_asr_decoded_results, ov_genai_asr_decoded_results_free> stored(stored_raw,
                                                                                   ov_genai_asr_decoded_results_free);
    EXPECT_EQ(read_result(result.get()), read_result(stored.get()));
    pipeline.reset();
    EXPECT_EQ(retained_text, read_result(defaults.get()));
    EXPECT_EQ(static_cast<std::string>(expected), read_result(result.get()));

    if (whisper) {
        ov::genai::WhisperPipeline old(model, "CPU", ov::genai::word_timestamps(true));
        auto old_config = old.get_generation_config();
        old_config.language = "<|en|>";
        old_config.return_timestamps = true;
        old_config.word_timestamps = true;
        const auto old_result = old.generate(speech, old_config);
        EXPECT_EQ(old_result.texts, expected.texts);
        EXPECT_EQ(old_result.scores, expected.scores);
        ASSERT_TRUE(old_result.chunks.has_value());
        ASSERT_TRUE(old_result.words.has_value());
        ASSERT_EQ(old_result.chunks->size(), expected.chunks->at(0).size());
        ASSERT_EQ(old_result.words->size(), expected.words->at(0).size());
        for (size_t i = 0; i < old_result.words->size(); ++i) {
            EXPECT_EQ(old_result.words->at(i).word, expected.words->at(0)[i].text);
            EXPECT_FLOAT_EQ(old_result.words->at(i).start_ts, expected.words->at(0)[i].start_ts);
            EXPECT_FLOAT_EQ(old_result.words->at(i).end_ts, expected.words->at(0)[i].end_ts);
        }
        // Compare the existing Whisper C bindings as well.
        ov_genai_whisper_pipeline* old_raw = nullptr;
        ASSERT_EQ(OK, ov_genai_whisper_pipeline_create(model, "CPU", 0, &old_raw));
        Handle<ov_genai_whisper_pipeline, ov_genai_whisper_pipeline_free> old_c(old_raw,
                                                                                ov_genai_whisper_pipeline_free);
        ov_genai_whisper_decoded_results* old_decoded_raw = nullptr;
        ASSERT_EQ(
            OK,
            ov_genai_whisper_pipeline_generate(old_c.get(), speech.data(), speech.size(), nullptr, &old_decoded_raw));
        Handle<ov_genai_whisper_decoded_results, ov_genai_whisper_decoded_results_free> old_decoded(
            old_decoded_raw,
            ov_genai_whisper_decoded_results_free);
        size_t size = 0;
        ASSERT_EQ(OK, ov_genai_whisper_decoded_results_get_string(old_decoded.get(), nullptr, &size));
        std::vector<char> text(size);
        ASSERT_EQ(OK, ov_genai_whisper_decoded_results_get_string(old_decoded.get(), text.data(), &size));
        EXPECT_EQ(retained_text, text.data());
    }
    std::cout << GetParam() << ": " << read_result(result.get()) << '\n';
}

INSTANTIATE_TEST_SUITE_P(Models,
                         ASRCAPIE2E,
                         testing::Values("ASR_TEST_WHISPER_MODEL", "ASR_TEST_QWEN_MODEL", "ASR_TEST_FUN_MODEL"));
}  // namespace
