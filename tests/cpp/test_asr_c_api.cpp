// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>

#include "openvino/genai/c/asr_pipeline.h"
#include "types_c.h"

namespace {
template <typename T, void (*Free)(T*)>
using Handle = std::unique_ptr<T, decltype(Free)>;

TEST(ASRCAPI, ConfigurationDefaultsAndBaseSettingsCopy) {
    ov_genai_asr_generation_config* raw = nullptr;
    ASSERT_EQ(OK, ov_genai_asr_generation_config_create(&raw));
    Handle<ov_genai_asr_generation_config, ov_genai_asr_generation_config_free> config(
        raw,
        ov_genai_asr_generation_config_free);
    ov::genai::ASRGenerationConfig defaults;
    EXPECT_EQ(defaults.max_length, config->object->max_length);
    EXPECT_EQ(defaults.apply_chat_template, config->object->apply_chat_template);
    ASSERT_EQ(OK, ov_genai_asr_generation_config_set_language(config.get(), "English"));
    ASSERT_EQ(OK, ov_genai_asr_generation_config_set_context(config.get(), "OpenVINO"));
    ASSERT_EQ(OK, ov_genai_asr_generation_config_set_word_timestamps(config.get(), true));
    ov_genai_generation_config* base_raw = nullptr;
    ASSERT_EQ(OK, ov_genai_asr_generation_config_get_generation_config(config.get(), &base_raw));
    Handle<ov_genai_generation_config, ov_genai_generation_config_free> base(base_raw, ov_genai_generation_config_free);
    ASSERT_EQ(OK, ov_genai_generation_config_set_max_new_tokens(base.get(), 42));
    EXPECT_NE(42, config->object->max_new_tokens);
    ASSERT_EQ(OK, ov_genai_asr_generation_config_set_generation_config(config.get(), base.get()));
    EXPECT_EQ(42, config->object->max_new_tokens);
    EXPECT_EQ("English", config->object->language);
    EXPECT_EQ("OpenVINO", config->object->context);
    EXPECT_TRUE(config->object->word_timestamps);
    EXPECT_EQ(INVALID_C_PARAM, ov_genai_asr_generation_config_set_generation_config(config.get(), nullptr));
}

TEST(ASRCAPI, OptionalStringsAndBufferBounds) {
    ov_genai_asr_generation_config* raw = nullptr;
    ASSERT_EQ(OK, ov_genai_asr_generation_config_create(&raw));
    Handle<ov_genai_asr_generation_config, ov_genai_asr_generation_config_free> config(
        raw,
        ov_genai_asr_generation_config_free);
    size_t size = 0;
    EXPECT_EQ(NOT_FOUND, ov_genai_asr_generation_config_get_context(config.get(), nullptr, &size));
    ASSERT_EQ(OK, ov_genai_asr_generation_config_set_context(config.get(), "OpenVINO"));
    ASSERT_EQ(OK, ov_genai_asr_generation_config_get_context(config.get(), nullptr, &size));
    EXPECT_EQ(9, size);
    char text[9] = {};
    size_t short_size = 8;
    EXPECT_EQ(OUT_OF_BOUNDS, ov_genai_asr_generation_config_get_context(config.get(), text, &short_size));
    ASSERT_EQ(OK, ov_genai_asr_generation_config_get_context(config.get(), text, &size));
    EXPECT_STREQ("OpenVINO", text);
    ASSERT_EQ(OK, ov_genai_asr_generation_config_set_context(config.get(), nullptr));
    EXPECT_EQ(NOT_FOUND, ov_genai_asr_generation_config_get_context(config.get(), nullptr, &size));
    EXPECT_EQ(INVALID_C_PARAM, ov_genai_asr_generation_config_get_context(config.get(), nullptr, nullptr));
}

TEST(ASRCAPI, NestedTimestampsBoundsAndIndependentOwnership) {
    ov_genai_asr_decoded_results* raw = nullptr;
    ASSERT_EQ(OK, ov_genai_asr_decoded_results_create(&raw));
    Handle<ov_genai_asr_decoded_results, ov_genai_asr_decoded_results_free> results(raw,
                                                                                    ov_genai_asr_decoded_results_free);
    size_t count = 99;
    bool has = true;
    EXPECT_EQ(OK, ov_genai_asr_decoded_results_has_chunks(results.get(), &has));
    EXPECT_FALSE(has);
    EXPECT_EQ(OK, ov_genai_asr_decoded_results_get_chunks_count(results.get(), 0, &count));
    EXPECT_EQ(0, count);
    ov_genai_asr_decoded_result_chunk* chunk_raw = nullptr;
    EXPECT_EQ(NOT_FOUND, ov_genai_asr_decoded_results_get_chunk_at(results.get(), 0, 0, &chunk_raw));
    results->object->texts = {"First", "Second"};
    results->object->scores = {0.1f, 0.2f};
    results->object->languages = {"en", "English"};
    results->object->chunks = {{{0.f, 1.f, "First", {1}}}, {{1.f, 2.f, "Second", {2}}, {2.f, 3.f, "tail", {3}}}};
    results->object->words = {{{0.f, 1.f, "First", {1}}}, {{1.f, 2.f, "Second", {2}}}};
    EXPECT_EQ(OK, ov_genai_asr_decoded_results_get_chunks_count(results.get(), 1, &count));
    EXPECT_EQ(2, count);
    EXPECT_EQ(OUT_OF_BOUNDS, ov_genai_asr_decoded_results_get_chunks_count(results.get(), 2, &count));
    EXPECT_EQ(OUT_OF_BOUNDS, ov_genai_asr_decoded_results_get_chunk_at(results.get(), 0, 1, &chunk_raw));
    EXPECT_EQ(OUT_OF_BOUNDS, ov_genai_asr_decoded_results_get_word_at(results.get(), 2, 0, &chunk_raw));
    EXPECT_EQ(OK, ov_genai_asr_decoded_results_has_words(results.get(), &has));
    EXPECT_TRUE(has);
    EXPECT_EQ(OK, ov_genai_asr_decoded_results_get_words_count(results.get(), 1, &count));
    EXPECT_EQ(1, count);
    ASSERT_EQ(OK, ov_genai_asr_decoded_results_get_word_at(results.get(), 1, 0, &chunk_raw));
    Handle<ov_genai_asr_decoded_result_chunk, ov_genai_asr_decoded_result_chunk_free> chunk(
        chunk_raw,
        ov_genai_asr_decoded_result_chunk_free);
    size_t size = 0;
    ASSERT_EQ(OK, ov_genai_asr_decoded_results_get_language_at(results.get(), 1, nullptr, &size));
    char language[8] = {};
    ASSERT_EQ(OK, ov_genai_asr_decoded_results_get_language_at(results.get(), 1, language, &size));
    EXPECT_STREQ("English", language);
    EXPECT_EQ(OUT_OF_BOUNDS, ov_genai_asr_decoded_results_get_language_at(results.get(), 2, nullptr, &size));
    float score = 0.f;
    ASSERT_EQ(OK, ov_genai_asr_decoded_results_get_score_at(results.get(), 1, &score));
    EXPECT_FLOAT_EQ(0.2f, score);
    ASSERT_EQ(OK, ov_genai_asr_decoded_results_get_text_at(results.get(), 1, nullptr, &size));
    EXPECT_EQ(7, size);
    char transcription[7] = {};
    size_t too_small = 6;
    EXPECT_EQ(OUT_OF_BOUNDS, ov_genai_asr_decoded_results_get_text_at(results.get(), 1, transcription, &too_small));
    ASSERT_EQ(OK, ov_genai_asr_decoded_results_get_text_at(results.get(), 1, transcription, &size));
    EXPECT_STREQ("Second", transcription);
    EXPECT_EQ(OUT_OF_BOUNDS, ov_genai_asr_decoded_results_get_score_at(results.get(), 2, &score));
    results.reset();
    float end = 0.f;
    ASSERT_EQ(OK, ov_genai_asr_decoded_result_chunk_get_end_ts(chunk.get(), &end));
    EXPECT_FLOAT_EQ(2.f, end);
    char text[7] = {};
    size = sizeof(text);
    ASSERT_EQ(OK, ov_genai_asr_decoded_result_chunk_get_text(chunk.get(), text, &size));
    EXPECT_STREQ("Second", text);
}

TEST(ASRCAPI, InvalidPipelineArguments) {
    ov_genai_asr_pipeline* pipeline = nullptr;
    EXPECT_EQ(INVALID_C_PARAM, ov_genai_asr_pipeline_create(nullptr, "CPU", 0, &pipeline));
    EXPECT_EQ(INVALID_C_PARAM, ov_genai_asr_pipeline_create("model", "CPU", 1, &pipeline));
    EXPECT_EQ(INVALID_C_PARAM, ov_genai_asr_pipeline_generate(nullptr, nullptr, 0, nullptr, nullptr));
    ov_genai_asr_generation_config* config = nullptr;
    EXPECT_EQ(INVALID_C_PARAM, ov_genai_asr_pipeline_get_generation_config(nullptr, &config));
    EXPECT_EQ(INVALID_C_PARAM, ov_genai_asr_pipeline_set_generation_config(nullptr, nullptr));
    ov_genai_asr_pipeline_free(nullptr);
    ov_genai_asr_decoded_results_free(nullptr);
    ov_genai_asr_decoded_result_chunk_free(nullptr);
    ov_genai_asr_generation_config_free(nullptr);
}
}  // namespace

namespace {
TEST(ASRCAPI, ScalarAndOptionalConfigurationAccessors) {
    ov_genai_asr_generation_config* raw = nullptr;
    ASSERT_EQ(OK, ov_genai_asr_generation_config_create(&raw));
    Handle<ov_genai_asr_generation_config, ov_genai_asr_generation_config_free> config(
        raw,
        ov_genai_asr_generation_config_free);
#define CHECK_SCALAR(field, type, value)                                                               \
    {                                                                                                  \
        type output{};                                                                                 \
        EXPECT_EQ(INVALID_C_PARAM, ov_genai_asr_generation_config_get_##field(nullptr, &output));      \
        EXPECT_EQ(INVALID_C_PARAM, ov_genai_asr_generation_config_get_##field(config.get(), nullptr)); \
        ASSERT_EQ(OK, ov_genai_asr_generation_config_set_##field(config.get(), value));                \
        ASSERT_EQ(OK, ov_genai_asr_generation_config_get_##field(config.get(), &output));              \
        EXPECT_EQ(value, output);                                                                      \
    }
    CHECK_SCALAR(decoder_start_token_id, int64_t, 101);
    CHECK_SCALAR(pad_token_id, int64_t, 102);
    CHECK_SCALAR(translate_token_id, int64_t, 103);
    CHECK_SCALAR(transcribe_token_id, int64_t, 104);
    CHECK_SCALAR(prev_sot_token_id, int64_t, 105);
    CHECK_SCALAR(no_timestamps_token_id, int64_t, 106);
    CHECK_SCALAR(max_initial_timestamp_index, size_t, 25);
    CHECK_SCALAR(is_multilingual, bool, false);
    CHECK_SCALAR(return_timestamps, bool, true);
    CHECK_SCALAR(word_timestamps, bool, true);
#undef CHECK_SCALAR
    using Setter = ov_status_e (*)(ov_genai_asr_generation_config*, const char*);
    using Getter = ov_status_e (*)(const ov_genai_asr_generation_config*, char*, size_t*);
    const std::pair<Setter, Getter> accessors[] = {
        {ov_genai_asr_generation_config_set_language, ov_genai_asr_generation_config_get_language},
        {ov_genai_asr_generation_config_set_task, ov_genai_asr_generation_config_get_task},
        {ov_genai_asr_generation_config_set_initial_prompt, ov_genai_asr_generation_config_get_initial_prompt},
        {ov_genai_asr_generation_config_set_hotwords, ov_genai_asr_generation_config_get_hotwords},
        {ov_genai_asr_generation_config_set_context, ov_genai_asr_generation_config_get_context},
    };
    for (const auto& [set, get] : accessors) {
        size_t size = 0;
        EXPECT_EQ(INVALID_C_PARAM, set(nullptr, "test"));
        ASSERT_EQ(OK, set(config.get(), "test"));
        ASSERT_EQ(OK, get(config.get(), nullptr, &size));
        EXPECT_EQ(5, size);
        char buffer[5] = {};
        ASSERT_EQ(OK, get(config.get(), buffer, &size));
        EXPECT_STREQ("test", buffer);
        ASSERT_EQ(OK, set(config.get(), nullptr));
        EXPECT_EQ(NOT_FOUND, get(config.get(), nullptr, &size));
    }
}

TEST(ASRCAPI, SuppressionArraysAndValidation) {
    ov_genai_asr_generation_config* raw = nullptr;
    ASSERT_EQ(OK, ov_genai_asr_generation_config_create(&raw));
    Handle<ov_genai_asr_generation_config, ov_genai_asr_generation_config_free> config(
        raw,
        ov_genai_asr_generation_config_free);
    using Setter = ov_status_e (*)(ov_genai_asr_generation_config*, const int64_t*, size_t);
    using Count = ov_status_e (*)(const ov_genai_asr_generation_config*, size_t*);
    using Getter = ov_status_e (*)(const ov_genai_asr_generation_config*, int64_t*, size_t);
    struct Accessors {
        Setter set;
        Count count;
        Getter get;
    };
    const Accessors accessors[] = {
        {ov_genai_asr_generation_config_set_begin_suppress_tokens,
         ov_genai_asr_generation_config_get_begin_suppress_tokens_count,
         ov_genai_asr_generation_config_get_begin_suppress_tokens},
        {ov_genai_asr_generation_config_set_suppress_tokens,
         ov_genai_asr_generation_config_get_suppress_tokens_count,
         ov_genai_asr_generation_config_get_suppress_tokens},
    };
    for (const auto& api : accessors) {
        int64_t input[] = {17, 23};
        ASSERT_EQ(OK, api.set(config.get(), input, 2));
        input[0] = 99;
        size_t count = 0;
        ASSERT_EQ(OK, api.count(config.get(), &count));
        EXPECT_EQ(2, count);
        int64_t output[2] = {};
        EXPECT_EQ(OUT_OF_BOUNDS, api.get(config.get(), output, 1));
        ASSERT_EQ(OK, api.get(config.get(), output, 2));
        EXPECT_EQ(17, output[0]);
        EXPECT_EQ(23, output[1]);
        EXPECT_EQ(INVALID_C_PARAM, api.set(config.get(), nullptr, 2));
        ASSERT_EQ(OK, api.count(config.get(), &count));
        EXPECT_EQ(2, count);
        ASSERT_EQ(OK, api.get(config.get(), output, 2));
        EXPECT_EQ(17, output[0]);
        EXPECT_EQ(23, output[1]);
        ASSERT_EQ(OK, api.set(config.get(), nullptr, 0));
        ASSERT_EQ(OK, api.count(config.get(), &count));
        EXPECT_EQ(0, count);
    }
    config->object->max_new_tokens = 16;
    ASSERT_EQ(OK, ov_genai_asr_generation_config_validate(config.get()));
    config->object->num_return_sequences = 2;
    EXPECT_EQ(UNKNOW_EXCEPTION, ov_genai_asr_generation_config_validate(config.get()));
    EXPECT_EQ(INVALID_C_PARAM, ov_genai_asr_generation_config_create(nullptr));
    EXPECT_EQ(INVALID_C_PARAM, ov_genai_asr_generation_config_create_from_json(nullptr, &raw));
    EXPECT_EQ(UNKNOW_EXCEPTION, ov_genai_asr_generation_config_create_from_json("/nonexistent/asr.json", &raw));
}

TEST(ASRCAPI, EmptyResultsAndMetricsOwnership) {
    ov_genai_asr_decoded_results* raw = nullptr;
    ASSERT_EQ(OK, ov_genai_asr_decoded_results_create(&raw));
    Handle<ov_genai_asr_decoded_results, ov_genai_asr_decoded_results_free> results(raw,
                                                                                    ov_genai_asr_decoded_results_free);
    size_t count = 99;
    EXPECT_EQ(OK, ov_genai_asr_decoded_results_get_texts_count(results.get(), &count));
    EXPECT_EQ(0, count);
    EXPECT_EQ(OUT_OF_BOUNDS, ov_genai_asr_decoded_results_get_text_at(results.get(), 0, nullptr, &count));
    EXPECT_EQ(OUT_OF_BOUNDS, ov_genai_asr_decoded_results_get_language_at(results.get(), 0, nullptr, &count));
    ASSERT_EQ(OK, ov_genai_asr_decoded_results_get_string(results.get(), nullptr, &count));
    EXPECT_EQ(1, count);
    char output = 'x';
    ASSERT_EQ(OK, ov_genai_asr_decoded_results_get_string(results.get(), &output, &count));
    EXPECT_EQ('\0', output);
    ov_genai_perf_metrics* metrics_raw = nullptr;
    ASSERT_EQ(OK, ov_genai_asr_decoded_results_get_perf_metrics(results.get(), &metrics_raw));
    Handle<ov_genai_perf_metrics, ov_genai_asr_decoded_results_perf_metrics_free> metrics(
        metrics_raw,
        ov_genai_asr_decoded_results_perf_metrics_free);
    results.reset();
    float load_time = -1.f;
    ASSERT_EQ(OK, ov_genai_perf_metrics_get_load_time(metrics.get(), &load_time));
    EXPECT_FLOAT_EQ(0.f, load_time);
    ov_genai_asr_decoded_results_perf_metrics_free(nullptr);
    ov_genai_asr_decoded_result_chunk* chunk_raw = nullptr;
    ASSERT_EQ(OK, ov_genai_asr_decoded_result_chunk_create(&chunk_raw));
    Handle<ov_genai_asr_decoded_result_chunk, ov_genai_asr_decoded_result_chunk_free> chunk(
        chunk_raw,
        ov_genai_asr_decoded_result_chunk_free);
    EXPECT_EQ(INVALID_C_PARAM, ov_genai_asr_decoded_result_chunk_get_start_ts(chunk.get(), nullptr));
}
}  // namespace

namespace {
TEST(ASRCAPI, LoadConfigurationFromJson) {
    const auto path =
        std::filesystem::temp_directory_path() /
        ("genai-asr-config-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::filesystem::remove(path);
        }
    } cleanup{path};
    std::ofstream(path) << R"({"max_new_tokens": 32, "decoder_start_token_id": 123,
                             "suppress_tokens": [17, 23], "is_multilingual": false})";
    ov_genai_asr_generation_config* raw = nullptr;
    ASSERT_EQ(OK, ov_genai_asr_generation_config_create_from_json(path.string().c_str(), &raw));
    Handle<ov_genai_asr_generation_config, ov_genai_asr_generation_config_free> config(
        raw,
        ov_genai_asr_generation_config_free);
    EXPECT_EQ(32, config->object->max_new_tokens);
    int64_t token = 0;
    ASSERT_EQ(OK, ov_genai_asr_generation_config_get_decoder_start_token_id(config.get(), &token));
    EXPECT_EQ(123, token);
    size_t count = 0;
    ASSERT_EQ(OK, ov_genai_asr_generation_config_get_suppress_tokens_count(config.get(), &count));
    EXPECT_EQ(2, count);
}
}  // namespace
