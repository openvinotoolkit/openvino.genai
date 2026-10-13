// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../whisper_speech_recognition/whisper_utils.h"
#include "openvino/genai/c/asr_pipeline.h"

#define SAMPLE_RATE_TOLERANCE 0.5f

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s <MODEL_DIR> <WAV_FILE_PATH> [DEVICE] [--language LANG] "
                "[--timestamps] [--word-timestamps] [--max-new-tokens N]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char* model_path = argv[1];
    const char* wav_file_path = argv[2];
    const char* device = "CPU";
    const char* language = NULL;
    size_t max_new_tokens = 0;
    bool timestamps = false;
    bool word_timestamps = false;
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--language") == 0 && i + 1 < argc) {
            language = argv[++i];
        } else if (strcmp(argv[i], "--max-new-tokens") == 0 && i + 1 < argc) {
            char* end = NULL;
            const char* value = argv[++i];
            errno = 0;
            unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end || parsed == 0 || parsed > SIZE_MAX) {
                fprintf(stderr, "Invalid maximum token count: %s\n", value);
                return EXIT_FAILURE;
            }
            max_new_tokens = (size_t)parsed;
        } else if (strcmp(argv[i], "--timestamps") == 0) {
            timestamps = true;
        } else if (strcmp(argv[i], "--word-timestamps") == 0) {
            word_timestamps = true;
        } else if (i == 3 && argv[i][0] != '-') {
            device = argv[i];
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }

    int exit_code = EXIT_SUCCESS;

    ov_genai_asr_pipeline* pipeline = NULL;
    ov_genai_asr_generation_config* config = NULL;
    ov_genai_generation_config* base_config = NULL;
    ov_genai_asr_decoded_results* results = NULL;
    ov_genai_asr_decoded_result_chunk* chunk = NULL;
    char* chunk_text = NULL;
    float* audio_data = NULL;
    size_t audio_length = 0;
    char* output = NULL;
    size_t output_size = 0;

    float file_sample_rate;
    if (load_wav_file(wav_file_path, &audio_data, &audio_length, &file_sample_rate) != 0) {
        // The shared loader releases its audio buffer on failure.
        audio_data = NULL;
        exit_code = EXIT_FAILURE;
        goto err;
    }

    if (fabsf(file_sample_rate - 16000.0f) > SAMPLE_RATE_TOLERANCE) {
        size_t resampled_length;
        float* resampled_audio =
            resample_audio(audio_data, audio_length, file_sample_rate, 16000.0f, &resampled_length);
        if (!resampled_audio) {
            fprintf(stderr, "Error: Failed to resample audio\n");
            exit_code = EXIT_FAILURE;
            goto err;
        }
        free(audio_data);
        audio_data = resampled_audio;
        audio_length = resampled_length;
    }

    ov_status_e status = ov_genai_asr_pipeline_create(model_path,
                                                      device,
                                                      2,
                                                      &pipeline,
                                                      "word_timestamps",
                                                      word_timestamps ? "YES" : "NO");
    if (status != OK) {
        if (status == UNKNOW_EXCEPTION) {
            fprintf(stderr, "Error: Failed to create ASR pipeline. Please check:\n");
            fprintf(stderr, "  - Model path exists and contains valid ASR model files\n");
            fprintf(stderr, "  - Device '%s' is available and supported\n", device);
            fprintf(stderr, "  - Model is compatible with OpenVINO GenAI\n");
        }
        CHECK_STATUS(status);
    }

    CHECK_STATUS(ov_genai_asr_pipeline_get_generation_config(pipeline, &config));
    CHECK_STATUS(ov_genai_asr_generation_config_set_language(config, language));
    CHECK_STATUS(ov_genai_asr_generation_config_set_word_timestamps(config, word_timestamps));
    CHECK_STATUS(ov_genai_asr_generation_config_set_return_timestamps(config, timestamps));
    if (max_new_tokens) {
        CHECK_STATUS(ov_genai_asr_generation_config_get_generation_config(config, &base_config));
        CHECK_STATUS(ov_genai_generation_config_set_max_new_tokens(base_config, max_new_tokens));
        CHECK_STATUS(ov_genai_asr_generation_config_set_generation_config(config, base_config));
    }
    CHECK_STATUS(ov_genai_asr_pipeline_generate(pipeline, audio_data, audio_length, config, &results));

    CHECK_STATUS(ov_genai_asr_decoded_results_get_string(results, NULL, &output_size));
    output = (char*)malloc(output_size);
    if (!output) {
        fprintf(stderr, "Error: Failed to allocate memory for output\n");
        exit_code = EXIT_FAILURE;
        goto err;
    }

    CHECK_STATUS(ov_genai_asr_decoded_results_get_string(results, output, &output_size));
    printf("%s\n", output);

    bool has_chunks = false;
    CHECK_STATUS(ov_genai_asr_decoded_results_has_chunks(results, &has_chunks));

    if (has_chunks) {
        size_t chunks_count = 0;
        CHECK_STATUS(ov_genai_asr_decoded_results_get_chunks_count(results, 0, &chunks_count));

        for (size_t i = 0; i < chunks_count; i++) {
            CHECK_STATUS(ov_genai_asr_decoded_results_get_chunk_at(results, 0, i, &chunk));

            float start_ts = 0.0f, end_ts = 0.0f;
            CHECK_STATUS(ov_genai_asr_decoded_result_chunk_get_start_ts(chunk, &start_ts));
            CHECK_STATUS(ov_genai_asr_decoded_result_chunk_get_end_ts(chunk, &end_ts));

            size_t chunk_text_size = 0;
            CHECK_STATUS(ov_genai_asr_decoded_result_chunk_get_text(chunk, NULL, &chunk_text_size));

            chunk_text = (char*)malloc(chunk_text_size);
            if (!chunk_text) {
                fprintf(stderr, "Warning: Failed to allocate memory for chunk text %zu\n", i);
                exit_code = EXIT_FAILURE;
                goto err;
            }

            CHECK_STATUS(ov_genai_asr_decoded_result_chunk_get_text(chunk, chunk_text, &chunk_text_size));

            printf("timestamps: [%.2f, %.2f] text: %s\n", start_ts, end_ts, chunk_text);

            free(chunk_text);
            chunk_text = NULL;
            ov_genai_asr_decoded_result_chunk_free(chunk);
            chunk = NULL;
        }
    }

    bool has_words = false;
    CHECK_STATUS(ov_genai_asr_decoded_results_has_words(results, &has_words));

    if (has_words) {
        size_t words_count = 0;
        CHECK_STATUS(ov_genai_asr_decoded_results_get_words_count(results, 0, &words_count));

        for (size_t i = 0; i < words_count; i++) {
            CHECK_STATUS(ov_genai_asr_decoded_results_get_word_at(results, 0, i, &chunk));

            float start_ts = 0.0f, end_ts = 0.0f;
            CHECK_STATUS(ov_genai_asr_decoded_result_chunk_get_start_ts(chunk, &start_ts));
            CHECK_STATUS(ov_genai_asr_decoded_result_chunk_get_end_ts(chunk, &end_ts));

            size_t chunk_text_size = 0;
            CHECK_STATUS(ov_genai_asr_decoded_result_chunk_get_text(chunk, NULL, &chunk_text_size));

            chunk_text = (char*)malloc(chunk_text_size);
            if (!chunk_text) {
                fprintf(stderr, "Warning: Failed to allocate memory for chunk text %zu\n", i);
                exit_code = EXIT_FAILURE;
                goto err;
            }

            CHECK_STATUS(ov_genai_asr_decoded_result_chunk_get_text(chunk, chunk_text, &chunk_text_size));

            printf("[%.2f, %.2f]: %s\n", start_ts, end_ts, chunk_text);

            free(chunk_text);
            chunk_text = NULL;
            ov_genai_asr_decoded_result_chunk_free(chunk);
            chunk = NULL;
        }
    }

err:
    ov_genai_generation_config_free(base_config);
    free(chunk_text);
    ov_genai_asr_decoded_result_chunk_free(chunk);
    if (pipeline)
        ov_genai_asr_pipeline_free(pipeline);
    if (config)
        ov_genai_asr_generation_config_free(config);
    if (results)
        ov_genai_asr_decoded_results_free(results);
    if (output)
        free(output);
    if (audio_data)
        free(audio_data);

    return exit_code;
}
