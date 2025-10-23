// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "openvino/genai/c/whisper_pipeline.h"
#include "whisper_utils.h"

int main(int argc, char* argv[]) {
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "Usage: %s <MODEL_DIR> \"<WAV_FILE_PATH>\" [DEVICE]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* model_path = argv[1];
    const char* wav_file_path = argv[2];
    const char* device = (argc == 4) ? argv[3] : "CPU";  // Default to CPU if no device is provided

    int exit_code = EXIT_SUCCESS;

    ov_genai_whisper_pipeline* pipeline = NULL;
    ov_genai_whisper_generation_config* config = NULL;
    ov_genai_whisper_decoded_results* results = NULL;
    float* audio_data = NULL;
    size_t audio_length = 0;
    char* output = NULL;
    size_t output_size = 0;

    float file_sample_rate;
    if (load_wav_file(wav_file_path, &audio_data, &audio_length, &file_sample_rate) != 0) {
        exit_code = EXIT_FAILURE;
        goto err;
    }

    if (file_sample_rate != 16000.0f) {
        size_t resampled_length;
        float* resampled_audio = resample_audio(audio_data, audio_length, file_sample_rate, 16000.0f, &resampled_length);
        if (!resampled_audio) {
            fprintf(stderr, "Error: Failed to resample audio\n");
            exit_code = EXIT_FAILURE;
            goto err;
        }
        free(audio_data);
        audio_data = resampled_audio;
        audio_length = resampled_length;
    }

    ov_status_e status = ov_genai_whisper_pipeline_create(model_path, device, 0, &pipeline);
    if (status != OK) {
        if (status == UNKNOW_EXCEPTION) {
            fprintf(stderr, "Error: Failed to create Whisper pipeline. Please check:\n");
            fprintf(stderr, "  - Model path exists and contains valid Whisper model files\n");
            fprintf(stderr, "  - Device '%s' is available and supported\n", device);
            fprintf(stderr, "  - Model is compatible with OpenVINO GenAI\n");
        }
        CHECK_STATUS(status);
    }

    CHECK_STATUS(ov_genai_whisper_generation_config_create(&config));
    CHECK_STATUS(ov_genai_whisper_generation_config_set_task(config, "transcribe"));
    CHECK_STATUS(ov_genai_whisper_generation_config_set_return_timestamps(config, true));
    CHECK_STATUS(ov_genai_whisper_pipeline_generate(pipeline, audio_data, audio_length, config, &results));

    CHECK_STATUS(ov_genai_whisper_decoded_results_get_string(results, NULL, &output_size));
    output = (char*)malloc(output_size);
    if (!output) {
        fprintf(stderr, "Error: Failed to allocate memory for output\n");
        exit_code = EXIT_FAILURE;
        goto err;
    }

    CHECK_STATUS(ov_genai_whisper_decoded_results_get_string(results, output, &output_size));
    printf("%s\n", output);

    bool has_chunks = false;
    CHECK_STATUS(ov_genai_whisper_decoded_results_has_chunks(results, &has_chunks));

    if (has_chunks) {
        size_t chunks_count = 0;
        CHECK_STATUS(ov_genai_whisper_decoded_results_get_chunks_count(results, &chunks_count));

        for (size_t i = 0; i < chunks_count; i++) {
            ov_genai_whisper_decoded_result_chunk* chunk = NULL;
            CHECK_STATUS(ov_genai_whisper_decoded_results_get_chunk_at(results, i, &chunk));

            float start_ts = 0.0f, end_ts = 0.0f;
            CHECK_STATUS(ov_genai_whisper_decoded_result_chunk_get_start_ts(chunk, &start_ts));
            CHECK_STATUS(ov_genai_whisper_decoded_result_chunk_get_end_ts(chunk, &end_ts));

            size_t chunk_text_size = 0;
            CHECK_STATUS(ov_genai_whisper_decoded_result_chunk_get_text(chunk, NULL, &chunk_text_size));

            char* chunk_text = (char*)malloc(chunk_text_size);
            if (!chunk_text) {
                fprintf(stderr, "Warning: Failed to allocate memory for chunk text %zu\n", i);
                ov_genai_whisper_decoded_result_chunk_free(chunk);
                exit_code = EXIT_FAILURE;
                goto err;
            }

            CHECK_STATUS(ov_genai_whisper_decoded_result_chunk_get_text(chunk, chunk_text, &chunk_text_size));

            printf("timestamps: [%.2f, %.2f] text: %s\n", start_ts, end_ts, chunk_text);

            free(chunk_text);
            ov_genai_whisper_decoded_result_chunk_free(chunk);
        }
    }

err:
    if (pipeline)
        ov_genai_whisper_pipeline_free(pipeline);
    if (config)
        ov_genai_whisper_generation_config_free(config);
    if (results)
        ov_genai_whisper_decoded_results_free(results);
    if (output)
        free(output);
    if (audio_data)
        free(audio_data);

    return exit_code;
}
