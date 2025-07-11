// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "openvino/genai/c/whisper_pipeline.h"
#include "whisper_utils.h"

int main(int argc, char* argv[]) {
    Options options;
    int result = parse_arguments(argc, argv, &options);
    if (result == 0) {
        return EXIT_SUCCESS;  // Help was printed
    } else if (result == -1) {
        return EXIT_FAILURE;  // Error in arguments
    }

    // Track exit status
    int exit_code = EXIT_SUCCESS;

    // Initialize variables
    ov_genai_whisper_pipeline* pipeline = NULL;
    ov_genai_whisper_generation_config* config = NULL;
    ov_genai_whisper_decoded_results* results = NULL;
    float* audio_data = NULL;
    float* resampled_audio = NULL;
    size_t audio_length = 0;
    char* output = NULL;
    size_t output_size = 0;
    
    // Load or generate audio
    if (options.audio_path) {
        float file_sample_rate;
        if (load_wav_file(options.audio_path, &audio_data, &audio_length, &file_sample_rate) != 0) {
            exit_code = EXIT_FAILURE;
            goto err;
        }
        
        // Resample to 16kHz if needed
        if (file_sample_rate != 16000.0f) {
            size_t resampled_length;
            resampled_audio = resample_audio(audio_data, audio_length, file_sample_rate, 16000.0f, &resampled_length);
            if (!resampled_audio) {
                fprintf(stderr, "Error: Failed to resample audio\n");
                exit_code = EXIT_FAILURE;
                goto err;
            }
            free(audio_data);
            audio_data = resampled_audio;
            audio_length = resampled_length;
            resampled_audio = NULL;
        }
    } else {
        // Generate synthetic audio
        audio_length = (size_t)(options.sample_rate * options.duration);
        audio_data = (float*)malloc(audio_length * sizeof(float));
        if (!audio_data) {
            fprintf(stderr, "Error: Failed to allocate memory for audio data\n");
            exit_code = EXIT_FAILURE;
            goto err;
        }
        generate_synthetic_audio(audio_data, audio_length, 440.0f, options.sample_rate);
    }
    
    // Create pipeline
    ov_status_e status = ov_genai_whisper_pipeline_create(options.model_path, options.device, 0, &pipeline);
    if (status != OK) {
        if (status == UNKNOW_EXCEPTION) {
            fprintf(stderr, "Error: Failed to create Whisper pipeline. Please check:\n");
            fprintf(stderr, "  - Model path exists and contains valid Whisper model files\n");
            fprintf(stderr, "  - Device '%s' is available and supported\n", options.device);
            fprintf(stderr, "  - Model is compatible with OpenVINO GenAI\n");
        }
        CHECK_STATUS(status);
    }
    
    // Create and configure generation config
    CHECK_STATUS(ov_genai_whisper_generation_config_create(&config));
    
    if (strlen(options.language) > 0) {
        CHECK_STATUS(ov_genai_whisper_generation_config_set_language(config, options.language));
    }
    
    CHECK_STATUS(ov_genai_whisper_generation_config_set_task(config, options.task));
    CHECK_STATUS(ov_genai_whisper_generation_config_set_return_timestamps(config, options.return_timestamps));
    
    if (options.initial_prompt) {
        CHECK_STATUS(ov_genai_whisper_generation_config_set_initial_prompt(config, options.initial_prompt));
    }
    
    // Generate transcription
    CHECK_STATUS(ov_genai_whisper_pipeline_generate(pipeline, audio_data, audio_length, config, &results));
    
    // Get and print results
    // Get the transcription text
    CHECK_STATUS(ov_genai_whisper_decoded_results_get_string(results, NULL, &output_size));
    output = (char*)malloc(output_size);
    if (!output) {
        fprintf(stderr, "Error: Failed to allocate memory for output\n");
        exit_code = EXIT_FAILURE;
        goto err;
    }
    
    CHECK_STATUS(ov_genai_whisper_decoded_results_get_string(results, output, &output_size));
    printf("%s\n", output);
    
    
    // Display timestamps if available
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
                continue;
            }
            
            CHECK_STATUS(ov_genai_whisper_decoded_result_chunk_get_text(chunk, chunk_text, &chunk_text_size));
            
            printf("timestamps: [%.2f, %.2f] text:%s\n", start_ts, end_ts, chunk_text);
            
            free(chunk_text);
            ov_genai_whisper_decoded_result_chunk_free(chunk);
        }
    }
    

err:
    // Cleanup
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
    if (resampled_audio)
        free(resampled_audio);
    
    return exit_code; // Return the tracked exit status
}