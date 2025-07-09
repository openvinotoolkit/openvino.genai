// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "openvino/genai/c/whisper_pipeline.h"

#define CHECK_STATUS(return_status)                                                      \
    if (return_status != OK) {                                                           \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); \
        goto err;                                                                        \
    }

// Simple function to generate a synthetic audio signal (sine wave)
// In a real application, you would load audio from a file
void generate_sample_audio(float* audio, size_t length, float frequency, float sample_rate) {
    for (size_t i = 0; i < length; i++) {
        audio[i] = sinf(2.0f * M_PI * frequency * (float)i / sample_rate);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <MODEL_DIR> [language] [task]\n", argv[0]);
        fprintf(stderr, "  MODEL_DIR: Path to the directory containing the Whisper model files\n");
        fprintf(stderr, "  language: Optional language code (e.g., \"en\", \"fr\", \"de\") - default: auto-detect\n");
        fprintf(stderr, "  task: Optional task (\"transcribe\" or \"translate\") - default: \"transcribe\"\n");
        return EXIT_FAILURE;
    }
    
    const char* model_dir = argv[1];
    const char* language = argc > 2 ? argv[2] : NULL;
    const char* task = argc > 3 ? argv[3] : "transcribe";

    ov_genai_whisper_pipeline* pipeline = NULL;
    ov_genai_whisper_generation_config* config = NULL;
    ov_genai_whisper_decoded_results* results = NULL;
    const char* device = "CPU";  // GPU, NPU can be used as well
    char* output = NULL;
    size_t output_size = 0;
    
    // Sample audio parameters
    const float sample_rate = 16000.0f;  // Whisper requires 16kHz sample rate
    const float duration = 2.0f;  // 2 seconds of audio
    const size_t audio_length = (size_t)(sample_rate * duration);
    float* audio_data = (float*)malloc(audio_length * sizeof(float));
    
    if (!audio_data) {
        fprintf(stderr, "Failed to allocate memory for audio data\n");
        return EXIT_FAILURE;
    }
    
    // Generate sample audio (440 Hz sine wave for 2 seconds)
    // In a real application, you would load audio from a file
    generate_sample_audio(audio_data, audio_length, 440.0f, sample_rate);
    
    printf("Creating Whisper pipeline...\n");
    CHECK_STATUS(ov_genai_whisper_pipeline_create(model_dir, device, 0, &pipeline));
    
    printf("Creating generation config...\n");
    CHECK_STATUS(ov_genai_whisper_generation_config_create(&config));
    
    // Configure the pipeline
    if (language) {
        printf("Setting language to: %s\n", language);
        CHECK_STATUS(ov_genai_whisper_generation_config_set_language(config, language));
    }
    
    printf("Setting task to: %s\n", task);
    CHECK_STATUS(ov_genai_whisper_generation_config_set_task(config, task));
    
    // Enable timestamps
    CHECK_STATUS(ov_genai_whisper_generation_config_set_return_timestamps(config, true));
    
    printf("Running speech recognition on sample audio...\n");
    CHECK_STATUS(ov_genai_whisper_pipeline_generate(pipeline, audio_data, audio_length, config, &results));
    
    // Get the transcription text
    CHECK_STATUS(ov_genai_whisper_decoded_results_get_string(results, NULL, &output_size));
    output = (char*)malloc(output_size);
    if (!output) {
        fprintf(stderr, "Failed to allocate memory for output\n");
        goto err;
    }
    
    CHECK_STATUS(ov_genai_whisper_decoded_results_get_string(results, output, &output_size));
    printf("Transcription: %s\n", output);
    
    // Display individual text results with scores
    size_t texts_count = 0;
    CHECK_STATUS(ov_genai_whisper_decoded_results_get_texts_count(results, &texts_count));
    
    printf("\nDetailed Results (%zu texts):\n", texts_count);
    for (size_t i = 0; i < texts_count; i++) {
        size_t text_size = 0;
        CHECK_STATUS(ov_genai_whisper_decoded_results_get_text_at(results, i, NULL, &text_size));
        
        char* text = (char*)malloc(text_size);
        if (!text) {
            fprintf(stderr, "Failed to allocate memory for text %zu\n", i);
            continue;
        }
        
        CHECK_STATUS(ov_genai_whisper_decoded_results_get_text_at(results, i, text, &text_size));
        
        float score = 0.0f;
        CHECK_STATUS(ov_genai_whisper_decoded_results_get_score_at(results, i, &score));
        
        printf("  [%zu] Score: %.4f, Text: %s\n", i, score, text);
        free(text);
    }
    
    // Display timestamps if available
    bool has_chunks = false;
    CHECK_STATUS(ov_genai_whisper_decoded_results_has_chunks(results, &has_chunks));
    
    if (has_chunks) {
        size_t chunks_count = 0;
        CHECK_STATUS(ov_genai_whisper_decoded_results_get_chunks_count(results, &chunks_count));
        
        printf("\nTimestamp Information (%zu chunks):\n", chunks_count);
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
                fprintf(stderr, "Failed to allocate memory for chunk text %zu\n", i);
                ov_genai_whisper_decoded_result_chunk_free(chunk);
                continue;
            }
            
            CHECK_STATUS(ov_genai_whisper_decoded_result_chunk_get_text(chunk, chunk_text, &chunk_text_size));
            
            printf("  [%zu] %.2fs - %.2fs: %s\n", i, start_ts, end_ts, chunk_text);
            
            free(chunk_text);
            ov_genai_whisper_decoded_result_chunk_free(chunk);
        }
    } else {
        printf("\nNo timestamp information available.\n");
    }
    
    printf("\nSpeech recognition completed successfully!\n");

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
    
    return EXIT_SUCCESS;
}