// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file vlm_pipeline.c
 * @brief Example demonstrating how to use OpenVINO GenAI VLM Pipeline C API
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openvino/genai/c/vlm_pipeline.h"
#include "load_image.h"

#define MAX_PROMPT_LENGTH 64

// Callback function for streaming results
ov_genai_streaming_status_e stream_callback(const char* str, void* args) {
    printf("%s", str);
    fflush(stdout);
    return OV_GENAI_STREAMING_STATUS_RUNNING;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Usage: %s <models_path> <device> <image_path> \n", argv[0]);
        printf("Example: %s ./models CPU ./image.jpg \n", argv[0]);
        return -1;
    }

    const char* models_path = argv[1];
    const char* device = argv[2];
    const char* image_path = argv[3];

    size_t tensor_count;
    const ov_tensor_t** tensors = load_images(image_path, &tensor_count);

    // Create VLM pipeline
    ov_genai_vlm_pipeline* pipeline = NULL;
    ov_genai_vlm_pipeline_create(models_path, device, 0, &pipeline);

    // Set up streaming callback
    streamer_callback callback = {
        .callback_func = stream_callback,
        .args = NULL
    };

    // Generate response
    ov_genai_vlm_decoded_results* results = NULL;
    ov_genai_generation_config* config = NULL;
    ov_genai_generation_config_create(&config);
    ov_genai_generation_config_set_max_new_tokens(config, 100);
    char prompt[MAX_PROMPT_LENGTH];

    ov_genai_vlm_pipeline_start_chat(pipeline);
    printf("question:\n");

    if (fgets(prompt, MAX_PROMPT_LENGTH, stdin)) {
        prompt[strcspn(prompt, "\n")] = 0;
        if (strlen(prompt) > 0) {
            ov_genai_vlm_pipeline_generate(pipeline, prompt, tensors, tensor_count, config, &callback, &results);
            printf("\n----------\nquestion:\n");
        }
    }

    while (fgets(prompt, MAX_PROMPT_LENGTH, stdin)) {
        prompt[strcspn(prompt, "\n")] = 0;
        if (strlen(prompt) == 0) {
            continue; 
        }
        ov_genai_vlm_pipeline_generate(pipeline, prompt, NULL, 0, config, &callback, &results);
        printf("\n----------\nquestion:\n");
    }
    ov_genai_vlm_pipeline_finish_chat(pipeline);


    // Get performance metrics
    ov_genai_perf_metrics* metrics = NULL;
    ov_genai_vlm_decoded_results_get_perf_metrics(results, &metrics);

    // Get final result string
    size_t output_size = 0;
    ov_genai_vlm_decoded_results_get_string(results, NULL, &output_size);
    if (output_size > 0) {
        char* output = (char*)malloc(output_size);
        if (output) {
            ov_genai_vlm_decoded_results_get_string(results, output, &output_size);
            free(output);
        }
    }

    // Cleanup
    ov_genai_vlm_decoded_results_free(results);
    ov_genai_generation_config_free(config);
    ov_genai_vlm_pipeline_free(pipeline);

    return 0;
}
