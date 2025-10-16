// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file vlm_pipeline_example.c
 * @brief Example demonstrating how to use OpenVINO GenAI VLM Pipeline C API
 */

#include <stdio.h>
#include <stdlib.h>
#include "openvino/genai/c/vlm_pipeline.h"
#include "load_image.h"

// Callback function for streaming results
ov_genai_streaming_status_e stream_callback(const char* str, void* args) {
    printf("%s", str);
    fflush(stdout);
    return OV_GENAI_STREAMING_STATUS_RUNNING;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Usage: %s <models_path> <device> <image_path> [text_prompt]\n", argv[0]);
        printf("Example: %s ./models CPU ./image.jpg \"Describe this image\"\n", argv[0]);
        return -1;
    }

    const char* models_path = argv[1];
    const char* device = argv[2];
    const char* image_path = argv[3];
    const char* text_prompt = (argc > 4) ? argv[4] : "Describe this image";

    size_t tensor_count;
    const ov_tensor_t** tensors = load_images(image_path, &tensor_count);

    // Create VLM pipeline
    ov_genai_vlm_pipeline* pipeline = NULL;
    ov_genai_vlm_pipeline_create(models_path, device, 0, &pipeline);

    printf("VLM Pipeline created successfully!\n");
    printf("Models path: %s\n", models_path);
    printf("Device: %s\n", device);
    printf("Image path: %s\n", image_path);
    printf("Text prompt: %s\n", text_prompt);
    printf("\nGenerating response...\n\n");

    // Set up streaming callback
    streamer_callback callback = {
        .callback_func = stream_callback,
        .args = NULL
    };

    // Generate response
    ov_genai_vlm_decoded_results* results = NULL;
    ov_genai_vlm_pipeline_generate(pipeline, text_prompt, tensors, tensor_count, NULL, &callback, &results);

    printf("\n\nGeneration completed successfully!\n");

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
    ov_genai_vlm_pipeline_free(pipeline);

    printf("Example completed successfully!\n");
    return 0;
}
