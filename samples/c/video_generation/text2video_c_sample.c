// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "openvino/genai/c/text2video_pipeline.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <MODEL_DIR> '<PROMPT>'\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* models_dir = argv[1];
    const char* prompt = argv[2];
    const char* device = "CPU";
    const char* negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted";

    ov_genai_video_generation_config config = {0};
    config.width = 704;
    config.height = 480;
    config.num_frames = 161;
    config.frame_rate = 25;

    ov_genai_text2video_pipeline* pipe = NULL;
    text2video_custom_tensor output_tensor = {0};

    printf("Loading Text-to-Video models from: %s\n", models_dir);
    int status = ov_genai_text2video_pipeline_create(models_dir, device, &pipe);

    if (status != 0 || pipe == NULL) {
        fprintf(stderr, "Failed to create Text2Video pipeline.\n");
        return EXIT_FAILURE;
    }

    printf("Generating %d frames. This might take a while...\n", config.num_frames);

    ov_genai_text2video_pipeline_generate(
        pipe,
        prompt,
        negative_prompt,
        &config,
        &output_tensor);

    if (output_tensor.data != NULL && output_tensor.size > 0) {
        const char* output_filename = "generated_video.raw";
        FILE* f = fopen(output_filename, "wb");

        if (f != NULL) {
            fwrite(output_tensor.data, 1, output_tensor.size, f);
            fclose(f);

            printf("\nSuccessfully generated %d frames.\n", config.num_frames);
            printf("Raw video data saved to: %s\n", output_filename);
        } else {
            fprintf(stderr, "Failed to open %s for writing.\n", output_filename);
        }
    } else {
        fprintf(stderr, "Output tensor data is null or size is 0. Generation failed.\n");
    }

    ov_genai_text2video_free_tensor(&output_tensor);

    if (pipe != NULL) {
        ov_genai_text2video_pipeline_destroy(pipe);
    }
    return EXIT_SUCCESS;
}