// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Include your custom C API Header
#include "openvino/genai/c/text2video_pipeline.h"

int main(int argc, char* argv[]) {
    // Ensure the user passes the correct arguments
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <MODEL_DIR> '<PROMPT>'\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* models_dir = argv[1];
    const char* prompt = argv[2];
    const char* device = "CPU";

    // Define parameters matching the header signature
    const char* negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted";
    int width = 704;
    int height = 480;
    int num_frames = 161;
    int frame_rate = 25; // 

    // Allocate pointers based on your custom header
    ov_genai_text2video_pipeline* pipe = NULL;
    
    // Allocate the custom tensor struct locally (the API will populate its fields)
    text2video_custom_tensor output_tensor = {0};

    // Create the pipeline
    printf("Loading Text-to-Video models from: %s\n", models_dir);
    int status = ov_genai_text2video_pipeline_create(models_dir, device, &pipe);
    
    if (status != 0 || pipe == NULL) {
        fprintf(stderr, "Failed to create Text2Video pipeline.\n");
        return EXIT_FAILURE;
    }

    // Run Generation
    printf("Generating %d frames. This might take a while...\n", num_frames);
    ov_genai_text2video_pipeline_generate(
        pipe, prompt, negative_prompt, width, height, num_frames, frame_rate, &output_tensor);

    // The Video Saving Logic 
    if (output_tensor.data != NULL && output_tensor.size > 0) {
        const char* output_filename = "generated_video.raw";
        FILE* f = fopen(output_filename, "wb");
        
        if (f != NULL) {
            fwrite(output_tensor.data, 1, output_tensor.size, f);
            fclose(f);
            
            printf("\nSuccessfully generated %d frames.\n", num_frames);
            printf("Raw video data saved to: %s\n", output_filename);
            printf("To validate and play the output, use FFmpeg:\n");
            // Assuming RGB24 format. If your pipeline outputs something else, adjust pixel_format.
            printf("ffplay -f rawvideo -pixel_format rgb24 -video_size %dx%d -framerate %d %s\n", 
                   width, height, frame_rate, output_filename);
        } else {
            fprintf(stderr, "Failed to open %s for writing.\n", output_filename);
        }
    } else {
        fprintf(stderr, "Output tensor data is null or size is 0. Generation failed.\n");
    }

    // Cleanup memory using your custom destructors
    ov_genai_text2video_free_tensor(&output_tensor);
    
    if (pipe != NULL) {
        ov_genai_text2video_pipeline_destroy(pipe);
    }

    return EXIT_SUCCESS;
}
