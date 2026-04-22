// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openvino/genai/c/text2video_pipeline.h"
#include "openvino/c/ov_tensor.h"

#define MAX_PROMPT_LENGTH 1024

#define CHECK_STATUS(return_status)                                                      \
    if (return_status != 0) {                                                            \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); \
        goto err;                                                                        \
    }

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <MODEL_DIR> [DEVICE]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* models_path = argv[1];
    const char* device = (argc == 3) ? argv[2] : "CPU";  // GPU, NPU can be used as well

    ov_genai_text2video_pipeline* pipeline = NULL;
    ov_genai_video_generation_result* results = NULL;
    ov_tensor_t* video_tensor = NULL;
    char prompt[MAX_PROMPT_LENGTH];

    // Initialize the Text-to-Video pipeline
    CHECK_STATUS(ov_genai_text2video_pipeline_create_with_device(models_path, device, 0, &pipeline));

    printf("Enter a video description:\n");
    while (fgets(prompt, MAX_PROMPT_LENGTH, stdin)) {
        // Remove newline character
        prompt[strcspn(prompt, "\n")] = 0;
        
        // Skip empty lines
        if (strlen(prompt) == 0) {
            printf("Enter a video description:\n");
            continue;
        }

        printf("\nGenerating video (this may take a while)...\n");

        // The Text2Video generate call does not use streaming like text generation.
        // It processes the prompt and returns the result at the end.
        results = NULL;
        CHECK_STATUS(ov_genai_text2video_pipeline_generate(pipeline, prompt, 0, &results));

        if (results) {
            printf("Generation complete!\n");
            
            CHECK_STATUS(ov_genai_video_generation_result_get_video(results, &video_tensor));
            
            if (video_tensor) {
                // Print the tensor shape to verify the video dimensions (Frames, Channels, Height, Width, etc.)
                ov_shape_t shape;
                if (ov_tensor_get_shape(video_tensor, &shape) == 0) {
                    printf("Generated Video Tensor Shape: [");
                    for (size_t i = 0; i < shape.rank; i++) {
                        printf("%lld%s", shape.dims[i], (i < shape.rank - 1) ? ", " : "]\n");
                    }
                    ov_shape_free(&shape);
                } else {
                    fprintf(stderr, "[ERROR] Failed to get video tensor shape.\n");
                }
                
                // Typical C behavior: To save this to an MP4 or GIF, you'd integrate 
                // FFmpeg or OpenCV here to write the raw tensor buffer to a file.
                
                ov_tensor_free(video_tensor);
                video_tensor = NULL;
            }

            ov_genai_video_generation_result_free(results);
            results = NULL;
        }

        printf("\n----------\nEnter a video description:\n");
    }

err:
    if (video_tensor)
        ov_tensor_free(video_tensor);
    if (results)
        ov_genai_video_generation_result_free(results);
    if (pipeline)
        ov_genai_text2video_pipeline_free(pipeline);

    return EXIT_SUCCESS;
}
