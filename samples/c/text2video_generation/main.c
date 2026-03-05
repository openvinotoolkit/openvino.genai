// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openvino/genai/c/text2video_pipeline.h"
// 1. CHANGE: Include the Video Config header instead of the LLM one
#include "openvino/genai/c/video_generation_config.h" 
#include "openvino/c/openvino.h"


int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <MODEL_DIR> \"<PROMPT>\"\n", argv[0]);
        printf("Example: %s ./models/sora \"A cat running in a park\"\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* models_path = argv[1];
    const char* prompt = argv[2];
    const char* device = "CPU"; // Defaulting to CPU for the sample

    ov_status_e status;

    // 1. Initialize the Text2Video Pipeline
    printf("--- Initializing Pipeline ---\n");
    ov_genai_text2video_pipeline* pipe = NULL;
    status = ov_genai_text2video_pipeline_create(models_path, device, 0, &pipe);
    if (status != OK) {
        printf("Failed to create pipeline. Error code: %d\n", status);
        return EXIT_FAILURE;
    }

    // 2. Setup the Generation Configuration
    printf("--- Setting up Video Configuration ---\n");
    // 2. CHANGE: Use the video_generation_config type
    ov_genai_video_generation_config* config = NULL;
    ov_genai_video_generation_config_create(&config);
    
    // 3. CHANGE: Use the video-prefixed setter functions
    ov_genai_video_generation_config_set_width(config, 512);
    ov_genai_video_generation_config_set_height(config, 512);
    ov_genai_video_generation_config_set_num_frames(config, 16);
    ov_genai_video_generation_config_set_num_inference_steps(config, 20);
    // ov_genai_video_generation_config_set_guidance_scale(config, 7.5f);

    // 3. Generate the Video Tensor
    printf("--- Generating Video ---\n");
    printf("Prompt: '%s'\n", prompt);
    
    ov_tensor_t* video_tensor = NULL;
    status = ov_genai_text2video_pipeline_generate(pipe, prompt, config, &video_tensor);
    if (status != OK) {
        printf("Failed to generate video. Error code: %d\n", status);
        // 4. CHANGE: Use the video free function
        ov_genai_video_generation_config_free(config);
        ov_genai_text2video_pipeline_free(pipe);
        return EXIT_FAILURE;
    }

    // 4. Verify the Output
    ov_shape_t tensor_shape;
    ov_tensor_get_shape(video_tensor, &tensor_shape);
    
    printf("\nSuccess! Video tensor generated.\n");
    printf("Tensor Shape: [");
    for (size_t i = 0; i < tensor_shape.rank; ++i) {
        printf("%" PRId64 "%s", tensor_shape.dims[i], (i < tensor_shape.rank - 1) ? ", " : "");
    }
    printf("]\n");

    // 5. Clean up memory
    ov_shape_free(&tensor_shape);
    ov_tensor_free(video_tensor);
    // 5. CHANGE: Use the video free function here as well
    ov_genai_video_generation_config_free(config);
    ov_genai_text2video_pipeline_free(pipe);

    printf("--- Done ---\n");
    return EXIT_SUCCESS;
}