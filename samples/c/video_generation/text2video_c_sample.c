// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/text2video_pipeline.h"
#include <stdio.h>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <model_path> <device>\n", argv[0]);
        return 1;
    }

    ov_genai_text2video_pipeline* pipe = NULL;
    int status = ov_genai_text2video_pipeline_create(argv[1], argv[2], &pipe);

    if (status != 0) {
        printf("Failed to create pipeline\n");
        return 1;
    }

    printf("Generating video...\n");

    text2video_custom_tensor result = {0};
    ov_genai_text2video_pipeline_generate(pipe, "A cat eating pizza", &result);

    if (result.data != NULL) {
        printf("Success! Video generated.\n");
        printf("Dimensions: %zu x %zu, Bytes: %zu\n", result.width, result.height, result.size);
        ov_genai_text2video_free_tensor(&result);
    } else {
        printf("Error: Generation failed.\n");
    }

    ov_genai_text2video_pipeline_destroy(pipe);
    printf("Done!\n");
    return 0;
}