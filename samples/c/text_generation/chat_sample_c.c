// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>

#include "openvino/genai/openvino_genai_c.h"

#define MAX_PROMPT_LENGTH 64
#define MAX_OUTPUT_LENGTH 256

void streamer(const char* word) {
    printf("%s", word);
    fflush(stdout);
}
int main(int argc, char* argv[]) {
    printf("This is a C API example for OpenVINO GenAI.\n");
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <MODEL_DIR>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char prompt[MAX_PROMPT_LENGTH], output[MAX_OUTPUT_LENGTH];
    const char* models_path = argv[1];
    const char* device = "CPU";  // GPU, NPU can be used as well
    LLMPipelineHandle* pipeline = CreateLLMPipeline(models_path, "CPU");
    if (pipeline == NULL) {
        fprintf(stderr, "Failed to create LLM pipeline\n");
        return EXIT_FAILURE;
    }

    GenerationConfigHandle* config = CreateGenerationConfig();
    GenerationConfigSetMaxNewTokens(config, 100);

    LLMPipelineStartChat(pipeline);
    printf("question:\n");
    while (fgets(prompt, MAX_PROMPT_LENGTH, stdin)) {
        prompt[strcspn(prompt, "\n")] = 0;

        LLMPipelineGenerate(pipeline, prompt, output, sizeof(output), config);
        streamer(output);

        printf("\n----------\nquestion:\n");
    }
    LLMPipelineFinishChat(pipeline);

    DestroyLLMPipeline(pipeline);
    DestroyGenerationConfig(config);

    return EXIT_SUCCESS;
}
