// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openvino/genai/c/llm_pipeline_c.h"

#define MAX_PROMPT_LENGTH 64
#define MAX_OUTPUT_LENGTH 1024

#define CHECK_STATUS(return_status)                                                      \
    if (return_status == OUT_OF_BOUNDS) {                                                \
        fprintf(stderr, "[WARNING] output buffer is too small, line %d\n", __LINE__);    \
    } else if (return_status != OK) {                                                    \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); \
        return return_status;                                                            \
    }
void print_callback(const char* args) {
    if (args) {
        fprintf(stdout, "%s", args);
        fflush(stdout);
    } else {
        printf("Callback executed with NULL message!\n");
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <MODEL_DIR>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char prompt[MAX_PROMPT_LENGTH], output[MAX_OUTPUT_LENGTH];
    const char* models_path = argv[1];
    const char* device = "CPU";  // GPU, NPU can be used as well
    ov_genai_llm_pipeline* pipeline = NULL;
    CHECK_STATUS(ov_genai_llm_pipeline_create(models_path, "CPU", &pipeline));
    if (pipeline == NULL) {
        fprintf(stderr, "Failed to create LLM pipeline\n");
        return EXIT_FAILURE;
    }

    ov_genai_generation_config* config = NULL;
    CHECK_STATUS(ov_genai_generation_config_create(&config));
    CHECK_STATUS(ov_genai_generation_config_set_max_new_tokens(config, 100));

    CHECK_STATUS(ov_genai_llm_pipeline_start_chat(pipeline));
    printf("question:\n");
    while (fgets(prompt, MAX_PROMPT_LENGTH, stdin)) {
        prompt[strcspn(prompt, "\n")] = 0;

        stream_callback callback;
        callback.callback_func = print_callback;
        ov_genai_llm_pipeline_generate(pipeline, prompt, config, &callback, output, sizeof(output));

        printf("\n----------\nquestion:\n");
    }
    CHECK_STATUS(ov_genai_llm_pipeline_finish_chat(pipeline));
    ov_genai_llm_pipeline_free(pipeline);
    ov_genai_generation_config_free(config);

    return EXIT_SUCCESS;
}
