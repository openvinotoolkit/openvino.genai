// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <stdio.h>
#include <stdlib.h>

#include "openvino/genai/c/llm_pipeline_c.h"

#define CHECK_STATUS(return_status)                                                      \
    if (return_status != OK) {                                                           \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); \
        return return_status;                                                            \
    }

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <MODEL_DIR> \"<PROMPT>\"\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* model_dir = argv[1];
    const char* prompt = argv[2];

    ov_genai_llm_pipeline* pipeline = NULL;
    CHECK_STATUS(ov_genai_llm_pipeline_create(model_dir, "CPU", &pipeline));
    if (pipeline == NULL) {
        fprintf(stderr, "Failed to create LLM pipeline\n");
        return EXIT_FAILURE;
    }
    ov_genai_generation_config* config = NULL;
    CHECK_STATUS(ov_genai_generation_config_create(&config));
    CHECK_STATUS(ov_genai_generation_config_set_max_new_tokens(config, 100));

    char output[1024];
    CHECK_STATUS(ov_genai_llm_pipeline_generate(pipeline, prompt, config, NULL, output, sizeof(output)));

    printf("Generated text: %s\n", output);

    ov_genai_llm_pipeline_free(pipeline);
    ov_genai_generation_config_free(config);

    return EXIT_SUCCESS;
}
