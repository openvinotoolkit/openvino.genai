// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openvino/genai/c/llm_pipeline.h"

#define MAX_PROMPT_LENGTH 64
#define MAX_OUTPUT_LENGTH 1024

#define CHECK_STATUS(return_status)                                                      \
    if (return_status != OK) {                                                           \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); \
        goto err;                                                                        \
    }

ov_genai_streamming_status_e print_callback(const char* args) {
    if (args) {
        fprintf(stdout, "%s", args);
        fflush(stdout);
        return OV_GENAI_STREAMMING_STATUS_RUNNING;
    } else {
        printf("Callback executed with NULL message!\n");
        return OV_GENAI_STREAMMING_STATUS_STOP;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <MODEL_DIR>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* models_path = argv[1];
    const char* device = "CPU";  // GPU, NPU can be used as well

    ov_genai_generation_config* config = NULL;
    ov_genai_llm_pipeline* pipeline = NULL;
    stream_callback streamer = &print_callback;
    char prompt[MAX_PROMPT_LENGTH];

    CHECK_STATUS(ov_genai_llm_pipeline_create(models_path, device, &pipeline));
    CHECK_STATUS(ov_genai_generation_config_create(&config));
    CHECK_STATUS(ov_genai_generation_config_set_max_new_tokens(config, 100));

    CHECK_STATUS(ov_genai_llm_pipeline_start_chat(pipeline));
    printf("question:\n");
    while (fgets(prompt, MAX_PROMPT_LENGTH, stdin)) {
        prompt[strcspn(prompt, "\n")] = 0;
        CHECK_STATUS(ov_genai_llm_pipeline_generate(pipeline,
                                                    prompt,
                                                    config,
                                                    &streamer,
                                                    NULL));  // Only the streamer functionality is used here.
        printf("\n----------\nquestion:\n");
    }
    CHECK_STATUS(ov_genai_llm_pipeline_finish_chat(pipeline));

err:
    if (pipeline)
        ov_genai_llm_pipeline_free(pipeline);
    if (config)
        ov_genai_generation_config_free(config);

    return EXIT_SUCCESS;
}
