// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <stdio.h>
#include <stdlib.h>

#include "openvino/genai/c/llm_pipeline.h"

#define CHECK_STATUS(return_status)                                                      \
    if (return_status != OK) {                                                           \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); \
        goto err;                                                                        \
    }

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <MODEL_DIR> \"<PROMPT>\"\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* model_dir = argv[1];
    const char* prompt = argv[2];

    ov_genai_llm_pipeline* pipeline = NULL;
    ov_genai_generation_config* config = NULL;
    ov_genai_decoded_results* results = NULL;
    const char* device = "CPU";        // GPU, NPU can be used as well
    char* output = (char*)malloc(16);  // Firstly, malloc a small size (though it may not necessarily be that small),
                                       // used only for testing to allow realloc logic to run.
    size_t required_size = 0;          // Used to store the required size of the output buffer.

    if (!output) {
        fprintf(stderr, "[Error] Memory allocation failed (malloc 5 bytes).\n");
        goto err;
    }
    CHECK_STATUS(ov_genai_llm_pipeline_create(model_dir, device, &pipeline));
    CHECK_STATUS(ov_genai_generation_config_create(&config));
    CHECK_STATUS(ov_genai_generation_config_set_max_new_tokens(config, 100));
    CHECK_STATUS(ov_genai_llm_pipeline_generate_decoded_results(pipeline, prompt, config, NULL, &results));
    CHECK_STATUS(ov_genai_decoded_results_get_string(results,
                                                     NULL,
                                                     0,
                                                     &required_size));  // get the required size of the output buffer.
    if (required_size > sizeof(output)) {
        char* temp = (char*)realloc(output, required_size);
        if (!temp) {
            fprintf(stderr, "[Error] Memory allocation failed (malloc %zu bytes).\n", required_size);
            goto err;
        }
        output = temp;
    }
    CHECK_STATUS(ov_genai_decoded_results_get_string(results, output, required_size, NULL));
    printf("%s\n", output);

err:
    if (pipeline)
        ov_genai_llm_pipeline_free(pipeline);
    if (config)
        ov_genai_generation_config_free(config);
    if (output)
        free(output);

    return EXIT_SUCCESS;
}
