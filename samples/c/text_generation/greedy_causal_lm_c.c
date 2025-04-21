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
    const char* device = "CPU";  // GPU, NPU can be used as well
    char* output = NULL;  // The output of the generation function. The caller is responsible for allocating and freeing
                          // the memory.
    size_t output_size = 0;  // Used to store the required size of the output buffer.

    CHECK_STATUS(ov_genai_llm_pipeline_create(model_dir, device, 0, &pipeline));
    CHECK_STATUS(ov_genai_generation_config_create(&config));
    CHECK_STATUS(ov_genai_generation_config_set_max_new_tokens(config, 100));
    CHECK_STATUS(ov_genai_llm_pipeline_generate(pipeline, prompt, config, NULL, &results));

    // The function is called with NULL as the output to determine the required buffer size.
    CHECK_STATUS(ov_genai_decoded_results_get_string(results, NULL, &output_size));
    output = (char*)malloc(output_size);  // Allocate memory for the output string based on the determined size.
    if (!output) {
        fprintf(stderr, "Failed to allocate memory for output\n");
        goto err;
    }

    // Retrieve the actual output string from the results into the allocated buffer.
    CHECK_STATUS(ov_genai_decoded_results_get_string(results, output, &output_size));
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
