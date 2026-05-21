// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <stdio.h>
#include <stdlib.h>

#include "openvino/genai/c/text_rerank_pipeline.h"

#define CHECK_STATUS(return_status)                                                      \
    if (return_status != OK) {                                                           \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); \
        goto err;                                                                        \
    }

int main(int argc, char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <MODEL_DIR> \"<QUERY>\" \"<TEXT 1>\" [\"<TEXT 2>\" ...]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* model_dir = argv[1];
    const char* query = argv[2];
    const char** documents = (const char**)&argv[3];
    size_t documents_count = (size_t)(argc - 3);

    ov_genai_text_rerank_pipeline* pipeline = NULL;
    ov_genai_text_rerank_result* result = NULL;
    const char* device = "CPU";  // GPU can be used as well
    size_t result_size = 0;
    int exit_code = EXIT_FAILURE;

    // top_n=3 limits the result to the three highest-scoring documents.
    CHECK_STATUS(ov_genai_text_rerank_pipeline_create(model_dir, device, 2, &pipeline, "top_n", "3"));
    CHECK_STATUS(ov_genai_text_rerank_pipeline_rerank(pipeline, query, documents, documents_count, &result));

    CHECK_STATUS(ov_genai_text_rerank_result_get_size(result, &result_size));
    printf("Reranked documents:\n");
    for (size_t i = 0; i < result_size; ++i) {
        size_t index = 0;
        float score = 0.0f;
        CHECK_STATUS(ov_genai_text_rerank_result_get_item(result, i, &index, &score));
        printf("Document %zu (score: %.4f): %s\n", index, score, documents[index]);
    }

    exit_code = EXIT_SUCCESS;

err:
    if (result)
        ov_genai_text_rerank_result_free(result);
    if (pipeline)
        ov_genai_text_rerank_pipeline_free(pipeline);

    return exit_code;
}
