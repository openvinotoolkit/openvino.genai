// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <stdio.h>
#include <stdlib.h>

#include "openvino/genai/c/text_embedding_pipeline.h"

#define CHECK_STATUS(return_status)                                                      \
    if (return_status != OK) {                                                           \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); \
        goto err;                                                                        \
    }

static const char* dtype_name(ov_genai_embedding_dtype_e dtype) {
    switch (dtype) {
    case OV_GENAI_EMBEDDING_DTYPE_F32:
        return "f32";
    case OV_GENAI_EMBEDDING_DTYPE_I8:
        return "i8";
    case OV_GENAI_EMBEDDING_DTYPE_U8:
        return "u8";
    default:
        return "?";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <MODEL_DIR> \"<TEXT 1>\" [\"<TEXT 2>\" ...]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* model_dir = argv[1];
    const char** documents = (const char**)&argv[2];
    size_t documents_count = (size_t)(argc - 2);

    ov_genai_text_embedding_pipeline* pipeline = NULL;
    ov_genai_embedding_results* docs_embeddings = NULL;
    ov_genai_embedding_result* query_embedding = NULL;
    const char* device = "CPU";  // GPU can be used as well
    int exit_code = EXIT_FAILURE;

    CHECK_STATUS(
        ov_genai_text_embedding_pipeline_create(model_dir, device, 2, &pipeline, "pooling_type", "MEAN"));

    CHECK_STATUS(
        ov_genai_text_embedding_pipeline_embed_documents(pipeline, documents, documents_count, &docs_embeddings));

    ov_genai_embedding_dtype_e docs_dtype = OV_GENAI_EMBEDDING_DTYPE_F32;
    size_t docs_count = 0;
    CHECK_STATUS(ov_genai_embedding_results_get_dtype(docs_embeddings, &docs_dtype));
    CHECK_STATUS(ov_genai_embedding_results_get_count(docs_embeddings, &docs_count));
    printf("Documents: %zu embeddings of dtype %s\n", docs_count, dtype_name(docs_dtype));
    for (size_t i = 0; i < docs_count; ++i) {
        size_t embedding_size = 0;
        CHECK_STATUS(ov_genai_embedding_results_get_size_at(docs_embeddings, i, &embedding_size));
        printf("  [%zu] length=%zu\n", i, embedding_size);
    }

    CHECK_STATUS(
        ov_genai_text_embedding_pipeline_embed_query(pipeline, "What is the capital of France?", &query_embedding));

    ov_genai_embedding_dtype_e query_dtype = OV_GENAI_EMBEDDING_DTYPE_F32;
    size_t query_size = 0;
    CHECK_STATUS(ov_genai_embedding_result_get_dtype(query_embedding, &query_dtype));
    CHECK_STATUS(ov_genai_embedding_result_get_size(query_embedding, &query_size));
    printf("Query embedding: dtype=%s length=%zu\n", dtype_name(query_dtype), query_size);

    if (query_dtype == OV_GENAI_EMBEDDING_DTYPE_F32) {
        const float* data = NULL;
        size_t size = 0;
        CHECK_STATUS(ov_genai_embedding_result_get_data_f32(query_embedding, &data, &size));
        size_t preview = size < 8 ? size : 8;
        printf("  first %zu values: [", preview);
        for (size_t i = 0; i < preview; ++i) {
            printf("%s%.4f", i == 0 ? "" : ", ", data[i]);
        }
        printf("%s]\n", preview < size ? ", ..." : "");
    }

    exit_code = EXIT_SUCCESS;

err:
    if (query_embedding)
        ov_genai_embedding_result_free(query_embedding);
    if (docs_embeddings)
        ov_genai_embedding_results_free(docs_embeddings);
    if (pipeline)
        ov_genai_text_embedding_pipeline_free(pipeline);

    return exit_code;
}
