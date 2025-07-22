// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/text_embedding_pipeline.hpp"

int main(int argc, char* argv[]) try {
    auto documents = std::vector<std::string>(argv + 2, argv + argc);
    std::string models_path = argv[1];

    ov::genai::TextEmbeddingPipeline pipeline(
        models_path,
        "CPU",
        ov::genai::pooling_type(ov::genai::TextEmbeddingPipeline::PoolingType::MEAN),
        ov::genai::normalize(true)
    );

    ov::genai::EmbeddingResults embeddings = pipeline.embed_documents(documents);
    ov::genai::EmbeddingResult query_embedding = pipeline.embed_query("What is the capital of France?");
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}
