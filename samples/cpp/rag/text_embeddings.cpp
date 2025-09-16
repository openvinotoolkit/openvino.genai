// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/text_embedding_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<TEXT 1>' ['<TEXT 2>' ...]");
    }
    auto documents = std::vector<std::string>(argv + 2, argv + argc);
    std::string models_path = argv[1];

    std::string device = "CPU";  // GPU can be used as well

    ov::genai::TextEmbeddingPipeline::Config config;
    config.pooling_type = ov::genai::TextEmbeddingPipeline::PoolingType::MEAN;

    ov::genai::TextEmbeddingPipeline pipeline(models_path, device, config);

    ov::genai::EmbeddingResults documents_embeddings = pipeline.embed_documents(documents);
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
