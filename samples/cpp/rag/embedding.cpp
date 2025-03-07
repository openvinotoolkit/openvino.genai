// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag_components/embedding_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<TEXT 1>' ['<TEXT 2>' ...]");
    }
    auto documents = std::vector<std::string>(argv + 2, argv + argc);
    std::string models_path = argv[1];

    std::string device = "CPU";  // GPU can be used as well
    ov::genai::TextEmbeddingPipeline pipe(models_path, device);

    auto embeddings = pipe.embed_documents(documents);
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