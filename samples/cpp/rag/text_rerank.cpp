// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/text_rerank_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (argc < 4) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                 " <MODEL_DIR> '<QUERY>' '<TEXT 1>' ['<TEXT 2>' ...]");
    }

    auto documents = std::vector<std::string>(argv + 3, argv + argc);
    std::string models_path = argv[1];
    std::string query = argv[2];

    std::string device = "CPU";  // GPU can be used as well

    ov::genai::TextRerankPipeline::Config config;
    config.top_n = 3;

    ov::genai::TextRerankPipeline pipeline(models_path, device, config);

    std::vector<std::pair<size_t, float>> rerank_result = pipeline.rerank(query, documents);

    // print reranked documents
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Reranked documents:\n";
    for (const auto& [index, score] : rerank_result) {
        std::cout << "Document " << index << " (score: " << score << "): " << documents[index] << '\n';
    }
    std::cout << std::defaultfloat;

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
