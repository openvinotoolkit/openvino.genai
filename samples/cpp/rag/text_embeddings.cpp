// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/text_embedding_pipeline.hpp"

std::vector<std::string> texts = {
    "Machine learning is a subset of artificial intelligence focused on building systems that learn from data. Supervised learning involves labeled examples, while unsupervised learning looks for patterns without predefined categories. Popular algorithms include linear regression, decision trees, and gradient boosting.",
    "To brew good coffee, start with freshly roasted beans and grind them right before extraction. Water temperature should be between 92–96°C. For pour-over, aim for a brew time of 2.5–3.5 minutes. Adjust grind size to control extraction: finer for more bitterness, coarser for more acidity.",
    "Cycling improves cardiovascular health, increases lower-body strength, and enhances lung capacity. Regular riders often experience reduced stress and better sleep quality. Commuting by bike is also environmentally friendly, lowering CO₂ emissions compared with driving.",
    "Semantic versioning uses the format MAJOR.MINOR.PATCH. A major change introduces breaking updates, a minor version adds new functionality in a backward-compatible way, and a patch version includes small fixes. Following SemVer helps maintain predictability for users and developers.",
    "A balanced diet includes vegetables, fruits, whole grains, lean proteins, and healthy fats. Limiting added sugars and processed food reduces the risk of metabolic disorders. Hydration is also essential: adults should drink roughly 2–2.5 liters of water per day depending on activity level.",
    "Quantum computers use qubits instead of classical bits. Thanks to superposition and entanglement, they can explore many computational states simultaneously. Algorithms like Shor’s and Grover’s demonstrate significant theoretical speed-ups for factoring and search problems.",
    "Munich is known for its beer gardens, English Garden, and proximity to the Alps. Public transport is reliable and includes U-Bahn, S-Bahn, trams, and buses. Visitors should try local dishes such as Weißwurst, Brezn, and Kaiserschmarrn. The best time to visit is late spring or early autumn.",
    "In an emergency, first ensure the environment is safe. Check the victim’s responsiveness and breathing. Call emergency services if necessary. For heavy bleeding, apply direct pressure with a clean cloth. For burns, cool the area with running water for at least 10 minutes.",
};

std::vector<std::string> concat_strings(const std::vector<std::string>& origin) {
    std::stringstream ss;
    for (const auto& str : origin) {
        ss << str << " ";
    }
    return {ss.str()};
}

int main(int argc, char* argv[]) try {
        std::string models_path = argv[1];

        std::string device = "CPU";  // GPU can be used as well
        for (int i = 1; i <= texts.size(); i++){
            std::vector<std::string> documents;
            for (int j = 0; j < i; j++){
                documents.push_back(texts[j]);
            }
            std::cout << "Inputs size: " << documents.size()  << std::endl;
        {
            ov::genai::TextEmbeddingPipeline::Config config;
            config.pooling_type = ov::genai::TextEmbeddingPipeline::PoolingType::MEAN;
            config.batch_size = documents.size();

            ov::genai::TextEmbeddingPipeline pipeline(models_path, device, config);

            auto start = std::chrono::high_resolution_clock::now();
            ov::genai::EmbeddingResults documents_embeddings = pipeline.embed_documents(documents);
            auto end = std::chrono::high_resolution_clock::now();
            std::cout << "Multibatch time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
        }
        {
            ov::genai::TextEmbeddingPipeline::Config config;
            config.pooling_type = ov::genai::TextEmbeddingPipeline::PoolingType::MEAN;
            config.streams = documents.size();

            ov::genai::TextEmbeddingPipeline pipeline(models_path, device, config);

            auto start = std::chrono::high_resolution_clock::now();
            ov::genai::EmbeddingResults documents_embeddings = pipeline.embed_documents(documents);
            auto end = std::chrono::high_resolution_clock::now();
            std::cout << "Multistream time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl  << std::endl;
        }
    }
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
