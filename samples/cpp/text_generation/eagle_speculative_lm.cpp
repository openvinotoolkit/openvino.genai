// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <openvino/openvino.hpp>

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (4 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <EAGLE_MODEL_DIR> '<PROMPT>'");
    }

    std::string main_model_path = argv[1];
    std::string eagle_model_path = argv[2];
    std::string prompt = argv[3];

    // Configure devices - can run main and eagle models on different devices
    std::string main_device = "GPU", eagle_device = "GPU"; // currently only GPU is used during developing

    // Eagle Speculative settings
    ov::genai::GenerationConfig config = ov::genai::greedy();
    config.max_new_tokens = 100;
    // Eagle specific parameters
    config.eagle_model = true;
    config.num_return_sequences = 1; // only support 1

    //config.eagle_tree_width = 3;    // Number of candidate tokens to consider at each level
    //config.eagle_tree_depth = 4;    // How deep to explore the token tree

    // Create pipeline with eagle speculative enabled
    ov::genai::LLMPipeline pipe(
        main_model_path,
        main_device,
        ov::genai::draft_model(eagle_model_path, eagle_device),
        std::pair<std::string, ov::Any>("eagle_mode", ov::Any("EAGLE2"))  // Specify eagle2 mode for draft model
    );
    // Setup performance measurement
    auto start_time = std::chrono::high_resolution_clock::now();

    // Optional: Create a streaming callback for real-time token display
    auto streamer = [](std::string subword) {
        std::cout << subword << std::flush;
        return ov::genai::StreamingStatus::RUNNING;
    };

    // Run generation with eagle speculative decoding
    std::cout << "Generating with Eagle Speculative decoding:" << std::endl;
    pipe.generate(prompt, config, streamer);
    std::cout << std::endl;

    // Calculate and display performance metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "\nGeneration completed in " << duration.count() << " ms" << std::endl;

    // Run without Eagle for comparison
    std::cout << "\n-----------------------------" << std::endl;
    std::cout << "Generating without Eagle Speculative decoding:" << std::endl;

    // Disable Eagle mode
    /*config.eagle_model = false;
    
    start_time = std::chrono::high_resolution_clock::now();
    pipe.generate(prompt, config, streamer);
    std::cout << std::endl;
    */
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "\nStandard generation completed in " << duration.count() << " ms" << std::endl;

} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}