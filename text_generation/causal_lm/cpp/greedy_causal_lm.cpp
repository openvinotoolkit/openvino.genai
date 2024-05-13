// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (3 > argc || argc > 4)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\" <DEVICE>");

    std::string model_path = argv[1];
    std::string prompt = argv[2];
    
    // GPU can be used as well
    std::string device = "CPU";  
    if (argc > 3) device = argv[3];

    ov::LLMPipeline pipe(model_path, device);
    ov::GenerationConfig config = pipe.get_generation_config();
    config.max_new_tokens = 100;
    auto streamer = [](std::string subword){std::cout << subword << std::flush;};
    
    // since streamer is set results will be printed each time a new token is generated
    pipe.generate(prompt, config, streamer);
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
