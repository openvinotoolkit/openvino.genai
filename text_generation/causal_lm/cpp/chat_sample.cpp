// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include "openvino/genai/llm_pipeline.hpp"

using namespace std;

std::vector<string> questions = {
    "1+1=", 
    "what was the previous answer?", 
    "Why is the sky blue?", 
    "4+10=",
    "What is Intel OpenVINO?",
    "Can you briefly summarize what I asked you about during this session?",
};

int main(int argc, char* argv[]) try {
    std::string prompt;
    std::string accumulated_str = "";

    std::string model_path = argv[1];
    ov::genai::LLMPipeline pipe(model_path, "CPU");
    
    ov::genai::GenerationConfig config = pipe.get_generation_config();
    config.max_new_tokens = 10000;
    auto streamer = [](std::string word) { std::cout << word << std::flush; };

    pipe.start_chat();
    for (size_t i = 0; i < questions.size(); i++) {
        // std::getline(std::cin, prompt);
        prompt = questions[i];
        
        std::cout << "question:\n";
        cout << prompt << endl;

        // auto answer_str = pipe(prompt, config, streamer);
        auto answer_str = pipe.generate(prompt, ov::genai::max_new_tokens(10000), ov::genai::streamer(streamer));
        accumulated_str += answer_str;
        
        cout << "\n----------\n";
    }
    pipe.finish_chat();
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
