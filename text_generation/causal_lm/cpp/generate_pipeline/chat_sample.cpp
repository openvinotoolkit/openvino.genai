// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include "llm_pipeline.hpp"


std::string generate_chat_prompt(const LLMPipeline& pipe, std::string& input, bool first_iter = false, bool use_chat_template = true) {
    if (use_chat_template)
        return pipe.apply_chat_template(input);

    std::stringstream result_prompt;
    string prefix = (first_iter) ? "" : "<\n>";

    // Gemma-7b-it
    // result_prompt << "<bos><start_of_turn>user\n" << input << "<end_of_turn>\n<start_of_turn>model";
    
    // TinyLlama
    result_prompt << "<|user|>\n" << input << "</s>\n<|assistant|>\n";

    // LLama-2-7b
    // result_prompt << "<s>[INST] " << input << " [/INST]";
    return result_prompt.str();
}

int main(int argc, char* argv[]) try { 
    std::string prompt = "table is made of";
    std::string device = "CPU"; // can be replaced with GPU

    std::string model_path = argv[1];
    if (argc > 2)
        prompt = argv[2];
    if (argc > 3)
        device = argv[3];

    LLMPipeline pipe(model_path, device);

    GenerationConfig config = pipe.generation_config();
    config.max_new_tokens(10000);
    pipe.set_streamer([](std::string word) { std::cout << word << std::flush; });
    
    vector<string> questions = {
        "1+1=", 
        "what was the previous answer?", 
        "Why is the sky blue?", 
        "4+10=",
        "Who was Alan Turing?",
        "But why did he killed himself?",
        // "What is Intel OpenVINO?",
        // "4+10=", 
        // "sum up all the numeric answers in the current chat session"
        // "Why is the sky blue?",
        // "Please repeat all the questions I asked you.",
        "Can you briefly summarize what I asked you about during this session?",
    };

    std::string accumulated_str = "";
    pipe.start_conversation();
    for (size_t i = 0; i < questions.size(); i++) {
        prompt = questions[i];
        
        bool first_iter = (i == 0) ? true : false;
        bool last_iter = (i == questions.size() - 1) ? true : false;
        
        std::cout << "question:\n";
        // std::getline(std::cin, prompt);
        cout << prompt << endl;
        prompt = generate_chat_prompt(pipe, prompt, first_iter);
        accumulated_str += prompt;
        
        std::string prefix = (first_iter) ? "" : "</s>";
        auto answer_str = pipe.call(prefix + prompt, config);
        // auto answer_str = pipe(accumulated_str, config);
        accumulated_str += answer_str;
        cout << "\n----------\n";
    }
    pipe.stop_conversation();

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
