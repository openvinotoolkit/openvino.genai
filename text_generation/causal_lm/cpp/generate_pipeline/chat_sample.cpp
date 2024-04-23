// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include "llm_pipeline.hpp"


std::string generate_chat_prompt(const std::string& input) {
    std::stringstream result_prompt;
    // Gemma-7b-it
    // result_prompt << "<bos><start_of_turn>user\n" << input << "<end_of_turn>\n<start_of_turn>model";
    
    // TinyLlama
    result_prompt << "<|user|>\n" << input << "</s>\n<|assistant|>\n";
    return result_prompt.str();
}

int main(int argc, char* argv[]) try {
    // if (2 >= argc && argc <= 4)
    //     throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\" <DEVICE>");
    
    std::string prompt = "table is made of";
    std::string device = "CPU"; // can be replaced with GPU

    std::string model_path = argv[1];
    if (argc > 2)
        prompt = argv[2];
    if (argc > 3)
        device = argv[3];

    LLMPipeline pipe(model_path, device);
    GenerationConfig config = pipe.generation_config();
    config.do_reset_state(false);
    config.max_new_tokens(200);
    config.eos_token_id(2);
    pipe.set_streamer_callback([](std::string word) { std::cout << word; });
    
   std::string accumulated_str = "";

   std::cout << "Type keyword \"Stop!\" to stop the chat. \n";
   size_t max_len = 10000;
    for (size_t i = 0; i < max_len; i++) {
        std::cout << "question:\n";
        std::getline(std::cin, prompt);

        if (!prompt.compare("Stop!"))
            break;

        prompt = generate_chat_prompt(prompt);
        accumulated_str += prompt;
        // string prefix = (i != 0) ? "</s>" : ""; 
        string prefix = (i != 0) ? "</s>" : ""; 
        bool first_time = (i != 0) ? false : true;

        // auto answer_str = pipe.call(prompt, config.do_reset_state(false), first_time);
        auto answer_str = pipe(accumulated_str, config.do_reset_state(true));
        accumulated_str += answer_str;
    }

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
