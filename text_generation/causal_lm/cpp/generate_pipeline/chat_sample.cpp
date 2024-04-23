// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include "llm_pipeline.hpp"


std::string generate_chat_prompt(const std::string& input, bool first_iter = false) {
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
    config.reset_state(false);
    config.max_new_tokens(2000000);
    config.eos_token_id(2);
    pipe.set_streamer_callback([](std::string word) { std::cout << word; });
    
    vector<string> questions = {
        "1+1=", 
        "what was the previous answer?", 
        "Why is the sky blue?", 
        "4+10=",
        "Who was Alan Turing?",
        "But why did he killed himself?",
        // "4+10=", 
        // "sum up all the numeric answers in the current chat session"
        // "Why is the sky blue?",
        // "Please repeat all the questions I asked you.",
        "Can you briefly summarize what I asked you about during this session?",
    };

    std::string accumulated_str = "";
    for (size_t i = 0; i < questions.size(); i++) {
        prompt = questions[i];
        
        bool first_iter = (i == 0) ? true : false;
        bool last_iter = (i == questions.size() - 1) ? true : false;
        
        std::cout << "question:\n";
        // std::getline(std::cin, prompt);
        cout << prompt << endl;
        prompt = generate_chat_prompt(prompt, first_iter);
        accumulated_str += prompt;
        
        std::string prefix = (first_iter) ? "" : "</s>";
        auto answer_str = pipe.call(prefix + prompt, config.reset_state(false), first_iter);
        // auto answer_str = pipe(accumulated_str, config.reset_state(true));
        accumulated_str += answer_str;
        cout << "\n----------\n";
        
        // if (last_iter)
        //     cout << accumulated_str;
    }

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}

// using namespace inja;
// #include <inja.hpp>
// using namespace nlohmann;
// #include <jinja2cpp/template.h>
// string my_template = "{% for message in messages %}{% if message.role == 'user' %}{{ ' ' }}{% endif %}{{ message.content }}{% endfor %}";

// nlohmann::json data;
// data["messages"] = {
//     {{"role", "system"}, {"content", "You are a friendly chatbot who always responds in the style of a pirate"}},
//     {{"role", "user"}, {"content", "1+1="}},
// };
// data["eos_token"] = "</s>";

// cout << data.dump() << endl;
// auto res = inja::render(my_template, data);
// json data;
// data["messages"] = {"Jeff", "Tom", "Patrick"};
// auto res = render("{% for message in messages %}{{message}}{% endfor %}", data); // Turn up the music!