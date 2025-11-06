// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"
#include <Windows.h>

int main(int argc, char* argv[]) try {
#if defined(_DEBUG) || defined(RELWITHDEBINFO)
    MessageBox(nullptr, "------Enjoy Debug------", "Hi there", MB_OK);
    DebugBreak();
#endif  // DEBUG

    if ((2 != argc) && (3 != argc)) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR>" + " [OTD]");
    }
    std::string prompt;
    std::string models_path = argv[1];
    std::size_t offload_to_disk = 0;
    if (3 == argc) {
        long long val = std::stoll(argv[2]);
        if (val > 0) {
            offload_to_disk = val;
        }
    }

    std::string device = "GPU";  // GPU, NPU can be used as well
    ov::genai::LLMPipeline pipe(models_path, device, {}, offload_to_disk);
    
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 1000;

    auto streamer = [](std::string word) {
        std::cout << word << std::flush;
        // Return flag corresponds whether generation should be stopped.
        return ov::genai::StreamingStatus::RUNNING;
    };

#if 0
    pipe.start_chat();
    std::cout << "question:\n";
    while (std::getline(std::cin, prompt)) {
        pipe.generate(prompt, config, streamer);
        std::cout << "\n----------\n"
            "question:\n";
    }
    pipe.finish_chat();
#else
    config.apply_chat_template = false;
  //  config.stop_strings = {"</s>", "<|eot|>"};  // 根据模型实际 EOS token 调整
    std::cout << "question:\n";
    while (std::getline(std::cin, prompt)) {
        std::string formatted_prompt = "<|user|>\n" + prompt + "\n</s>\n<|assistant|>\n";
        pipe.generate(formatted_prompt, config, streamer);
        std::cout << "\n----------\nquestion:\n";
    }
#endif
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
