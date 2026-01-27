// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "load_image.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (argc < 4 || argc > 5) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                 " <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES> <PROMPT> <CHAT=YES or NO>");
    }
    std::string chat = (argc == 5) ? argv[4] : "NO";
    if (chat != "YES" && chat != "NO") {
        throw std::runtime_error(std::string{"Parameter is error: "} + "<CHAT> is not YES or NO, please reset.");
    }

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    // Define candidates number for candidate generation
    config.num_assistant_tokens = 5;
    // Define max_ngram_size
    config.max_ngram_size = 3;

    std::string model_path = argv[1];
    std::vector<ov::Tensor> rgbs = utils::load_images(argv[2]);
    std::string prompt = argv[3];

    std::string device = "CPU";
    bool enable_chat = chat == "YES";

    // Prompt lookup decoding in VLM pipeline enforses ContinuousBatching backend.
    ov::genai::VLMPipeline pipe(
        model_path,
        device,
        ov::genai::prompt_lookup(true));

    auto streamer = [](std::string subword) {
        std::cout << subword << std::flush;
        return ov::genai::StreamingStatus::RUNNING;
    };

    // Since the streamer is set, the results are
    // printed each time a new token is generated.
    if (enable_chat) {
        ov::genai::ChatHistory chat_history;

        std::cout << "question: " << prompt << "\n";
        chat_history.push_back({{"role", "user"}, {"content", prompt}});
        auto decoded_results = pipe.generate(chat_history,
                                             ov::genai::images(rgbs),
                                             ov::genai::generation_config(config),
                                             ov::genai::streamer(streamer));
        std::string output = decoded_results.texts[0];
        chat_history.push_back({{"role", "assistant"}, {"content", output}});

        std::cout << "\n----------\nquestion:\n";
        while (std::getline(std::cin, prompt)) {
            chat_history.push_back({{"role", "user"}, {"content", prompt}});
            auto decoded_results = pipe.generate(chat_history,
                                                 ov::genai::images(rgbs),
                                                 ov::genai::generation_config(config),
                                                 ov::genai::streamer(streamer));
            std::string output = decoded_results.texts[0];
            chat_history.push_back({{"role", "assistant"}, {"content", output}});
            std::cout << "\n----------\nquestion:\n";
        }
    } else {
        pipe.generate(prompt,
                      ov::genai::images(rgbs),
                      ov::genai::generation_config(config),
                      ov::genai::streamer(streamer));
    }
    std::cout << std::endl;
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
