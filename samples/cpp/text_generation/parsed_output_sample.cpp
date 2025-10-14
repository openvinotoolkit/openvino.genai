// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/parsers.hpp"
#include "openvino/genai/text_streamer.hpp"


class CurrentStreamer : public ov::genai::TextParserStreamer {
private:
public:
    CurrentStreamer(const ov::genai::Tokenizer& tokenizer)
        : ov::genai::TextParserStreamer(tokenizer) {}
    ov::genai::StreamingStatus write(ov::genai::ParsedMessage& message) {
       std::cout << message["content"].get_string() << std::flush;
        return ov::genai::StreamingStatus::RUNNING;
    }
};


int main(int argc, char* argv[]) try {
    if (argc < 2 || argc > 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <DEVICE>");
    }
    // std::string prompt = "<｜begin▁of▁sentence｜><｜User｜>Please think of a dificcult task to solve x**2 + y**2 = 1<｜Assistant｜><think>";
    std::string prompt = "<｜begin▁of▁sentence｜><｜User｜>Why is the Sky blue?<｜Assistant｜><think>";
    std::string models_path = argv[1];

    // Default device is CPU; can be overridden by the second argument
    std::string device = (argc == 3) ? argv[2] : "CPU";  // GPU, NPU can be used as well
    ov::genai::LLMPipeline pipe(models_path, device);
    
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 1000;

    auto tok = pipe.get_tokenizer();
    std::shared_ptr<CurrentStreamer> streamer = std::make_shared<CurrentStreamer>(tok);

    pipe.generate(prompt, config, streamer);


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
