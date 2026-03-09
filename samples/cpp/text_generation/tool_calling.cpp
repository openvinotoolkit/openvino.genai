// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/parsers/lfm2_tool_parser.hpp"
#include "openvino/genai/text_streamer.hpp"

std::string get_prompt(const std::string& file_path) {
    std::ifstream f(file_path, std::ios::in);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open prompt file: " + file_path);
    }
    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
}

void print_handle_response(const ov::genai::GenerationHandle& h,
                           ov::genai::Tokenizer& tokenizer,
                           const std::shared_ptr<ov::genai::Lfm2Parser>& parser) {
    const auto outputs = h->read_all();
    
    for (size_t out_idx = 0; out_idx < outputs.size(); ++out_idx) {
        const auto text = tokenizer.decode(outputs[out_idx].generated_ids);
        ov::genai::JsonContainer parsed_message;
        parsed_message["content"] = text;
        parser->parse(parsed_message);

        std::cout << "Response: " << text << std::endl;
        std::cout << "Parsed: " << parsed_message.to_json_string(2) << std::endl;

        if (parsed_message.contains("tool_calls")) {
            std::cout << "Tool calls: " << parsed_message["tool_calls"].to_json_string(2) << std::endl;
        }
    }
}


int main(int argc, char* argv[]) try {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> <prompt_file> \n";
        return EXIT_FAILURE;
    }

    const std::string model_dir = argv[1];
    const std::string prompt_file1 = argv[2];

    const std::string prompt1 = get_prompt(prompt_file1);

    const std::string device = "CPU";

    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.enable_prefix_caching = true;

    ov::genai::ContinuousBatchingPipeline pipe(model_dir, scheduler_config, device);

    ov::genai::GenerationConfig config;
    config.temperature = 0.0f;
    config.max_new_tokens = 10;
    config.apply_chat_template = false;
    auto parser = std::make_shared<ov::genai::Lfm2Parser>();
    config.parsers.push_back(parser);
    auto tokenizer = pipe.get_tokenizer();

    auto h1 = pipe.add_request(5, prompt1, config);

    while (pipe.has_non_finished_requests()) {
        pipe.step();
    }
    std::cout << "Generation finished for Unary call\n";
    print_handle_response(h1, tokenizer, parser);


    std::cout << "Adding second request with streaming\n";
    parser->reset();
    auto streaming_callback = [parser](std::string chunk) -> ov::genai::StreamingStatus {
        std::string delta_text = chunk;
        ov::genai::JsonContainer delta_message;
        const std::string filtered_text = parser->parseChunk(delta_message, delta_text);

        std::cout << "Delta text: " << delta_text << std::endl;
        std::cout << "Filtered text: " << filtered_text << std::endl;
        std::cout << "Delta message: " << delta_message.to_json_string(2) << std::endl;

        if (delta_message.contains("tool_calls")) {
            std::cout << "Tool calls: " << delta_message["tool_calls"].to_json_string(2) << std::endl;
        }
        return ov::genai::StreamingStatus::RUNNING;
    };

    auto text_streamer = std::make_shared<ov::genai::TextStreamer>(tokenizer, streaming_callback);
    std::vector<std::string> prompts = {prompt1};
    std::vector<ov::genai::GenerationConfig> sampling_params = {config};
    pipe.generate(prompts, sampling_params, text_streamer);
    std::cout << "Generation finished for streaming call\n";



    return EXIT_SUCCESS;
} catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return EXIT_FAILURE;
}
