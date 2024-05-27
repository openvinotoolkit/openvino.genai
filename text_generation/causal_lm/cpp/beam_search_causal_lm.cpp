// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/genai/llm_pipeline.hpp>

namespace {
    enum SPECIAL_TOKEN { PAD_TOKEN = 2 };
}

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<PROMPT 1>' ['<PROMPT 2>' ...]");
    }
    auto prompts = std::vector<std::string>(argv + 2, argv + argc);
    
    std::string model_path = argv[1];
    std::string device = "CPU";  // GPU can be used as well

    ov::genai::LLMPipeline pipe(model_path, device);
    ov::genai::GenerationConfig config = pipe.get_generation_config();
    config.max_new_tokens = 20;
    config.num_beam_groups = 3;
    config.num_beams = 15;
    config.num_return_sequences = config.num_beams * prompts.size();
    
    // workaround until pad_token_id is not written into IR
    pipe.get_tokenizer().set_pad_token_id(PAD_TOKEN);
    
    auto beams = pipe.generate(prompts, config);
    for (int i = 0; i < beams.scores.size(); i++)
        std::cout << beams.scores[i] << ": " << beams.texts[i] << '\n';

    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
