// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (4 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <ADAPTER_SAFETENSORS> \"<PROMPT>\"");

    std::string model_path = argv[1];
    std::string adapter_path = argv[2];
    std::string prompt = argv[3];
    std::string device = "CPU";  // GPU can be used as well

    // Prepare Adapter object before creation of LLMPipeline
    ov::genai::Adapter adapter1 = ov::genai::Adapter(adapter_path);
    //ov::genai::Adapter adapter2 = ov::genai::Adapter(adapter_path);
    ov::genai::AdapterConfig adapters_config({{adapter1, /*alpha = */ 1}/*, {adapter2, 0.5}*/}, false);
    adapters_config.fuse = true;

    // Pass AdapterConfig to LLMPipeline to be able to dynamically connect adapter in following generate calls
    ov::genai::LLMPipeline pipe(model_path, device, adapters_config);

    // Create generation config as usual or take it from an LLMPipeline, adjust config as usualy required in your app
    ov::genai::GenerationConfig config = pipe.get_generation_config();
    config.max_new_tokens = 10;
    // Note: If a GenerationConfig object is created from scratch and not given by `get_generation_config`
    // you need to set AdapterConfig manually to it, the adapters won't be applied otherwise.

    std::cout << "*** Generation with LoRA adapter applied: ***\n";
    std::string result = pipe.generate(prompt, config);
    std::cout << result << std::endl;
    result = pipe.generate(prompt, config);
    std::cout << result << std::endl;

    if(!adapters_config.is_dynamic) {
        return 0;
    }

    std::cout << "*** Generation without LoRA adapter: ****\n";
    // Set alpha to 0 for a paticular adapter to temporary disable it.
    //config.adapters.set_alpha(adapter1, 0);
    config.adapters.remove(adapter1);
    //config.adapters.set_alpha(adapter2, 0);
    result = pipe.generate(prompt, config);
    std::cout << result << std::endl;
    result = pipe.generate(prompt, config);
    std::cout << result << std::endl;

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
