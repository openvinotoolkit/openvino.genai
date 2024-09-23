// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (4 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <ADAPTER_SAFETENSORS> \"<PROMPT>\"");

    std::string model_path = argv[1];
    std::string adapter_path = argv[2];
    std::string adapter_path1 = adapter_path;
    std::string adapter_path2 = adapter_path;
    std::string prompt = argv[3];
    std::string device = "CPU";  // GPU can be used as well

    using namespace ov::genai;

    std::cout << "MODE_AUTO" << std::endl;
    {
        // Create Adapter object inline if you are not going to refer to it later
        LLMPipeline pipe(model_path, device, adapters(Adapter(adapter_path), 0.75));
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
    }

    std::cout << "MODE_AUTO/implicit alpha = 1" << std::endl;
    {
        // Create Adapter object inline if you are not going to refer to it later
        LLMPipeline pipe(model_path, device, adapters(Adapter(adapter_path)));
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
    }

    std::cout << "MODE_AUTO/explicit alpha" << std::endl;
    {
        // Create Adapter object inline if you are not going to refer to it later
        LLMPipeline pipe(model_path, device, adapters(Adapter(adapter_path), 0.75));
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
    }

    std::cout << "MODE_AUTO/two adapters" << std::endl;
    {
        Adapter adapter1(adapter_path1);
        Adapter adapter2(adapter_path2);
        LLMPipeline pipe(model_path, device, adapters({adapter1, adapter2}));
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
    }

    std::cout << "MODE_AUTO/switching between adapters" << std::endl;
    {
        Adapter adapter1(adapter_path1);
        Adapter adapter2(adapter_path2);
        LLMPipeline pipe(model_path, device, adapters({adapter1, adapter2}));
        std::cout << pipe.generate(prompt, max_new_tokens(100), adapters(adapter1, 0.25)) << std::endl;
        std::cout << pipe.generate(prompt, max_new_tokens(100), adapters(adapter2, 0.75)) << std::endl;
    }

    std::cout << "MODE_AUTO/switching on/off" << std::endl;
    {
        Adapter adapter(adapter_path);
        LLMPipeline pipe(model_path, device, adapters(adapter));
        std::cout << pipe.generate(prompt, max_new_tokens(100), adapters(adapter, 0.75)) << std::endl;
        std::cout << pipe.generate(prompt, max_new_tokens(100), adapters()) << std::endl;
    }

    std::cout << "MODE_AUTO/blended with late alpha set" << std::endl;
    {
        Adapter adapter1 = Adapter(adapter_path1);
        Adapter adapter2 = Adapter(adapter_path2);
        LLMPipeline pipe(model_path, device, adapters({{adapter1, 0.5}, {adapter2, 0.25}}));
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
    }

    std::cout << "MODE_AUTO/blended with late alpha set changed in config" << std::endl;
    {
        Adapter adapter1 = Adapter(adapter_path1);
        Adapter adapter2 = Adapter(adapter_path2);
        LLMPipeline pipe(model_path, device, adapters({{adapter1, 0.2}, {adapter2, 0.5}}));
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
        auto config = pipe.get_generation_config();
        config.adapters.set_alpha(adapter1, 0.6).set_alpha(adapter2, 0.8);
        config.max_new_tokens = 100;
        std::cout << pipe.generate(prompt, config) << std::endl;
    }

    std::cout << "MODE_AUTO/blended with late alpha set in generate" << std::endl;
    {
        Adapter adapter1 = Adapter(adapter_path1);
        Adapter adapter2 = Adapter(adapter_path2);
        LLMPipeline pipe(model_path, device, adapters({{adapter1, 0.2}, {adapter2, 0.5}}));
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
        std::cout << pipe.generate(prompt, adapters({{adapter1, 0.6}, {adapter2, 0.8}}), max_new_tokens(100)) << std::endl;
    }

    // -----------------------------------------------------------------------------------------------------------------------
    // Low-level mode manipulation to test plugin capabilities and tune for better performance if dynamic LoRA is not required

    std::cout << "MODE_STATIC" << std::endl;
    try {
        Adapter adapter(adapter_path);
        LLMPipeline pipe(model_path, device, adapters(adapter, 0.75, AdapterConfig::MODE_STATIC));
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
    } catch (const ov::Exception& exception) {
        std::cout << "[ ERROR ] Cannot run MODE_STATIC: " << exception.what() << "\n";
    }

    std::cout << "MODE_STATIC_RANK" << std::endl;
    try {
        Adapter adapter(adapter_path);
        LLMPipeline pipe(model_path, device, adapters(adapter, 0.75, AdapterConfig::MODE_STATIC_RANK));
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
    } catch (const ov::Exception& exception) {
        std::cout << "[ ERROR ] Cannot run MODE_STATIC_RANK: " << exception.what() << "\n";
    }

    std::cout << "MODE_DYNAMIC" << std::endl;
    try {
        Adapter adapter(adapter_path);
        LLMPipeline pipe(model_path, device, adapters(adapter, 0.75, AdapterConfig::MODE_DYNAMIC));
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
    } catch (const ov::Exception& exception) {
        std::cout << "[ ERROR ] Cannot run MODE_DYNAMIC: " << exception.what() << "\n";
    }

    std::cout << "MODE_FUSE" << std::endl;
    try {
        Adapter adapter(adapter_path);
        LLMPipeline pipe(model_path, device, adapters(adapter, AdapterConfig::MODE_FUSE));
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
    } catch (const ov::Exception& exception) {
        std::cout << "[ ERROR ] Cannot run MODE_FUSE: " << exception.what() << "\n";
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
