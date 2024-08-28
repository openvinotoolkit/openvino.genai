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

    std::cout << "Run on devide: " << device << "\n";

    using namespace ov::genai;

    {
        LLMPipeline pipe(model_path, device, Adapter(adapter_path));
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
    }

    {
        LLMPipeline pipe(model_path, device, Adapter(adapter_path, 0.75));
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
    }

    {
        Adapter adapter1(adapter_path1);
        Adapter adapter2(adapter_path2);
        LLMPipeline pipe(model_path, device, {adapter1, adapter2});
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
    }

    {
        Adapter adapter1(adapter_path1, 0.3);
        Adapter adapter2(adapter_path2, 0.6);
        LLMPipeline pipe(model_path, device, {adapter1, adapter2});
        std::cout << pipe.generate(prompt, adapters(adapter1)) << std::endl;
        std::cout << pipe.generate(prompt, adapters(adapter2)) << std::endl;
    }

    {
        Adapter adapter1 = Adapter(adapter_path1);
        Adapter adapter2 = Adapter(adapter_path2);
        LLMPipeline pipe(model_path, device, {{adapter1, 0.2}, {adapter2, 0.5}});
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
    }

    {
        Adapter adapter1 = Adapter(adapter_path1);
        Adapter adapter2 = Adapter(adapter_path2);
        LLMPipeline pipe(model_path, device, {{adapter1, 0.2}, {adapter2, 0.5}});
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
        auto config = pipe.get_generation_config();
        config.adapters.set_alpha(adapter1, 0.6).set_alpha(adapter2, 0.8);
        config.max_new_tokens = 100;
        std::cout << pipe.generate(prompt, config) << std::endl;
    }

    {
        Adapter adapter(adapter_path);
        LLMPipeline pipe(model_path, device, AdapterConfig(adapter, AdapterConfig::MODE_STATIC));
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
    }

    {
        Adapter adapter(adapter_path);
        LLMPipeline pipe(model_path, device, AdapterConfig(adapter, AdapterConfig::MODE_FUSE));
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
    }

    {
        Adapter adapter(adapter_path);
        LLMPipeline pipe(model_path, device, AdapterConfig(adapter, AdapterConfig::MODE_AUTO));
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
    }

    {
        Adapter adapter(adapter_path);
        LLMPipeline pipe(model_path, device, AdapterConfig(adapter, AdapterConfig::MODE_DYNAMIC));
        std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
    }

    {
        Adapter adapter1 = Adapter(adapter_path);
        Adapter adapter2 = Adapter(adapter_path);
        LLMPipeline pipe1(model_path, device,
            AdapterConfig({{adapter1, 0.2}, {adapter2, 0.5}}, AdapterConfig::MODE_STATIC)
        );
        std::cout << pipe1.generate(prompt, max_new_tokens(100)) << std::endl;
        LLMPipeline pipe2(model_path, device,
            AdapterConfig({{adapter1, 0.1}, {adapter2, 0.9}}, AdapterConfig::MODE_STATIC)
        );
        std::cout << pipe2.generate(prompt, max_new_tokens(100)) << std::endl;
    }

#if 0
    {
        Adapter adapter1 = Adapter(adapter_path);
        AdapterConfig adapters({adapter1}, true);
    AdapterConfig adapters({{adapter1, /*alpha = */ 1}/*, {adapter2, 0.5}*/}, true);
    adapters.fuse = false;
    adapters.is_dynamic_rank = false;

    // Pass AdapterConfig to LLMPipeline to be able to dynamically connect adapter in following generate calls
    LLMPipeline pipe(model_path, device, adapters);

    // Create generation config as usual or take it from an LLMPipeline, adjust config as usualy required in your app
    GenerationConfig config = pipe.get_generation_config();
    config.max_new_tokens = 10;
    config.adapters = adapters;
    // Note: If a GenerationConfig object is created from scratch and not given by `get_generation_config`
    // you need to set AdapterConfig manually to it, the adapters won't be applied otherwise.

    std::cout << "*** Generation with LoRA adapter applied: ***\n";
    std::string result = pipe.generate(prompt, config);
    std::cout << result << std::endl;
    result = pipe.generate(prompt, config);
    std::cout << result << std::endl;

    if(!adapters.is_dynamic) {
        return 0;
    }

    return 0;

    std::cout << "*** Generation without LoRA adapter: ****\n";
    // Set alpha to 0 for a paticular adapter to temporary disable it.
    //config.adapters.set_alpha(adapter1, 0);
    config.adapters.add(adapter1);
    //config.adapters.set_alpha(adapter2, 0);
    result = pipe.generate(prompt, config);
    std::cout << result << std::endl;
    result = pipe.generate(prompt, config);
    std::cout << result << std::endl;
#endif

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
