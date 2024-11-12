// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/text2image_pipeline.hpp"

#include "imwrite.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 3 && (argc - 3) % 2 == 0, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>' [<LORA_SAFETENSORS> <ALPHA> ...]]");

    const std::string models_path = argv[1], prompt = argv[2];
    const std::string device = "CPU";  // GPU can be used as well

    // MODE_FUSE instructs pipeline to fuse adapter tensors into original model weights loaded into memory
    // giving the same performance level for inference as for the original model. After doing it you cannot
    // change adapter dynamically without re-initializing the pipeline from scratch.
    ov::genai::AdapterConfig adapter_config(ov::genai::AdapterConfig::MODE_FUSE);

    // Multiple LoRA adapters applied simultaneously are supported, parse them all and corresponding alphas from cmd parameters:
    for(size_t i = 0; i < (argc - 3)/2; ++i) {
        ov::genai::Adapter adapter(argv[3 + 2*i]);
        float alpha = std::atof(argv[3 + 2*i + 1]);
        adapter_config.add(adapter, alpha);
    }

    // LoRA adapters passed to the constructor will be activated by default in next generates
    ov::genai::Text2ImagePipeline pipe(models_path, device, ov::genai::adapters(adapter_config));

    std::cout << "Generating image with LoRA adapters fused into original weights, resulting image will be in lora.bmp\n";
    ov::Tensor image = pipe.generate(prompt,
        ov::genai::generator(std::make_shared<ov::genai::CppStdGenerator>(42)),
        ov::genai::width(512),
        ov::genai::height(896),
        ov::genai::num_inference_steps(20));
    imwrite("lora.bmp", image, true);

    return EXIT_SUCCESS;
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
