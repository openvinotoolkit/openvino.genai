// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/text2image/pipeline.hpp"

#include "imwrite.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 3 && (argc - 3) % 2 == 0, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>' [<LORA_SAFETENSORS> <ALPHA> ...]]");

    const std::string models_path = argv[1], prompt = argv[2];
    const std::string device = "CPU";  // GPU, NPU can be used as well

    ov::genai::AdapterConfig adapter_config;
    // Multiple LoRA adapters applied simultaniously are supported, register them all
    for(size_t i = 0; i < (argc - 3)/2; ++i) {
        std::cerr << std::atof(argv[3 + i + 1]) << "\n";
        adapter_config.add(ov::genai::Adapter(argv[3 + i]), std::atof(argv[3 + i + 1]));
    }

    ov::genai::Text2ImagePipeline pipe(models_path, device, ov::genai::adapters(adapter_config));
    ov::Tensor image = pipe.generate(prompt,
        ov::genai::width(512),
        ov::genai::height(512),
        ov::genai::num_inference_steps(20));

    imwrite("image.bmp", image, true);

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
