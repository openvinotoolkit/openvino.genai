// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <openvino/genai/module_genai/pipeline.hpp>
#include <filesystem>
#include <chrono>
#include "module_genai/utils/profiler.hpp"

TEST(Z_Image_Trubo, disable_tiling) {
    GTEST_SKIP() << "Skip Z Image Turbo test";
    std::string config_yaml = "test_models/Z-Image-Turbo-fp16-ov/config.yaml";

    ov::AnyMap inputs;
    inputs["prompt"] = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.";
    inputs["width"] = 16 * 65;
    inputs["height"] = 16 * 65;
    inputs["num_inference_steps"] = 9;
    inputs["guidance_scale"] = 0.0f;
    inputs["max_sequence_length"] = 512;

    auto config_path = std::filesystem::path(config_yaml);

    ov::genai::module::ModulePipeline pipe(config_path);

    pipe.generate(inputs);
    
    auto output = pipe.get_output("generated_image");

    std::cout << std::endl;
}

TEST(Z_Image_Trubo, enable_tiling) {
    GTEST_SKIP() << "Skip Z Image Turbo test";
    std::string config_yaml = "test_models/Z-Image-Turbo-fp16-ov/config_tiling.yaml";

    ov::AnyMap inputs;
    inputs["prompt"] = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.";
    inputs["width"] = 16 * 65;
    inputs["height"] = 16 * 65;
    inputs["num_inference_steps"] = 9;
    inputs["guidance_scale"] = 0.0f;
    inputs["max_sequence_length"] = 512;

    auto config_path = std::filesystem::path(config_yaml);

    ov::genai::module::ModulePipeline pipe(config_path);

    pipe.generate(inputs);
    
    auto output = pipe.get_output("generated_image");

    std::cout << std::endl;
}
