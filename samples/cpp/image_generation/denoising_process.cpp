// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>

#include "imwrite_video.hpp"
#include "openvino/genai/image_generation/text2image_pipeline.hpp"
#include "progress_bar.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc == 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>'");

    const std::string models_path = argv[1], prompt = argv[2];
    const std::string device = "CPU";  // GPU can be used as well

    constexpr size_t rng_seed = 42;
    constexpr size_t num_inference_steps = 20;
    constexpr int64_t image_width = 512;
    constexpr int64_t image_height = 512;
    constexpr size_t channels = 3;
    constexpr float fps = 5.0f;

    ov::genai::Text2ImagePipeline pipe(models_path, device);

    ov::Tensor frames(ov::element::u8,
                      ov::Shape{1,
                                num_inference_steps,
                                static_cast<size_t>(image_height),
                                static_cast<size_t>(image_width),
                                channels});
    uint8_t* frames_data = frames.data<uint8_t>();
    const size_t frame_bytes = static_cast<size_t>(image_height) * static_cast<size_t>(image_width) * channels;

    auto denoising_callback = [&](size_t step, size_t num_steps, ov::Tensor& latent) -> bool {
        OPENVINO_ASSERT(step < num_inference_steps,
                        "Denoising callback step ",
                        step,
                        " is out of range for ",
                        num_inference_steps,
                        " inference steps");
        const ov::Tensor decoded = pipe.decode(latent);
        std::memcpy(frames_data + step * frame_bytes, decoded.data<const uint8_t>(), frame_bytes);
        progress_bar(step, num_steps, latent);
        return false;
    };

    pipe.generate(prompt,
                  ov::genai::width(image_width),
                  ov::genai::height(image_height),
                  ov::genai::num_inference_steps(num_inference_steps),
                  ov::genai::num_images_per_prompt(1),
                  ov::genai::rng_seed(rng_seed),
                  ov::genai::callback(denoising_callback));

    save_video("denoising_process.avi", frames, fps);

    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}
