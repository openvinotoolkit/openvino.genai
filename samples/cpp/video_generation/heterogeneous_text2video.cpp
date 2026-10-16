// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>
#include <filesystem>

#include "progress_bar.hpp"
#include "imwrite_video.hpp"

#include <openvino/genai/video_generation/text2video_pipeline.hpp>

int main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 3 && argc <= 6,
                    "Usage: ",
                    argv[0],
                    " <MODEL_DIR> '<PROMPT>' [ <TXT_ENCODE_DEVICE> <DENOISER_DEVICE> <VAE_DEVICE> ]");

    std::filesystem::path models_dir = argv[1];
    std::string prompt = argv[2];

    const int width = 704;
    const int height = 480;
    const int num_frames = 161;
    const float frame_rate = 25.0f;
    const int number_of_videos_to_generate = 1;
    const int number_of_inference_steps_per_video = 25;

    // Set devices to command-line args if specified, otherwise default to CPU.
    // Note that these can be set to CPU, GPU, or NPU.
    const std::string text_encoder_device = (argc > 3) ? argv[3] : "CPU";
    const std::string denoiser_device = (argc > 4) ? argv[4] : "CPU";
    const std::string vae_decoder_device = (argc > 5) ? argv[5] : "CPU";

    std::cout << "text_encoder_device: " << text_encoder_device << std::endl;
    std::cout << "denoiser_device: " << denoiser_device << std::endl;
    std::cout << "vae_decoder_device: " << vae_decoder_device << std::endl;

    // this is the path to where compiled models will get cached
    // (so that the 'compile' method run much faster 2nd+ time)
    std::string ov_cache_dir = "./cache";

    //
    // Step 1: Create the initial Text2VideoPipeline, given the model path
    //
    ov::genai::Text2VideoPipeline pipe(models_dir);

    //
    // Step 2: Reshape the pipeline given number of videos, frames, height, width and guidance scale.
    //
    pipe.reshape(number_of_videos_to_generate, num_frames, height, width, pipe.get_generation_config().guidance_scale);

    //
    // Step 3: Compile the pipeline with the specified devices, and properties (like cache dir)
    //
    ov::AnyMap properties = {ov::cache_dir(ov_cache_dir)};

    // Note that if there are device-specific properties that are needed, they can
    // be added using ov::device::properties groups, like this:
    //ov::AnyMap properties = {ov::device::properties("CPU", ov::cache_dir("cpu_cache")),
    //                         ov::device::properties("GPU", ov::cache_dir("gpu_cache")),
    //                         ov::device::properties("NPU", ov::cache_dir("npu_cache"))};

    pipe.compile(text_encoder_device, denoiser_device, vae_decoder_device, properties);

    //
    // Step 4: Use the Text2VideoPipeline to generate 'number_of_videos_to_generate' videos.
    //
    for (int videoi = 0; videoi < number_of_videos_to_generate; videoi++) {
        std::cout << "Generating video " << videoi << std::endl;

        auto output = pipe.generate(
            prompt,
            ov::genai::negative_prompt("worst quality, inconsistent motion, blurry, jittery, distorted"),
            ov::genai::num_inference_steps(number_of_inference_steps_per_video),
            ov::genai::callback(progress_bar),
            ov::genai::frame_rate(frame_rate)
        );

        save_video("video_" + std::to_string(videoi) + ".avi", output.video, frame_rate);
    }

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
