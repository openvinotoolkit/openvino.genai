// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>
#include <random>
#include <filesystem>

#include "progress_bar.hpp"

#include <openvino/genai/video_generation/text2video_pipeline.hpp>

int main(int32_t argc, char* argv[]) {
    OPENVINO_ASSERT(argc == 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>'");

    std::filesystem::path models_dir = argv[1];
    std::string prompt = argv[2];
    // TODO: Test GPU, NPU, HETERO, MULTI, AUTO, different steps on different devices
    // TODO: OpenCV?
    // TODO: describe algo
    // scheduler needs extra dim?
    // upgrade diffusers, take from master
    // optimum-intel vs diffusers
    // update validation tools later
    // Vide instead of images because of video generation confing
    // new classes LTXVideoTransformer3DModel AutoencoderKLLTXVideo
    // private copy constructors
    // const VideoGenerationConfig& may outlive VideoGenerationConfig?
    // hide negative prompt
    // LoRA?
    // How is vedeo inpainting mask specified
    // WIll v::Tensor decode(const ov::Tensor latent); stay the same - yes, just an extra dim in Tensor
    // using VideoGenerationPerfMetrics = ImageGenerationPerfMetrics;
    // wasn't need so far:
    //     TODO: OVLTXPipeline allows prompt_embeds and prompt_attention_mask instead of prompt; Same for negative_prompt_embeds and negative_prompt_attention_mask
    //     TODO: OVLTXPipeline allows batched generation with multiple prompts
    // Tests:
    //     Functional
    //     Sample
    // Cover all config members in sample
    const std::string device = "CPU";  // GPU can be used as well

    ov::genai::Text2VideoPipeline pipe(models_dir, device);
    ov::genai::VideoGenerationConfig config = pipe.get_generation_config();
    config.num_frames = 1;
    pipe.set_generation_config(config);
    ov::Tensor video = pipe.generate(
        prompt,
        "worst quality, inconsistent motion, blurry, jittery, distorted",
        ov::genai::height(512),
        ov::genai::width(704),
        ov::genai::num_inference_steps(50),
        ov::genai::num_images_per_prompt(1),
        ov::genai::callback(progress_bar)
    );
    return EXIT_SUCCESS;
// } catch (const std::exception& error) {
//     try {
//         std::cerr << error.what() << '\n';
//     } catch (const std::ios_base::failure&) {}
//     return EXIT_FAILURE;
// } catch (...) {
//     try {
//         std::cerr << "Non-exception object thrown\n";
//     } catch (const std::ios_base::failure&) {}
//     return EXIT_FAILURE;
}
