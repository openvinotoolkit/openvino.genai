// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>
#include <random>
#include <filesystem>

#include "progress_bar.hpp"

#include "imwrite_video.hpp"

#include <openvino/genai/video_generation/text2video_pipeline.hpp>

int main(int32_t argc, char* argv[]) {
    OPENVINO_ASSERT(argc == 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>'");

    std::filesystem::path models_dir = argv[1];
    std::string prompt = argv[2];
    // Compare with https://github.com/Lightricks/LTX-Video
    // TODO: Test GPU, NPU, HETERO, MULTI, AUTO, different steps on different devices
    // TODO: describe algo to generate a video in docs and docstrings
    // TODO: explain in docstrings available perf metrics
    // scheduler needs extra dim?
    // Present that will update validation tools later
    // new classes LTXVideoTransformer3DModel AutoencoderKLLTXVideo
    // private copy constructors to implement clone()
    // const VideoGenerationConfig& may outlive VideoGenerationConfig?
    // Move negative_prompt to Property
    // Allow selecting different models to export from optimum-intel, for example ltxv-2b-0.9.8-distilled.safetensors
    // LoRA later: https://huggingface.co/Lightricks/LTX-Video-ICLoRA-depth-13b-0.9.7, https://huggingface.co/Lightricks/LTX-Video-ICLoRA-pose-13b-0.9.7, https://huggingface.co/Lightricks/LTXV-LoRAs Check https://github.com/Lightricks/LTX-Video for updates
    // Wasn't need so far so not going to implement:
    //     OVLTXPipeline allows prompt_embeds and prompt_attention_mask instead of prompt; Same for negative_prompt_embeds and negative_prompt_attention_mask
    //     OVLTXPipeline allows batched generation with multiple prompts
    // Tests:
    //     Functional
    //     Sample
    // Cover all config members in sample. Use default values explicitly
    // Prefer patching optimum-intel to include more stuff into a model instead of implementing it in C++
    // Add video-to-video, inpainting
    // image to video described in https://huggingface.co/Lightricks/LTX-Video (class LTXConditionPipeline)
    // Optimum doesn't have LTXLatentUpsamplePipeline class
    // Controlled video from https://github.com/Lightricks/LTX-Video
    // TODO: decode, perf metrics, set_scheduler, set/get_generation_config, reshape, compile, clone()
    // TODO: Rename image->video everywhere
    // TODO: test multiple videos per prompt
    // TODO: test with different config values
    // TODO: test log prompts to check truncation
    // TODO: throw if num_frames isn't devisable by 8 + 1. Similar value for resolution. The model works on resolutions that are divisible by 32 and number of frames that are divisible by 8 + 1 (e.g. 257). The model works best on resolutions under 720 x 1280 and number of frames below 257.
    // OVLTXPipeline()(num_inference_steps=1) fails. 2 passes. Would be nice to avoid that bug in genai.
    // Verify tiny resolution like 32x32
    const std::string device = "CPU";  // GPU can be used as well

    ov::genai::Text2VideoPipeline pipe(models_dir, device);
    auto output = pipe.generate(
        prompt,
        ov::genai::negative_prompt("worst quality, inconsistent motion, blurry, jittery, distorted"),
        ov::genai::height(512),  // OVLTXPipeline's default
        ov::genai::width(704),  // OVLTXPipeline's default
        ov::genai::num_frames(65),
        ov::genai::num_inference_steps(15),
        ov::genai::num_images_per_prompt(1),
        ov::genai::callback(progress_bar),
        ov::genai::frame_rate(25),
        ov::genai::guidance_scale(3)
        // num_frames: int = 161,
        // frame_rate: int = 25,
    );

    imwrite_video("5_genai_video.avi", output.video, 25);

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



