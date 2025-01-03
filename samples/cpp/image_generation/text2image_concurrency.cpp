// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/text2image_pipeline.hpp"

#include "imwrite.hpp"
#include <thread>
#include <future>


void runPipeline(std::string prompt, std::filesystem::path root_dir, ov::genai::CLIPTextModel & text_encoder, ov::genai::UNet2DConditionModel & unet, ov::genai::AutoencoderKL & vae,  std::promise<ov::Tensor> & Tensor_prm){
    std::cout << "create pipeline" << prompt << std::endl;
    auto scheduler = ov::genai::Scheduler::from_config(root_dir / "scheduler/scheduler_config.json");
    auto pipe2 = ov::genai::Text2ImagePipeline::stable_diffusion(scheduler, text_encoder, unet, vae);
    std::cout << "start generate " << prompt << std::endl;
    try{
    ov::Tensor image = pipe2.generate(prompt,
        ov::genai::width(512),
        ov::genai::height(512),
        ov::genai::guidance_scale(0.75f),
        ov::genai::num_inference_steps(10));
    Tensor_prm.set_value(image);
    std::cout << "finished generate" << std::endl;
    }
    catch (const std::exception& error) {
        try {
            std::cerr << error.what() << '\n';
        } catch (const std::ios_base::failure&) {}
    } catch (...) {
        try {
            std::cerr << "Non-exception object thrown\n";
        } catch (const std::ios_base::failure&) {}
    }

}

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc == 2, "Usage: ", argv[0], " <MODEL_DIR>");

    const std::string models_path = argv[1];
    std::filesystem::path root_dir = models_path;
    const std::string device = "CPU";  // GPU can be used as well
    auto scheduler = ov::genai::Scheduler::from_config(root_dir / "scheduler/scheduler_config.json");
    auto text_encoder = ov::genai::CLIPTextModel(root_dir / "text_encoder");
    text_encoder.compile("CPU");
    auto unet = ov::genai::UNet2DConditionModel(root_dir / "unet");
    if (device == "NPU") {
        // The max_position_embeddings config from text encoder will be used as a parameter to unet reshape.
        int max_position_embeddings = text_encoder.get_config().max_position_embeddings;
        unet.reshape(1, 512, 512, max_position_embeddings);
    }
    unet.compile("CPU");

    auto vae = ov::genai::AutoencoderKL(root_dir / "vae_decoder");
    vae.compile("CPU");
    std::cout << "models loaded" << std::endl;

    std::promise<ov::Tensor> Tensor1_prm;
    std::promise<ov::Tensor> Tensor2_prm;

    std::thread t1(&runPipeline, std::string("a bucket of red roses"), root_dir, std::ref(text_encoder), std::ref(unet), std::ref(vae), std::ref(Tensor1_prm));
    std::thread t2(&runPipeline, std::string("a glass of water on a wooden table"), root_dir, std::ref(text_encoder), std::ref(unet), std::ref(vae), std::ref(Tensor2_prm));


    std::cout << "threads started" << std::endl;
    std::future<ov::Tensor> T1_ftr = Tensor1_prm.get_future();
    std::future<ov::Tensor> T2_ftr = Tensor2_prm.get_future();

    ov::Tensor image1 = T1_ftr.get();
    ov::Tensor image2 = T2_ftr.get();
    t1.join();
    t2.join();
    // writes `num_images_per_prompt` images by pattern name
    imwrite("image1_%d.bmp", image1, true);
    imwrite("image2_%d.bmp", image2, true);

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
