// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/text2image_pipeline.hpp"

#include "imwrite.hpp"
#include "progress_bar.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 3 && argc <= 6,
                    "Usage: ",
                    argv[0],
                    " <MODEL_DIR> '<PROMPT>' [ <TXT_ENCODE_DEVICE> <UNET_DEVICE> <VAE_DEVICE> ]");

    const std::string models_path = argv[1], prompt = argv[2];

    std::filesystem::path root_dir = models_path;

    const int width = 512;
    const int height = 512;
    const float guidance_scale = 7.5f;
    const int number_of_images_to_generate = 1;
    const int number_of_inference_steps_per_image = 20;

    // Set devices to command-line args if specified, otherwise default to CPU.
    // Note that these can be set to CPU, GPU, or NPU.
    const std::string text_encoder_device = (argc > 3) ? argv[3] : "CPU";
    const std::string unet_device = (argc > 4) ? argv[4] : "CPU";
    const std::string vae_decoder_device = (argc > 5) ? argv[5] : "CPU";

    std::cout << "text_encoder_device: " << text_encoder_device << std::endl;
    std::cout << "unet_device: " << unet_device << std::endl;
    std::cout << "vae_decoder_device: " << vae_decoder_device << std::endl;

    // this is the path to where compiled models will get cached
    // (so that the 'compile' method run much faster 2nd+ time)
    std::string ov_cache_dir = "./cache";

    //
    // Step 1: Prepare each Text2Image subcomponent (scheduler, text encoder, unet, vae) separately.
    //

    // Create the scheduler from the details listed in the json.
    auto scheduler = ov::genai::Scheduler::from_config(root_dir / "scheduler/scheduler_config.json");

    // Note that we could have created the scheduler by specifying specific type (for example EULER_DISCRETE), like
    // this: auto scheduler = ov::genai::Scheduler::from_config(root_dir / "scheduler/scheduler_config.json",
    //                                                    ov::genai::Scheduler::Type::EULER_DISCRETE);
    // This can be useful when a particular type of Scheduler is not yet supported natively by OpenVINO GenAI.
    // (even though we are actively working to support most commonly used ones)

    // Create unet object
    auto unet = ov::genai::UNet2DConditionModel(root_dir / "unet");

    // Set batch size based on classifier free guidance condition.
    int unet_batch_size = unet.do_classifier_free_guidance(guidance_scale)  ? 2 : 1;

    // Create the text encoder.
    auto text_encoder = ov::genai::CLIPTextModel(root_dir / "text_encoder");

    // In case of NPU, we need to reshape the model to have static shapes
    if (text_encoder_device == "NPU") {
        text_encoder.reshape(unet_batch_size);
    }

    // Compile text encoder for the specified device
    text_encoder.compile(text_encoder_device, ov::cache_dir(ov_cache_dir));

    // In case of NPU, we need to reshape the model to have static shapes
    if (unet_device == "NPU") {
        // The max_postiion_embeddings config from text encoder will be used as a parameter to unet reshape.
        int max_position_embeddings = text_encoder.get_config().max_position_embeddings;

        unet.reshape(unet_batch_size, height, width, max_position_embeddings);
    }

    // Compile unet for specified device
    unet.compile(unet_device, ov::cache_dir(ov_cache_dir));

    // Create the vae decoder.
    auto vae = ov::genai::AutoencoderKL(root_dir / "vae_decoder");

    // In case of NPU, we need to reshape the model to have static shapes
    if (vae_decoder_device == "NPU") {
        // We set batch-size to '1' here, as we're configuring our pipeline to return 1 image per 'generate' call.
        vae.reshape(1, height, width);
    }

    // Compile vae decoder for the specified device
    vae.compile(vae_decoder_device, ov::cache_dir(ov_cache_dir));

    //
    // Step 2: Create a Text2ImagePipeline from the individual subcomponents
    //
    auto pipe = ov::genai::Text2ImagePipeline::stable_diffusion(scheduler, text_encoder, unet, vae);

    //
    // Step 3: Use the Text2ImagePipeline to generate 'number_of_images_to_generate' images.
    //
    for (int imagei = 0; imagei < number_of_images_to_generate; imagei++) {
        std::cout << "Generating image " << imagei << std::endl;

        ov::Tensor image = pipe.generate(prompt,
                                         ov::genai::width(width),
                                         ov::genai::height(height),
                                         ov::genai::guidance_scale(guidance_scale),
                                         ov::genai::num_inference_steps(number_of_inference_steps_per_image),
                                         ov::genai::callback(progress_bar));

        imwrite("image_" + std::to_string(imagei) + ".bmp", image, true);
    }

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
