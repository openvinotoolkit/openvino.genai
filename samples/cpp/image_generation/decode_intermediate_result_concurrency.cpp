// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <queue>
#include <thread>

#include "openvino/genai/image_generation/text2image_pipeline.hpp"

#include "imwrite.hpp"
#include "progress_bar.hpp"


void decode_latent(std::queue<ov::Tensor>* intermediate_latent, 
                   std::function<ov::Tensor(const ov::Tensor)>& decoder, int32_t num_steps) {
    int i = 0;
    //
    // Callback doesn't decode latest latent which let image generation pipeline handle. 
    //
    while (i < (num_steps - 1)) {
        if (!intermediate_latent->empty()) {
            ov::Tensor intermediate_image = decoder(intermediate_latent->front());
            imwrite("intermediate_image_" + std::to_string(i+1) + ".bmp", intermediate_image, true);
            intermediate_latent->pop();
            i++;
        }
    }
}


int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 3 && argc <= 6,
                    "Usage: ",
                    argv[0],
                    " <MODEL_DIR> '<PROMPT>' [ <TXT_ENCODE_DEVICE> <UNET_DEVICE> <VAE_DEVICE> ]");

    const std::string models_path = argv[1], prompt = argv[2];

    std::filesystem::path root_dir = models_path;

    const int width = 512;
    const int height = 512;
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
    // Step 1: Create the initial Text2ImagePipeline, given the model path
    //
    ov::genai::Text2ImagePipeline pipe(models_path);

    //
    // Step 2: Reshape the pipeline given number of images, height, width and guidance scale.
    //
    pipe.reshape(1, height, width, pipe.get_generation_config().guidance_scale);

    //
    // Step 3: Compile the pipeline with the specified devices, and properties (like cache dir)
    //
    ov::AnyMap properties = {ov::cache_dir(ov_cache_dir)};

    // Note that if there are device-specific properties that are needed, they can
    // be added using ov::device::properties groups, like this:
    // ov::AnyMap properties = {ov::device::properties("CPU", ov::cache_dir("cpu_cache")),
    //                          ov::device::properties("GPU", ov::cache_dir("gpu_cache")),
    //                          ov::device::properties("NPU", ov::cache_dir("npu_cache"))};

    pipe.compile(text_encoder_device, unet_device, vae_decoder_device, properties);

    //
    // Step 4: Use the std::queue to handle intermediate latent result
    //

    std::queue<ov::Tensor>* intermediate_latent = new std::queue<ov::Tensor>();

    std::function<ov::Tensor(const ov::Tensor)> decoder_func = 
        [&](const ov::Tensor& latent) {
            return pipe.decode(latent);
        };

    //
    // Step 5: Initial thread for decode intermediate latent and save as image
    //
    std::thread decode_thread(decode_latent, std::ref(intermediate_latent), decoder_func, number_of_inference_steps_per_image);

    //
    // Step 6: Callback function for add lantent into intermediate latent queue
    //
    auto callback = [&](size_t step, size_t num_steps, ov::Tensor& latent) -> bool {
        std::cout << "Image generation step: " << step + 1 << " / " << num_steps << std::endl;
        if (step < num_steps - 1) {
            std::cout << "Add latent into intermediate latent queue." << std::endl;
            intermediate_latent->push(latent);
        } else if (step == num_steps - 1) {
            decode_thread.join();
        }
        return false;
    };

    //
    // Step 7: Use the Text2ImagePipeline to generate 'number_of_images_to_generate' images.
    //
    for (int imagei = 0; imagei < number_of_images_to_generate; imagei++) {
        std::cout << "Generating image " << imagei << std::endl;

        ov::Tensor image = pipe.generate(prompt,
                                         ov::genai::num_inference_steps(number_of_inference_steps_per_image),
                                         ov::genai::callback(callback));

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
