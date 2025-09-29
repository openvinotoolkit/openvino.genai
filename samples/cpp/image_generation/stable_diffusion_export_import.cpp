// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "imwrite.hpp"
#include "openvino/genai/image_generation/text2image_pipeline.hpp"
#include "progress_bar.hpp"

void pipeline_export_import(const std::filesystem::path& root_dir) {
    ov::genai::Text2ImagePipeline pipe(root_dir, "CPU");
    pipe.export_model(root_dir / "exported");
    // pipeline models are exported to dedicated subfolders
    // for stable diffusion xl:
    // exported/
    // ├── text_encoder/
    // │   └── openvino_model.blob
    // ├── text_encoder_2/
    // │   └── openvino_model.blob
    // ├── unet/
    // │   └── openvino_model.blob
    // └── vae_decoder/
    //     └── openvino_model.blob

    // during import, specify blob_path property to point to the exported model location
    ov::genai::Text2ImagePipeline imported_pipe(root_dir, "CPU", ov::genai::blob_path(root_dir / "exported"));
};

void dedicated_models_export_import(const std::filesystem::path& root_dir) {
    const auto blob_path = root_dir / "exported";
    const auto device = "CPU";

    // instantiate models and export them individually
    auto text_encoder = ov::genai::CLIPTextModel(root_dir / "text_encoder", device);
    text_encoder.export_model(blob_path / "text_encoder");

    auto text_encoder_2 = ov::genai::CLIPTextModelWithProjection(root_dir / "text_encoder_2", device);
    text_encoder_2.export_model(blob_path / "text_encoder_2");

    auto unet = ov::genai::UNet2DConditionModel(root_dir / "unet", device);
    unet.export_model(blob_path / "unet");

    auto vae = ov::genai::AutoencoderKL(root_dir / "vae_decoder", "CPU", ov::AnyMap{});
    vae.export_model(blob_path);
    // AutoencoderKL can be composed with decoder and encoder models
    // exported/
    // └── vae_decoder/
    //     └── openvino_model.blob
    // └── vae_encoder/
    //     └── openvino_model.blob

    // create pipeline from the exported models
    auto imported_pipe = ov::genai::Text2ImagePipeline::stable_diffusion_xl(
        ov::genai::Scheduler::from_config(root_dir / "scheduler" / "scheduler_config.json"),
        ov::genai::CLIPTextModel(root_dir / "text_encoder", device, ov::genai::blob_path(blob_path / "text_encoder")),
        ov::genai::CLIPTextModelWithProjection(root_dir / "text_encoder_2",
                                               device,
                                               ov::genai::blob_path(blob_path / "text_encoder_2")),
        ov::genai::UNet2DConditionModel(root_dir / "unet", device, ov::genai::blob_path(blob_path / "unet")),
        ov::genai::AutoencoderKL(root_dir / "vae_decoder", device, ov::genai::blob_path(blob_path)));
};

void export_import_with_reshape(const std::filesystem::path& root_dir, const std::string& prompt) {
    const auto device = "CPU";

    const int width = 512;
    const int height = 512;
    const int number_of_images_to_generate = 1;
    const int number_of_inference_steps_per_image = 20;

    // reshape before export
    ov::genai::Text2ImagePipeline pipe(root_dir);
    pipe.reshape(1, height, width, pipe.get_generation_config().guidance_scale);
    pipe.compile(device);
    pipe.export_model(root_dir / "exported");

    ov::genai::Text2ImagePipeline imported_pipe(root_dir, device, ov::genai::blob_path(root_dir / "exported"));

    // update generation config according to the new shape parameters
    auto config = imported_pipe.get_generation_config();
    config.num_images_per_prompt = number_of_images_to_generate;
    config.height = height;
    config.width = width;
    imported_pipe.set_generation_config(config);

    for (int imagei = 0; imagei < number_of_images_to_generate; imagei++) {
        std::cout << "Generating image " << imagei << std::endl;

        ov::Tensor image = imported_pipe.generate(prompt,
                                                  ov::genai::num_inference_steps(number_of_inference_steps_per_image),
                                                  ov::genai::callback(progress_bar));

        imwrite("image_" + std::to_string(imagei) + ".bmp", image, true);
    }
}

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc == 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>'");

    const std::string models_path = argv[1], prompt = argv[2];

    std::filesystem::path root_dir = models_path;

    pipeline_export_import(root_dir);
    dedicated_models_export_import(root_dir);
    export_import_with_reshape(root_dir, prompt);

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
