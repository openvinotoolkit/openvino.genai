// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "imwrite.hpp"
#include "openvino/genai/image_generation/text2image_pipeline.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "progress_bar.hpp"

namespace {

const size_t NUMBER_OF_RUNS = 10;

void test_load_unet_from_ir(const std::filesystem::path& models_path, const std::string& device) {
    auto duration = std::chrono::milliseconds::zero();
    for (size_t i = 0; i < NUMBER_OF_RUNS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto unet = ov::genai::UNet2DConditionModel(models_path / "unet", device);
        auto end = std::chrono::high_resolution_clock::now();
        duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }
    auto average_duration = duration.count() / static_cast<float>(NUMBER_OF_RUNS);
    std::cout << "UNet load time from IR: " << average_duration << " ms\n";
}

void test_load_unet_from_blob(const std::filesystem::path& models_path, const std::string& device) {
    ov::AnyMap import_blob_properties{{ov::genai::blob_path.name(), models_path / "blobs" / "unet"}};

    auto duration = std::chrono::milliseconds::zero();
    for (size_t i = 0; i < NUMBER_OF_RUNS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto unet = ov::genai::UNet2DConditionModel(models_path / "unet", device, import_blob_properties);
        auto end = std::chrono::high_resolution_clock::now();
        duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }
    auto average_duration = duration.count() / static_cast<float>(NUMBER_OF_RUNS);
    std::cout << "UNet load time from Blob: " << average_duration << " ms\n";
}

void test_load_full_pipe_from_ir(const std::filesystem::path& models_path, const std::string& device) {
    auto duration = std::chrono::milliseconds::zero();
    for (size_t i = 0; i < NUMBER_OF_RUNS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto pipe = ov::genai::Text2ImagePipeline(models_path, device);
        auto end = std::chrono::high_resolution_clock::now();
        duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }
    auto average_duration = duration.count() / static_cast<float>(NUMBER_OF_RUNS);
    std::cout << "Full pipeline load time from IR: " << average_duration << " ms\n";
}

void test_load_full_pipe_from_blob(const std::filesystem::path& models_path, const std::string& device) {
    ov::AnyMap import_blob_properties{{ov::genai::blob_path.name(), models_path / "blobs"}};

    auto duration = std::chrono::milliseconds::zero();
    for (size_t i = 0; i < NUMBER_OF_RUNS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        ov::genai::Text2ImagePipeline pipe(models_path, device, import_blob_properties);
        auto end = std::chrono::high_resolution_clock::now();
        duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }
    auto average_duration = duration.count() / static_cast<float>(NUMBER_OF_RUNS);
    std::cout << "Full pipeline load time from Blob: " << average_duration << " ms\n";
}

}  // namespace

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc == 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>'");

    const std::filesystem::path models_path = argv[1];
    const std::string prompt = argv[2];
    const std::string device = "CPU";  // GPU can be used as well

    // full pipeline export/import example
    {
        auto pipe = ov::genai::Text2ImagePipeline(models_path, device);
        pipe.export_model(models_path / "blobs");

        // unet model saved at:
        // models_path/ 
        // └── blobs/
        //     └── unet/
        //         └── openvino_model.blob

        // another approach can be the use of export_blob property
        // ov::AnyMap import_blob_properties{
        //     {ov::genai::blob_path.name(), models_path / "blobs"},
        //     {ov::genai::export_blob.name(), true},
        // };
        // auto pipe = ov::genai::Text2ImagePipeline(models_path, device, import_blob_properties);
    }

    {
        ov::AnyMap import_blob_properties{{ov::genai::blob_path.name(), models_path / "blobs"}};
        auto pipe = ov::genai::Text2ImagePipeline(models_path, device, import_blob_properties);
    }

    // unet model export/import examples
    {
        auto unet = ov::genai::UNet2DConditionModel(models_path / "unet", device);
        unet.export_model(models_path / "blobs" / "unet");
    }

    {
        ov::AnyMap import_blob_properties{{ov::genai::blob_path.name(), models_path / "blobs" / "unet"}};
        auto unet = ov::genai::UNet2DConditionModel(models_path / "unet", device, import_blob_properties);

        auto pipe = ov::genai::Text2ImagePipeline::stable_diffusion(
            ov::genai::Scheduler::from_config(models_path / "scheduler" / "scheduler_config.json"),
            ov::genai::CLIPTextModel(models_path / "text_encoder", device),
            ov::genai::UNet2DConditionModel(models_path / "unet", device, import_blob_properties),
            ov::genai::AutoencoderKL(models_path / "vae_decoder", device));
    }

    test_load_unet_from_ir(models_path, device);
    test_load_unet_from_blob(models_path, device);
    // UNet load time from IR: 3414.7 ms
    // UNet load time from Blob: 579.4 ms

    test_load_full_pipe_from_ir(models_path, device);
    test_load_full_pipe_from_blob(models_path, device);
    // Full pipeline load time from IR: 6138.7 ms
    // Full pipeline load time from Blob: 3365.2 ms

    // ov::genai::Text2ImagePipeline pipe(models_path, device);
    // ov::Tensor image = pipe.generate(prompt,
    //                                  ov::genai::width(512),
    //                                  ov::genai::height(512),
    //                                  ov::genai::num_inference_steps(20),
    //                                  ov::genai::num_images_per_prompt(1),
    //                                  ov::genai::callback(progress_bar));

    // writes `num_images_per_prompt` images by pattern name
    // imwrite("image_%d.bmp", image, true);

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
