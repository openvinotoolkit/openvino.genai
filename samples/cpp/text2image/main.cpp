// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/text2image/pipeline.hpp"

#include "imwrite.hpp"

namespace {

    void imwrite_output_imgs(const ov::Tensor& output) {
        ov::Shape out_shape = output.get_shape();

        if (out_shape[0] == 1) {
            imwrite("image.bmp", output, true);
            return;
        }

        ov::Shape img_shape = {1, out_shape[1], out_shape[2], out_shape[3]};
        size_t img_size = output.get_size() / out_shape[0];

        ov::Tensor image(output.get_element_type(), img_shape);
        uint8_t* out_data = output.data<uint8_t>();
        uint8_t* img_data = image.data<uint8_t>();

        for (int img_num = 0; img_num < out_shape[0]; ++img_num) {
            std::memcpy(img_data, out_data + img_size * img_num, img_size * sizeof(uint8_t));

            char img_name[25];
            sprintf(img_name, "image_%d.bmp", img_num);

            imwrite(img_name, image, true);
        }
    }

} //namespace

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc == 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>'");

    const std::string models_path = argv[1], prompt = argv[2];
    const std::string device = "CPU";  // GPU, NPU can be used as well

    ov::genai::Text2ImagePipeline pipe(models_path, device);
    ov::Tensor image = pipe.generate(prompt,
        ov::genai::width(512),
        ov::genai::height(512),
        ov::genai::num_inference_steps(20),
        ov::genai::num_images_per_prompt(1));

    imwrite_output_imgs(image);

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
