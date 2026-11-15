// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <optional>
#include <string>

#include "imwrite.hpp"
#include "load_image.hpp"
#include "openvino/genai/bagel/pipeline.hpp"

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc == 5 || argc == 6,
                    "Usage: ",
                    argv[0],
                    " <MODEL_DIR> <DEVICE> <vqa|t2i|edit> '<PROMPT>' [IMAGE]");
    const std::string models_path = argv[1], device = argv[2], task = argv[3], prompt = argv[4];
    OPENVINO_ASSERT(task == "vqa" || task == "t2i" || task == "edit", "Unknown task ", task);
    OPENVINO_ASSERT(task != "edit" || argc == 6, "edit requires an IMAGE");

    ov::genai::BagelPipeline pipe(models_path, device);

    if (task == "vqa") {
        std::optional<ov::Tensor> image;
        if (argc == 6) {
            image = utils::load_image(argv[5]);
        }
        std::cout << pipe.generate_text(prompt, image) << std::endl;
    } else if (task == "t2i") {
        imwrite("image_%d.bmp", pipe.generate_image(prompt), true);
    } else {
        imwrite("image_%d.bmp", pipe.edit_image(utils::load_image(argv[5]), prompt), true);
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
