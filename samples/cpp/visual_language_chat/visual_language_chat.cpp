// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/vlm_pipeline.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>

bool print_subword(std::string&& subword) {
    return !(std::cout << subword << std::flush);
}

int main(int argc, char* argv[]) {
    // std::string prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image_id>0</image_id><image><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk></image><slice><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk></slice><slice><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk></slice>\n<slice><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk></slice><slice><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk></slice>\nWhat is unusual on this image?<|im_end|>\n<|im_start|>assistant";
    // auto core = ov::Core{};
    // core.add_extension("/home/vzlobin/r/ov/build/openvino_genai/libopenvino_tokenizers.so");
    // auto ireq = core.compile_model(argv[1] + std::string{"openvino_tokenizer.xml"}, "CPU").create_infer_request();
    // ireq.set_input_tensor(ov::Tensor{ov::element::string, {1}, &prompt});
    // ireq.infer();
    // std::cout << "AAAAAAAAAAAAAAAAAAAAAAAAAa\n";
    if (3 != argc) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE>");
    }
    ov::Tensor image = utils::load_image(argv[2]);
    std::string device = "CPU";  // GPU can be used as well
    ov::AnyMap enable_compile_cache;
    if ("GPU" == device) {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
    }
    ov::genai::VLMPipeline pipe(argv[1], device, enable_compile_cache);
    std::string prompt;

    pipe.start_chat();
    std::cout << "question:\n";
    if (!std::getline(std::cin, prompt)) {
        throw std::runtime_error("std::cin failed");
    }
    pipe.generate(
        prompt,
        ov::genai::image(std::move(image)),
        ov::genai::streamer(print_subword)
    );
    std::cout << "\n----------\n"
        "question:\n";
    while (std::getline(std::cin, prompt)) {
        pipe.generate(prompt, ov::genai::streamer(print_subword));
        std::cout << "\n----------\n"
            "question:\n";
    }
    pipe.finish_chat();
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
