// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>

bool print_subword(std::string&& subword) {
    return !(std::cout << subword << std::flush);
}

#include <fstream>
#include <iostream>
#include <openvino/runtime/tensor.hpp>
#include <string>

template <typename T>
void print_array(T* array, size_t size) {
    std::cout << " => [ ";
    for (size_t i = 0; i < std::min(size, size_t(10)); ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << " ] " << std::endl;
}

template <typename T>
void print_tensor(ov::Tensor tensor) {
    const auto shape = tensor.get_shape();
    const size_t rank = shape.size();
    const auto* data = tensor.data<T>();

    if (rank > 3) {
        print_array(data, tensor.get_size());
        return;
    }

    const size_t batch_size = shape[0];
    const size_t seq_length = shape[1];

    std::cout << " => [ \n";
    for (size_t batch = 0; batch < batch_size; ++batch) {
        std::cout << "  [ ";
        const size_t batch_offset = batch * seq_length;

        if (rank == 2) {
            for (size_t j = 0; j < std::min(seq_length, size_t(10)); ++j) {
                std::cout << data[batch_offset + j] << " ";
            }
            std::cout << "]\n";
            continue;
        }

        const size_t hidden_size = shape[2];

        for (size_t seq = 0; seq < seq_length; ++seq) {
            if (seq != 0)
                std::cout << "    ";
            std::cout << "[ ";
            const size_t seq_offset = (batch_offset + seq) * hidden_size;
            for (size_t h = 0; h < std::min(hidden_size, size_t(10)); ++h) {
                std::cout << data[seq_offset + h] << " ";
            }
            std::cout << "]\n";
        }
    }
    std::cout << " ]" << std::endl;
}

inline void print_tensor(std::string name, ov::Tensor tensor) {
    std::cout << name;
    std::cout << " " << tensor.get_shape().to_string();
    if (tensor.get_element_type() == ov::element::i32) {
        print_tensor<int>(tensor);
    } else if (tensor.get_element_type() == ov::element::i64) {
        print_tensor<int64_t>(tensor);
    } else if (tensor.get_element_type() == ov::element::f32) {
        print_tensor<float>(tensor);
    } else if (tensor.get_element_type() == ov::element::boolean) {
        print_tensor<bool>(tensor);
    } else if (tensor.get_element_type() == ov::element::f16) {
        print_tensor<ov::float16>(tensor);
    }
}
int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES> <DEVICE>");
    }

    std::vector<ov::Tensor> rgbs = utils::load_images(argv[2]);

    // GPU and NPU can be used as well.
    // Note: If NPU is selected, only language model will be run on NPU
    std::string device = (argc == 4) ? argv[3] : "CPU";
    ov::AnyMap enable_compile_cache;
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
    }
    ov::genai::VLMPipeline pipe(argv[1], device, {{"ATTENTION_BACKEND", "SDPA"}});
    // print_tensor("tokenized", pipe.get_tokenizer().encode("<|im_start|>user\n<image>./</image>\nAnswer the question posed in the mobile screenshot.<|im_end|>\n<|im_start|>assistant\n").input_ids);  // [ 151644 872 198 7 151665 1725 151666 340 16141 279 ]  22
    // return 0;


    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;
    generation_config.apply_chat_template = false;

    std::string prompt;

    // pipe.start_chat();
    std::cout << "question:\n";

    std::getline(std::cin, prompt);
    pipe.generate("<|im_start|>user\n<image>./</image>\nAnswer the question posed in the mobile screenshot.<|im_end|>\n<|im_start|>assistant\n",
                  ov::genai::images(rgbs),
                  ov::genai::generation_config(generation_config),
                  ov::genai::streamer(print_subword));
    std::cout << "\n----------\n"
        "question:\n";
    while (std::getline(std::cin, prompt)) {
        pipe.generate(prompt,
                      ov::genai::generation_config(generation_config),
                      ov::genai::streamer(print_subword));
        std::cout << "\n----------\n"
            "question:\n";
    }
    // pipe.finish_chat();
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
