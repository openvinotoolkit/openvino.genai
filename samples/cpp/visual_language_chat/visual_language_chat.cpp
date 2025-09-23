// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <openvino/genai/continuous_batching_pipeline.hpp>
#include <filesystem>

bool print_subword(std::string&& subword) {
    return !(std::cout << subword << std::flush);
}

int main(int argc, char* argv[]) {
    // if (argc < 3 || argc > 4) {
    //     throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES> <DEVICE>");
    // }

    std::vector<ov::Tensor> rgbs = utils::load_images("/home/panas/319483352-d5fbbd1a-d484-415c-88cb-9986625b7b11.jpg");

    // GPU and NPU can be used as well.
    // Note: If NPU is selected, only language model will be run on NPU
    std::string device = (argc == 4) ? argv[3] : "CPU";
    ov::AnyMap enable_compile_cache;
        enable_compile_cache["ATTENTION_BACKEND"] = "SDPA";
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
    }
   // ov::genai::VLMPipeline pipe("/home/panas/test_models/Qwen2.5-VL-3B-Instruct-int8/", device, enable_compile_cache);

    ov::genai::SchedulerConfig sch_config;
    sch_config.max_num_batched_tokens = 256;
    ov::genai::ContinuousBatchingPipeline pipe("/home/panas/test_models/Qwen2.5-VL-3B-Instruct-int8/", sch_config, device);

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;

    std::string prompt = "What is in the image?";



   // pipe.start_chat();
            // std::cout << std::endl;
            // print_tensor("position_ids", position_ids);
            // std::cout << std::endl;
    
    //auto results = pipe.generate( {"What is in the image?", "What is in the image?", "What is in the image?", "What is in the image?"}, {rgbs,rgbs,rgbs,rgbs}, {generation_config,generation_config,generation_config,generation_config});
    //auto results = pipe.generate( {"What is in the image?", "What is in the image?" }, {rgbs,rgbs}, {generation_config, generation_config});
    auto r1 = pipe.add_request(0, "What is in the image?", rgbs, generation_config);
    auto r2 = pipe.add_request(1, "What is in the image?", rgbs, generation_config);
    while (pipe.has_non_finished_requests()) {
      pipe.step();
    }
    auto r_gen1 = r1->read_all();
    auto r_gen2 = r2->read_all();


    ov::genai::Tokenizer tok("/home/panas/test_models/Qwen2.5-VL-3B-Instruct-int8/");
    std::cout << tok.decode(r_gen1[0].generated_ids) << std::endl;
    std::cout << tok.decode(r_gen2[0].generated_ids) << std::endl;


    // size_t idx = 0;
    // for (auto res: results) {
    //   std::cout << "res "<<idx << std::endl;
    //   std::cout << res.texts[0] << std::endl;
    // }
    // // pipe.generate(prompt,
    //               ov::genai::images(rgbs),
    //               ov::genai::generation_config(generation_config),
    //               ov::genai::streamer(print_subword));
    // std::cout << "\n----------\n"
    //     "question:\n";
    // while (std::getline(std::cin, prompt)) {
    //     pipe.generate(prompt,
    //                   ov::genai::generation_config(generation_config),
    //                   ov::genai::streamer(print_subword));
    //     std::cout << "\n----------\n"
    //         "question:\n";
    // }
   // pipe.finish_chat();

}
