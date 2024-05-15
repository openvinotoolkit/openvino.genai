// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <cxxopts.hpp>

#include "continuous_batching_pipeline.hpp"

int main(int argc, char* argv[]) try {
    // Command line options

    cxxopts::Options options("accuracy_sample", "Help command");

    options.add_options()
    ("n,num_prompts", "A number of prompts", cxxopts::value<size_t>()->default_value("1"))
    ("dynamic_split_fuse", "Whether to use dynamic split-fuse or vLLM scheduling", cxxopts::value<bool>()->default_value("false"))
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>()->default_value("."))
    ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    const size_t num_prompts = result["num_prompts"].as<size_t>();
    const bool dynamic_split_fuse = result["dynamic_split_fuse"].as<bool>();
    const std::string models_path = result["model"].as<std::string>();

    // create dataset

    std::vector<std::string> prompt_examples = {
        "What is OpenVINO?",
        "How are you?",
        "What is OpenVINO?",
        "What is the current time",
        "What is OpenVINO?",
    };

    std::vector<GenerationConfig> sampling_params_examples {
        GenerationConfig::beam_search(),
        // GenerationConfig::greedy(),
        // GenerationConfig::multinomial(),
    };

    std::vector<std::string> prompts(num_prompts);
    std::vector<GenerationConfig> sampling_params(num_prompts);

    for (size_t request_id = 0; request_id < num_prompts; ++request_id) {
        prompts[request_id] = prompt_examples[request_id % prompt_examples.size()];
        sampling_params[request_id] = sampling_params_examples[request_id % sampling_params_examples.size()];
    }

    // Perform the inference

    SchedulerConfig scheduler_config {
        // batch size
        .max_num_batched_tokens = 32,
        // cache params
        .num_kv_blocks = 364,
        .block_size = 32,
        // mode - vLLM or dynamic_split_fuse
        .dynamic_split_fuse = dynamic_split_fuse,
        // vLLM specific params
        .max_num_seqs = 2,
        .max_paddings = 8,
    };

    ContinuousBatchingPipeline pipe(models_path, scheduler_config);
    std::vector<GenerationResult> generation_results = pipe.generate(prompts, sampling_params);

    for (size_t request_id = 0; request_id < generation_results.size(); ++request_id) {
        const GenerationResult & generation_result = generation_results[request_id];

        std::cout << "Question: " << prompts[request_id] << std::endl;
        for (size_t output_id = 0; output_id < generation_result.m_generation_ids.size(); ++output_id) {
            std::cout << "Answer " << output_id << " (" << generation_result.m_scores[output_id] << ") : " << generation_result.m_generation_ids[output_id] << std::endl;
        }
        std::cout << std::endl;
    }

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
