// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "continuous_batching_pipeline.hpp"

int main(int argc, char* argv[]) try {
    // create dataset

    const size_t dataset_size = 1;
    std::vector<std::string> prompt_examples = {
        "What is OpenVINO?",
        "How are you?",
        "What is OpenVINO?",
        "What is the current time",
        "What is OpenVINO?",
    };

    std::vector<SamplingParameters> sampling_params_examples {
        SamplingParameters::beam_search(),
        // SamplingParameters::greedy(),
        // SamplingParameters::multinomial(),
    };

    std::vector<std::string> prompts(dataset_size);
    std::vector<SamplingParameters> sampling_params(dataset_size);

    for (size_t request_id = 0; request_id < dataset_size; ++request_id) {
        prompts[request_id] = prompt_examples[request_id % prompt_examples.size()];
        sampling_params[request_id] = sampling_params_examples[request_id % sampling_params_examples.size()];
    }

    // Perform the inference

    SchedulerConfig scheduler_config {
        // batch size
        .max_num_batched_tokens = 32,
        // cache params
        .num_kv_blocks = 36400,
        .block_size = 16,
        // mode - vLLM or dynamic_split_fuse
        .dynamic_split_fuse = false,
        // vLLM specific params
        .max_num_seqs = 2,
        .max_paddings = 8,
    };

    ContinuousBatchingPipeline pipe("/home/sandye51/Documents/Programming/git_repo/vllm/", scheduler_config);
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
