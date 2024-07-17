// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <cxxopts.hpp>

#include "openvino/genai/continuous_batching_pipeline.hpp"

void print_generation_result(const ov::genai::GenerationResult& generation_result) {
    for (size_t output_id = 0; output_id < generation_result.m_generation_ids.size(); ++output_id) {
        std::cout << "Answer " << output_id << " (" << generation_result.m_scores[output_id] << ") : " << generation_result.m_generation_ids[output_id] << std::endl;
    }
}

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

    const size_t num_prompts = 1;
    const bool dynamic_split_fuse = false;
    const std::string models_path = "/home/panas/llm/models/opt-125m/";

    // create dataset

    std::vector<std::string> prompt_examples = {
        "What is OpenVINO?\n",
        "Explain this in more details?\n",
        "What is your name?\n",
        "Tell me something about Canada\n",
        "What is OpenVINO?\n",
    };

    std::vector<ov::genai::GenerationConfig> sampling_params_examples {
        ov::genai::greedy(),
    };

    size_t num_chat_iterations = 10;

    std::vector<std::string> prompts(num_prompts);
    std::vector<ov::genai::GenerationConfig> sampling_params(num_prompts);

    for (size_t request_id = 0; request_id < num_prompts; ++request_id) {
        prompts[request_id] = prompt_examples[request_id % prompt_examples.size()];
        sampling_params[request_id] = sampling_params_examples[request_id % sampling_params_examples.size()];
    }

    ov::genai::SchedulerConfig scheduler_config {
        // batch size
        .max_num_batched_tokens = 64,
        // cache params
        .num_kv_blocks = 364,
        .block_size = 32,
        // mode - vLLM or dynamic_split_fuse
        .dynamic_split_fuse = dynamic_split_fuse,
        // vLLM specific params
        .max_num_seqs = 2,
        .enable_prefix_caching = true,
    };

    ov::genai::ContinuousBatchingPipeline pipe(models_path, scheduler_config);
    std::string conversation_history = "";
    for(size_t i = 0; i<num_chat_iterations; i++) {
        std::string question = conversation_history + prompt_examples[i % prompt_examples.size()];

        std::cout <<"Iteration "<< i << std::endl << "History: " << question << std::endl;
        std::vector<ov::genai::GenerationResult> generation_results = pipe.generate({question}, sampling_params);
        
        for (size_t request_id = 0; request_id < generation_results.size(); ++request_id) {
            const ov::genai::GenerationResult & generation_result = generation_results[request_id];
            
            switch (generation_result.m_status)
            {
            case ov::genai::GenerationStatus::FINISHED:
                print_generation_result(generation_result);
                break;
            case ov::genai::GenerationStatus::IGNORED:
                std::cout << "Request was ignored due to lack of memory." <<std::endl;
                if (generation_result.m_generation_ids.size() > 0) {
                    std::cout << "Partial result:" << std::endl;
                    print_generation_result(generation_result);
                }
                break;
            case ov::genai::GenerationStatus::DROPPED_BY_PIPELINE:
                std::cout << "Request was aborted." <<std::endl;
                if (generation_result.m_generation_ids.size() > 0) {
                    std::cout << "Partial result:" << std::endl;
                    print_generation_result(generation_result);
                }
                break;   
            default:
                break;
            }
            std::cout << std::endl;
            conversation_history = question + generation_result.m_generation_ids[0] + "\n";
        }
    }

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
