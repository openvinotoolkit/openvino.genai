// Copyright (C) 2023-2025 Intel Corporation
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
    ("d,device", "Target device to run the model", cxxopts::value<std::string>()->default_value("CPU"))
    ("use_prefix", "Whether to use a prefix or not", cxxopts::value<bool>()->default_value("false"))
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
    const std::string device = result["device"].as<std::string>();
    const bool use_prefix = result["use_prefix"].as<bool>();

    std::string prefix_str =
        "You are an advanced language model designed to assist users by providing accurate, "
        "relevant, and helpful information. Your responses should be accurate, concise, contextual, "
        "respectful, and helpful. The request is: ";

    // create dataset

    std::vector<std::string> prompt_examples = {
        "What is OpenVINO?",
        "How are you?",
        "What is your name?",
        "Tell me something about Canada",
        "What is OpenVINO?",
    };

    std::vector<ov::genai::GenerationConfig> sampling_params_examples {
        ov::genai::beam_search(),
        ov::genai::greedy(),
        ov::genai::multinomial(),
    };

    std::vector<std::string> prompts(num_prompts);
    std::vector<ov::genai::GenerationConfig> sampling_params(num_prompts);

    for (size_t request_id = 0; request_id < num_prompts; ++request_id) {
        prompts[request_id] = use_prefix ? prefix_str + prompt_examples[request_id % prompt_examples.size()]
                                         : prompt_examples[request_id % prompt_examples.size()];
        sampling_params[request_id] = sampling_params_examples[request_id % sampling_params_examples.size()];
    }

    ov::genai::SchedulerConfig scheduler_config;
    // batch size
    scheduler_config.max_num_batched_tokens = use_prefix ? 256 : 32;
    // cache params
    scheduler_config.num_kv_blocks = 364;
    // mode - vLLM or dynamic_split_fuse
    scheduler_config.dynamic_split_fuse = dynamic_split_fuse;
    // vLLM specific params
    scheduler_config.max_num_seqs = 2;
    scheduler_config.enable_prefix_caching = use_prefix;

    // It's possible to construct a Tokenizer from a different path.
    // If the Tokenizer isn't specified, it's loaded from the same folder.
    ov::genai::ContinuousBatchingPipeline pipe(models_path, ov::genai::Tokenizer{models_path}, scheduler_config, device);

    if (use_prefix) {
        std::cout << "Running inference for prefix to compute the shared prompt's KV cache..." << std::endl;
        std::vector<ov::genai::GenerationResult> generation_results = pipe.generate({prefix_str}, {ov::genai::greedy()});
        ov::genai::GenerationResult& generation_result = generation_results.front();
        OPENVINO_ASSERT(generation_result.m_status == ov::genai::GenerationStatus::FINISHED);
    }

    std::vector<ov::genai::GenerationResult> generation_results = pipe.generate(prompts, sampling_params);

    for (size_t request_id = 0; request_id < generation_results.size(); ++request_id) {
        const ov::genai::GenerationResult & generation_result = generation_results[request_id];
        std::cout << "Question: " << prompts[request_id] << std::endl;
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
        case ov::genai::GenerationStatus::STOP:
        case ov::genai::GenerationStatus::CANCEL:
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
    }
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
