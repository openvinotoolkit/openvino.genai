// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <cxxopts.hpp>

#include "openvino/genai/continuous_batching_pipeline.hpp"

void print_cb_generation_result(const ov::genai::GenerationResult& generation_result) {
    for (size_t output_id = 0; output_id < generation_result.m_generation_ids.size(); ++output_id) {
        std::cout << "Answer " << output_id << " (" << generation_result.m_scores[output_id] << ") : " << generation_result.m_generation_ids[output_id] << std::endl;
    }
}

std::vector<ov::genai::GenerationConfig> get_spec_decoding_generation_config_examples() {
    
    // sampling param for speulative decoding
    ov::genai::GenerationConfig generation_config_greedy_constant = ov::genai::greedy();
    {
        generation_config_greedy_constant.num_assistant_tokens = 5;
    }

    ov::genai::GenerationConfig generation_config_multinomial_constant = ov::genai::multinomial();
    {
        generation_config_multinomial_constant.num_assistant_tokens = 5;
        generation_config_multinomial_constant.num_return_sequences = 1;
    }

    ov::genai::GenerationConfig generation_config_greedy_dynamic = ov::genai::greedy();
    {
        generation_config_greedy_dynamic.assistant_confidence_threshold = 0.8f;
    }

    ov::genai::GenerationConfig generation_config_multinomial_dynamic = ov::genai::multinomial();
    {
        generation_config_multinomial_dynamic.assistant_confidence_threshold = 0.8f;
    }

    return {
        generation_config_greedy_constant,
        generation_config_multinomial_constant,
        generation_config_greedy_dynamic,
        generation_config_multinomial_dynamic,
    };
}

int main(int argc, char* argv[]) try {
    // Command line options

    cxxopts::Options options("accuracy_sample", "Help command");

    options.add_options()
    ("n,num_prompts", "A number of prompts", cxxopts::value<size_t>()->default_value("1"))
    ("dynamic_split_fuse", "Whether to use dynamic split-fuse or vLLM scheduling", cxxopts::value<bool>()->default_value("false"))
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>()->default_value("."))
    ("a,draft_model", "Path to assisting model base directory", cxxopts::value<std::string>()->default_value("."))
    ("d,device", "Target device to run the model", cxxopts::value<std::string>()->default_value("CPU"))
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
    const std::string draft_models_path = result["draft_model"].as<std::string>();
    const std::string device = result["device"].as<std::string>();

    std::vector<std::string> prompt_examples = {
        "What is OpenVINO?",
        "How are you?",
        "What is your name?",
        "Tell me something about Canada",
        "What is OpenVINO?",
    };

    auto generation_config = get_spec_decoding_generation_config_examples();
    auto default_config_size = generation_config.size();
    for (size_t i = default_config_size; i < num_prompts; ++i) {
        generation_config.push_back(generation_config[i % default_config_size]);
    }

    std::vector<std::string> prompts(num_prompts);
    for (size_t i = 0; i < num_prompts; ++i) {
        prompts[i] = prompt_examples[i % prompt_examples.size()];
    }

    ov::genai::SchedulerConfig scheduler_config;
    // batch size
    scheduler_config.max_num_batched_tokens = 32;
    // cache params
    scheduler_config.num_kv_blocks = 364;
    // mode - vLLM or dynamic_split_fuse
    scheduler_config.dynamic_split_fuse = dynamic_split_fuse;
    // vLLM specific params
    scheduler_config.max_num_seqs = 2;
    
    ov::genai::ContinuousBatchingPipeline pipe(models_path, scheduler_config, device, {ov::genai::draft_model(draft_models_path, device)});
    std::vector<ov::genai::GenerationResult> generation_results = pipe.generate(prompts, generation_config);

    for (size_t request_id = 0; request_id < generation_results.size(); ++request_id) {
        const ov::genai::GenerationResult & generation_result = generation_results[request_id];
        std::cout << "Question: " << prompts[request_id] << std::endl;
        switch (generation_result.m_status)
        {
        case ov::genai::GenerationStatus::FINISHED:
            print_cb_generation_result(generation_result);
            break;
        case ov::genai::GenerationStatus::IGNORED:
            std::cout << "Request was ignored due to lack of memory." <<std::endl;
            if (generation_result.m_generation_ids.size() > 0) {
                std::cout << "Partial result:" << std::endl;
                print_cb_generation_result(generation_result);
            }
            break;
        case ov::genai::GenerationStatus::STOP:
        case ov::genai::GenerationStatus::CANCEL:
            std::cout << "Request was aborted." <<std::endl;
            if (generation_result.m_generation_ids.size() > 0) {
                std::cout << "Partial result:" << std::endl;
                print_cb_generation_result(generation_result);
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
