// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <cxxopts.hpp>

#include "openvino/genai/continuous_batching_pipeline.hpp"

std::string gen_random(const int len) {
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
                                   " !@#$%^&*()_-+";
    std::string tmp_s;
    tmp_s.reserve(len);

    for (int i = 0; i < len; ++i) {
        tmp_s += alphanum[rand() % (sizeof(alphanum) - 1)];
    }

    return tmp_s;
}

#include <ctime>
#include <iostream>
#include <unistd.h>
int main(int argc, char* argv[]) {
    std::vector<std::string> prompts = {"be5nfEKe6a"};

    for (const auto& p : prompts) {
        {
            // Command line options
            srand((unsigned)time(NULL) * getpid());
            cxxopts::Options options("accuracy_sample", "Help command");

            options.add_options()("n,num_prompts", "A number of prompts", cxxopts::value<size_t>()->default_value("1"))(
                "dynamic_split_fuse",
                "Whether to use dynamic split-fuse or vLLM scheduling",
                cxxopts::value<bool>()->default_value("false"))("m,model",
                                                                "Path to model and tokenizers base directory",
                                                                cxxopts::value<std::string>()->default_value("."))(
                "d,device",
                "Target device to run the model",
                cxxopts::value<std::string>()->default_value("GPU"))(
                "use_prefix",
                "Whether to use a prefix or not",
                cxxopts::value<bool>()->default_value("false"))("h,help", "Print usage");

            cxxopts::ParseResult result;
            try {
                result = options.parse(argc, argv);
            } catch (const cxxopts::exceptions::exception& e) {
                std::cout << e.what() << "\n\n";
                std::cout << options.help() << std::endl;
                return EXIT_FAILURE;
            }

            const size_t num_prompts = 1;
            const std::string models_path = result["model"].as<std::string>();
            const std::string device = "CPU";


            // create dataset
            std::vector<std::string> prompt_examples;

            // Generate random prompt
            // auto p = //gen_random(len);
            std::cout << "prompt: " << p << std::endl;
            prompt_examples.push_back(p);

            std::vector<ov::genai::GenerationConfig> sampling_params_examples{
                ov::genai::greedy(),
            };

            std::vector<std::string> prompts(num_prompts);
            std::vector<ov::genai::GenerationConfig> sampling_params(num_prompts);

            for (size_t request_id = 0; request_id < num_prompts; ++request_id) {
                prompts[request_id] = prompt_examples[request_id % prompt_examples.size()];
                sampling_params[request_id] = sampling_params_examples[request_id % sampling_params_examples.size()];
            }

            ov::genai::SchedulerConfig scheduler_config;
            // batch size
            scheduler_config.max_num_batched_tokens = 256;
            // cache params
            scheduler_config.num_kv_blocks = 364;
            // mode - vLLM or dynamic_split_fuse
            // vLLM specific params
            scheduler_config.max_num_seqs = 2;


            std::cout << "Continuous Batching PIPE (PagedAttention) STARTED: " << std::endl;
            ov::genai::ContinuousBatchingPipeline pipe(models_path,
                                                       ov::genai::Tokenizer{models_path},
                                                       scheduler_config,
                                                       device);

            std::vector<ov::genai::GenerationResult> generation_results = pipe.generate(prompts, sampling_params);
            std::cout << std::string(generation_results.front().m_generation_ids[0]) << std::endl;

            std::cout << "VANILLA LLM PIPE (SDPA) STARTED: " << std::endl;
            ov::genai::LLMPipeline llm_pipe(models_path, device);
            ov::genai::DecodedResults res = llm_pipe.generate(prompts, sampling_params[0]);
            std::cout << res << std::endl;

            if (std::string(generation_results.front().m_generation_ids[0]) != std::string(res)) {
                std::cout << "ERRROR" << std::endl;
                std::cout << "CB : " << std::string(generation_results.front().m_generation_ids[0]) << std::endl;
                std::cout << "LLM : " << std::string(res) << std::endl;
                return 0;
            }
        }
    }
}
