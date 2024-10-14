// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <cxxopts.hpp>

#include "openvino/genai/continuous_batching_pipeline.hpp"

void print_generation_result(const std::vector<std::string>& texts, const std::vector<float>& log_probs) {
    for (size_t output_id = 0; output_id < texts.size(); ++output_id) {
        std::cout << "Answer " << output_id << " (" << log_probs[output_id] << ") : " << texts[output_id] << std::endl;
    }
}

std::vector<ov::genai::GenerationConfig> get_spec_decoding_generation_config_examples() {
    
    // sampling param for speulative decoding
    ov::genai::GenerationConfig generation_config_greedy_constant = ov::genai::greedy();
    {
        generation_config_greedy_constant.num_assistant_tokens_schedule = ov::genai::NumAssistatantTokensScheduleType::CONSTANT;
        generation_config_greedy_constant.num_assistant_tokens = 5;
    }

    ov::genai::GenerationConfig generation_config_multinomial_constant = ov::genai::multinomial();
    {
        generation_config_multinomial_constant.num_assistant_tokens_schedule = ov::genai::NumAssistatantTokensScheduleType::CONSTANT;
        generation_config_multinomial_constant.num_assistant_tokens = 5;
        generation_config_multinomial_constant.num_return_sequences = 1;
    }

    ov::genai::GenerationConfig generation_config_greedy_dynamic = ov::genai::greedy();
    {
        generation_config_greedy_dynamic.num_assistant_tokens_schedule = ov::genai::NumAssistatantTokensScheduleType::HEURISTIC;
        generation_config_greedy_dynamic.assistant_confidence_threshold = 0.4f;
    }

    ov::genai::GenerationConfig generation_config_multinomial_dynamic = ov::genai::multinomial();
    {
        generation_config_multinomial_dynamic.num_assistant_tokens_schedule = ov::genai::NumAssistatantTokensScheduleType::HEURISTIC;
        generation_config_multinomial_dynamic.assistant_confidence_threshold = 0.4f;
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
    const std::string model_path = result["model"].as<std::string>();
    const std::string draft_model_path = result["draft_model"].as<std::string>();
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

    auto generation_config = get_spec_decoding_generation_config_examples();
    auto default_config_size = generation_config.size();
    for (size_t i = default_config_size; i < num_prompts; ++i) {
        generation_config.push_back(generation_config[i % default_config_size]);
    }

    std::vector<std::string> prompts(num_prompts);
    for (size_t i = 0; i < num_prompts; ++i) {
        prompts[i] = prompt_examples[i % prompt_examples.size()];
    }

    // Perform the inference
    auto get_default_block_size = [](const std::string& device) {
        const size_t cpu_block_size = 32;
        const size_t gpu_block_size = 16;

        bool is_gpu = device.find("GPU") != std::string::npos;

        return is_gpu ? gpu_block_size : cpu_block_size;
    };

    ov::genai::SchedulerConfig scheduler_config;
    // batch size
    scheduler_config.max_num_batched_tokens = use_prefix ? 256 : 32;
    // cache params
    scheduler_config.num_kv_blocks = 364;
    scheduler_config.block_size = get_default_block_size(device);
    // mode - vLLM or dynamic_split_fuse
    scheduler_config.dynamic_split_fuse = dynamic_split_fuse;
    // vLLM specific params
    scheduler_config.max_num_seqs = 2;
    scheduler_config.enable_prefix_caching = use_prefix;

    ov::AnyMap plugin_config{
        { ov::genai::scheduler_config.name(), scheduler_config },
        // device to run draft pipeline. Can be different with the main pipeline.
        // in case of same devices, plugin_config will be reused for both pipeline, KV cache will be splitted for main and draft pipeline
        { ov::genai::draft_model.name(), ov::genai::ModelDesc(draft_model_path, device) },
    };

    // It's possible to construct a Tokenizer from a different path.
    // If the Tokenizer isn't specified, it's loaded from the same folder.
    ov::genai::LLMPipeline pipe(model_path, ov::genai::Tokenizer{model_path}, device, plugin_config);

    if (use_prefix) {
        std::cout << "Running inference for prefix to compute the shared prompt's KV cache..." << std::endl;
        auto generation_results = pipe.generate(prefix_str, ov::genai::greedy());
    }

    for (size_t request_id = 0; request_id < prompts.size(); ++request_id) {
        ov::genai::DecodedResults generation_results = pipe.generate(prompts[request_id], generation_config[request_id]);
        std::cout << "Question: " << prompts[request_id] << std::endl;
        const std::vector<std::string>& text_results = generation_results.texts;
        const std::vector<float>& log_prob_results = generation_results.scores;
        print_generation_result(text_results, log_prob_results);
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
