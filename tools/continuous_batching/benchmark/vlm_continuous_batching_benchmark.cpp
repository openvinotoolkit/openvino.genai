// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <cstdlib>
#include <chrono>
#include <ostream>
#include <stdexcept>
#include <thread>

#include <nlohmann/json.hpp>
#include <cxxopts.hpp>

#include "openvino/genai/cache_eviction.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "benchmark_utils.hpp"
#include "load_image.hpp"

namespace {

using benchmark_utils::Dataset;
using benchmark_utils::GenerationInfoCollector;

Dataset parse_vlm_dataset(const std::string& models_path, const std::string& dataset_path, const size_t num_prompts, const size_t max_output_len) {
    std::ifstream json_file(dataset_path.c_str());
    OPENVINO_ASSERT(json_file.is_open(), "Cannot open dataset file:", dataset_path);

    nlohmann::json json_dataset = nlohmann::json::parse(json_file);
    Dataset dataset;
    dataset.reserve(num_prompts);

    ov::genai::Tokenizer tokenizer(models_path);

    for (auto json_data_iterator = json_dataset.begin(); json_data_iterator != json_dataset.end() && dataset.size() < num_prompts; ++json_data_iterator) {
        auto & json_data = *json_data_iterator;

        std::string prompt = json_data["prompt"];
        std::string image_path = json_data["image"];

        ov::Tensor input_ids_prompt = tokenizer.encode(prompt).input_ids;
        size_t prompt_input_len = input_ids_prompt.get_size();

        ov::genai::GenerationConfig greedy_search;
        greedy_search.max_new_tokens = max_output_len;
        greedy_search.ignore_eos = true;

        dataset.push_data(prompt, greedy_search, image_path);
        dataset.push_lens(prompt_input_len, max_output_len);
    }

    return dataset;
}

void statisticsReporter(GenerationInfoCollector* generations_info_collector, size_t num_prompts) {
    size_t num_finished = 0;
    while (num_finished < num_prompts) {
        num_finished = generations_info_collector->run();
    }
    std::cout << "Benchmark finished, summarizing statistics..." << std::endl;
    generations_info_collector->print_statistics(true);

    std::cout << "Exiting statistics reporter thread." << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) try {
    //
    // Command line options
    //

    cxxopts::Options options("benchmark_sample", "Help command");

    options.add_options()
    ("n,num_prompts", "A number of prompts", cxxopts::value<size_t>()->default_value("1"))
    ("b,max_batch_size", "A maximum number of batched tokens", cxxopts::value<size_t>()->default_value("256"))
    ("dynamic_split_fuse", "Whether to use dynamic split-fuse or vLLM scheduling", cxxopts::value<bool>()->default_value("true"))
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>()->default_value("."))
    ("dataset", "Path to dataset .json file", cxxopts::value<std::string>())
    ("max_output_len", "Max output length", cxxopts::value<size_t>()->default_value("128"))
    ("cache_size", "Size of memory used for KV cache in GB. Default: 16", cxxopts::value<size_t>()->default_value("16"))
    ("device", "Target device to run the model. Default: CPU", cxxopts::value<std::string>()->default_value("CPU"))
    ("use_cache_eviction", "Whether to use cache eviction", cxxopts::value<bool>()->default_value("false"))
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

    if (!result.count("dataset")) {
        std::cout << "The option '--dataset' is required!" << std::endl;
        return EXIT_FAILURE;
    }

    if (result["dataset"].as<std::string>().empty()) {
        std::cout << "The value of option '--dataset' is empty!" << std::endl;
        return EXIT_FAILURE;
    }

    if (result["num_prompts"].as<size_t>() == 0) {
        std::cout << "The num of prompts should be greater than 0!" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << ov::get_openvino_version() << std::endl;

    const size_t num_prompts = result["num_prompts"].as<size_t>();
    const size_t max_batch_size = result["max_batch_size"].as<size_t>();
    const bool dynamic_split_fuse = result["dynamic_split_fuse"].as<bool>();
    const std::string models_path = result["model"].as<std::string>();
    const std::string dataset_path = result["dataset"].as<std::string>();
    const size_t max_output_len = result["max_output_len"].as<size_t>();
    const size_t cache_size = result["cache_size"].as<size_t>();
    const std::string device = result["device"].as<std::string>();
    const bool use_cache_eviction = result["use_cache_eviction"].as<bool>();

    // Create requests for generation
    Dataset dataset = parse_vlm_dataset(models_path, dataset_path, num_prompts, max_output_len);
    const size_t prompt_nums = std::min(num_prompts, dataset.size());

    // Configure scheduler and pipeline settings
    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.enable_prefix_caching = false;
    scheduler_config.max_num_batched_tokens = max_batch_size;
    scheduler_config.cache_size = cache_size;
    scheduler_config.dynamic_split_fuse = dynamic_split_fuse;
    scheduler_config.max_num_seqs = 256;
    if (use_cache_eviction) {
        scheduler_config.use_cache_eviction = true;
        scheduler_config.cache_eviction_config = ov::genai::CacheEvictionConfig(32,
                                                                                 32,
                                                                                 128,
                                                                                 ov::genai::AggregationMode::NORM_SUM,
                                                                                 false,
                                                                                 8,
                                                                                 ov::genai::KVCrushConfig(0, ov::genai::KVCrushAnchorPointMode::MEAN));
    }

    std::cout << "Benchmarking parameters: " << std::endl;
    std::cout << "\tMax number of batched tokens: " << scheduler_config.max_num_batched_tokens << std::endl;
    std::cout << "\tScheduling type: " << (scheduler_config.dynamic_split_fuse ? "dynamic split-fuse" : "vLLM") << std::endl;
    if (!scheduler_config.dynamic_split_fuse) {
        std::cout << "\tMax number of batched sequences: " << scheduler_config.max_num_seqs << std::endl;
    }
    std::cout << "\tNum prompts: " << prompt_nums << std::endl;
    std::cout << "\tTarget device: " << device << std::endl;
    
    // Benchmarking
    std::cout << "Loading models, creating pipelines, preparing environment..." << std::endl;
    ov::genai::ContinuousBatchingPipeline pipe(models_path, scheduler_config, device);

    GenerationInfoCollector generation_info_collector(true);
    generation_info_collector.set_start_time(std::chrono::steady_clock::now());
    for (size_t request_id = 0; request_id < prompt_nums; ++request_id) {
        std::vector<ov::Tensor> images = utils::load_images(dataset.m_image_path[request_id]);
        generation_info_collector.add_generation(&pipe, &dataset, request_id, false, &images);
    }

    std::thread statisticsReporterThread(statisticsReporter, &generation_info_collector, prompt_nums);
    
    while (pipe.has_non_finished_requests()) {
        pipe.step();
    }

    statisticsReporterThread.join();

    auto tokenizer = pipe.get_tokenizer();
    generation_info_collector.output_generated_text(tokenizer);

    return EXIT_SUCCESS;
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
