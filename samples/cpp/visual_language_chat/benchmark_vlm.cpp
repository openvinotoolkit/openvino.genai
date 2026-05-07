// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cxxopts.hpp>
#include <filesystem>

#include "load_image.hpp"
#include <openvino/genai/speculative_decoding/perf_metrics.hpp>
#include <openvino/genai/visual_language/pipeline.hpp>
#include "../text_generation/read_prompt_from_file.h"

int main(int argc, char* argv[]) try {
    cxxopts::Options options("benchmark_vlm", "Help command");

    options.add_options()
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>()->default_value("."))
    ("dm,draft_model", "Path to draft model and tokenizers base directory", cxxopts::value<std::string>()->default_value(""))
    ("p,prompt", "Prompt", cxxopts::value<std::string>()->default_value(""))
    ("pf,prompt_file", "Read prompt from file", cxxopts::value<std::string>())
    ("i,image", "Image", cxxopts::value<std::string>()->default_value("image.jpg"))
    ("nw,num_warmup", "Number of warmup iterations", cxxopts::value<size_t>()->default_value(std::to_string(1)))
    ("n,num_iter", "Number of iterations", cxxopts::value<size_t>()->default_value(std::to_string(3)))
    ("mt,max_new_tokens", "Maximal number of new tokens", cxxopts::value<size_t>()->default_value(std::to_string(20)))
    ("d,device", "device", cxxopts::value<std::string>()->default_value("CPU"))
    ("pr,pruning_ratio", "(optional): Percentage of visual tokens to prune (valid range: 0-100); if this option is not provided, pruning is disabled.", cxxopts::value<size_t>())
    ("rw,relevance_weight", "(optional): Float value from 0 to 1, controls the trade-off between diversity and relevance for visual tokens pruning; a value of 0 disables relevance weighting, while higher values (up to 1.0) emphasize relevance, making pruning more conservative on borderline tokens.", cxxopts::value<float>())
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

    std::string prompt;
    if (result.count("prompt") && result.count("prompt_file")) {
        std::cout << "Prompt and prompt file should not exist together!" << std::endl;
        return EXIT_FAILURE;
    } else {
        if (result.count("prompt_file")) {
            prompt = utils::read_prompt(result["prompt_file"].as<std::string>());
        } else {
            prompt = result["prompt"].as<std::string>().empty() ? "What is on the image?" : result["prompt"].as<std::string>();
        }
    }
    if (prompt.empty()) {
        std::cout << "Prompt is empty!" << std::endl;
        return EXIT_FAILURE;
    } 

    const std::string models_path = result["model"].as<std::string>();
    const std::string draft_models_path = result["draft_model"].as<std::string>();
    const std::string image_path = result["image"].as<std::string>();
    std::string device = result["device"].as<std::string>();

    if (device == "NPU" && !draft_models_path.empty()) {
        std::cout << "--draft_model is not supported when --device is NPU" << std::endl;
        return EXIT_FAILURE;
    }

    size_t num_warmup = result["num_warmup"].as<size_t>();
    size_t num_iter = result["num_iter"].as<size_t>();
    std::vector<ov::Tensor> images = utils::load_images(image_path);

    ov::genai::GenerationConfig config;
    if (result.count("pruning_ratio")) {
        config.pruning_ratio = result["pruning_ratio"].as<size_t>();
    }
    if (result.count("relevance_weight")) {
        config.relevance_weight = result["relevance_weight"].as<float>();
    }
    config.max_new_tokens = result["max_new_tokens"].as<size_t>();
    config.ignore_eos = true;

     ov::AnyMap properties;
    if (!draft_models_path.empty()) {
        properties.insert(ov::genai::draft_model(draft_models_path, device));
    }

    std::cout << ov::get_openvino_version() << std::endl;

    std::unique_ptr<ov::genai::VLMPipeline> pipe;
    if (device == "NPU")
        pipe = std::make_unique<ov::genai::VLMPipeline>(models_path, device);
    else {
        // Setting of Scheduler config will trigger usage of ContinuousBatching pipeline, which is not default for Qwen2VL, Qwen2.5VL, Gemma3 due to accuracy issues.
        ov::genai::SchedulerConfig scheduler_config;
        scheduler_config.enable_prefix_caching = false;
        scheduler_config.max_num_batched_tokens = std::numeric_limits<std::size_t>::max();
        properties.insert(ov::genai::scheduler_config(scheduler_config));
        pipe = std::make_unique<ov::genai::VLMPipeline>(models_path, device, properties);
    }

    auto input_data = pipe->get_tokenizer().encode(prompt);
    size_t prompt_token_size = input_data.input_ids.get_shape()[1];
    std::cout << "Number of images:" << images.size() << ", prompt token size:" << prompt_token_size << std::endl;

    for (size_t i = 0; i < num_warmup; i++)
        pipe->generate(prompt, ov::genai::images(images), ov::genai::generation_config(config));
    
    auto res = pipe->generate(prompt, ov::genai::images(images), ov::genai::generation_config(config));
    auto metrics = res.perf_metrics;
    for (size_t i = 0; i < num_iter - 1; i++) {
        res = pipe->generate(prompt, ov::genai::images(images), ov::genai::generation_config(config));
        metrics = metrics + res.perf_metrics;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Output token size:" << res.perf_metrics.get_num_generated_tokens() << std::endl;
    std::cout << "Load time: " << metrics.get_load_time() << " ms" << std::endl;
    std::cout << "Generate time: " << metrics.get_generate_duration().mean << " ± " << metrics.get_generate_duration().std << " ms" << std::endl;
    std::cout << "Tokenization time: " << metrics.get_tokenization_duration().mean << " ± " << metrics.get_tokenization_duration().std << " ms" << std::endl;
    std::cout << "Detokenization time: " << metrics.get_detokenization_duration().mean << " ± " << metrics.get_detokenization_duration().std << " ms" << std::endl;
    std::cout << "Embeddings preparation time: " << metrics.get_prepare_embeddings_duration().mean << " ± " << metrics.get_prepare_embeddings_duration().std << " ms" << std::endl;
    std::cout << "TTFT: " << metrics.get_ttft().mean  << " ± " << metrics.get_ttft().std << " ms" << std::endl;
    std::cout << "TPOT: " << metrics.get_tpot().mean  << " ± " << metrics.get_tpot().std << " ms/token " << std::endl;
    std::cout << "Throughput: " << metrics.get_throughput().mean  << " ± " << metrics.get_throughput().std << " tokens/s" << std::endl;

    auto sd_perf_metrics = std::dynamic_pointer_cast<ov::genai::SDPerModelsPerfMetrics>(res.extended_perf_metrics);
    if (sd_perf_metrics) {
        auto main_model_metrics = sd_perf_metrics->main_model_metrics;
        std::cout << "\nMAIN MODEL " << std::endl;
        std::cout << "  Generate time: " << main_model_metrics.get_generate_duration().mean << " ms" << std::endl;
        std::cout << "  TTFT: " << main_model_metrics.get_ttft().mean  << " ± " << main_model_metrics.get_ttft().std << " ms" << std::endl;
        std::cout << "  TTST: " << main_model_metrics.get_ttst().mean  << " ± " << main_model_metrics.get_ttst().std << " ms/token " << std::endl;
        std::cout << "  TPOT: " << main_model_metrics.get_tpot().mean  << " ± " << main_model_metrics.get_tpot().std << " ms/iteration " << std::endl;
        std::cout << "  AVG Latency: " << main_model_metrics.get_latency().mean  << " ± " << main_model_metrics.get_latency().std << " ms/token " << std::endl;
        std::cout << "  Num generated token: " << main_model_metrics.get_num_generated_tokens() << " tokens" << std::endl;
        std::cout << "  Total iteration number: " << main_model_metrics.raw_metrics.m_durations.size() << std::endl;
        std::cout << "  Num accepted token: " << sd_perf_metrics->get_num_accepted_tokens() << " tokens" << std::endl;

        auto draft_model_metrics = sd_perf_metrics->draft_model_metrics;
        std::cout << "\nDRAFT MODEL " << std::endl;
        std::cout << "  Generate time: " << draft_model_metrics.get_generate_duration().mean << " ms" << std::endl;
        std::cout << "  TTFT: " << draft_model_metrics.get_ttft().mean  << " ms" << std::endl;
        std::cout << "  TTST: " << draft_model_metrics.get_ttst().mean  << " ms/token " << std::endl;
        std::cout << "  TPOT: " << draft_model_metrics.get_tpot().mean  << " ± " << draft_model_metrics.get_tpot().std << " ms/token " << std::endl;
        std::cout << "  AVG Latency: " << draft_model_metrics.get_latency().mean  << " ± " << draft_model_metrics.get_latency().std << " ms/iteration " << std::endl;
        std::cout << "  Num generated token: " << draft_model_metrics.get_num_generated_tokens() << " tokens" << std::endl;
        std::cout << "  Total iteration number: " << draft_model_metrics.raw_metrics.m_durations.size() << std::endl;
        const float accept_length = main_model_metrics.raw_metrics.m_durations.empty()
            ? 0.f
            : static_cast<float>(sd_perf_metrics->get_num_generated_tokens()) /
                static_cast<float>(main_model_metrics.raw_metrics.m_durations.size());
        std::cout << "  Accept length: " << accept_length << std::endl;
    }

    return 0;
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
