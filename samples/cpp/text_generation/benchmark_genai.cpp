// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"
#include <openvino/runtime/properties.hpp>
#include <cxxopts.hpp>
#include "read_prompt_from_file.h"

int main(int argc, char* argv[]) try {
    cxxopts::Options options("benchmark_vanilla_genai", "Help command");

    options.add_options()
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>())
    ("p,prompt", "Prompt", cxxopts::value<std::string>()->default_value(""))
    ("pf,prompt_file", "Read prompt from file", cxxopts::value<std::string>())
    ("nw,num_warmup", "Number of warmup iterations", cxxopts::value<size_t>()->default_value(std::to_string(1)))
    ("n,num_iter", "Number of iterations", cxxopts::value<size_t>()->default_value(std::to_string(3)))
    ("mt,max_new_tokens", "Maximal number of new tokens", cxxopts::value<size_t>()->default_value(std::to_string(20)))
    ("d,device", "device", cxxopts::value<std::string>()->default_value("CPU"))
    ("pc,enable_prefix_caching", "Enable prefix caching (true/false)", cxxopts::value<std::string>()->default_value("true"))
    ("kv_load_dir", "Directory to load pre-computed KV cache from disk", cxxopts::value<std::string>()->default_value(""))
    ("kv_dump_dir", "Directory to dump KV cache to disk after prefill", cxxopts::value<std::string>()->default_value(""))
    ("kv_cache_precision", "KV cache precision: u8, f16, f32 (default: device default)", cxxopts::value<std::string>()->default_value(""))
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
            prompt = result["prompt"].as<std::string>().empty() ? "The Sky is blue because" : result["prompt"].as<std::string>();
        }
    }
    if (prompt.empty()) {
        std::cout << "Prompt is empty!" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string models_path = result["model"].as<std::string>();
    std::string device = result["device"].as<std::string>();
    size_t num_warmup = result["num_warmup"].as<size_t>();
    size_t num_iter = result["num_iter"].as<size_t>();
    
    // Parse prefix caching parameter (string to bool conversion)
    std::string pc_str = result["enable_prefix_caching"].as<std::string>();
    bool enable_prefix_caching = (pc_str == "true" || pc_str == "1" || pc_str == "yes");

    ov::genai::GenerationConfig config;
    config.max_new_tokens = result["max_new_tokens"].as<size_t>();

    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.enable_prefix_caching = enable_prefix_caching;
    scheduler_config.max_num_batched_tokens = std::numeric_limits<std::size_t>::max();
    
    // Configure KV cache size (use cache_size in GB for auto-calculation)
    scheduler_config.num_kv_blocks = 0;  // Let it be auto-calculated from cache_size
    scheduler_config.cache_size = 4;     // 4GB cache size
    
    // KV cache load/dump via SchedulerConfig (alternative to environment variables)
    // These take precedence over OV_GENAI_LOAD_KV_DIR and OV_GENAI_DUMP_KV_DIR env vars
    std::string kv_load_dir = result["kv_load_dir"].as<std::string>();
    std::string kv_dump_dir = result["kv_dump_dir"].as<std::string>();
    if (!kv_load_dir.empty()) {
        scheduler_config.kv_cache_load_dir = kv_load_dir;
        std::cout << "[Benchmark] KV cache load directory: " << kv_load_dir << std::endl;
    }
    if (!kv_dump_dir.empty()) {
        scheduler_config.kv_cache_dump_dir = kv_dump_dir;
        std::cout << "[Benchmark] KV cache dump directory: " << kv_dump_dir << std::endl;
    }
    
    // Parse KV cache precision
    std::string kv_prec_str = result["kv_cache_precision"].as<std::string>();
    ov::element::Type kv_cache_precision = ov::element::dynamic;  // dynamic means use device default
    if (kv_prec_str == "u8") {
        kv_cache_precision = ov::element::u8;
    } else if (kv_prec_str == "f16") {
        kv_cache_precision = ov::element::f16;
    } else if (kv_prec_str == "f32") {
        kv_cache_precision = ov::element::f32;
    } else if (!kv_prec_str.empty()) {
        std::cout << "Warning: Unknown kv_cache_precision '" << kv_prec_str << "', using device default" << std::endl;
    }
    
    // IMPORTANT: KV cache precision is determined at model compile time and cached in model_cache/
    // If you change --kv_cache_precision, you MUST delete the model_cache directory first!
    // Otherwise, the cached model with the old precision will be loaded instead.
    if (!kv_prec_str.empty()) {
        std::cout << "\n[WARNING] KV cache precision is set to '" << kv_prec_str << "'." << std::endl;
        std::cout << "[WARNING] If this is different from previous runs, delete the 'model_cache' directory" << std::endl;
        std::cout << "[WARNING] to force recompilation with the new precision.\n" << std::endl;
    }
    
    std::cout << "[Benchmark] Configuration:" << std::endl;
    std::cout << "[Benchmark] - cache_size=" << scheduler_config.cache_size << "GB" << std::endl;
    std::cout << "[Benchmark] - enable_prefix_caching=" << (enable_prefix_caching ? "true" : "false") << std::endl;
    std::cout << "[Benchmark] - kv_cache_precision=" << (kv_prec_str.empty() ? "default" : kv_prec_str) << std::endl;

    std::cout << ov::get_openvino_version() << std::endl;
    std::cout << "Prefix caching: " << (enable_prefix_caching ? "ENABLED" : "DISABLED") << std::endl;

    std::unique_ptr<ov::genai::LLMPipeline> pipe;
    if (device == "NPU") {
        pipe = std::make_unique<ov::genai::LLMPipeline>(models_path, device);
    } else if (kv_cache_precision != ov::element::dynamic) {
        // Create pipeline with explicit KV cache precision
        pipe = std::make_unique<ov::genai::LLMPipeline>(
            models_path, 
            device, 
            ov::genai::scheduler_config(scheduler_config),
            ov::hint::kv_cache_precision(kv_cache_precision)
        );
    } else {
        pipe = std::make_unique<ov::genai::LLMPipeline>(models_path, device, ov::genai::scheduler_config(scheduler_config));
    }

    auto input_data = pipe->get_tokenizer().encode(prompt);
    size_t prompt_token_size = input_data.input_ids.get_shape()[1];
    std::cout << "Prompt token size:" << prompt_token_size << std::endl;
    
    // Debug: Print first 20 tokens of current prompt
    std::cout << "Current prompt tokens (first 20): ";
    for (size_t i = 0; i < std::min(size_t(20), prompt_token_size); i++) {
        std::cout << input_data.input_ids.data<int64_t>()[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Skip warmup" << std::endl;
    /*
    for (size_t i = 0; i < num_warmup; i++)
        pipe->generate(prompt, config);
    */

    std::cout << "[Benchmark] Starting generation..." << std::endl;
    ov::genai::DecodedResults res = pipe->generate(prompt, config);
    std::cout << "[Benchmark] Generation completed successfully" << std::endl;
    
    std::cout << "[Benchmark] Extracting metrics..." << std::endl;
    ov::genai::PerfMetrics metrics = res.perf_metrics;
    std::cout << "[Benchmark] Metrics extracted successfully" << std::endl;
    
    // Print the generated result to verify correctness
    std::cout << "\n=== GENERATION RESULT ===" << std::endl;
    std::cout << "Generated text: \"" << res.texts[0] << "\"" << std::endl;
    std::cout << "========================\n" << std::endl;
    
    std::cout << "Skip num_iter" << std::endl;
    /*
    for (size_t i = 0; i < num_iter - 1; i++) {
        res = pipe->generate(prompt, config);
        metrics = metrics + res.perf_metrics;
    }
    */
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Output token size:" << res.perf_metrics.get_num_generated_tokens() << std::endl;
    std::cout << "Load time: " << metrics.get_load_time() << " ms" << std::endl;
    std::cout << "Generate time: " << metrics.get_generate_duration().mean << " ± " << metrics.get_generate_duration().std << " ms" << std::endl;
    std::cout << "Tokenization time: " << metrics.get_tokenization_duration().mean << " ± " << metrics.get_tokenization_duration().std << " ms" << std::endl;
    std::cout << "Detokenization time: " << metrics.get_detokenization_duration().mean << " ± " << metrics.get_detokenization_duration().std << " ms" << std::endl;
    std::cout << "TTFT: " << metrics.get_ttft().mean  << " ± " << metrics.get_ttft().std << " ms" << std::endl;
    std::cout << "TPOT: " << metrics.get_tpot().mean  << " ± " << metrics.get_tpot().std << " ms/token " << std::endl;
    std::cout << "Throughput: " << metrics.get_throughput().mean  << " ± " << metrics.get_throughput().std << " tokens/s" << std::endl;

    std::cout << "\n[Benchmark] Cleaning up pipeline..." << std::endl;
    
    try {
        // Explicitly reset the pipeline to catch destruction errors
        pipe.reset();
        std::cout << "[Benchmark] Pipeline cleanup completed successfully" << std::endl;
    } catch (const ov::AssertFailure& e) {
        std::cerr << "[Benchmark] ERROR: OpenVINO assertion failure during pipeline cleanup: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "[Benchmark] ERROR: Exception during pipeline cleanup: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

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
