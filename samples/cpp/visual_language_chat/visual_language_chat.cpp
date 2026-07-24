// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>
#include <openvino/genai/speculative_decoding/perf_metrics.hpp>
ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}

int main(int argc, char* argv[]) try {
    if (argc < 3 || argc > 10) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES> [DEVICE] [PROMPT_LOOKUP] [DRAFT_MODEL_DIR] [CM_PATH] [NUM_ASSISTANT_TOKENS] [BRANCHING_FACTOR] [TREE_DEPTH]");
    }

    std::vector<ov::Tensor> rgbs = utils::load_images(argv[2]);

    // GPU and NPU can be used as well.
    // Note: If NPU is selected, only language model will be run on NPU
    std::string device = (argc >= 4) ? argv[3] : "CPU";
    std::string lookup = (argc >= 5) ? argv[4] : "false";
    bool prompt_lookup = (lookup == "true");
    std::string draft_model_dir = (argc >= 6) ? argv[5] : "";
    std::string cm_path_arg = (argc >= 7) ? argv[6] : "false";
    bool use_cm_path = (cm_path_arg == "true");
    size_t num_assistant_tokens = (argc >= 8) ? std::stoul(argv[7]) : 15;
    size_t branching_factor = (argc >= 9) ? std::stoul(argv[8]) : 8;
    size_t tree_depth = (argc == 10) ? std::stoul(argv[9]) : 4;
    if (prompt_lookup && !draft_model_dir.empty()) {
        throw std::runtime_error("PROMPT_LOOKUP and DRAFT_MODEL_DIR are mutually exclusive");
    }

    // Prompt lookup decoding in VLM pipeline enforces ContinuousBatching backend
    ov::AnyMap properties = {ov::genai::prompt_lookup(prompt_lookup)};
    if (!draft_model_dir.empty()) {
        // Speculative decoding (e.g. EAGLE3) — also enforces ContinuousBatching backend
        properties.insert(ov::genai::draft_model(draft_model_dir, device));
    }

    if (use_cm_path && device != "NPU") {
        // CM PA path w/o sparse — XATTENTION mode with a high threshold
        ov::genai::SchedulerConfig scheduler_config;
        scheduler_config.enable_prefix_caching = false;
        scheduler_config.max_num_batched_tokens = std::numeric_limits<std::size_t>::max();
        ov::genai::SparseAttentionConfig sparse_attention_config;
        sparse_attention_config.mode = ov::genai::SparseAttentionMode::XATTENTION;
        sparse_attention_config.xattention_threshold = 100.0f;
        scheduler_config.use_sparse_attention = true;
        scheduler_config.sparse_attention_config = sparse_attention_config;
        properties.insert(ov::genai::scheduler_config(scheduler_config));
        properties.insert(ov::hint::kv_cache_precision(ov::element::i8));
    }
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        //properties.insert({ov::cache_dir("vlm_cache")});
    }
    //properties.insert(ov::hint::kv_cache_precision(ov::element::f16));
    ov::genai::VLMPipeline pipe(argv[1], device, properties);

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;
    if (prompt_lookup) {
        // Define candidates number for candidate generation
        generation_config.num_assistant_tokens = 5;
        // Define max_ngram_size
        generation_config.max_ngram_size = 3;
    } else if (!draft_model_dir.empty()) {
        // EAGLE3 tree-drafting parameters
        generation_config.num_assistant_tokens = num_assistant_tokens;
        generation_config.branching_factor = branching_factor;
        generation_config.tree_depth = tree_depth;
    }

    std::string prompt;

    ov::genai::ChatHistory history;
    
    std::cout << "question:\n";
    std::getline(std::cin, prompt);

    history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
    ov::genai::VLMDecodedResults decoded_results = pipe.generate(
        history,
        ov::genai::images(rgbs),
        ov::genai::generation_config(generation_config),
        ov::genai::streamer(print_subword)
    );
    auto sd_perf_metrics = std::dynamic_pointer_cast<ov::genai::SDPerModelsPerfMetrics>(decoded_results.extended_perf_metrics);
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
    history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});
    std::cout << "\n----------\n"
                 "question:\n";
    while (std::getline(std::cin, prompt)) {
        history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
        // New images and videos can be passed at each turn
        ov::genai::VLMDecodedResults decoded_results = pipe.generate(
            history,
            ov::genai::generation_config(generation_config),
            ov::genai::streamer(print_subword)
        );
        history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});
        std::cout << "\n----------\n"
                     "question:\n";
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
