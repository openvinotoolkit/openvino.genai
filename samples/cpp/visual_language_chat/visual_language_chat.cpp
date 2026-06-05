// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>

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
    use_cm_path = true;
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
    }
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        //properties.insert({ov::cache_dir("vlm_cache")});
    }
    properties.insert(ov::hint::kv_cache_precision(ov::element::f16));
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
