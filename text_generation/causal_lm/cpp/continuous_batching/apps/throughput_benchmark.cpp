// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <cstdlib>

#include <openvino/openvino.hpp>
#include <nlohmann/json.hpp>
#include <cxxopts.hpp>

#include "tokenizer.hpp"
#include "timer.hpp"
#include "continuous_batching_pipeline.hpp"

namespace {

std::vector<std::pair<std::string, SamplingParameters>> filtered_dataset(const std::string& models_path, const std::string& dataset_path, const size_t num_prompts, const size_t max_input_len, const size_t max_output_len) {
    std::ifstream json_file(dataset_path.c_str());
    OPENVINO_ASSERT(json_file.is_open(), "Cannot open dataset file");

    // from vLLM tput benchmark
    const float dataset_size_coeff = 1.2f;

    nlohmann::json json_dataset = nlohmann::json::parse(json_file);
    std::vector<std::pair<std::string, SamplingParameters>> sampled_dataset, dataset;
    const size_t num_prompt_candidates = static_cast<size_t>(num_prompts * dataset_size_coeff);
    sampled_dataset.reserve(num_prompt_candidates);
    dataset.reserve(num_prompt_candidates);

    Tokenizer tokenizer(models_path);
    std::map<std::string, double> perf_counters;

    for (auto json_data_iterator = json_dataset.begin(); json_data_iterator != json_dataset.end() && dataset.size() < num_prompt_candidates; ++json_data_iterator) {
        auto & json_data = *json_data_iterator;

        // Filter out the conversations with less than 2 turns.
        if (json_data["conversations"].size() < 2)
            continue;

        // Only keep the first two turns of each conversation.
        std::string human_question = json_data["conversations"][0]["value"];
        std::string gpt_answer = json_data["conversations"][1]["value"];

        ov::Tensor _input_ids_prompt = tokenizer.encode(human_question);
        ov::Tensor _input_ids_answer = tokenizer.encode(gpt_answer);

        size_t input_len = _input_ids_prompt.get_size(), output_len = _input_ids_answer.get_size();

        // Prune too short sequences.
        if (input_len < 4 || output_len < 4)
            continue;
        // Prune too long sequences.
        if (input_len > max_input_len || (input_len + output_len) > 2048)
            continue;

        SamplingParameters greedy_search = SamplingParameters::greedy();
        greedy_search.max_new_tokens = std::min(max_output_len, output_len);

        dataset.push_back({ human_question, greedy_search });
    }

    // sample dataset
    srand(42);
    std::generate_n(std::back_inserter(sampled_dataset), num_prompts, [&] {
        return dataset[rand() % dataset.size()];
    });

    return sampled_dataset;
}

}  // namespace

int main(int argc, char* argv[]) try {
    //
    // Command line options
    //

    cxxopts::Options options("benchmark_sample", "Help command");

    options.add_options()
    ("n,num_prompts", "A number of prompts", cxxopts::value<size_t>()->default_value("1000"))
    ("b,max_batch_size", "A maximum number of batched tokens", cxxopts::value<size_t>()->default_value("256"))
    ("dynamic_split_fuse", "Whether to use dynamic split-fuse or vLLM scheduling", cxxopts::value<bool>()->default_value("false"))
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>()->default_value("."))
    ("dataset", "Path to dataset .json file", cxxopts::value<std::string>()->default_value("./ShareGPT_V3_unfiltered_cleaned_split.json"))
    ("max_input_len", "Max input length take from dataset", cxxopts::value<size_t>()->default_value("1024"))
    ("max_output_len", "Max output length", cxxopts::value<size_t>()->default_value("2048"))
    ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    const size_t num_prompts = result["num_prompts"].as<size_t>();
    const size_t max_batch_size = result["max_batch_size"].as<size_t>();
    const bool dynamic_split_fuse = result["dynamic_split_fuse"].as<bool>();
    const std::string models_path = result["model"].as<std::string>();
    const std::string dataset_path = result["dataset"].as<std::string>();
    const size_t max_input_len = result["max_input_len"].as<size_t>();
    const size_t max_output_len = result["max_output_len"].as<size_t>();

    // Create requests for generation
    std::vector<std::pair<std::string, SamplingParameters>> dataset = filtered_dataset(models_path, dataset_path, num_prompts, max_input_len, max_output_len);

    // Perform the first inference
    SchedulerConfig scheduler_config {
        .max_num_batched_tokens = max_batch_size,
        .num_kv_blocks = 36800,
        .block_size = 16,
        .dynamic_split_fuse = dynamic_split_fuse,
        .max_num_seqs = 256, // not used if dynamic_split_fuse=True
        .max_paddings = 256, // not used if dynamic_split_fuse=True
    };

    std::cout << "Benchmarking parameters: " << std::endl;
    std::cout << "\tMax number of batched tokens: " << scheduler_config.max_num_batched_tokens << std::endl;
    std::cout << "\tScheduling type: " << (scheduler_config.dynamic_split_fuse ? "dynamic split-fuse" : "vLLM") << std::endl;
    if (!scheduler_config.dynamic_split_fuse) {
        std::cout << "\tMax number of batched sequences: " << scheduler_config.max_num_seqs << std::endl;
        std::cout << "\tMax number of padding tokens within prompt batch: " << scheduler_config.max_paddings << std::endl;
    }
    std::cout << "Dataset parameters: " << std::endl;
    std::cout << "\tNum prompts: " << num_prompts << std::endl;
    std::cout << "\tMax input length: " << max_input_len << std::endl;
    std::cout << "\tMax output length: " << max_output_len << std::endl;

    // Benchmarking
    ContinuousBatchingPipeline pipe(models_path, scheduler_config/*, ov::enable_profiling(true)*/);

    for (size_t request_id = 0; request_id < dataset.size(); ++request_id) {
        pipe.add_request(request_id, dataset[request_id].first, dataset[request_id].second);
    }

    Timer timer;
    size_t total_input_tokens = 0, total_output_tokens = 0;
    double paged_attention_time_ms = 0.0, matmul_time_ms = 0.0, infer_total_ms = 0.0;

    for (size_t num_finished = 0; pipe.has_running_requests(); ) {
        std::vector<GenerationResult> results = pipe.step();
        if (!results.empty()) {
            num_finished += results.size();
            for (size_t output_id = 0; output_id < results.size(); ++output_id) {
                size_t output_len = results[output_id].m_generation_ids[0].size();
                size_t input_len = dataset[results[output_id].m_request_id].first.size();
                // check correctness of generated length
                OPENVINO_ASSERT(dataset[results[output_id].m_request_id].second.max_new_tokens == output_len);
                // accumulate input tokens
                total_input_tokens += input_len;
                // accumulate output tokens
                total_output_tokens += output_len;
            }
            std::cout << "Finished: " << num_finished << std::endl;
        }

        // collect performance metrics
        // std::vector<ov::ProfilingInfo> profiling_info = request.get_profiling_info();
        // for (const ov::ProfilingInfo& info : profiling_info) {
        //     double current_time = info.real_time.count();
        //     if (info.node_type == "PagedAttentionExtension") {
        //         paged_attention_time_ms += current_time;
        //     } else if (info.node_type == "FullyConnected") {
        //         matmul_time_ms += current_time;
        //     }
        //     infer_total_ms += current_time;
        // }
    }

    double total_time_in_ms = timer.current_in_milli();
    infer_total_ms /= 1000;
    paged_attention_time_ms /= 1000;
    matmul_time_ms /= 1000;

    std::cout << "Total input tokens: " << total_input_tokens << std::endl;
    std::cout << "Total output tokens: " << total_output_tokens << std::endl;
    std::cout << "Average input len: " << total_input_tokens / static_cast<float>(num_prompts) << " tokens" << std::endl;
    std::cout << "Average output len: " << total_output_tokens / static_cast<float>(num_prompts) << " tokens" << std::endl;
    std::cout << "Total execution time secs: " << total_time_in_ms / 1000. << " secs" << std::endl;
    std::cout << "Tput: " << (total_input_tokens + total_output_tokens) / (total_time_in_ms / 1000.) << " tokens / sec " << std::endl << std::endl;

    std::cout << "Paged attention % of inference execution: " << (paged_attention_time_ms / infer_total_ms) * 100 << std::endl;
    std::cout << "MatMul % of inference execution: " << (matmul_time_ms / infer_total_ms) * 100 << std::endl;
    std::cout << "Total inference execution secs: " << infer_total_ms / 1000. << std::endl;
    std::cout << "Inference % of total execution: " << (infer_total_ms / total_time_in_ms) * 100 << std::endl;

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
