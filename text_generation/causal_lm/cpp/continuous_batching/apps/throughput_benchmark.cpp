// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <cstdlib>

#include <openvino/openvino.hpp>
#include <nlohmann/json.hpp>
#include <cxxopts.hpp>

#include "tokenizer.hpp"
#include "continuous_batching_pipeline.hpp"

namespace {

class AutoStartTimer {
    const decltype(std::chrono::steady_clock::now()) m_start;
public:
    AutoStartTimer() :
        m_start(std::chrono::steady_clock::now()) {
    }

    double current_in_milli() const {
        auto m_end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(m_end - m_start).count();
    }
};

struct Dataset {
    std::vector<std::string> m_prompts;
    std::vector<GenerationConfig> m_sampling_params;
    std::vector<size_t> m_input_lens, m_output_lens;

    size_t m_total_input_len = 0;
    size_t m_total_output_len = 0;

    void reserve(const size_t size) {
        m_prompts.reserve(size);
        m_sampling_params.reserve(size);
        m_input_lens.reserve(size);
        m_output_lens.reserve(size);
    }

    void push_data(std::string prompt, GenerationConfig sampling_params) {
        m_prompts.push_back(prompt);
        m_sampling_params.push_back(sampling_params);
    }

    void push_lens(size_t input_len, size_t output_len) {
        m_input_lens.push_back(input_len);
        m_output_lens.push_back(output_len);

        m_total_input_len += input_len;
        m_total_output_len += output_len;
    }

    float get_average_input_len() const {
        OPENVINO_ASSERT(!empty());
        return static_cast<float>(m_total_input_len / size());
    }

    float get_average_output_len() const {
        OPENVINO_ASSERT(!empty());
        return static_cast<float>(m_total_output_len / size());
    }

    bool empty() const {
        return size() == 0;
    }

    size_t size() const {
        return m_prompts.size();
    }
};

Dataset filtered_dataset(const std::string& models_path, const std::string& dataset_path, const size_t num_prompts, const size_t max_input_len, const size_t max_output_len) {
    std::ifstream json_file(dataset_path.c_str());
    OPENVINO_ASSERT(json_file.is_open(), "Cannot open dataset file");

    // from vLLM tput benchmark
    const float dataset_size_coeff = 1.2f;

    nlohmann::json json_dataset = nlohmann::json::parse(json_file);
    Dataset sampled_dataset, dataset;
    const size_t num_prompt_candidates = static_cast<size_t>(num_prompts * dataset_size_coeff);
    sampled_dataset.reserve(num_prompt_candidates);
    dataset.reserve(num_prompt_candidates);

    Tokenizer tokenizer(models_path);

    for (auto json_data_iterator = json_dataset.begin(); json_data_iterator != json_dataset.end() && dataset.size() < num_prompt_candidates; ++json_data_iterator) {
        auto & json_data = *json_data_iterator;

        // Filter out the conversations with less than 2 turns.
        if (json_data["conversations"].size() < 2)
            continue;

        // Only keep the first two turns of each conversation.
        std::string human_question = json_data["conversations"][0]["value"];
        std::string gpt_answer = json_data["conversations"][1]["value"];

        ov::Tensor _input_ids_prompt = tokenizer.encode(human_question);
        size_t input_len = _input_ids_prompt.get_size();

        ov::Tensor _input_ids_answer = tokenizer.encode(gpt_answer);
        size_t output_len = _input_ids_answer.get_size();

        // Prune too short sequences.
        if (input_len < 4 || output_len < 4)
            continue;
        // Prune too long sequences.
        if (input_len > max_input_len || (input_len + output_len) > 2048)
            continue;

        GenerationConfig greedy_search = GenerationConfig::greedy();
        greedy_search.max_new_tokens = std::min(max_output_len, output_len);

        dataset.push_data(human_question, greedy_search);
        dataset.push_lens(input_len, output_len);
    }

    // sample dataset
    srand(42);

    for (size_t selected_index = rand() % dataset.size(); sampled_dataset.size() < num_prompts; selected_index = rand() % dataset.size()) {
        sampled_dataset.push_data(dataset.m_prompts[selected_index], dataset.m_sampling_params[selected_index]);
        sampled_dataset.push_lens(dataset.m_input_lens[selected_index], dataset.m_output_lens[selected_index]);
    }

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

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    const size_t num_prompts = result["num_prompts"].as<size_t>();
    const size_t max_batch_size = result["max_batch_size"].as<size_t>();
    const bool dynamic_split_fuse = result["dynamic_split_fuse"].as<bool>();
    const std::string models_path = result["model"].as<std::string>();
    const std::string dataset_path = result["dataset"].as<std::string>();
    const size_t max_input_len = result["max_input_len"].as<size_t>();
    const size_t max_output_len = result["max_output_len"].as<size_t>();

    // Create requests for generation
    Dataset dataset = filtered_dataset(models_path, dataset_path, num_prompts, max_input_len, max_output_len);

    // Perform the first inference
    SchedulerConfig scheduler_config {
        .max_num_batched_tokens = max_batch_size,
        .num_kv_blocks = 36800,
        .block_size = 32,
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
    ContinuousBatchingPipeline pipe(models_path, scheduler_config);

    for (size_t request_id = 0; request_id < dataset.size(); ++request_id) {
        pipe.add_request(request_id, dataset.m_prompts[request_id], dataset.m_sampling_params[request_id]);
    }

    AutoStartTimer timer;
    while (pipe.has_running_requests())
        pipe.step();
    double total_time_in_ms = timer.current_in_milli();

    std::cout << "Total input tokens: " << dataset.m_total_input_len << std::endl;
    std::cout << "Total output tokens: " << dataset.m_total_output_len << std::endl;
    std::cout << "Average input len: " << dataset.get_average_input_len() << " tokens" << std::endl;
    std::cout << "Average output len: " << dataset.get_average_output_len() << " tokens" << std::endl;
    std::cout << "Total execution time secs: " << total_time_in_ms / 1000. << " secs" << std::endl;
    std::cout << "Tput: " << (dataset.m_total_input_len + dataset.m_total_output_len) / (total_time_in_ms / 1000.) << " tokens / sec " << std::endl << std::endl;

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
