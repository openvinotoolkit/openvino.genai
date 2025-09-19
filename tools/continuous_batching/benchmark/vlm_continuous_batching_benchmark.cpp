// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <cstdlib>
#include <chrono>
#include <ostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <atomic>

#include <nlohmann/json.hpp>
#include <cxxopts.hpp>

#include "openvino/genai/cache_eviction.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "load_image.hpp"

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

struct VLMDataset {
    std::vector<std::string> m_prompts;
    std::vector<std::string> m_image_path;
    std::vector<ov::genai::GenerationConfig> m_sampling_params;
    std::vector<size_t> m_input_lens, m_output_lens;

    size_t m_total_input_len = 0;
    size_t m_total_output_len = 0;

    void reserve(const size_t size) {
        m_prompts.reserve(size);
        m_image_path.reserve(size);
        m_sampling_params.reserve(size);
        m_input_lens.reserve(size);
        m_output_lens.reserve(size);
    }

    void push_data(std::string prompt, std::string image_path, ov::genai::GenerationConfig sampling_params) {
        m_prompts.push_back(prompt);
        m_image_path.push_back(image_path);
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

VLMDataset parse_vlm_dataset(const std::string& models_path, const std::string& dataset_path, const size_t num_prompts, const size_t max_new_tokens) {
    std::ifstream json_file(dataset_path.c_str());
    OPENVINO_ASSERT(json_file.is_open(), "Cannot open dataset file:", dataset_path);

    nlohmann::json json_dataset = nlohmann::json::parse(json_file);
    VLMDataset dataset;
    dataset.reserve(num_prompts);

    ov::genai::Tokenizer tokenizer(models_path);

    for (auto json_data_iterator = json_dataset.begin(); json_data_iterator != json_dataset.end() && dataset.size() < num_prompts; ++json_data_iterator) {
        auto & json_data = *json_data_iterator;

        std::string prompt = json_data["prompt"];
        std::string image_path = json_data["image"];

        ov::Tensor _input_ids_prompt = tokenizer.encode(prompt).input_ids;
        size_t prompt_input_len = _input_ids_prompt.get_size();

        ov::genai::GenerationConfig greedy_search = ov::genai::greedy();
        greedy_search.max_new_tokens = max_new_tokens;
        greedy_search.ignore_eos = true;

        dataset.push_data(prompt, image_path, greedy_search);
        dataset.push_lens(prompt_input_len, max_new_tokens);
    }

    return dataset;
}

class GenerationInfo {

    struct SequenceInfo {
        std::chrono::milliseconds ttft;
        std::chrono::milliseconds cumulated_tpot;
        std::chrono::milliseconds mean_tpot;
        size_t num_output_tokens;
        std::vector<int64_t> generated_ids;
    
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point last_read_time;

        SequenceInfo(std::chrono::steady_clock::time_point& start_time) {
            num_output_tokens = 0;
            ttft = std::chrono::milliseconds::zero();
            cumulated_tpot = std::chrono::milliseconds::zero();
            this->start_time = start_time;
        }

        void update() {
            std::chrono::steady_clock::time_point new_read_time = std::chrono::steady_clock::now();
            if (last_read_time.time_since_epoch() == std::chrono::milliseconds::zero()) {
                ttft = std::chrono::duration_cast<std::chrono::milliseconds>(new_read_time - start_time);
            } else {
                cumulated_tpot += std::chrono::duration_cast<std::chrono::milliseconds>(new_read_time - last_read_time);
                if (num_output_tokens > 0)
                    mean_tpot = cumulated_tpot / num_output_tokens;

            }
            num_output_tokens++;
            last_read_time = new_read_time;
        }

        void update_generated_ids(std::vector<int64_t>& gen_ids) {
            generated_ids.insert(generated_ids.end(), gen_ids.begin(), gen_ids.end());
        }
    };

    struct GenerationMetrics {
        std::chrono::milliseconds mean_ttft = std::chrono::milliseconds::zero();
        std::chrono::milliseconds mean_tpot = std::chrono::milliseconds::zero();
        size_t num_output_tokens = 0;
        size_t num_input_tokens;
    };

    ov::genai::GenerationHandle generation_handle;
    std::chrono::steady_clock::time_point start_time;
    std::unordered_map<int64_t, SequenceInfo> sequences_info;
    bool active = true;
    size_t input_len;

public:
    GenerationInfo(ov::genai::GenerationHandle generation_handle, size_t input_len) : input_len(input_len)
    {
        this->generation_handle = std::move(generation_handle);
        start_time = std::chrono::steady_clock::now();
    }

    std::unordered_map<int64_t, SequenceInfo> get_all_seq_info() {
        return sequences_info;
    }

    void update_sequence(int64_t sequence_id, ov::genai::GenerationOutput& gen_output) {
        if (sequences_info.find(sequence_id) == sequences_info.end())
            sequences_info.emplace(sequence_id, SequenceInfo(start_time));
        sequences_info.at(sequence_id).update();
        sequences_info.at(sequence_id).update_generated_ids(gen_output.generated_ids);
    }

    void update(ov::genai::GenerationOutputs& outputs){
        for (auto& output: outputs) {
            update_sequence(output.first, output.second);
        }
    }

    ov::genai::GenerationOutputs read() {
        return generation_handle->read();
    }

    bool can_read() {
        return generation_handle->can_read();
    }

    bool is_finished() {
        return generation_handle->get_status() == ov::genai::GenerationStatus::FINISHED;
    }

    void set_inactive() {
        active = false;
    }

    bool is_active() {
        return active;
    }

    GenerationMetrics get_metrics() {
        GenerationMetrics generation_metrics;
        if (!sequences_info.empty()) {
            for (auto& sequenceInfoPair : sequences_info) {
                generation_metrics.mean_ttft += sequenceInfoPair.second.ttft;
                generation_metrics.mean_tpot += sequenceInfoPair.second.mean_tpot;
                generation_metrics.num_output_tokens += sequenceInfoPair.second.num_output_tokens;
            }
            generation_metrics.mean_ttft /= sequences_info.size();
            generation_metrics.mean_tpot /= sequences_info.size();
            generation_metrics.num_input_tokens = input_len;
        }
        return generation_metrics;
    }
};

class GenerationInfoCollector {
    std::vector<GenerationInfo> generations_info;
    size_t num_finished = 0;
    std::chrono::steady_clock::time_point start_time;

public:

    void set_start_time(std::chrono::steady_clock::time_point start_time) {
        this->start_time = start_time;
    }

    void add_generation(ov::genai::ContinuousBatchingPipeline* pipe, VLMDataset* dataset, size_t request_id) {
        auto sampling_params = dataset->m_sampling_params[request_id];

        std::vector<ov::Tensor> images = utils::load_images(dataset->m_image_path[request_id]);
        ov::genai::GenerationHandle generation_handle = pipe->add_request(request_id, dataset->m_prompts[request_id], images, sampling_params);
        generations_info.emplace_back(std::move(generation_handle), dataset->m_input_lens[request_id]);
    }

    size_t run() {
        for (GenerationInfo& generation_info : generations_info) {
            if (!generation_info.is_active())
                continue;
            
            if (generation_info.is_finished()) {
                num_finished++;
                generation_info.set_inactive();
            } else if (generation_info.can_read()) {
                auto outputs = generation_info.read();
                generation_info.update(outputs);
            }
        }
        return num_finished;
    }

    void output_generated_text(ov::genai::Tokenizer& tokenizer) {
        int request_idx = 0;
        for (GenerationInfo& generation_info : generations_info) {
            auto outputs = generation_info.get_all_seq_info();
            for (const auto& output : outputs) {
                auto text = tokenizer.decode(output.second.generated_ids);
                std::cout << "[" << request_idx << "] generated text:" << text << std::endl;
            }
            request_idx++;
        }
    }

    void print_statistics() {
        std::chrono::seconds total_duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time);
        std::chrono::milliseconds mean_ttft = std::chrono::milliseconds::zero();
        std::chrono::milliseconds mean_tpot = std::chrono::milliseconds::zero();
        size_t total_input_len = 0;
        size_t total_output_len = 0;

        for (GenerationInfo& generation_info : generations_info) {
            auto generation_metrics = generation_info.get_metrics();
            mean_ttft += generation_metrics.mean_ttft;
            mean_tpot += generation_metrics.mean_tpot;
            total_input_len += generation_metrics.num_input_tokens;
            total_output_len += generation_metrics.num_output_tokens;
        }
        mean_ttft /= generations_info.size();
        mean_tpot /= generations_info.size();
        std::cout << "Benchmark duration: " << total_duration.count() << " s" << std::endl;
        std::cout << "Total number of input prompt tokens: " << total_input_len << std::endl;
        std::cout << "Total number of output tokens: " << total_output_len << std::endl;

        if (total_duration.count() > 0) {
            std::cout << "Input throughput: " << total_input_len / total_duration.count() << " tokens / s" << std::endl;
            std::cout << "Output throughput: " << total_output_len / total_duration.count() << " tokens / s" << std::endl;            
        }
        
        if (mean_ttft.count() > 0)
            std::cout << "Mean TTFT: " << mean_ttft.count() << " ms" << std::endl;
        
        if (mean_tpot.count() > 0)
            std::cout << "Mean TPOT: " << mean_tpot.count() << " ms" << std::endl; 
    }
};
}  // namespace

void statisticsReporter(GenerationInfoCollector* generations_info_collector, int num_prompts) {
    int num_finished = 0;
    while (num_finished < num_prompts) {
        num_finished = generations_info_collector->run();
    }
    std::cout << "Benchmark finished, summarizing statistics..." << std::endl;
    generations_info_collector->print_statistics();

    std::cout << "Exiting statistics reporter thread." << std::endl;
}

int main(int argc, char* argv[]) try {
    //
    // Command line options
    //

    cxxopts::Options options("benchmark_sample", "Help command");

    options.add_options()
    ("n,num_prompts", "A number of prompts", cxxopts::value<size_t>()->default_value("1"))
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>()->default_value("."))
    ("dataset", "Path to dataset .json file", cxxopts::value<std::string>())
    ("mt,max_new_tokens", "Maximal number of output tokens", cxxopts::value<size_t>()->default_value("128"))
    ("device", "Target device to run the model. Default: CPU", cxxopts::value<std::string>()->default_value("CPU"))
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

    if (result["dataset"].as<std::string>().empty()) {
        std::cout << "The value of option '--dataset' is empty!" << std::endl;
        return EXIT_FAILURE;
    }

    if (result["num_prompts"].as<size_t>() <= 0) {
        std::cout << "The num of prompts should be greater than 0!" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << ov::get_openvino_version() << std::endl;

    const size_t num_prompts = result["num_prompts"].as<size_t>();
    const std::string models_path = result["model"].as<std::string>();
    const std::string dataset_path = result["dataset"].as<std::string>();
    const size_t max_new_tokens = result["max_new_tokens"].as<size_t>();
    const std::string device = result["device"].as<std::string>();

    // Create requests for generation
    VLMDataset dataset = parse_vlm_dataset(models_path, dataset_path, num_prompts, max_new_tokens);
    const size_t prompt_nums = std::min(num_prompts, dataset.size());

    // Perform the first inference
    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.enable_prefix_caching = false;
    scheduler_config.max_num_batched_tokens = std::numeric_limits<std::size_t>::max();

    std::cout << "Benchmarking parameters: " << std::endl;
    std::cout << "\tMax number of batched tokens: " << scheduler_config.max_num_batched_tokens << std::endl;
    std::cout << "\tNum prompts: " << prompt_nums << std::endl;
    std::cout << "\tTarget device: " << device << std::endl;
    
    // Benchmarking
    std::cout << "Loading models, creating pipelines, preparing environment..." << std::endl;
    ov::genai::ContinuousBatchingPipeline pipe(models_path, scheduler_config, device);

    GenerationInfoCollector generation_info_collector;
    generation_info_collector.set_start_time(std::chrono::steady_clock::now());
    for (size_t request_id = 0; request_id < prompt_nums; ++request_id) {
        generation_info_collector.add_generation(&pipe, &dataset, request_id);
    }

    std::thread statisticsReporterThread(statisticsReporter, &generation_info_collector, prompt_nums);
    
    while (pipe.has_non_finished_requests()) {
        pipe.step();
    }

    statisticsReporterThread.join();

    auto tokenizer = pipe.get_tokenizer();
    generation_info_collector.output_generated_text(tokenizer);

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
