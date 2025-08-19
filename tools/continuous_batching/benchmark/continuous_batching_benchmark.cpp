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
    std::vector<ov::genai::GenerationConfig> m_sampling_params;
    std::vector<size_t> m_input_lens, m_output_lens;

    size_t m_total_input_len = 0;
    size_t m_total_output_len = 0;

    void reserve(const size_t size) {
        m_prompts.reserve(size);
        m_sampling_params.reserve(size);
        m_input_lens.reserve(size);
        m_output_lens.reserve(size);
    }

    void push_data(std::string prompt, ov::genai::GenerationConfig sampling_params) {
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

    ov::genai::Tokenizer tokenizer(models_path);

    for (auto json_data_iterator = json_dataset.begin(); json_data_iterator != json_dataset.end() && dataset.size() < num_prompt_candidates; ++json_data_iterator) {
        auto & json_data = *json_data_iterator;

        // Filter out the conversations with less than 2 turns.
        if (json_data["conversations"].size() < 2)
            continue;

        // Only keep the first two turns of each conversation.
        std::string human_question = json_data["conversations"][0]["value"];
        std::string gpt_answer = json_data["conversations"][1]["value"];

        ov::Tensor _input_ids_prompt = tokenizer.encode(human_question).input_ids;
        size_t input_len = _input_ids_prompt.get_size();

        ov::Tensor _input_ids_answer = tokenizer.encode(gpt_answer).input_ids;
        size_t output_len = _input_ids_answer.get_size();

        // Prune too short sequences.
        if (input_len < 4 || output_len < 4)
            continue;
        // Prune too long sequences.
        if (input_len > max_input_len || (input_len + output_len) > 2048)
            continue;

        ov::genai::GenerationConfig greedy_search = ov::genai::greedy();
        greedy_search.max_new_tokens = std::min(max_output_len, output_len);
        greedy_search.ignore_eos = true;

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

class GenerationInfo {

    struct SequenceInfo {
        std::chrono::milliseconds ttft;
        std::chrono::milliseconds cumulated_tpot;
        std::chrono::milliseconds mean_tpot;
        size_t num_output_tokens;
    
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
                mean_tpot = cumulated_tpot / num_output_tokens;

            }
            num_output_tokens++;
            last_read_time = new_read_time;
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

    void update_sequence(int64_t sequence_id) {
        if (sequences_info.find(sequence_id) == sequences_info.end())
            sequences_info.emplace(sequence_id, SequenceInfo(start_time));
        sequences_info.at(sequence_id).update();
    }

    void update(ov::genai::GenerationOutputs& outputs){
        for (auto const& output: outputs) {
            update_sequence(output.first);
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
    std::mutex mutex;
    std::vector<GenerationInfo> generations_info;
    size_t num_finished = 0;
    std::chrono::steady_clock::time_point start_time;

public:

    void set_start_time(std::chrono::steady_clock::time_point start_time) {
        this->start_time = start_time;
    }

    void add_generation(ov::genai::ContinuousBatchingPipeline* pipe, Dataset* dataset, size_t request_id, bool is_speculative_decoding_enabled) {
        auto sampling_params = dataset->m_sampling_params[request_id];
        if (is_speculative_decoding_enabled) {
            // to enable static speculative decoding
            sampling_params.num_assistant_tokens = 5;
            // to enable dynamic speculative decoding
            // sampling_params.assistant_confidence_threshold = 0.4f;
        }
        ov::genai::GenerationHandle generation_handle = pipe->add_request(request_id, dataset->m_prompts[request_id], sampling_params);
        std::lock_guard<std::mutex> lock(mutex);
        generations_info.emplace_back(std::move(generation_handle), dataset->m_input_lens[request_id]);
    }

    size_t run() {
        std::lock_guard<std::mutex> lock(mutex);
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

    void print_statistics() {
        std::chrono::seconds total_duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time);
        std::chrono::milliseconds mean_ttft = std::chrono::milliseconds::zero();
        std::chrono::milliseconds mean_tpot = std::chrono::milliseconds::zero();
        size_t total_input_len = 0;
        size_t total_output_len = 0;
        
    
        for (GenerationInfo& generation_info : generations_info){
            auto generation_metrics = generation_info.get_metrics();
            mean_ttft += generation_metrics.mean_ttft;
            mean_tpot += generation_metrics.mean_tpot;
            total_input_len += generation_metrics.num_input_tokens;
            total_output_len += generation_metrics.num_output_tokens;
        }
        mean_ttft /= generations_info.size();
        mean_tpot /= generations_info.size();
        std::cout << "Benchmark duration: " << total_duration.count() << " s" << std::endl;
        std::cout << "Total number of input tokens: " << total_input_len << std::endl;
        std::cout << "Total number of output tokens: " << total_output_len << std::endl;
        std::cout << "Input throughput: " << total_input_len / total_duration.count() << " tokens / s" << std::endl;
        std::cout << "Output throughput: " << total_output_len / total_duration.count() << " tokens / s" << std::endl;
        std::cout << "Mean TTFT: " << mean_ttft.count() << " ms" << std::endl;
        std::cout << "Mean TPOT: " << mean_tpot.count() << " ms" << std::endl; 
    }
};

void trafficSimulator(ov::genai::ContinuousBatchingPipeline* pipe, Dataset* dataset, std::string request_rate, GenerationInfoCollector* generation_info_collector, bool is_speculative_decoding_enabled) {
    double numeric_request_rate;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::exponential_distribution<> distribution;

    if (request_rate == "inf") {
        numeric_request_rate = -1.0;
    } else {
        numeric_request_rate = std::stod(request_rate);
        if (numeric_request_rate < 0)
            throw std::invalid_argument("request_rate cannot be a negative number");

        distribution = std::exponential_distribution<>(numeric_request_rate);
    }

    /*
    std::cout << "Total input tokens: " << dataset->m_total_input_len << std::endl;
    std::cout << "Total output tokens: " << dataset->m_total_output_len << std::endl;
    std::cout << "Average input len: " << dataset->get_average_input_len() << " tokens" << std::endl;
    std::cout << "Average output len: " << dataset->get_average_output_len() << " tokens" << std::endl;
    */

    std::cout << "Launching traffic simulator thread with request_rate: " << request_rate << std::endl;
    generation_info_collector->set_start_time(std::chrono::steady_clock::now());
    for (size_t request_id = 0; request_id < dataset->size(); ++request_id) {
        std::cout << "Traffic thread adding request to the queue..." << std::endl;
        generation_info_collector->add_generation(pipe, dataset, request_id, is_speculative_decoding_enabled);
        if (numeric_request_rate > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(int(distribution(gen) * 1000)));
    }
    std::cout << "All requests sent, traffic simulation finished. Exiting thread." << std::endl;
}

void llmEngineLoop(ov::genai::ContinuousBatchingPipeline* pipe, Dataset* dataset, std::atomic<bool>* finishThread) {
    std::cout << "Launching LLM engine thread" << std::endl;
    size_t num_finished = 0;

    while (!(*finishThread)) {
        while (pipe->has_non_finished_requests()) {
            pipe->step();
        }
    }
    std::cout << "All requests processed, LLM Engine loop escaped. Exiting thread." << std::endl;
}

void statisticsReporter(GenerationInfoCollector* generations_info_collector, int num_prompts) {
    int num_finished = 0;
    while (num_finished < num_prompts) {
        num_finished = generations_info_collector->run();
    }
    std::cout << "Benchmark finished, summarizing statistics..." << std::endl;
    generations_info_collector->print_statistics();

    std::cout << "Exiting statistics reporter thread." << std::endl;
}

bool parse_plugin_config_json(nlohmann::json& node, ov::AnyMap& device_config_map) {
    if (!node.is_object()) {
        std::cout << "Error: nlohmann json object is not an object." << std::endl;
        return false;
    }
    for (auto& element : node.items()) {
        if (element.value().is_string()) {
            device_config_map[std::string(element.key())] = element.value().get<std::string>();
            std::cout << "Setting plugin config: " << element.key() << " : " << element.value().get<std::string>() << std::endl;
        } else if (element.value().is_number_integer()) {
            device_config_map[std::string(element.key())] = element.value().get<std::int64_t>();
            std::cout << "Setting plugin config: " << element.key() << " : " << element.value().get<std::int64_t>() << std::endl;
        } else if (element.value().is_number_float()) {
            device_config_map[std::string(element.key())] = element.value().get<float>();
            std::cout << "Setting plugin config: " << element.key() << " : " << element.value().get<float>() << std::endl;
        } else if (element.value().is_number_unsigned()) {
            device_config_map[std::string(element.key())] = element.value().get<uint64_t>();
            std::cout << "Setting plugin config: " << element.key() << " : " << element.value().get<float>() << std::endl;
        } else if (element.value().is_boolean()) {
            device_config_map[std::string(element.key())] = element.value().get<bool>();
            std::cout << "Setting plugin config: " << element.key() << " : " << element.value().get<bool>() << std::endl;
        } else {
            std::cout << "Error: nlohmann json type not supported for: " << element.key() << std::endl;
            return false;
        }
    }

    return true;
}

bool parse_plugin_config_string(const std::string& config_string, ov::AnyMap& device_config_map) {
    if (config_string.empty()) {
        std::cout << "Empty plugin config string. " << std::endl;
        return true;
    }

    nlohmann::json node;
    try {
        node = nlohmann::json::parse(config_string);
    } catch (const nlohmann::json::parse_error& e) {
        std::cout << "ERROR: Plugin config json parser error - message: " << e.what() << '\n'
                << "exception id: " << e.id << '\n'
                << "byte position of error: " << e.byte << std::endl;
                return false;
    } catch (...) {
        std::cout << "ERROR: Plugin config json parser error - message: " << std::endl;
        return false;
    }

    if (node.is_null()) {
        std::cout << "Error: nlohmann json object is null." << std::endl;
        return false;
    }

    return parse_plugin_config_json(node, device_config_map);
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
    ("dynamic_split_fuse", "Whether to use dynamic split-fuse or vLLM scheduling", cxxopts::value<bool>()->default_value("true"))
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>()->default_value("."))
    ("draft_model", "Path to assistant model directory", cxxopts::value<std::string>()->default_value(""))
    ("dataset", "Path to dataset .json file", cxxopts::value<std::string>()->default_value("./ShareGPT_V3_unfiltered_cleaned_split.json"))
    ("max_input_len", "Max input length take from dataset", cxxopts::value<size_t>()->default_value("1024"))
    ("max_output_len", "Max output length", cxxopts::value<size_t>()->default_value("2048"))
    ("request_rate", "Number of requests per second. If this is inf, then all the requests are sent at time 0. Otherwise, we use Poisson process to synthesize the request arrival times.", cxxopts::value<std::string>()->default_value("inf"))
    ("cache_size", "Size of memory used for KV cache in GB. Default: 16", cxxopts::value<size_t>()->default_value("16"))
    ("device", "Target device to run the model. Default: CPU", cxxopts::value<std::string>()->default_value("CPU"))
    ("device_config", "Plugin configuration JSON. Example: '{\"MODEL_DISTRIBUTION_POLICY\":\"TENSOR_PARALLEL\",\"PERF_COUNT\":true}' Default: {\"PERF_COUNT\":true}", cxxopts::value<std::string>()->default_value("{\"PERF_COUNT\":true}"))
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

    const size_t num_prompts = result["num_prompts"].as<size_t>();
    const size_t max_batch_size = result["max_batch_size"].as<size_t>();
    const bool dynamic_split_fuse = result["dynamic_split_fuse"].as<bool>();
    const std::string models_path = result["model"].as<std::string>();
    const std::string draft_model_path = result["draft_model"].as<std::string>();
    const std::string dataset_path = result["dataset"].as<std::string>();
    const size_t max_input_len = result["max_input_len"].as<size_t>();
    const size_t max_output_len = result["max_output_len"].as<size_t>();
    const std::string request_rate = result["request_rate"].as<std::string>();
    const std::string device = result["device"].as<std::string>();
    const std::string device_config = result["device_config"].as<std::string>();
    const size_t cache_size = result["cache_size"].as<size_t>();
    const bool use_cache_eviction = result["use_cache_eviction"].as<bool>();

    bool is_speculative_decoding_enabled = !draft_model_path.empty();

    // Create requests for generation
    Dataset dataset = filtered_dataset(models_path, dataset_path, num_prompts, max_input_len, max_output_len);

    // Perform the first inference
    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = max_batch_size,
    scheduler_config.cache_size = cache_size,
    scheduler_config.dynamic_split_fuse = dynamic_split_fuse,
    scheduler_config.max_num_seqs = 256; // not used if dynamic_split_fuse=True
    if (use_cache_eviction) {
        scheduler_config.use_cache_eviction = true;
        scheduler_config.cache_eviction_config = ov::genai::CacheEvictionConfig(32, 32, 128, ov::genai::AggregationMode::NORM_SUM, false, 8, ov::genai::KVCrushConfig(0, ov::genai::KVCrushAnchorPointMode::MEAN));
    }

    std::cout << "Benchmarking parameters: " << std::endl;
    std::cout << "\tMax number of batched tokens: " << scheduler_config.max_num_batched_tokens << std::endl;
    std::cout << "\tScheduling type: " << (scheduler_config.dynamic_split_fuse ? "dynamic split-fuse" : "vLLM") << std::endl;
    if (!scheduler_config.dynamic_split_fuse) {
        std::cout << "\tMax number of batched sequences: " << scheduler_config.max_num_seqs << std::endl;
    }
    std::cout << "Dataset parameters: " << std::endl;
    std::cout << "\tNum prompts: " << num_prompts << std::endl;
    std::cout << "\tMax input length: " << max_input_len << std::endl;
    std::cout << "\tMax output length: " << max_output_len << std::endl;
    std::cout << "\tTarget device: " << device << std::endl;
    std::cout << "\tPlugin configuration JSON: " << device_config << std::endl;

    ov::AnyMap device_config_map = {};
    if (is_speculative_decoding_enabled) {
        device_config_map.insert({ ov::genai::draft_model(draft_model_path) });
    }
    if (!parse_plugin_config_string(device_config, device_config_map)) {
        std::cout << "ERROR: Wrong json parameter in device_config." << std::endl;
        return EXIT_FAILURE;
    }
    
    // Benchmarking
    std::cout << "Loading models, creating pipelines, preparing environment..." << std::endl;
    ov::genai::ContinuousBatchingPipeline pipe(models_path, scheduler_config, device, device_config_map);

    std::cout << "Setup finished, launching LLM executor, traffic simulation and statistics reporter threads" << std::endl;

    GenerationInfoCollector generation_info_collector;

    std::atomic<bool> finishGenerationThread{false};
    if (request_rate == "inf") {
        std::thread trafficSimulatorThread(trafficSimulator, &pipe, &dataset, request_rate, &generation_info_collector, is_speculative_decoding_enabled);
        trafficSimulatorThread.join();
    }
    
    std::thread lmmEngineThread(llmEngineLoop, &pipe, &dataset, &finishGenerationThread);
    std::thread statisticsReporterThread(statisticsReporter, &generation_info_collector, num_prompts);
    if (request_rate != "inf") {
        std::thread trafficSimulatorThread(trafficSimulator, &pipe, &dataset, request_rate, &generation_info_collector, is_speculative_decoding_enabled);
        trafficSimulatorThread.join();
    }
    statisticsReporterThread.join();
    finishGenerationThread = true;
    lmmEngineThread.join();

    std::cout << "Benchmark finished" << std::endl;
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
