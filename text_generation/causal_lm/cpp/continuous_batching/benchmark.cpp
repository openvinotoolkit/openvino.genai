// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <cstdlib>

#include <openvino/openvino.hpp>
#include <nlohmann/json.hpp>
#include <cxxopts.hpp>

#include "llm_engine.hpp"

namespace {

constexpr size_t BATCH_SIZE = 1;

std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string prompt) {
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt});
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

std::string detokenize(ov::InferRequest& detokenizer, std::vector<int64_t> tokens) {
    constexpr size_t BATCH_SIZE = 1;
    detokenizer.set_input_tensor(ov::Tensor{ov::element::i64, {BATCH_SIZE, tokens.size()}, tokens.data()});
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}

std::string detokenize(ov::InferRequest& detokenizer, ov::Tensor tokens) {
    detokenizer.set_input_tensor(tokens);
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}


std::vector<std::pair<ov::Tensor, SamplingParameters>> filtered_dataset(const std::string& models_path, const std::string& dataset_path, const size_t num_prompts, const size_t max_input_len, const size_t max_output_len) {
    std::ifstream json_file(dataset_path.c_str());
    OPENVINO_ASSERT(json_file.is_open(), "Cannot open dataset file");

    nlohmann::json json_dataset = nlohmann::json::parse(json_file);
    std::vector<std::pair<ov::Tensor, SamplingParameters>> sampled_dataset, dataset;
    sampled_dataset.reserve(num_prompts);
    dataset.reserve(json_dataset.size());

    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt

    ov::InferRequest tokenizer = core.compile_model(
        models_path + "/openvino_tokenizer.xml", "CPU").create_infer_request();

    for (auto json_data_iterator = json_dataset.begin(); json_data_iterator != json_dataset.end(); ++json_data_iterator) {
        auto & json_data = *json_data_iterator;

        // Filter out the conversations with less than 2 turns.
        if (json_data["conversations"].size() < 2)
            continue;

        // Only keep the first two turns of each conversation.
        std::string human_question = json_data["conversations"][0]["value"];
        std::string gpt_answer = json_data["conversations"][1]["value"];

        auto [_input_ids_prompt, _attention_mask_prompt] = tokenize(tokenizer, human_question);
        ov::Tensor _input_ids_prompt_clone(_input_ids_prompt.get_element_type(), _input_ids_prompt.get_shape());
        _input_ids_prompt.copy_to(_input_ids_prompt_clone);

        auto [_input_ids_answer, _attention_mask_answer] = tokenize(tokenizer, gpt_answer);
        size_t input_len = _input_ids_prompt_clone.get_size(), output_len = _input_ids_answer.get_size();

        // Prune too short sequences.
        if (input_len < 4 || output_len < 4)
            continue;
        // Prune too long sequences.
        if (input_len > max_input_len || (input_len + output_len) > 2048)
            continue;

        SamplingParameters greedy_search = SamplingParameters::greedy();
        greedy_search.max_new_tokens = std::min(max_output_len, output_len);

        dataset.push_back({ _input_ids_prompt_clone, greedy_search });
    }

    // sample dataset
    size_t total_i = 0, total_o = 0;
    for (size_t i = 0; i < num_prompts; ++i) {
        auto sample = dataset[rand() % dataset.size()];
        total_i += sample.first.get_size();
        total_o += sample.second.max_new_tokens;
        sampled_dataset.push_back(sample);
    }

    std::cout << "Total inputs: " << total_i << std::endl;
    std::cout << "Total outputs: " << total_o << std::endl;

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

    //
    // Compile models
    //

    ov::Core core;
    core.add_extension("libuser_ov_extensions.so");

    // The model can be compiled for GPU as well
    std::shared_ptr<ov::Model> model = core.read_model(models_path + "/vllm_optimum_openvino_model.xml");
    const ov::ParameterVector& parameters = model->get_parameters();
    for (size_t decoder_layer_id = 0; decoder_layer_id < NUM_DECODER_LAYERS; ++decoder_layer_id) {
        parameters[2 + 2 * decoder_layer_id]->set_element_type(kv_cache_precision);
        parameters[2 + 2 * decoder_layer_id + 1]->set_element_type(kv_cache_precision);
    }
    model->validate_nodes_and_infer_types();
    ov::InferRequest request = core.compile_model(model, "CPU", ov::enable_profiling(true)).create_infer_request();

    //
    // Create requests for generation
    //

    std::vector<std::pair<ov::Tensor, SamplingParameters>> dataset = filtered_dataset(models_path, dataset_path, num_prompts, max_input_len, max_output_len);

    //
    // Perform the first inference
    //

    SchedulerConfig scheduler_config {
        .max_num_batched_tokens = max_batch_size,
        .num_kv_blocks = NUM_BLOCKS,
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

    //
    // Benchmarking
    //

    LLMEngine engine(request, scheduler_config);

    for (size_t request_id = 0; request_id < dataset.size(); ++request_id) {
        engine.add_request(request_id, dataset[request_id].first, dataset[request_id].second);
    }

    Timer timer;
    size_t total_input_tokens = 0, total_output_tokens = 0;
    double paged_attention_time_ms = 0.0, matmul_time_ms = 0.0, infer_total_ms = 0.0;

    for (size_t num_finished = 0; engine.has_running_requests(); ) {
        std::vector<GenerationResult> results = engine.step();
        if (!results.empty()) {
            num_finished += results.size();
            for (size_t output_id = 0; output_id < results.size(); ++output_id) {
                size_t output_len = results[output_id].m_generation_ids[0].size();
                size_t input_len = dataset[results[output_id].m_request_id].first.get_size();
                // check correctness of generated length
                OPENVINO_ASSERT(dataset[results[output_id].m_request_id].second.max_new_tokens == output_len);
                // accumulate input tokens
                total_input_tokens += dataset[results[output_id].m_request_id].first.get_size();
                // accumulate output tokens
                total_output_tokens += output_len;
            }
        }

        // collect performance metrics
        std::vector<ov::ProfilingInfo> profiling_info = request.get_profiling_info();
        for (const ov::ProfilingInfo& info : profiling_info) {
            double current_time = info.real_time.count();
            if (info.node_type == "PagedAttentionExtension") {
                paged_attention_time_ms += current_time;
            } else if (info.node_type == "FullyConnected") {
                matmul_time_ms += current_time;
            }
            infer_total_ms += current_time;
        }
    }

    double total_time_in_ms = timer.current_in_milli();
    infer_total_ms /= 1000;
    paged_attention_time_ms /= 1000;
    matmul_time_ms /= 1000;

    std::cout << "Total input tokens: " << total_input_tokens << std::endl;
    std::cout << "Total output tokens: " << total_output_tokens << std::endl;
    std::cout << "Total execution time secs: " << total_time_in_ms / 1000. << " secs" << std::endl;
    std::cout << "Tput: " << (total_input_tokens + total_output_tokens) / (total_time_in_ms / 1000.) << " tokens / sec " << std::endl << std::endl;

    std::cout << "Paged attention % of inference execution: " << (paged_attention_time_ms / infer_total_ms) * 100 << std::endl;
    std::cout << "MatMul % of inference execution: " << (matmul_time_ms / infer_total_ms) * 100 << std::endl;
    std::cout << "Total inference execution ms: " << infer_total_ms << std::endl;
    std::cout << "Inference % of total execution: " << (infer_total_ms / total_time_in_ms) * 100 << std::endl;

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
