// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include <openvino/openvino.hpp>
#include <nlohmann/json.hpp>

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

std::vector<std::pair<ov::Tensor, SamplingParameters>> filtered_dataset(const std::string& dataset_path, const size_t num_prompts) {
    std::ifstream json_file(dataset_path.c_str());
    OPENVINO_ASSERT(json_file.is_open(), "Cannot open dataset file");

    nlohmann::json json_dataset = nlohmann::json::parse(json_file);
    std::vector<std::pair<ov::Tensor, SamplingParameters>> dataset;
    dataset.reserve(num_prompts);

    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt

    ov::InferRequest tokenizer = core.compile_model(
        "/home/sandye51/Documents/Programming/git_repo/vllm/openvino_tokenizer.xml", "CPU").create_infer_request();
    ov::InferRequest detokenizer = core.compile_model(
        "/home/sandye51/Documents/Programming/git_repo/vllm/openvino_detokenizer.xml", "CPU").create_infer_request();

    for (auto json_data_iterator = json_dataset.begin(); dataset.size() < num_prompts && json_data_iterator != json_dataset.end(); ++json_data_iterator) {
        auto & json_data = *json_data_iterator;

        // Filter out the conversations with less than 2 turns.
        if (json_data["conversations"].size() < 2)
            continue;

        // Only keep the first two turns of each conversation.
        std::string human_question = json_data["conversations"][0]["value"];
        std::string gpt_answer = json_data["conversations"][1]["value"];

        auto [_input_ids_prompt, _attention_mask_prompt] = tokenize(tokenizer, human_question);
        auto [_input_ids_answer, _attention_mask_answer] = tokenize(tokenizer, gpt_answer);

        size_t input_len = _input_ids_prompt.get_size(), output_len = _input_ids_answer.get_size();

        // Prune too short sequences.
        if (input_len < 4 || input_len < 4)
            continue;
        // Prune too long sequences.
        if (input_len > 1024 || (input_len + output_len) > 2048)
            continue;

        SamplingParameters greedy_search = SamplingParameters::greedy();
        greedy_search.max_new_tokens = output_len;

        dataset.push_back({ _input_ids_prompt, greedy_search });
    }

    return dataset;
}

class Timer {
    const decltype(std::chrono::steady_clock::now()) m_start;
public:
    Timer() :
        m_start(std::chrono::steady_clock::now()) {
    }

    double current_milli() const {
        auto m_end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(m_end - m_start).count();
    }
};


}  // namespace

int main(int argc, char* argv[]) try {
    //
    // Compile models
    //

    ov::Core core;
    core.add_extension("libuser_ov_extensions.so");

    // The model can be compiled for GPU as well
    std::shared_ptr<ov::Model> model = core.read_model("/home/sandye51/Documents/Programming/git_repo/vllm/vllm_optimum_openvino_model.xml");
    const ov::ParameterVector& parameters = model->get_parameters();
    for (size_t decoder_layer_id = 0; decoder_layer_id < NUM_DECODER_LAYERS; ++decoder_layer_id) {
        parameters[2 + 2 * decoder_layer_id]->set_element_type(kv_cache_precision);
        parameters[2 + 2 * decoder_layer_id + 1]->set_element_type(kv_cache_precision);
    }
    model->validate_nodes_and_infer_types();
    ov::InferRequest request = core.compile_model(model, "CPU").create_infer_request();

    //
    // Create requests for generation
    //

    const size_t dataset_size = 300;
    std::string dataset_path = "/home/sandye51/Documents/Programming/git_repo/openvino.genai/text_generation/causal_lm/cpp/build/ShareGPT_V3_unfiltered_cleaned_split.json";

    std::vector<std::pair<ov::Tensor, SamplingParameters>> dataset = filtered_dataset(dataset_path, dataset_size);

    //
    // Perform the first inference
    //

    SchedulerConfig scheduler_config {
        .max_num_batched_tokens = 1024,
        .num_kv_blocks = NUM_BLOCKS,
        .dynamic_split_fuse = false,
        .max_num_seqs = 256, // not used if dynamic_split_fuse=True
        .max_paddings = 256, // not used if dynamic_split_fuse=True
    };

    LLMEngine engine(request, scheduler_config);

    for (size_t request_id = 0; request_id < dataset.size(); ++request_id) {
        engine.add_request(request_id, dataset[request_id].first, dataset[request_id].second);
    }

    Timer timer;
    size_t total_input_tokens = 0, total_output_tokens = 0;

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

            std::cout << "Num finished " << num_finished << std::endl;
        }
    }

    double total_time_in_secs = timer.current_milli() / 1000.;
    std::cout << "Total input tokens: " << total_input_tokens << std::endl;
    std::cout << "Total output tokens: " << total_output_tokens << std::endl;
    std::cout << "Total execution time: " << total_time_in_secs << " secs" << std::endl;
    std::cout << "Tput: " << (total_input_tokens + total_output_tokens) / total_time_in_secs << " tokens / sec " << std::endl;

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
