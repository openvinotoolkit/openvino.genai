// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

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

// The following reasons require TextStreamer to keep a cache of previous tokens:
// detokenizer removes starting ' '. For example detokenize(tokenize(" a")) == "a",
// but detokenize(tokenize("prefix a")) == "prefix a"
// 1 printable token may consist of 2 token ids: detokenize(incomplete_token_idx) == "�"
struct TextStreamer {
    ov::InferRequest detokenizer;
    std::vector<int64_t> token_cache;
    size_t print_len = 0;

    void put(int64_t token) {
        token_cache.push_back(token);
        std::string text = detokenize(detokenizer, token_cache);
        if (!text.empty() && '\n' == text.back()) {
            // Flush the cache after the new line symbol
            std::cout << std::string_view{text.data() + print_len, text.size() - print_len};
            token_cache.clear();
            print_len = 0;
            return;
        }
        if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
            // Don't print incomplete text
            return;
        }
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
        print_len = text.size();
    }

    void end() {
        std::string text = detokenize(detokenizer, token_cache);
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << '\n';
        token_cache.clear();
        print_len = 0;
    }
};

}  // namespace

int main(int argc, char* argv[]) try {
    //
    // Compile models
    //

    ov::Core core;
    core.add_extension("libuser_ov_extensions.so");
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    // tokenizer and detokenizer work on CPU only
    ov::InferRequest tokenizer = core.compile_model(
        "/home/sandye51/Documents/Programming/git_repo/vllm/openvino_tokenizer.xml", "CPU").create_infer_request();
    ov::InferRequest detokenizer = core.compile_model(
        "/home/sandye51/Documents/Programming/git_repo/vllm/openvino_detokenizer.xml", "CPU").create_infer_request();
    // The model can be compiled for GPU as well
    std::shared_ptr<ov::Model> model = core.read_model("/home/sandye51/Documents/Programming/git_repo/vllm/vllm_optimum_openvino_model.xml");
    ov::InferRequest request = core.compile_model(model, "CPU").create_infer_request();

    //
    // Create requests for generation
    //

    const size_t dataset_size = 1;

    std::vector<std::string> prompt_examples = {
        "What is OpenVINO?",
        "How are you?",
        "What is the current time",
        "What is OpenVINO?",
    };

    std::vector<SamplingParameters> sampling_params_examples {
        // SamplingParameters::greedy(),
        // SamplingParameters::multimomial(),
        SamplingParameters::beam_search()
    };

    std::vector<ov::Tensor> input_ids;
    std::vector<SamplingParameters> sampling_params;

    input_ids.reserve(dataset_size);
    sampling_params.reserve(dataset_size);

    for (size_t request_id = 0; request_id < dataset_size; ++request_id) {
        auto [_input_ids, _attention_mask] = tokenize(tokenizer, prompt_examples[request_id % prompt_examples.size()]);
        ov::Tensor input_id(_input_ids.get_element_type(), _input_ids.get_shape());
        _input_ids.copy_to(input_id);
        input_ids.push_back(input_id);
        sampling_params.push_back(sampling_params_examples[request_id % sampling_params_examples.size()]);
    }

    //
    // Perform the first inference
    //

    SchedulerConfig scheduler_config {
        .max_tokens_to_batch = 16,
        .num_kv_blocks = NUM_BLOCKS
    };

    LLMEngine engine(request, scheduler_config);
    std::vector<GenerationResult> generation_results = engine.generate(input_ids, sampling_params);

    for (size_t request_id = 0; request_id < generation_results.size(); ++request_id) {
        const GenerationResult & generation_result = generation_results[request_id];

        std::cout << "Question: " << detokenize(detokenizer, input_ids[request_id]) << std::endl;
        for (size_t output_id = 0; output_id < generation_result.m_generation_ids.size(); ++output_id) {
            std::cout << "Answer " << output_id << ": " << detokenize(detokenizer, generation_result.m_generation_ids[output_id]) << std::endl;
        }
        std::cout << std::endl;
    }

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
