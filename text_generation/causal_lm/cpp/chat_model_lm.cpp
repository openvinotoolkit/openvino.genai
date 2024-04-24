// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "group_beam_searcher.hpp"
#include "openvino/openvino.hpp"
#include <iostream>
#include <fstream>

namespace {
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string&& prompt) {
    constexpr size_t BATCH_SIZE = 1;
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt});
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

std::string detokenize(ov::InferRequest& detokenizer, const std::vector<int64_t>& tokens) {
    constexpr size_t BATCH_SIZE = 1;
    ov::Tensor inp = detokenizer.get_input_tensor();
    inp.set_shape({BATCH_SIZE, tokens.size()});
    for (size_t idx = 0; idx < tokens.size(); ++idx) {
        inp.data<int64_t>()[idx] = tokens.at(idx);
    }
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}

std::string generate_chat_prompt_gemma(const std::string& input) {
    std::stringstream result_prompt;
    result_prompt << "<bos><start_of_turn>user\n" << input << "<end_of_turn>\n<start_of_turn>model";
    return result_prompt.str();
}
}  // namespace

int main(int argc, char* argv[]) try {
    if (argc != 2) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> ");
    }
    // Compile models
    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    auto tokenizer_model = core.read_model(std::string{argv[1]} + "/openvino_tokenizer.xml");
    // tokenizer and detokenizer work on CPU only
    ov::InferRequest tokenizer =
        core.compile_model(tokenizer_model, "CPU").create_infer_request();
    ov::InferRequest detokenizer =
        core.compile_model(std::string{argv[1]} + "/openvino_detokenizer.xml", "CPU").create_infer_request();
    // The model can be compiled for GPU as well
    ov::InferRequest lm =
        core.compile_model(std::string{argv[1]} + "/openvino_model.xml", "CPU").create_infer_request();

    // Get the runtime info from the tokenizer model that we read earlier
    auto rt_info = tokenizer_model->get_rt_info(); //Get the runtime info for the model
    int64_t SPECIAL_EOS_TOKEN;

    if (rt_info.count("eos_token_id") > 0) { //check if the runtime information has a valid EOS token ID
        SPECIAL_EOS_TOKEN = rt_info["eos_token_id"].as<int64_t>();

    } else {
        throw std::runtime_error("EOS token ID not found in model's runtime information.");
    }

    int64_t total_positions = 0;
    int32_t global_beam_idx = 0;
    std::string prompt;
    const size_t seq_len_dim_idx = 1;

    std::cout << "Type keyword \"Stop!\" to stop the chat. \n";
    for (;;) {
        std::cout << "User prompt:\n";
        std::getline(std::cin, prompt);
        std::cout << "\n";

        if (!prompt.compare("Stop!"))
            break;

        prompt = generate_chat_prompt_gemma(prompt);

        auto [input_ids, new_attention_mask] = tokenize(tokenizer, std::move(prompt));

        // Initialize inputs
        lm.set_tensor("input_ids", input_ids);

        auto attention_mask = lm.get_tensor("attention_mask");
        // attention to all previous tokens + new tokens
        ov::Shape mask_shape{1, attention_mask.get_shape().at(seq_len_dim_idx) + new_attention_mask.get_shape().at(seq_len_dim_idx)};
        attention_mask.set_shape(mask_shape);
        std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);

        auto position_ids = lm.get_tensor("position_ids");
        position_ids.set_shape(input_ids.get_shape());
        // increment position_ids for every token sent to model
        std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), total_positions);
        total_positions += position_ids.get_size();

        lm.set_tensor("beam_idx", ov::Tensor{ov::element::i32, {1}, &global_beam_idx});

        const int64_t* prompt_data = input_ids.data<const int64_t>();
        Parameters parameters{{{prompt_data, prompt_data + input_ids.get_size()}}, SPECIAL_EOS_TOKEN};
        GroupBeamSearcher group_beam_searcher{parameters};
        std::vector<int64_t> next_tokens;
        std::vector<int32_t> next_beams;
        lm.infer();

        for (size_t length_count = 0; length_count < parameters.max_new_tokens; ++length_count) {
            std::tie(next_tokens, next_beams) = group_beam_searcher.select_next_tokens(lm.get_tensor("logits"));
            if (next_tokens.empty()) {
                break;
            }
            size_t batch_size = next_tokens.size();
            // Set pointers
            lm.set_tensor("input_ids", ov::Tensor{ov::element::i64, {batch_size, 1}, next_tokens.data()});
            lm.set_tensor("beam_idx", ov::Tensor{ov::element::i32, {batch_size}, next_beams.data()});
            // Set auxiliary inputs
            ov::Tensor attention_mask = lm.get_tensor("attention_mask");
            ov::Shape mask_shape{batch_size, attention_mask.get_shape().at(seq_len_dim_idx) + 1};
            attention_mask.set_shape(mask_shape);
            std::fill_n(attention_mask.data<int64_t>(), ov::shape_size(mask_shape), 1);
            lm.get_tensor("position_ids").set_shape({batch_size, 1});
            std::fill_n(lm.get_tensor("position_ids").data<int64_t>(), batch_size, total_positions++);
            lm.infer();
        }

        Beam answer;
        float highest_score = std::numeric_limits<float>().lowest();
        auto all_groups = finalize(std::move(group_beam_searcher));
        for (const std::vector<Beam>& group : all_groups[0]) {
            for (const Beam& beam : group) {
                if (beam.score > highest_score) {
                    highest_score = beam.score;
                    answer = std::move(beam);
                }
            }
        }

        auto answer_str = detokenize(detokenizer, answer.tokens);
        //answer_str = answer_str.substr(0, answer_str.find("<eos>"));
        std::cout << "Answer: " << answer_str << "\n_______\n";
        global_beam_idx = answer.global_beam_idx;

        // Model is stateful which means that context (kv-cache) which belongs to a particular
        // text sequence is accumulated inside the model during the generation loop above.
        // This context should NOT be reset before processing the next text sequence.
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
