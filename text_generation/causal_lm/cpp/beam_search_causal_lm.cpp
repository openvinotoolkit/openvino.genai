// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <group_beam_searcher.hpp>
#include <openvino/openvino.hpp>

namespace {

enum SPECIAL_TOKEN { PAD_TOKEN = 2 };

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

std::pair<ov::Tensor, ov::Tensor> pad_left(ov::Tensor&& input_ids, ov::Tensor&& attention_mask) {
    const size_t batch_size = input_ids.get_shape().at(0);
    const size_t sequence_length = input_ids.get_shape().at(1);
    int64_t* inputs_data = input_ids.data<int64_t>();
    int64_t* attention_mask_data = attention_mask.data<int64_t>();

    for (size_t batch = 0; batch < batch_size; batch++) {
        const size_t batch_offset = batch * sequence_length;

        // last token in the sequence is not a PAD_TOKEN, skipping
        if (inputs_data[batch_offset + sequence_length - 1] != SPECIAL_TOKEN::PAD_TOKEN) {
            continue;
        }

        size_t pad_tokens_number = 0;
        for (int i = sequence_length - 1; i >= 0; i--) {
            const size_t token_offset = batch_offset + i;

            if (inputs_data[token_offset] == SPECIAL_TOKEN::PAD_TOKEN) {
                continue;
            }

            if (pad_tokens_number == 0) {
                pad_tokens_number = sequence_length - i - 1;
            }

            std::swap(inputs_data[token_offset], inputs_data[token_offset + pad_tokens_number]);
            std::swap(attention_mask_data[token_offset], attention_mask_data[token_offset + pad_tokens_number]);
        }
    }

    return {input_ids, attention_mask};
}

std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::vector<std::string> prompts) {
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {prompts.size()}, prompts.data()});

    tokenizer.infer();

    pad_left(tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask"));

    // fix mask filled with '2' instead of '0'
    ov::Tensor attention_mask = tokenizer.get_tensor("attention_mask");
    int64_t* attention_mask_data = attention_mask.data<int64_t>();
    std::replace(attention_mask_data, attention_mask_data + attention_mask.get_size(), 2, 0);

    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

void initialize_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask) {
    const size_t batch_size = attention_mask.get_shape().at(0);
    const size_t sequence_length = attention_mask.get_shape().at(1);

    const int64_t* attention_mask_data = attention_mask.data<int64_t>();
    int64_t* position_ids_data = position_ids.data<int64_t>();

    for (size_t batch = 0; batch < batch_size; batch++) {
        const size_t batch_offset = batch * sequence_length;
        size_t sum = 0;

        for (size_t i = 0; i < sequence_length; i++) {
            const size_t element_offset = batch_offset + i;
            position_ids_data[element_offset] = sum;
            if (attention_mask_data[element_offset] == 1) {
                sum += 1;
            }
        }
    }
}

void initialize_inputs(const ov::Tensor& input_ids, const ov::Tensor& attention_mask, ov::InferRequest& request) {
    request.set_tensor("input_ids", input_ids);
    request.set_tensor("attention_mask", attention_mask);

    ov::Shape input_shape = input_ids.get_shape();

    ov::Tensor position_ids = request.get_tensor("position_ids");
    position_ids.set_shape(input_shape);
    initialize_position_ids(position_ids, attention_mask);

    ov::Tensor beam_idx = request.get_tensor("beam_idx");
    beam_idx.set_shape({input_shape.at(0)});
    std::fill_n(beam_idx.data<int32_t>(), input_shape.at(0), 0);
}

void set_attention_mask(ov::Tensor&& attention_mask, std::vector<int32_t> next_beams) {
    ov::Tensor original_mask{ov::element::i64, attention_mask.get_shape()};
    ov::Shape original_shape = original_mask.get_shape();
    attention_mask.copy_to(original_mask);

    ov::Shape new_shape{next_beams.size(), original_mask.get_shape().at(1) + 1};
    attention_mask.set_shape(new_shape);

    for (size_t beam_id = 0; beam_id < next_beams.size(); beam_id++) {
        const size_t original_prompt_offset = next_beams.at(beam_id) * original_shape.at(1);
        const size_t result_prompt_offset = beam_id * new_shape.at(1);

        int64_t* dest = attention_mask.data<int64_t>() + result_prompt_offset;
        const int64_t* src = original_mask.data<int64_t>() + original_prompt_offset;

        std::memcpy(dest, src, original_shape.at(1) * sizeof(int64_t));
        attention_mask.data<int64_t>()[result_prompt_offset + new_shape.at(1) - 1] = 1;
    }
}

void set_position_ids(ov::Tensor&& position_ids, const ov::Tensor&& attention_mask) {
    const size_t batch_size = attention_mask.get_shape().at(0);
    const size_t sequence_length = attention_mask.get_shape().at(1);
    position_ids.set_shape({batch_size, 1});

    for (size_t batch = 0; batch < batch_size; batch++) {
        int64_t* mask_start = attention_mask.data<int64_t>() + batch * sequence_length;
        position_ids.data<int64_t>()[batch] = std::accumulate(mask_start, mask_start + sequence_length - 1, 0);
    }
}

std::vector<std::string> prompts_arguments_to_vector(int argc, char* argv[]) {
    std::vector<std::string> prompts;
    prompts.reserve(argc - 2);
    for (size_t i = 2; i < argc; i++) {
        prompts.push_back(std::string{argv[i]});
    }
    return prompts;
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<PROMPT 1>' ['<PROMPT 2>' ...]");
    }

    // Compile models
    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    // Read the tokenizer model information from the file to later get the runtime information
    auto tokenizer_model = core.read_model(std::string{argv[1]} + "/openvino_tokenizer.xml");
    // tokenizer and detokenizer work on CPU only
    ov::InferRequest tokenizer = core.compile_model(tokenizer_model, "CPU").create_infer_request();
    ov::InferRequest detokenizer =
        core.compile_model(std::string{argv[1]} + "/openvino_detokenizer.xml", "CPU").create_infer_request();
    // The model can be compiled for GPU as well
    ov::InferRequest lm =
        core.compile_model(std::string{argv[1]} + "/openvino_model.xml", "CPU").create_infer_request();

    auto [input_ids, attention_mask] = tokenize(tokenizer, prompts_arguments_to_vector(argc, argv));

    // Initialize beam search
    const int64_t* prompt_data = input_ids.data<const int64_t>();
    std::vector<std::vector<int64_t>> prompts;
    prompts.reserve(input_ids.get_shape().at(0));
    for (size_t batch = 0; batch < input_ids.get_shape().at(0); batch++) {
        size_t sequence_length = input_ids.get_shape().at(1);
        size_t batch_offset = batch * sequence_length;
        const int64_t* prompt_start = prompt_data + batch_offset;
        prompts.push_back(std::vector<int64_t>{prompt_start, prompt_start + sequence_length});
    }

    // Get the runtime info from the tokenizer model that we read earlier
    auto rt_info = tokenizer_model->get_rt_info();  // Get the runtime info for the model
    int64_t SPECIAL_EOS_TOKEN;

    if (rt_info.count("eos_token_id") > 0) {  // check if the runtime information has a valid EOS token ID
        SPECIAL_EOS_TOKEN = rt_info["eos_token_id"].as<int64_t>();

    } else {
        throw std::runtime_error("EOS token ID not found in model's runtime information.");
    }

    Parameters parameters{std::move(prompts), SPECIAL_EOS_TOKEN};
    GroupBeamSearcher group_beam_searcher{parameters};

    initialize_inputs(input_ids, attention_mask, lm);

    std::vector<int64_t> next_tokens;
    std::vector<int32_t> next_beams;

    for (size_t length_count = 0; length_count < parameters.max_new_tokens; ++length_count) {
        lm.infer();

        std::tie(next_tokens, next_beams) = group_beam_searcher.select_next_tokens(lm.get_tensor("logits"));
        if (next_tokens.empty()) {
            break;
        }
        size_t batch_size = next_tokens.size();
        // Set pointers
        lm.set_tensor("input_ids", ov::Tensor{ov::element::i64, {batch_size, 1}, next_tokens.data()});
        lm.set_tensor("beam_idx", ov::Tensor{ov::element::i32, {batch_size}, next_beams.data()});
        // Set auxiliary inputs
        set_attention_mask(lm.get_tensor("attention_mask"), next_beams);
        set_position_ids(lm.get_tensor("position_ids"), lm.get_tensor("attention_mask"));
    }

    for (const std::vector<std::vector<Beam>>& prompt_group : finalize(std::move(group_beam_searcher))) {
        std::cout << "Prompt:\n";
        for (const std::vector<Beam> group : prompt_group) {
            std::cout << "Group:\n";
            for (const Beam& beam : group) {
                std::cout << beam.score << ": " << detokenize(detokenizer, beam.tokens) << '\n';
            }
        }
    }
    // Model is stateful which means that context (kv-cache) which belongs to a particular
    // text sequence is accumulated inside the model during the generation loop above.
    // This context should be reset before processing the next text sequence.
    // While it is not required to reset context in this sample as only one batch of sequences is processed,
    // it is called for education purposes:
    lm.reset_state();
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
