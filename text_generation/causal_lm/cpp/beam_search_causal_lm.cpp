// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <group_beam_searcher.hpp>
#include <openvino/openvino.hpp>

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
}

int main(int argc, char* argv[]) try {
    if (argc != 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<PROMPT>'");
    }
    // Compile models
    ov::Core core;
    core.add_extension(USER_OV_EXTENSIONS_PATH);  // USER_OV_EXTENSIONS_PATH is defined in CMakeLists.txt
    ov::InferRequest tokenizer = core.compile_model(
        std::string{argv[1]} + "/openvino_tokenizer.xml", "CPU").create_infer_request();
    auto [input_ids, attention_mask] = tokenize(tokenizer, argv[2]);
    ov::InferRequest detokenizer = core.compile_model(
        std::string{argv[1]} + "/openvino_detokenizer.xml", "CPU").create_infer_request();
    ov::InferRequest lm = core.compile_model(
        std::string{argv[1]} + "/openvino_model.xml", "CPU").create_infer_request();

    const int64_t* prompt_data = input_ids.data<const int64_t>();
    Parameters parameters{std::vector<int64_t>{prompt_data, prompt_data + input_ids.get_size()}};
    parameters.max_new_tokens = 2;
    for (size_t group_size : {3, 5, 4}) {
        std::cout << "Group size: " << group_size << '\n';
        parameters.group_size = group_size;
        GroupBeamSearcher group_beam_searcher{parameters};

        // Initialize inputs
        size_t FULL_BATCH_SIZE = parameters.n_groups * parameters.group_size;
        ov::Shape input_ids_shape = input_ids.get_shape();
        input_ids_shape.front() = FULL_BATCH_SIZE;
        for (const char* name : {"input_ids", "attention_mask", "position_ids"}) {
            ov::Tensor input;
            if (true) {
                input = ov::Tensor{ov::element::i64, input_ids_shape};
                lm.set_tensor(name, input);
            } else {
                input = lm.get_tensor(name);
                // set_shape() fails for second group_size value of global loop
                // Exception from src/core/src/runtime/ov_tensor.cpp:71:
                // Check 'shape_size(new_shape) <= ov::shape_size(m_capacity)' failed at src/inference/src/dev/make_tensor.cpp:60:
                // Could set new shape: [6,4]
            }
            input.set_shape(input_ids_shape);
            std::fill_n(input.data<int64_t>(), input.get_size(), 0);
        }
        ov::Tensor beam_idx{ov::element::i32, {FULL_BATCH_SIZE}};
        std::fill_n(beam_idx.data<int32_t>(), beam_idx.get_size(), 0);
        lm.set_tensor("beam_idx", beam_idx);
        // Set inputs
        input_ids.copy_to(ov::Tensor{lm.get_tensor("input_ids"), {0, 0}, {1, input_ids_shape.at(1)}});
        attention_mask.copy_to(ov::Tensor{lm.get_tensor("attention_mask"), {0, 0}, {1, input_ids_shape.at(1)}});
        ov::Tensor position_ids = lm.get_tensor("position_ids");
        std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + input_ids_shape.at(1), 0);
        lm.get_tensor("beam_idx").data<int32_t>()[0] = 0;

        std::vector<int64_t> next_tokens;
        std::vector<int32_t> next_beams;
        for (size_t length_count = 0; length_count < parameters.max_new_tokens; ++length_count) {
            lm.infer();
            std::tie(next_tokens, next_beams) = group_beam_searcher.process(lm.get_tensor("logits"));
            if (next_tokens.empty()) {
                break;
            }
            size_t batch_size = next_tokens.size();
            // Set pointers
            lm.set_tensor("input_ids", ov::Tensor{ov::element::i64, {batch_size, 1}, next_tokens.data()});
            lm.set_tensor("beam_idx", ov::Tensor{ov::element::i32, {batch_size}, next_beams.data()});
            // Set auxiliary inputs
            ov::Tensor attention_mask = lm.get_tensor("attention_mask");
            ov::Shape mask_shape{batch_size, attention_mask.get_shape().at(1) + 1};
            attention_mask.set_shape(mask_shape);
            std::fill_n(attention_mask.data<int64_t>(), ov::shape_size(mask_shape), 1);
            lm.get_tensor("position_ids").set_shape({batch_size, 1});
            std::fill_n(lm.get_tensor("position_ids").data<int64_t>(), batch_size, mask_shape.at(1) - 1);
        }
        for (const std::vector<Beam>& group : finalize(std::move(group_beam_searcher))) {
            std::cout << "Group:\n";
            for (const Beam& beam : group) {
                std::cout << beam.score << ": " << detokenize(detokenizer, beam.tokens) << '\n';
            }
        }
        lm.reset_state();
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
