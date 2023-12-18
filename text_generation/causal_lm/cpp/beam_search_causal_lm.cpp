// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <group_beam_searcher.hpp>
#include <openvino/openvino.hpp>
#include <openvino_extensions/strings.hpp>

namespace {
std::tuple<ov::InferRequest, ov::InferRequest, ov::InferRequest> compile_models(const std::string model_dir) {
    ov::Core core;
    core.add_extension(USER_OV_EXTENSIONS_PATH);  // USER_OV_EXTENSIONS_PATH is defined in CMakeLists.txt
    return {
        core.compile_model(model_dir + "/openvino_model.xml", "CPU").create_infer_request(),
        core.compile_model(model_dir + "/openvino_tokenizer.xml", "CPU").create_infer_request(),
        core.compile_model(model_dir + "/openvino_detokenizer.xml", "CPU").create_infer_request()
    };
}

std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string_view prompt) {
    ov::Tensor destination = tokenizer.get_input_tensor();
    openvino_extensions::pack_strings(std::array{prompt}, destination);
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
    return openvino_extensions::unpack_strings(detokenizer.get_output_tensor()).front();
}

void initialize_inputs(ov::InferRequest& lm, const ov::Tensor& input_ids, const ov::Tensor& attention_mask) {
    lm.set_tensor("input_ids", input_ids);
    lm.set_tensor("attention_mask", attention_mask);
    ov::Tensor position_ids = lm.get_tensor("position_ids");
    position_ids.set_shape(input_ids.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);
    lm.get_tensor("beam_idx").set_shape({1});
    lm.get_tensor("beam_idx").data<int32_t>()[0] = 0;
}

void set_pointers(
        ov::InferRequest& lm, std::vector<int64_t>& next_tokens, std::vector<int32_t>& next_beams) {
    size_t batch_size = next_tokens.size();
    lm.set_tensor("input_ids", ov::Tensor{ov::element::i64, {batch_size, 1}, next_tokens.data()});
    lm.set_tensor("beam_idx", ov::Tensor{ov::element::i32, {batch_size}, next_beams.data()});
}

void set_auxiliary_inputs(ov::InferRequest& lm) {
    size_t batch_size = lm.get_tensor("input_ids").get_shape().front();
    ov::Tensor attention_mask = lm.get_tensor("attention_mask");
    ov::Shape mask_shape{batch_size, attention_mask.get_shape().at(1) + 1};
    attention_mask.set_shape(mask_shape);
    std::fill_n(attention_mask.data<int64_t>(), ov::shape_size(mask_shape), 1);
    lm.get_tensor("position_ids").set_shape({batch_size, 1});
    std::fill_n(lm.get_tensor("position_ids").data<int64_t>(), batch_size, mask_shape.at(1) - 1);
}
}

int main(int argc, char* argv[]) try {
    if (argc != 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<PROMPT>'");
    }
    auto [lm, tokenizer, detokenizer] = compile_models(argv[1]);
    auto [input_ids, attention_mask] = tokenize(tokenizer, argv[2]);
    initialize_inputs(lm, input_ids, attention_mask);
    const int64_t* prompt_data = input_ids.data<const int64_t>();
    Parameters parameters{std::vector<int64_t>{prompt_data, prompt_data + input_ids.get_size()}};
    GroupBeamSearcher group_beam_searcher{parameters};
    std::vector<int64_t> next_tokens;
    std::vector<int32_t> next_beams;
    for (size_t length_count = 0; length_count < parameters.max_new_tokens; ++length_count) {
        lm.infer();
        std::tie(next_tokens, next_beams) = group_beam_searcher.process(lm.get_tensor("logits"));
        if (next_tokens.empty()) {
            break;
        }
        set_pointers(lm, next_tokens, next_beams);
        set_auxiliary_inputs(lm);
    }
    for (const std::vector<Beam>& group : finalize(std::move(group_beam_searcher))) {
        std::cout << "Group:\n";
        for (const Beam& beam : group) {
            std::cout << beam.score << ": " << detokenize(detokenizer, beam.tokens) << '\n';
        }
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
