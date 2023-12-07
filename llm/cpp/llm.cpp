// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <group_beam_searcher.hpp>
#include <openvino/openvino.hpp>
#include <openvino_extensions/strings.hpp>

namespace {
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest&& tokenizer, std::string_view prompt) {
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
}

int main(int argc, char* argv[]) try {
    if (argc != 5) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0]
            + " <openvino_model.xml> <tokenizer.xml> <detokenizer.xml> '<prompt>'");
    }
    ov::Core core;
    core.add_extension(USER_OV_EXTENSIONS_PATH);  // USER_OV_EXTENSIONS_PATH is defined in root CMakeLists.txt
    std::string_view prompt = argv[4];
    auto [input_ids, mask] = tokenize(core.compile_model(argv[2], "CPU").create_infer_request(), prompt);
    ov::InferRequest detokenizer = core.compile_model(argv[3], "CPU").create_infer_request();
    std::shared_ptr<ov::Model> model = core.read_model(argv[1]);
    std::map<size_t, ov::PartialShape> shapes = {
        {0, ov::PartialShape{
            -1, -1
        }},
        {1, ov::PartialShape{
            -1, -1
        }},
        {2, ov::PartialShape{
            -1, -1
        }}
    };
    std::vector<ov::Output<ov::Node>> inputs = model->inputs();
    for (size_t idx = 3; idx < inputs.size(); ++idx) {
        ov::PartialShape shape = inputs.at(idx).get_partial_shape();
        shape[0] = -1;
        shapes.emplace(idx, shape);
    }
    model->reshape(shapes);
    ov::InferRequest ireq = core.compile_model(model, "CPU", ov::cache_dir("llm-cache")).create_infer_request();
    ireq.set_tensor("input_ids", input_ids);
    ireq.set_tensor("attention_mask", mask);
    ov::Tensor position_ids = ireq.get_tensor("position_ids");
    position_ids.set_shape(input_ids.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);
    for (size_t idx = 3; idx < inputs.size(); ++idx) {
        ov::Shape shape = inputs.at(idx).get_partial_shape().get_min_shape();
        shape.at(0) = 1;
        ireq.get_input_tensor(idx).set_shape(shape);
    }
    Parameters parameters;
    const int64_t* prompt_data = input_ids.data<const int64_t>();
    parameters.prompt = std::vector<int64_t>{prompt_data, prompt_data + input_ids.get_size()};
    GroupBeamSearcher group_beam_searcher{parameters};
    for (size_t length_count = 0; length_count < parameters.max_new_tokens; ++length_count) {
        ireq.infer();
        std::vector<TokenToBeam> next_tokens = group_beam_searcher.process(ireq.get_tensor("logits"));
        if (next_tokens.empty()) {
            break;
        }
        size_t batch_size = next_tokens.size();
        ireq.get_tensor("input_ids").set_shape({batch_size, 1});
        ov::Tensor attention_mask = ireq.get_tensor("attention_mask");
        ov::Shape mask_shape = attention_mask.get_shape();
        mask_shape.at(0) = batch_size;
        ++mask_shape.at(1);
        attention_mask.set_shape(mask_shape);
        std::fill_n(attention_mask.data<int64_t>(), shape_size(mask_shape), 1);
        ireq.get_tensor("position_ids").set_shape({batch_size, 1});
        std::fill_n(ireq.get_tensor("position_ids").data<int64_t>(), batch_size, mask_shape.at(1) - 1);
        for (size_t tensor_idx = 3; tensor_idx < inputs.size(); ++tensor_idx) {
            ov::Shape shape = ireq.get_output_tensor(tensor_idx - 2).get_shape();
            shape.at(0) = batch_size;
            ireq.get_input_tensor(tensor_idx).set_shape(shape);
        }
        for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            ireq.get_tensor("input_ids").data<int64_t>()[batch_idx] = next_tokens.at(batch_idx).token_idx;
            for (size_t tensor_idx = 3; tensor_idx < inputs.size(); ++tensor_idx) {
                ov::Tensor present = ireq.get_output_tensor(tensor_idx - 2);
                ov::Shape present_begin = {next_tokens.at(batch_idx).beam_idx, 0, 0, 0};
                ov::Shape present_end = present.get_shape();
                present_end.at(0) = next_tokens.at(batch_idx).beam_idx + 1;
                ov::Tensor past = ireq.get_input_tensor(tensor_idx);
                ov::Shape past_begin = {batch_idx, 0, 0, 0};
                ov::Shape past_end = past.get_shape();
                past_end.at(0) = batch_idx + 1;
                ov::Tensor{present, present_begin, present_end}.copy_to(ov::Tensor{past, past_begin, past_end});
            }
        }
    }
    for (Group& group : group_beam_searcher.groups) {
        if (!group.done) {
            for (Beam& beam : group.ongoing) {
                group.finish(std::move(beam), parameters);
            }
        }
        std::cout << "Group:\n";
        for (const Beam& beam : group.min_heap) {
            std::string detokenized = detokenize(detokenizer, beam.tokens);
            if (detokenized.size() < prompt.size()) {
                throw std::runtime_error("Detokenized sequence became smaller than the prompt which must be included");
            }
            std::string_view generated{detokenized.data() + prompt.size(), detokenized.size() - prompt.size()};
            std::cout << beam.score << ": " << generated << '\n';
        }
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
