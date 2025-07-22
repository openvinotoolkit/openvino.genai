// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer/add_second_input_pass.hpp"
#include "tokenizer/tokenizers_path.hpp"
#include "gguf_utils/gguf_tokenizer.hpp"
#include <gtest/gtest.h>
#include <openvino/pass/visualize_tree.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/select.hpp>
#include <openvino/op/equal.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/multiply.hpp>
#include <filesystem>
#include <memory>


using namespace ov::genai;
using namespace ov;
using namespace ov::op;


ScopedVar env_manager(tokenizers_relative_to_genai());
std::filesystem::path ov_tokenizer_filesystem_path;

#ifdef _WIN32
        const wchar_t* ov_tokenizer_path_w = _wgetenv(ScopedVar::ENVIRONMENT_VARIABLE_NAME_W);
#else
        const char* ov_tokenizer_path = getenv(ScopedVar::ENVIRONMENT_VARIABLE_NAME);
#endif

auto shared_object_ov_tokenizers = load_shared_object(std::filesystem::path(ov_tokenizer_path));

FactoryCreateType create_func = reinterpret_cast<FactoryCreateType>(get_symbol(shared_object_ov_tokenizers, "create_tokenizer_node"));


TEST(AddSecondInputTest, add_second_input_test_1) {
    std::shared_ptr<Model> model;
    auto parameter_1 = std::make_shared<v0::Parameter>(element::string, Shape{2});
    OutputVector outputs = create_func("StringTensorUnpack", {parameter_1}, {});
    
    // Prepare all necessary BPETokenizer inputs according to evaluate() logic

    auto ragged_begins = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{0});
    auto ragged_ends = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{3});
    
    auto vocab_begins = std::make_shared<v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{0, 1});
    auto vocab_ends = std::make_shared<v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{1, 2});
    auto vocab_chars = std::make_shared<v0::Constant>(element::u8, Shape{2}, std::vector<uint8_t>{'a', 'b'});
    
    auto merges_begins = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{0});
    auto merges_ends = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{3});
    auto merges_chars = std::make_shared<v0::Constant>(element::u8, Shape{3}, std::vector<uint8_t>{'a', 'b', 'c'});

    // Compose the input vector for BPETokenizer (11 inputs for minimal case)
    std::vector<ov::Output<ov::Node>> bpe_inputs = {
        ragged_begins, ragged_ends, outputs[0], outputs[1], outputs[2],
        vocab_begins, vocab_ends, vocab_chars,
        merges_begins, merges_ends, merges_chars
    };

    auto BPETokenizer = create_func("BPETokenizer", bpe_inputs, {});

    auto max_length = std::make_shared<v0::Constant>(element::i32, Shape{}, std::vector<int32_t>({10}));
    // Create trunc_side constant: "right" as u8 chars
    auto trunc_side = std::make_shared<v0::Constant>(
        element::u8, Shape{5}, std::vector<uint8_t>{'r', 'i', 'g', 'h', 't'}
    );
    auto truncate = create_func("Truncate", {BPETokenizer[0], BPETokenizer[1], BPETokenizer[2], max_length, trunc_side}, {});

    int32_t eos_token_id = 42;
    auto eos_begins = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{0});
    auto eos_ends = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{1});
    auto eos_token = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{eos_token_id});

    // IDs tensor (required as the last input for CombineSegment)
    auto ids = std::make_shared<v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{0, 1});

    // Prepare CombineSegment inputs: [begins1, ends1, data1, begins2, ends2, data2, ..., ids]
    ov::OutputVector combine_inputs = {
        // Main BPETokenizer outputs which went through Truncate
        truncate[0], truncate[1], truncate[2],
        // EOS token tensors
        eos_begins, eos_ends, eos_token,
        // IDs tensor
        ids
    };

    auto CombineSegments = create_func("CombineSegments", {combine_inputs}, {});
    model = std::make_shared<ov::Model>(OutputVector{CombineSegments}, ParameterVector{parameter_1});
    
    // Add a valid post_processor to model's rt_info
    static const std::string PROCESSED_POST_PROCESSOR_NAME = "processed_post_processor_template";
    // Signature for single input [first_seq, eos]
    std::vector<int> input_signature = {-1, eos_token_id};
    // Signature for pair [first_seq, eos_seq, second_seq, eos]
    std::vector<int> pair_signature = {-1, eos_token_id, -1, eos_token_id};
    
    nlohmann::json post_processor = {
        {"single", {{"ids", input_signature}}},
        {"pair", {{"ids", pair_signature}, {"type_ids", std::vector<int>{0, 0, 1, 1}}}}
    };

    model->get_rt_info()[PROCESSED_POST_PROCESSOR_NAME] = post_processor.dump();

    std::ostringstream pass_errors;
    
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::VisualizeTree>("graph_before.svg");
    manager.register_pass<ov::genai::AddSecondInputPass>(shared_object_ov_tokenizers, pass_errors);
    manager.register_pass<ov::pass::VisualizeTree>("graph_after.svg");
    manager.run_passes(model);

    // Check that resulting graph contains required fields.
    ASSERT_EQ(model->get_parameters().size(), 2);
    // Find combine segment op and assert that it has 2 extra inputs
    auto results = model->get_results();
    // ASSERT_EQ(results.size(), 1);
    auto combine_segments_node = results[0]->get_input_node_shared_ptr(0);
    ASSERT_NE(combine_segments_node, nullptr);
    ASSERT_EQ(combine_segments_node->get_input_size(), combine_inputs.size() + 6); // 6 extra inputs for the second input and eos token
}
