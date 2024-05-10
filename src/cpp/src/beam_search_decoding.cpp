// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "generation_config_helper.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "group_beam_searcher.hpp"

namespace ov {

EncodedResults beam_search(ov::InferRequest& model_runner, ov::Tensor prompts, ov::Tensor attentin_mask, GenerationConfig config) {
    GenerationConfigHelper config_helper = config;

    ov::Shape prompts_shape = prompts.get_shape();
    size_t batch_size = prompts_shape[0];
    // todo: implement for batch > 1
    OPENVINO_ASSERT(batch_size == 1);

    // initialize inputs
    auto attention_mask = ov::Tensor{ov::element::i64, prompts.get_shape()};
    std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);
    auto position_ids = ov::Tensor{ov::element::i64, prompts.get_shape()};
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);
    auto prompt_len = prompts.get_shape()[1];

    model_runner.set_tensor("input_ids", prompts);
    model_runner.set_tensor("attention_mask", attention_mask);
    model_runner.set_tensor("position_ids", position_ids);

    // set beam_idx for stateful model: no beam search is used and BATCH_SIZE = 1
    model_runner.get_tensor("beam_idx").set_shape({batch_size});
    model_runner.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    const int64_t* prompt_data = prompts.data<const int64_t>();
    
    // todo: remove this duplication and use the same SamplingParameters for both greedy and beam
    Parameters parameters{{std::vector<int64_t>{prompt_data, prompt_data + prompts.get_size()}}};
    parameters.n_groups = config.num_beam_groups;
    parameters.diversity_penalty = config.diversity_penalty;
    parameters.group_size = config.num_beams / config.num_beam_groups;
    OPENVINO_ASSERT(config.num_beams % config.num_beam_groups == 0, "number of beams should be divisible by number of groups");

    
    GroupBeamSearcher group_beam_searcher{parameters};
    std::vector<int64_t> next_tokens;
    std::vector<int32_t> next_beams;
    for (size_t length_count = 0; length_count < config_helper.get_max_new_tokens(prompt_len); ++length_count) {
        model_runner.infer();
        std::tie(next_tokens, next_beams) = group_beam_searcher.select_next_tokens(model_runner.get_tensor("logits"));
        if (next_tokens.empty()) {
            break;
        }
        size_t batch_size = next_tokens.size();
        // Set pointers
        model_runner.set_tensor("input_ids", ov::Tensor{ov::element::i64, {batch_size, 1}, next_tokens.data()});
        model_runner.set_tensor("beam_idx", ov::Tensor{ov::element::i32, {batch_size}, next_beams.data()});
        // Set auxiliary inputs
        ov::Tensor attention_mask = model_runner.get_tensor("attention_mask");
        ov::Shape mask_shape{batch_size, attention_mask.get_shape()[1] + 1};
        attention_mask.set_shape(mask_shape);
        std::fill_n(attention_mask.data<int64_t>(), ov::shape_size(mask_shape), 1);

        model_runner.get_tensor("position_ids").set_shape({batch_size, 1});
        std::fill_n(model_runner.get_tensor("position_ids").data<int64_t>(), batch_size, mask_shape[1] - 1);
        
        // todo: pass streamer here
        // m_streamer.put(token_iter_results[0]);

    }

    std::vector<Beam> beams;
    for (const std::vector<std::vector<Beam>>& prompt_group : finalize(std::move(group_beam_searcher))) {
        for (const std::vector<Beam> group : prompt_group) {
            for (const Beam& beam : group) {
                beams.emplace_back(beam);
            }
        }
    }

    auto compare_scores = [](Beam left, Beam right) { return (left.score > right.score); };
    std::sort(beams.begin(), beams.end(), compare_scores);
    
    ov::EncodedResults results;
    for (auto beam = beams.begin(); beam != beams.begin() + config.num_return_sequences; ++beam) {
        // todo: convert to string 
        results.scores.emplace_back(beam->score);
        results.tokens.emplace_back(beam->tokens);
    }
    return results;
}

} // namespace ov