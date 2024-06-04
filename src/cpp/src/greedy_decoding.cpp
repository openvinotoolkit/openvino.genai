// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

EncodedResults greedy_decoding(
    ov::InferRequest& m_model_runner, 
    ov::Tensor input_ids, 
    ov::Tensor attention_mask, 
    const ov::genai::GenerationConfig generation_config, 
    const std::shared_ptr<StreamerBase> streamer, 
    const bool is_chat_conversation,
    const bool is_cache_empty
) {
    ov::Shape prompts_shape = input_ids.get_shape();
    const size_t batch_size = prompts_shape[0];
    size_t running_batch_size = batch_size;
    size_t prompt_len = prompts_shape[1];
       
    auto num_inputs = m_model_runner.get_compiled_model().inputs().size();
    bool position_ids_available = num_inputs == 4;
    ov::Tensor position_ids;

    EncodedResults results;
    results.scores.resize(running_batch_size);
    results.tokens.resize(running_batch_size);
    std::fill(results.scores.begin(), results.scores.end(), 0);
    
    int64_t kv_cache_len = 0;
    if (is_chat_conversation && !is_cache_empty) {
        OPENVINO_ASSERT(batch_size == 1, "continuation of generation is possible only for batch 1");
        
        // between subsequent runs attention_mask should not be modified
        auto atten_mask_history = m_model_runner.get_tensor("attention_mask");
        kv_cache_len = atten_mask_history.get_shape()[1];

        size_t prompt_len = attention_mask.get_shape()[1];
        ov::Tensor new_atten_mask =  ov::Tensor{ov::element::i64, {batch_size, kv_cache_len + prompt_len}};

        std::copy(atten_mask_history.data<int64_t>(), atten_mask_history.data<int64_t>() + kv_cache_len,
                  new_atten_mask.data<int64_t>());
        std::copy(attention_mask.data<int64_t>(), attention_mask.data<int64_t>() + prompt_len,
                  new_atten_mask.data<int64_t>() + kv_cache_len);

        m_model_runner.set_tensor("attention_mask", new_atten_mask);
    } else if (!is_cache_empty) {
        OPENVINO_THROW("KV cache contains initial values but generate is run not in chat scenario. "
                        "Initial KV cache can contain values only if start_chat() is called.");
    } else {
        m_model_runner.set_tensor("attention_mask", attention_mask);
    }

    if (position_ids_available) {
        position_ids = ov::Tensor{ov::element::i64, input_ids.get_shape()};
        utils::initialize_position_ids(position_ids, attention_mask, kv_cache_len);
    }
    
    m_model_runner.set_tensor("input_ids", input_ids);
    if (position_ids_available)
        m_model_runner.set_tensor("position_ids", position_ids);

    m_model_runner.get_tensor("beam_idx").set_shape({running_batch_size});
    auto beam_data = m_model_runner.get_tensor("beam_idx").data<int32_t>();
    std::iota(beam_data, beam_data + running_batch_size, 0);

    m_model_runner.infer();
    auto logits = m_model_runner.get_tensor("logits");
    ov::Shape logits_shape = logits.get_shape();
    size_t seq_len = logits_shape[1], vocab_size = logits_shape[2];
    m_model_runner.get_tensor("input_ids").set_shape({running_batch_size, 1});

    std::vector<int64_t> token_iter_results(running_batch_size);  // results of a single infer request
    std::vector<int> eos_met(running_batch_size, 0);  // use int because can not use std::all_of with vector<bool>
    for (size_t batch = 0; batch < running_batch_size; ++batch) {
        auto out_token = utils::argmax(logits, batch);
        results.tokens[batch].emplace_back(out_token);

        token_iter_results[batch] = out_token;
        eos_met[batch] = (out_token == generation_config.eos_token_id);
        m_model_runner.get_tensor("input_ids").data<int64_t>()[batch] = out_token;
    }
    if (streamer && streamer->put(token_iter_results[0])) {
        return results;
    }

    bool all_are_eos = std::all_of(eos_met.begin(), eos_met.end(), [](int elem) { return elem == 1; });
    if (!generation_config.ignore_eos && all_are_eos)
        return results;
    
    size_t max_tokens = generation_config.get_max_new_tokens(prompt_len);
    for (size_t i = 0; i < max_tokens - 1; ++i) {
        if (position_ids_available)
            utils::update_position_ids(m_model_runner.get_tensor("position_ids"), m_model_runner.get_tensor("attention_mask"));
        m_model_runner.set_tensor("attention_mask", utils::extend_attention(m_model_runner.get_tensor("attention_mask")));

        m_model_runner.infer();
        auto logits = m_model_runner.get_tensor("logits");
        ov::Shape logits_shape = logits.get_shape();
        size_t seq_len = logits_shape[1], vocab_size = logits_shape[2];
        
        std::vector<int64_t> token_iter_results(running_batch_size);  // results of a single infer request
        std::vector<int> eos_met(running_batch_size, 0);  // use int because can not use std::all_of with vector<bool>
        for (size_t batch = 0; batch < running_batch_size; ++batch) {
            auto out_token = ov::genai::utils::argmax(logits, batch);
            results.tokens[batch].emplace_back(out_token);

            token_iter_results[batch] = out_token;
            eos_met[batch] = (out_token == generation_config.eos_token_id);
            
            m_model_runner.get_tensor("input_ids").data<int64_t>()[batch] = out_token;
        }

        if (streamer && streamer->put(token_iter_results[0]))
            return results;

        if (generation_config.ignore_eos)
            continue;
        
        // stop generation when EOS is met in all batches
        bool all_are_eos = std::all_of(eos_met.begin(), eos_met.end(), [](int elem) { return elem == 1; });
        if (all_are_eos)
            break;

        // Filter out batches where eos is met
        std::vector<int32_t> beam_idx(running_batch_size);
        std::iota(beam_idx.begin(), beam_idx.end(), 0);
        auto end_it = std::remove_if(beam_idx.begin(), beam_idx.end(), [&eos_met](int idx) { return eos_met[idx]; });
        beam_idx.erase(end_it, beam_idx.end());  // Remove the eos met indices

        m_model_runner.get_tensor("beam_idx").set_shape({beam_idx.size()});
        auto beam_data = m_model_runner.get_tensor("beam_idx").data<int32_t>();
        std::copy(beam_idx.begin(), beam_idx.end(), beam_data);
        running_batch_size = beam_idx.size();
    }
    if (streamer) {
        streamer->end();
    }
    return results;
}

}  //namespace genai
}  //namespace ov