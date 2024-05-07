// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "generation_config_helper.hpp"
#include "llm_pipeline.hpp"

ov::Tensor trimm_tensor(ov::Tensor& tensor, uint64_t seq_len_axis, uint64_t new_seq_len);
void update_kv_cache(ov::InferRequest request, uint64_t seq_len_axis, uint64_t new_seq_len);

ov::Tensor trimm_tensor(ov::Tensor& tensor, uint64_t seq_len_axis, uint64_t new_seq_len) {
    // Copy elements from the old to a new tensor and return it.
    // It's assumed that key/values tensor has a shape [BATCH_SIZE, num_kv_heads, seq_len, head_size] or [seq_len, ...],
    // It that's not the case for your model please implement your own trim method.
    OPENVINO_ASSERT(seq_len_axis == 2 || seq_len_axis == 0, "Cannot trim key/values with sequence length axis = ", seq_len_axis);
    
    auto old_tensor_data = tensor.data<float>();
    auto shape = tensor.get_shape();
    size_t batch_size = shape[0];
    size_t num_kv_heads = shape[1];
    size_t old_seq_len = shape[2];
    size_t head_size = shape[3];
    
    OPENVINO_ASSERT(new_seq_len <= old_seq_len);
    
    // if new_seq_len equal to old one no need to copy tensor, return as is
    if (old_seq_len == new_seq_len)
        return tensor;

    if (seq_len_axis == 0) {
        shape[0] = new_seq_len;
        tensor.set_shape(shape);
    }

    // if seq_len_axis == 2, then data is not contiguous, in order to trim need to repack tensor
    auto new_tensor = ov::Tensor{ov::element::f32, {batch_size, num_kv_heads, new_seq_len, head_size}};
    auto new_tensor_data = new_tensor.data<float>();
    for (size_t batch = 0; batch < batch_size; ++batch){
        for (size_t i = 0; i < num_kv_heads; ++i) {
            for (size_t j = 0; j < new_seq_len; ++j) {
                auto dst_ptr = new_tensor_data + num_kv_heads * new_seq_len * head_size * batch + new_seq_len * head_size * i +  head_size * j;
                auto src_ptr = old_tensor_data + num_kv_heads * new_seq_len * head_size * batch + old_seq_len * head_size * i +  head_size * j;
                std::memcpy(dst_ptr, src_ptr, head_size * sizeof(float));
            }
        }
    }
    return new_tensor;
}

void update_kv_cache(ov::InferRequest request, uint64_t seq_len_axis, uint64_t new_seq_len) {
    // trim kv_cache values up to the new_seq_len
    for (auto& state: request.query_state()) {
        ov::Tensor old_tensor = state.get_state();
        state.set_state(trimm_tensor(old_tensor, seq_len_axis, new_seq_len));
    }
}

/* Speculative decoding works the following way. The draft model predicts the next K
tokens one by one in an autoregressive manner, while the main model validates these
predictions and corrects them if necessary. We go through each predicted token, and
if a difference is detected between the draft and main model, we stop and keep the
last token predicted by the main model. Then the draft model gets the latest main
prediction and again tries to predict the next K tokens, repeating the cycle.

This approach reduces the need for multiple infer requests to the main model,
enhancing performance. For instance, in more predictable parts of text generation,
the draft model can, in best-case scenarios, generate the next K tokens that exactly
match the target. In tha caste the are validated in a single inference request to
the main model (which is bigger, more accurate but slower) instead of running K
subsequent requests. 
*/

namespace ov {
ov::EncodedResults assistive_decoding(ov::InferRequest& m_model_runner, ov::Tensor input_ids, ov::Tensor attention_mask, ov::GenerationConfig generation_config) {
    ov::GenerationConfigHelper config_helper = generation_config;

    auto batch_size = input_ids.get_shape()[0];
    OPENVINO_ASSERT(batch_size == 1);
    auto draft_model = config_helper.get_assistant_model(); // todo: add config getting m_device, m_plugin_config
    auto main_model = m_model_runner;
    
    auto draft_input_ids = ov::Tensor{input_ids.get_element_type(), input_ids.get_shape()};
    input_ids.copy_to(draft_input_ids);
    auto draft_attention_mask = ov::Tensor{input_ids.get_element_type(), input_ids.get_shape()};
    
    draft_model.set_tensor("input_ids", draft_input_ids);
    draft_model.set_tensor("attention_mask", draft_attention_mask);
    
    ov::Tensor draft_position_ids = draft_model.get_tensor("position_ids");
    draft_position_ids.set_shape(draft_input_ids.get_shape());
    std::iota(draft_position_ids.data<int64_t>(), draft_position_ids.data<int64_t>() + draft_position_ids.get_size(), 0);
    uint64_t seq_len = draft_input_ids.get_shape()[1];

    // Input tensors for the main model should not be mixed with draft.
    // Do not feed the same draft_postion_ids to the main, but copy input_ids from the draft_input_ids
    // auto input_ids = main_model.get_tensor("input_ids");
    // input_ids.set_shape(draft_input_ids.get_shape());
    // draft_input_ids.copy_to(input_ids);

    // auto attention_mask = main_model.get_tensor("attention_mask");
    // attention_mask.set_shape(draft_input_ids.get_shape());
    // std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);

    auto position_ids = main_model.get_tensor("position_ids");
    position_ids.set_shape(draft_input_ids.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);
    
    // set beam_idx for stateful model: no beam search is used and batch_size = 1
    draft_model.get_tensor("beam_idx").set_shape({batch_size});
    draft_model.get_tensor("beam_idx").data<int32_t>()[0] = 0;
    main_model.get_tensor("beam_idx").set_shape({batch_size});
    main_model.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    main_model.set_tensor("input_ids", input_ids);
    main_model.set_tensor("attention_mask", attention_mask);
    main_model.set_tensor("position_ids", position_ids);

    // To coollect kv-cache for the <PROMPT> and to get the next token run the very first infer request
    draft_model.infer();
    main_model.infer();

    size_t vocab_size = draft_model.get_tensor("logits").get_shape().back();
    OPENVINO_ASSERT(vocab_size == main_model.get_tensor("logits").get_shape().back(), "vocab size should be the same for the both models");
    
    // logits shape is [batch_size, seq_len, vocab_size]
    auto logits = main_model.get_tensor("logits");
    auto data_logits = logits.data<float>() + (seq_len - 1) * vocab_size;
    int64_t out_token = std::max_element(data_logits, data_logits + vocab_size) - data_logits;
    
    // the first token which is fed to both draft and main netwoks on each iteration
    auto first_token = out_token;

    ov::EncodedResults results;
    results.tokens.resize(batch_size);

    results.tokens[0].emplace_back(out_token);
    
    // run K infer requests on draft model and get next K prediction tokens on each iteration
    uint64_t K = config_helper.num_assistant_tokens;
    std::vector<int64_t> draft_tokens;

    // The draft model predicts tokens one by one in an auto-regressive manner, draft_input_ids length should be 1.
    draft_input_ids.set_shape({batch_size, 1});
    draft_position_ids.set_shape({batch_size, 1});

    int max_sequence_length = generation_config.max_new_tokens;
    auto eos_token = generation_config.eos_token_id;
    
    while (out_token != eos_token && seq_len < max_sequence_length) {
        // infer the K next tokens with draft model
        for (int i = 0; i < K; ++i) {
            draft_input_ids.data<int64_t>()[0] = out_token;
            draft_attention_mask.set_shape({batch_size, seq_len + i + 1});
            std::fill_n(draft_attention_mask.data<int64_t>(), draft_attention_mask.get_size(), 1);
            draft_position_ids.data<int64_t>()[0] = int64_t(draft_attention_mask.get_size() - 1);

            draft_model.infer();

            auto draft_logits = draft_model.get_tensor("logits").data<float>();
            int64_t arg_max_token = std::max_element(draft_logits, draft_logits + vocab_size) - draft_logits;
            out_token = arg_max_token;
            draft_tokens.emplace_back(arg_max_token);
        }

        // For the main network, K tokens will be fed at once in a single infer request.
        input_ids.set_shape({batch_size, K});
        // Set the first token for the main model to be the same as for the draft model.
        input_ids.data<int64_t>()[0] = first_token;
        for (int i = 0; i < K - 1; i++)
            input_ids.data<int64_t>()[i + 1] = draft_tokens[i];

        attention_mask.set_shape({batch_size, seq_len + K});
        std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);

        position_ids.set_shape({batch_size, K});
        std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), seq_len);

        main_model.infer();

        data_logits = logits.data<float>();  // [batch_size, K, vocab_size]
        size_t disagree_idx = K - 1;
        // Iterate through the predicted tokens from the main model and compare them with draft predictions.
        // In the worst-case scenario (disagreement at the beginning), iter will increase by 1.
        // In the best-case scenario, all elements match, and K predicted tokens will be taken.
        for (size_t i = 0; i < K; i++) {
            auto start = data_logits + vocab_size * i;
            auto stop = data_logits + vocab_size * (i + 1);
            out_token = std::max_element(start, stop) - start;
            results.tokens[0].emplace_back(out_token);

            // m_streamer->put(out_token);

            disagree_idx = i;                
            if (out_token != draft_tokens[i] || out_token == eos_token || seq_len + disagree_idx + 1 >= max_sequence_length)
                break;
        }

        // After the inference request, key/values have shape [batch_size, seq_len + K, vocab_size].
        // Increment the sequence length by the number of matched tokens, and
        // trim the KV cache to match the new sequence length.
        seq_len += disagree_idx + 1;
        update_kv_cache(draft_model, config_helper.seq_len_axis, seq_len);
        update_kv_cache(main_model, config_helper.seq_len_axis, seq_len);
        
        draft_tokens.clear();
        first_token = out_token;
    }

    return results;
}

} // namespace ov