// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include "generate_pipeline/llm_pipeline.hpp"
#include "group_beam_searcher.hpp"
#include <filesystem>
#include <jinja2cpp/template.h>
#include <jinja2cpp/template_env.h>

using GenerationResult = std::vector<std::pair<float, std::vector<int64_t>>>;
using namespace std;

std::pair<ov::Tensor, ov::Tensor> pad_left(ov::Tensor&& input_ids, ov::Tensor&& attention_mask, int64_t pad_token) {
    const size_t batch_size = input_ids.get_shape()[0];
    const size_t sequence_length = input_ids.get_shape()[1];
    int64_t* inputs_data = input_ids.data<int64_t>();
    int64_t* attention_mask_data = attention_mask.data<int64_t>();

    for (size_t batch = 0; batch < batch_size; batch++) {
        const size_t batch_offset = batch * sequence_length;

        // last token in the sequence is not a PAD_TOKEN, skipping
        if (inputs_data[batch_offset + sequence_length - 1] != pad_token)
            continue;

        size_t pad_tokens_number = 0;
        for (int i = sequence_length - 1; i >= 0; i--) {
            const size_t token_offset = batch_offset + i;

            if (inputs_data[token_offset] == pad_token)
                continue;

            if (pad_tokens_number == 0)
                pad_tokens_number = sequence_length - i - 1;

            std::swap(inputs_data[token_offset], inputs_data[token_offset + pad_tokens_number]);
            std::swap(attention_mask_data[token_offset], attention_mask_data[token_offset + pad_tokens_number]);
        }
    }

    return {input_ids, attention_mask};
}

void update_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask) {
    const size_t batch_size = attention_mask.get_shape()[0];
    const size_t atten_length = attention_mask.get_shape()[1];
    position_ids.set_shape({batch_size, 1});

    for (size_t batch = 0; batch < batch_size; batch++) {
        int64_t* start = attention_mask.data<int64_t>() + batch * atten_length;
        position_ids.data<int64_t>()[batch] = std::accumulate(start, start + atten_length, 0);
    }
}

void initialize_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask, int64_t start_pos = 0) {
    const size_t batch_size = attention_mask.get_shape()[0];
    const size_t seq_length = attention_mask.get_shape()[1];

    const int64_t* attention_mask_data = attention_mask.data<int64_t>();
    int64_t* position_ids_data = position_ids.data<int64_t>();

    for (size_t batch = 0; batch < batch_size; batch++) {
        size_t sum = start_pos;
        for (size_t i = 0; i < seq_length; i++) {
            const size_t element_offset = batch * seq_length + i;
            position_ids_data[element_offset] = sum;
            if (attention_mask_data[element_offset] == 1) {
                sum += 1;
            }
        }
    }
}

ov::Tensor init_attention_mask(ov::Tensor& position_ids) {
    auto shape = position_ids.get_shape();
    auto attention_mask = ov::Tensor{position_ids.get_element_type(), shape};
    std::fill_n(attention_mask.data<int64_t>(), shape[0] * shape[1], 1);
    return attention_mask;
}

ov::Tensor extend_attention(ov::Tensor attention_mask) {
    auto shape = attention_mask.get_shape();
    auto batch_size = shape[0];
    auto seq_len = shape[1];

    ov::Tensor new_atten_mask = ov::Tensor{attention_mask.get_element_type(), {batch_size, seq_len + 1}};
    auto old_data = attention_mask.data<int64_t>();
    auto new_data = new_atten_mask.data<int64_t>();
    for (size_t batch = 0; batch < batch_size; ++batch) {
        std::memcpy(new_data + batch * (seq_len + 1), old_data + batch * seq_len, seq_len * sizeof(int64_t));
        new_data[batch * (seq_len + 1) + seq_len] = 1;
    }
    return new_atten_mask;
}

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
   
LLMPipeline::LLMPipeline(
    std::string& model_path,
    std::string& tokenizer_path,
    std::string& detokenizer_path,
    std::string device,
    const ov::AnyMap& config
) {
    m_device = device;
    m_config = config;
    ov::Core core;
    
    auto is_xml = [](std::string path) -> bool { return path.compare(path.length() - 4, 4, ".xml") == 0;};
    
    std::string full_path = model_path;
    if (!is_xml(full_path))
        full_path += "/openvino_model.xml";
    m_model_runner = core.compile_model(full_path, device, config).create_infer_request();
    
    // todo: add loading Tokenizers from separate folders
}

LLMPipeline::LLMPipeline(std::string& path, std::string device, const ov::AnyMap& config) {
    std::string tokenizer_config_fname = "tokenizer_config.json";
    std::string generation_config_fname = "generation_config.json";

    if (std::filesystem::exists(path + "/" + generation_config_fname)) {
        m_sampling_parameters = GenerationConfig(path + "/" + generation_config_fname);
    }
    if (std::filesystem::exists(path + "/" + tokenizer_config_fname)) {
        std::ifstream f(path + "/" + tokenizer_config_fname);
        nlohmann::json data = nlohmann::json::parse(f);
        m_chat_template = data.value("chat_template", "");

    }
    
    m_device = device;

    ov::Core core;
    auto model_request = core.compile_model(path + "/openvino_model.xml", device, config).create_infer_request();
    m_model_runner = model_request;

    m_tokenizer = Tokenizer(path);
}

GenerationConfig LLMPipeline::generation_config() const {
    return m_sampling_parameters;
}

void print_tensor(const ov::Tensor& tensor) {
    std::vector<int64_t> res;

    auto t_shape = tensor.get_shape();
    // cout << "[";
    // for (size_t i = 0; i < t_shape[1]; ++i) {
    //     if (tensor.get_element_type() == ov::element::i64) {
    //         res.emplace_back(tensor.data<int64_t>()[i]);
    //         cout << tensor.data<int64_t>()[i] << " ";
    //     }
    // }
    // cout << "]" << endl;
    // cout << "---------" << endl;
}

GenerationResult LLMPipeline::greedy_search(ov::Tensor input_ids, 
                                ov::Tensor attention_mask, 
                                GenerationConfig sampling_params) {
    ov::Shape prompts_shape = input_ids.get_shape();
    size_t batch_size = prompts_shape[0];
    size_t prompt_len = prompts_shape[1];
    
    auto kv_cache_len = m_model_runner.query_state()[0].get_state().get_shape()[2];
    
    auto position_ids = ov::Tensor{ov::element::i64, input_ids.get_shape()};
    initialize_position_ids(position_ids, attention_mask, kv_cache_len);

    GenerationResult results(batch_size);
    
    if (is_chat_conversation && kv_cache_len > 0) {
        // m_attentions_mask_cache extent with attention_mask;

        size_t new_prompt_len = attention_mask.get_shape()[1];
        size_t context_len = m_attentions_mask_cache.get_shape()[1];
        ov::Tensor new_attention_mask =  ov::Tensor{ov::element::i64, {1, context_len + new_prompt_len}};
        // print_tensor(m_attentions_mask_cache);
        // print_tensor(attention_mask);

        for (size_t i = 0; i < context_len; ++i) {
            auto r = m_attentions_mask_cache.data<int64_t>()[i];
            new_attention_mask.data<int64_t>()[i] = m_attentions_mask_cache.data<int64_t>()[i];
        }
        for (size_t i = context_len; i < context_len + new_prompt_len; ++i) {
            auto r = attention_mask.data<int64_t>()[i];
            new_attention_mask.data<int64_t>()[i] = attention_mask.data<int64_t>()[i - context_len];
        }
        // attention_mask = new_attention_mask;
    // }
        m_model_runner.set_tensor("attention_mask", new_attention_mask);
    } else {
        m_model_runner.set_tensor("attention_mask", attention_mask);
    }
    
    // todo: make this work even if position_ids are not specified

    auto atten_shape = attention_mask.get_shape();
    auto pos_shape = position_ids.get_shape();
    auto input_ids_shape = input_ids.get_shape();

    m_model_runner.set_tensor("input_ids", input_ids);
    // print_tensor(m_model_runner.get_tensor("input_ids"));
    // m_model_runner.set_tensor("attention_mask", attention_mask);
    m_model_runner.set_tensor("position_ids", position_ids);

    m_model_runner.get_tensor("beam_idx").set_shape({batch_size});
    auto beam_data = m_model_runner.get_tensor("beam_idx").data<int32_t>();
    std::iota(beam_data, beam_data + batch_size, 0);

    size_t max_tokens = sampling_params.get_max_new_tokens(prompt_len);
    for (size_t i = 0; i < max_tokens; ++i) {
        // print_tensor(m_model_runner.query_state()[0].get_state());
        // print_tensor(m_model_runner.get_tensor("attention_mask"));
        // print_tensor(m_model_runner.get_tensor("position_ids"));
        print_tensor(m_model_runner.get_tensor("input_ids"));
        
        // todo: consider replacing with start_async and run callback right after that
        m_model_runner.infer();
        auto logits = m_model_runner.get_tensor("logits");
        ov::Shape logits_shape = logits.get_shape();
        size_t seq_len = logits_shape[1], vocab_size = logits_shape[2];

        m_model_runner.get_tensor("input_ids").set_shape({batch_size, 1});
        
        m_attentions_mask_cache = ov::Tensor{attention_mask.get_element_type(),  m_model_runner.get_tensor("attention_mask").get_shape()};
        m_model_runner.get_tensor("attention_mask").copy_to(m_attentions_mask_cache);
        // m_attentions_mask_cache = m_model_runner.get_tensor("attention_mask");
        
        update_position_ids(position_ids, m_model_runner.get_tensor("attention_mask"));
        m_model_runner.set_tensor("attention_mask", extend_attention(m_model_runner.get_tensor("attention_mask")));
        
        std::vector<int64_t> token_iter_results(batch_size);  // results of a single infer request
        std::vector<int> eos_met(batch_size, 0);  // use int because can not use std::all_of with vector<bool>
        for (size_t batch = 0; batch < batch_size; ++batch) {
            const float * logits_data = logits.data<const float>() + seq_len * vocab_size * batch + (seq_len - 1) * vocab_size;
            int64_t out_token = std::max_element(logits_data, logits_data + vocab_size) - logits_data;
            results[batch].second.emplace_back(out_token);
            token_iter_results[batch] = out_token;
            eos_met[batch] = (out_token == sampling_params.m_eos_token_id);

            m_model_runner.get_tensor("input_ids").data<int64_t>()[batch] = out_token;
        }
        // place
        // sampling_params.m_callback(std::move(token_iter_results), *this);
        
        if (is_streamer_set) {
            m_streamer_callback(m_streamer.put(token_iter_results[0]));
        }
        
        // stop generation when EOS is met in all batches
        bool all_are_eos = std::all_of(eos_met.begin(), eos_met.end(), [](int elem) { return elem == 1; });
        if (!sampling_params.m_ignore_eos && all_are_eos)
            break;
        // if (i != sampling_params.get_max_new_tokens(prompt_len) - 1)
        //     kv_cache_len += 1;
    }
    return results;
}

GenerationResult LLMPipeline::beam_search(ov::Tensor prompts, ov::Tensor attentin_mask, GenerationConfig sampling_params) {
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

    m_model_runner.set_tensor("input_ids", prompts);
    m_model_runner.set_tensor("attention_mask", attention_mask);
    m_model_runner.set_tensor("position_ids", position_ids);

    // set beam_idx for stateful model: no beam search is used and BATCH_SIZE = 1
    m_model_runner.get_tensor("beam_idx").set_shape({batch_size});
    m_model_runner.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    const int64_t* prompt_data = prompts.data<const int64_t>();
    
    // todo: remove this duplication and use the same SamplingParameters for both greedy and beam
    Parameters parameters{{std::vector<int64_t>{prompt_data, prompt_data + prompts.get_size()}}};
    parameters.n_groups = sampling_params.m_num_groups;
    parameters.diversity_penalty = sampling_params.m_diversity_penalty;
    parameters.group_size = sampling_params.m_group_size;
    
    GroupBeamSearcher group_beam_searcher{parameters};
    std::vector<int64_t> next_tokens;
    std::vector<int32_t> next_beams;
    for (size_t length_count = 0; length_count < sampling_params.get_max_new_tokens(prompt_len); ++length_count) {
        m_model_runner.infer();
        std::tie(next_tokens, next_beams) = group_beam_searcher.select_next_tokens(m_model_runner.get_tensor("logits"));
        if (next_tokens.empty()) {
            break;
        }
        size_t batch_size = next_tokens.size();
        // Set pointers
        m_model_runner.set_tensor("input_ids", ov::Tensor{ov::element::i64, {batch_size, 1}, next_tokens.data()});
        m_model_runner.set_tensor("beam_idx", ov::Tensor{ov::element::i32, {batch_size}, next_beams.data()});
        // Set auxiliary inputs
        ov::Tensor attention_mask = m_model_runner.get_tensor("attention_mask");
        ov::Shape mask_shape{batch_size, attention_mask.get_shape()[1] + 1};
        attention_mask.set_shape(mask_shape);
        std::fill_n(attention_mask.data<int64_t>(), ov::shape_size(mask_shape), 1);

        m_model_runner.get_tensor("position_ids").set_shape({batch_size, 1});
        std::fill_n(m_model_runner.get_tensor("position_ids").data<int64_t>(), batch_size, mask_shape[1] - 1);
        
        // sampling_params.m_callback(std::move(next_tokens), *this);
        // m_callback(std::move(next_tokens);
        if (is_streamer_set) {
            m_streamer.put(next_tokens[0]);
        }

    }

    std::vector<Beam> beams;
    // for (const std::vector<Beam>& group : finalize(std::move(group_beam_searcher))) {
    //     for (const Beam& beam : group) {
    //         beams.emplace_back(beam);
    //     }
    // }

    auto compare_scores = [](Beam left, Beam right) { return (left.score > right.score); };
    std::sort(beams.begin(), beams.end(), compare_scores);
    
    GenerationResult results;
    for (auto beam = beams.begin(); beam != beams.begin() + sampling_params.m_num_return_sequences; ++beam) {
        results.emplace_back(std::pair(beam->score, beam->tokens));
    }
    return results;
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
GenerationResult LLMPipeline::speculative_sampling(ov::Tensor input_ids, ov::Tensor attention_mask, GenerationConfig sampling_params) {
    auto batch_size = input_ids.get_shape()[0];
    OPENVINO_ASSERT(batch_size == 1);
    auto draft_model = sampling_params.get_assistant_model(m_device, m_config);
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

    GenerationResult results(batch_size);
    results[0].second.emplace_back(out_token);
    
    // run K infer requests on draft model and get next K prediction tokens on each iteration
    uint64_t K = sampling_params.m_num_assistant_tokens;
    std::vector<int64_t> draft_tokens;

    // The draft model predicts tokens one by one in an auto-regressive manner, draft_input_ids length should be 1.
    draft_input_ids.set_shape({batch_size, 1});
    draft_position_ids.set_shape({batch_size, 1});

    int max_sequence_length = sampling_params.m_max_new_tokens;
    auto eos_token = sampling_params.m_eos_token_id;
    
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
            results[0].second.emplace_back(out_token);

            disagree_idx = i;                
            if (out_token != draft_tokens[i] || out_token == eos_token || seq_len + disagree_idx + 1 >= max_sequence_length)
                break;
        }

        // After the inference request, key/values have shape [batch_size, seq_len + K, vocab_size].
        // Increment the sequence length by the number of matched tokens, and
        // trim the KV cache to match the new sequence length.
        seq_len += disagree_idx + 1;
        update_kv_cache(draft_model, sampling_params.m_seq_len_axis, seq_len);
        update_kv_cache(main_model, sampling_params.m_seq_len_axis, seq_len);
        
        draft_tokens.clear();
        first_token = out_token;
    }

    return results;
}

GenerationResult LLMPipeline::multinomial_sampling(ov::Tensor prompts, GenerationConfig sampling_params) {
    // todo: implement
    GenerationResult results;
    return results;
}

std::string LLMPipeline::call(std::string text) {
    auto [input_ids, attention_mask] = m_tokenizer.tokenize(text);

    auto generate_results = generate(input_ids, attention_mask, m_sampling_parameters);

    return m_tokenizer.detokenize(generate_results)[0];
}

std::string LLMPipeline::call(std::string text, GenerationConfig generation_config, bool first_time) {
    auto [input_ids, attention_mask] = m_tokenizer.tokenize(text);

    // todo: W/A If sentence begins with a special tokens (<bos>, <s>, etc.) openvino_tokenizer inserts 2 special extra tokens <bos> and "▁",
    // but HF does not do that. Moreover openvino_tokenizer always inserts <bos> but in chat scenario HF does not do that because skip_special_tokens=True.
    // Need to remove both of that tokens manually to get exact token by token alignment with HF
    auto size = input_ids.get_shape();
    int64_t* inputs_data = input_ids.data<int64_t>();
    std::vector<int64_t> tmp_ids(inputs_data, inputs_data + input_ids.get_size()); // todo: works only for batch 1
    tmp_ids.erase(tmp_ids.begin());

    auto attention_mask_data = attention_mask.data<int64_t>();
    std::vector<float> tmp_attn_mask(attention_mask_data, attention_mask_data + attention_mask.get_size());
    tmp_attn_mask.erase(tmp_attn_mask.begin());

    std::vector<std::string> prefixes_to_exclude = {"<s>", "</s>"};  // todo: for TinyLlama, need to get them form generation_config
    auto prefix_match = [&text](std::string prefix) { return text.substr(0, prefix.length()) == prefix; };
    if (std::any_of(prefixes_to_exclude.begin(), prefixes_to_exclude.end(), prefix_match)) {
        tmp_ids.erase(tmp_ids.begin());
        tmp_attn_mask.erase(tmp_attn_mask.begin());
    }

    input_ids = ov::Tensor(input_ids.get_element_type(), {1, tmp_ids.size()});
    for (size_t i = 0; i < tmp_ids.size(); i++)
        input_ids.data<int64_t>()[i] = tmp_ids.data()[i];
    attention_mask = ov::Tensor(attention_mask.get_element_type(), {1, tmp_attn_mask.size()});
    for (size_t i = 0; i < tmp_attn_mask.size(); i++)
        attention_mask.data<int64_t>()[i] = tmp_attn_mask.data()[i];

    // to keep config specified during LLMPipeline creation need to get existing 
    // and modify only after that, e.g.:
    // GenerationConfig config = pipe.generation_config();
    // config.do_sample(false).max_new_tokens(20);
    // print_tensor(input_ids);
    auto generate_results = generate(input_ids, attention_mask, generation_config, first_time);

    return m_tokenizer.detokenize(generate_results)[0];
}

std::vector<std::string> LLMPipeline::call(std::vector<std::string> text, GenerationConfig sampling_parameters) {
    auto [input_ids, attention_mask] = m_tokenizer.tokenize(text);

    auto generate_results = generate(input_ids, attention_mask, sampling_parameters);

    return m_tokenizer.detokenize(generate_results);
}

std::string LLMPipeline::operator()(std::string text) {
    return call(text);
}

std::string LLMPipeline::operator()(std::string text, GenerationConfig sampling_parameters) {
    return call(text, sampling_parameters);
}

std::vector<std::string> LLMPipeline::operator()(std::vector<std::string> text, GenerationConfig sampling_parameters) {
    return call(text, sampling_parameters);
}

std::vector<std::string> LLMPipeline::operator()(std::initializer_list<std::string> text, GenerationConfig sampling_parameters) {
    return call(text, sampling_parameters);
}

GenerationResult LLMPipeline::generate(ov::Tensor input_ids, ov::Tensor attention_mask, GenerationConfig generation_config, bool first_time) {
    GenerationResult result;

    if (generation_config.is_gready_sampling()) {
        result = greedy_search(input_ids, attention_mask, generation_config);
    } else if (generation_config.is_beam_search()) {
        result = beam_search(input_ids, attention_mask, generation_config);
    } else if (generation_config.is_multimomial()) {
        result = multinomial_sampling(input_ids, generation_config);
    } else { // speculative
        result = speculative_sampling(input_ids, attention_mask, generation_config);
    }

    if (!is_chat_conversation)
        reset_state();

    return result;
}

GenerationResult LLMPipeline::generate(ov::Tensor input_ids, ov::Tensor attention_mask) {
    return generate(input_ids, attention_mask, m_sampling_parameters);
}

GenerationResult LLMPipeline::generate(ov::Tensor input_ids, GenerationConfig sampling_params) {

    return generate(input_ids, init_attention_mask(input_ids), sampling_params);
}

GenerationResult LLMPipeline::generate(ov::Tensor input_ids) {
    return generate(input_ids, init_attention_mask(input_ids), m_sampling_parameters);
}

Tokenizer LLMPipeline::get_tokenizer() {
    return m_tokenizer;
}

std::string LLMPipeline::apply_chat_template(std::string prompt, std::string role) const {
    jinja2::TemplateEnv env;
    env.GetSettings().lstripBlocks = true;
    env.GetSettings().trimBlocks = true;
    jinja2::Template tpl(&env);
    tpl.Load(m_chat_template);
    
    jinja2::ValuesMap message {{"role", role}, {"content", prompt}};
    jinja2::ValuesMap params = {
        {"messages", jinja2::ValuesList({message})},
        {"eos_token", "</s>"},  // todo: load from config
        {"add_generation_prompt", true},
    };
 
    return tpl.RenderAsString(params).value();
}

void LLMPipeline::set_streamer_callback(std::function<void (std::string)> callback) {
    is_streamer_set = true;
    m_streamer_callback = callback;
    m_streamer = TextCoutStreamer(m_tokenizer);
}

void LLMPipeline::start_conversation() {
    is_chat_conversation = true;
}

void LLMPipeline::stop_conversation() {
    is_chat_conversation = false;
    reset_state();
}

void LLMPipeline::reset_state() {
    m_model_runner.reset_state();
}
