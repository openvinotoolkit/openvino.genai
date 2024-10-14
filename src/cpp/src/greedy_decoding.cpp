// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/perf_metrics.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

EncodedResults greedy_decoding(
    ov::InferRequest& m_model_runner, 
    ov::Tensor input_ids, 
    ov::Tensor attention_mask, 
    const ov::genai::GenerationConfig generation_config, 
    const std::shared_ptr<StreamerBase> streamer,
    std::optional<ov::Tensor> position_ids
) {
    ov::Shape prompts_shape = input_ids.get_shape();
    const size_t batch_size = prompts_shape[0];
    size_t running_batch_size = batch_size;
    size_t prompt_len = prompts_shape[1];
    size_t max_new_tokens = generation_config.get_max_new_tokens(prompt_len);

    // Initialize results and performance metrics.
    EncodedResults results;
    auto& raw_perf_counters = results.perf_metrics.raw_metrics;
    raw_perf_counters.m_new_token_times.reserve(max_new_tokens);
    raw_perf_counters.m_batch_sizes.reserve(max_new_tokens);
    raw_perf_counters.m_token_infer_durations.reserve(max_new_tokens);
    raw_perf_counters.m_inference_durations = {{ MicroSeconds(0.0f) }};

    results.scores.resize(running_batch_size);
    results.tokens.resize(running_batch_size);
    std::fill(results.scores.begin(), results.scores.end(), 0);

    m_model_runner.set_tensor("input_ids", input_ids);
    m_model_runner.set_tensor("attention_mask", attention_mask);
    if (position_ids.has_value())
        m_model_runner.set_tensor("position_ids", *position_ids);

    m_model_runner.get_tensor("beam_idx").set_shape({running_batch_size});
    auto beam_data = m_model_runner.get_tensor("beam_idx").data<int32_t>();
    std::iota(beam_data, beam_data + running_batch_size, 0);

    const auto infer_start = std::chrono::steady_clock::now();
    m_model_runner.infer();
    const auto infer_ms = PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
    raw_perf_counters.m_inference_durations[0] = MicroSeconds(infer_ms);
    raw_perf_counters.m_token_infer_durations.emplace_back(infer_ms);
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
    raw_perf_counters.m_new_token_times.emplace_back(std::chrono::steady_clock::now());
    raw_perf_counters.m_batch_sizes.emplace_back(batch_size);
        
    if (streamer && streamer->put(token_iter_results[0])) {
        return results;
    }

    bool all_are_eos = std::all_of(eos_met.begin(), eos_met.end(), [](int elem) { return elem == 1; });
    if (!generation_config.ignore_eos && all_are_eos)
        return results;

    for (size_t i = 0; i < max_new_tokens - 1; ++i) {
        if (position_ids.has_value())
            utils::update_position_ids(m_model_runner.get_tensor("position_ids"), m_model_runner.get_tensor("attention_mask"));
        m_model_runner.set_tensor("attention_mask", utils::extend_attention(m_model_runner.get_tensor("attention_mask")));

        const auto infer_start = std::chrono::steady_clock::now();
        m_model_runner.infer();
        const auto infer_ms = PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
        raw_perf_counters.m_inference_durations[0] += MicroSeconds(infer_ms);
        raw_perf_counters.m_token_infer_durations.emplace_back(infer_ms);

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
        raw_perf_counters.m_new_token_times.emplace_back(std::chrono::steady_clock::now());
        raw_perf_counters.m_batch_sizes.emplace_back(batch_size);

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
