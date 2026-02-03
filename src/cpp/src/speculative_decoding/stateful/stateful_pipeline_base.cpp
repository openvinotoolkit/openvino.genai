// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "stateful_pipeline_base.hpp"

#include "continuous_batching/timer.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "utils.hpp"

namespace ov::genai {

template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

StatefulSpeculativePipelineBase::StatefulSpeculativePipelineBase(const Tokenizer& tokenizer,
                                                                 const GenerationConfig& generation_config)
    : LLMPipelineImplBase(tokenizer, generation_config) {
    m_sd_perf_metrics = SDPerModelsPerfMetrics();
}

void StatefulSpeculativePipelineBase::ensure_num_assistant_tokens_is_set(GenerationConfig& config) {
    OPENVINO_ASSERT(
        config.assistant_confidence_threshold == 0.f,
        "Stateful (non-Continuous Batching) Speculative Decoding pipeline only supports num_assistant_tokens, "
        "not assistant_confidence_threshold. Set assistant_confidence_threshold to 0.f or remove its specification.");

    if (config.num_assistant_tokens == 0) {
        config.num_assistant_tokens = DEFAULT_NUM_ASSISTANT_TOKENS;
    }
}

GenerationConfig StatefulSpeculativePipelineBase::resolve_generation_config(
    OptionalGenerationConfig generation_config) {
    GenerationConfig config = generation_config.value_or(m_generation_config);

    // Apply defaults from base config
    if (config.stop_token_ids.empty()) {
        config.stop_token_ids = m_generation_config.stop_token_ids;
    }
    if (config.eos_token_id == -1) {
        config.set_eos_token_id(m_generation_config.eos_token_id);
    }

    config.validate();
    return config;
}

TokenizedInputs StatefulSpeculativePipelineBase::tokenize(const std::string& prompt, const GenerationConfig& config) {
    ManualTimer encode_timer("Encode");
    encode_timer.start();

    TokenizedInputs tokenized_input;
    if (m_is_chat_active) {
        // In chat mode, append to history and apply template
        m_chat_history.push_back({{"role", "user"}, {"content", prompt}});
        constexpr bool add_generation_prompt = true;
        auto templated_prompt = m_tokenizer.apply_chat_template(m_chat_history, add_generation_prompt);
        // for chat ov::genai::add_special_tokens(false) is aligned with stateful pipeline and HF
        tokenized_input = m_tokenizer.encode(templated_prompt, ov::genai::add_special_tokens(false));
    } else {
        // Non-chat mode: check if chat template should be applied
        if (config.apply_chat_template && !m_tokenizer.get_chat_template().empty()) {
            ChatHistory history({{{"role", "user"}, {"content", prompt}}});
            constexpr bool add_generation_prompt = true;
            auto templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
            tokenized_input = m_tokenizer.encode(templated_prompt, ov::genai::add_special_tokens(false));
        } else {
            tokenized_input = m_tokenizer.encode(prompt, ov::genai::add_special_tokens(true));
        }
    }

    encode_timer.end();
    m_sd_perf_metrics.raw_metrics.tokenization_durations.emplace_back(encode_timer.get_duration_microsec());

    return tokenized_input;
}

std::vector<std::string> StatefulSpeculativePipelineBase::detokenize(const std::vector<std::vector<int64_t>>& tokens) {
    ManualTimer decode_timer("Decode");
    decode_timer.start();

    auto result = m_tokenizer.decode(tokens);

    decode_timer.end();
    m_sd_perf_metrics.raw_metrics.detokenization_durations.emplace_back(decode_timer.get_duration_microsec());

    return result;
}

void StatefulSpeculativePipelineBase::update_decoded_results_with_perf_metrics(DecodedResults& decoded_results,
                                                                               const EncodedResults& encoded_results,
                                                                               float generate_duration_us,
                                                                               TimePoint generate_start_time) {
    // Use encoded_results as base (contains model-level metrics from generate_tokens)
    decoded_results.perf_metrics = encoded_results.perf_metrics;
    decoded_results.extended_perf_metrics = encoded_results.extended_perf_metrics;

    // Update with the latest timing data from m_sd_perf_metrics
    auto& raw_counters = decoded_results.perf_metrics.raw_metrics;

    // Update generate_durations with total outer time
    raw_counters.generate_durations.clear();
    raw_counters.generate_durations.emplace_back(generate_duration_us);

    // Copy tokenization_durations from m_sd_perf_metrics (set by tokenize() or encode_timer)
    raw_counters.tokenization_durations = m_sd_perf_metrics.raw_metrics.tokenization_durations;

    // Copy detokenization_durations from m_sd_perf_metrics (set by detokenize())
    raw_counters.detokenization_durations = m_sd_perf_metrics.raw_metrics.detokenization_durations;

    // Re-evaluate statistics with updated timings
    decoded_results.perf_metrics.m_evaluated = false;
    decoded_results.perf_metrics.evaluate_statistics(generate_start_time);
}

DecodedResults StatefulSpeculativePipelineBase::generate(StringInputs inputs,
                                                         OptionalGenerationConfig generation_config,
                                                         StreamerVariant streamer) {
    ManualTimer generate_timer("StatefulSpeculativePipelineBase::generate(StringInputs)");
    generate_timer.start();

    // Extract prompt string
    std::string prompt = std::visit(
        overloaded{[](const std::string& prompt_str) {
                       return prompt_str;
                   },
                   [](std::vector<std::string>& prompts) {
                       OPENVINO_ASSERT(prompts.size() == 1u, "Currently only batch size=1 is supported");
                       return prompts.front();
                   }},
        inputs);

    GenerationConfig config = resolve_generation_config(generation_config);

    // Tokenize
    auto tokenized_input = tokenize(prompt, config);

    // Generate tokens
    auto encoded_results = generate_tokens(tokenized_input, config, streamer);

    // Detokenize
    DecodedResults decoded_results = {detokenize(encoded_results.tokens), encoded_results.scores};

    // Handle chat history
    if (m_is_chat_active) {
        if (m_streaming_was_cancelled) {
            m_chat_history.pop_back();  // Rollback on cancellation
        } else {
            m_chat_history.push_back({{"role", "assistant"}, {"content", decoded_results.texts[0]}});
        }
    }

    generate_timer.end();

    // Update performance metrics with outer layer timings
    update_decoded_results_with_perf_metrics(decoded_results,
                                             encoded_results,
                                             generate_timer.get_duration_microsec(),
                                             generate_timer.get_start_time());

    return decoded_results;
}

DecodedResults StatefulSpeculativePipelineBase::generate(const ChatHistory& history,
                                                         OptionalGenerationConfig generation_config,
                                                         StreamerVariant streamer) {
    ManualTimer generate_timer("StatefulSpeculativePipelineBase::generate(ChatHistory)");
    generate_timer.start();

    ManualTimer encode_timer("Encode");
    encode_timer.start();

    GenerationConfig config = resolve_generation_config(generation_config);

    OPENVINO_ASSERT(config.apply_chat_template,
                    "Chat template must be applied when using ChatHistory in generate method.");
    OPENVINO_ASSERT(!m_tokenizer.get_chat_template().empty(),
                    "Chat template must not be empty when using ChatHistory in generate method.");
    OPENVINO_ASSERT(!history.empty(), "Chat history must not be empty when using ChatHistory in generate method.");

    constexpr bool add_generation_prompt = true;
    // for chat ov::genai::add_special_tokens(false) is aligned with stateful pipeline and HF
    auto templated_chat_history = m_tokenizer.apply_chat_template(history, add_generation_prompt);
    auto tokenized_inputs = m_tokenizer.encode(templated_chat_history, ov::genai::add_special_tokens(false));

    encode_timer.end();
    m_sd_perf_metrics.raw_metrics.tokenization_durations.emplace_back(encode_timer.get_duration_microsec());

    // Generate tokens
    auto encoded_results = generate_tokens(tokenized_inputs, config, streamer);

    // Detokenize
    DecodedResults decoded_results = {detokenize(encoded_results.tokens), encoded_results.scores};

    generate_timer.end();

    // Update performance metrics with outer layer timings
    update_decoded_results_with_perf_metrics(decoded_results,
                                             encoded_results,
                                             generate_timer.get_duration_microsec(),
                                             generate_timer.get_start_time());

    return decoded_results;
}

EncodedResults StatefulSpeculativePipelineBase::generate(const EncodedInputs& inputs,
                                                         OptionalGenerationConfig generation_config,
                                                         StreamerVariant streamer) {
    // Resolve configuration
    GenerationConfig config = resolve_generation_config(generation_config);
    
    // Delegate to generate_tokens (implemented by child classes)
    return generate_tokens(inputs, config, streamer);
}

void StatefulSpeculativePipelineBase::start_chat(const std::string& system_message) {
    m_is_chat_active = true;
    m_chat_history.clear();
    if (!system_message.empty()) {
        m_chat_history.push_back({{"role", "system"}, {"content", system_message}});
    }
}

void StatefulSpeculativePipelineBase::finish_chat() {
    m_is_chat_active = false;
    m_chat_history.clear();
}

}  // namespace ov::genai
