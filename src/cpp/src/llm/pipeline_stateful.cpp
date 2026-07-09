
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include <nlohmann/json.hpp>

#include "llm/pipeline_stateful.hpp"

#include "json_utils.hpp"
#include "lora/helper.hpp"
#include "lm_encoding.hpp"
#include "openvino/genai/text_streamer.hpp"

#include "utils.hpp"

namespace {

// Mitigates a LongRoPE short/long-factor KV-cache-consistency issue: once the cumulative
// sequence length reaches original_max_position_embeddings, the model's RoPE branch
// selection switches from the "short" to the "long" factor table, but previously cached KV
// entries remain encoded with the "short" factor, corrupting attention. NPU only (see the
// m_is_npu checks at the two call sites below).
//
// Detects the threshold from the model's config.json ("rope_parameters", rope_type/type ==
// "longrope", e.g. Phi-4-mini). Returns std::nullopt for models that don't use LongRoPE, or
// if config.json isn't present/parseable.
std::optional<size_t> detect_longrope_threshold(const std::filesystem::path& models_path) {
    std::ifstream config_file(models_path / "config.json");
    if (!config_file.is_open())
        return std::nullopt;

    nlohmann::json data = nlohmann::json::parse(config_file, nullptr, /* allow_exceptions = */ false);
    if (data.is_discarded())
        return std::nullopt;

    std::string rope_type;
    ov::genai::utils::read_json_param(data, "rope_parameters.rope_type", rope_type);
    if (rope_type.empty())
        ov::genai::utils::read_json_param(data, "rope_parameters.type", rope_type);
    if (rope_type != "longrope")
        return std::nullopt;

    size_t threshold = 0;
    ov::genai::utils::read_json_param(data, "rope_parameters.original_max_position_embeddings", threshold);
    if (threshold == 0)
        ov::genai::utils::read_json_param(data, "original_max_position_embeddings", threshold);
    return threshold > 0 ? std::optional<size_t>(threshold) : std::nullopt;
}

// Wraps the caller-supplied streamer (if any) and, in addition to forwarding every token to
// it, requests ov::genai::StreamingStatus::STOP once the newly generated token's 0-based
// position reaches `threshold`. Lets the caller detect a mid-generation LongRoPE crossing via
// threshold_reached() and re-run generation from a fully re-prefilled, consistent KV-cache.
class LongRopeThresholdStreamer : public ov::genai::StreamerBase {
public:
    LongRopeThresholdStreamer(std::shared_ptr<ov::genai::StreamerBase> base, size_t prompt_length, size_t threshold)
        : m_base(std::move(base)), m_total_len(prompt_length), m_threshold(threshold) {}

    ov::genai::StreamingStatus write(int64_t token) override {
        ov::genai::StreamingStatus status = m_base ? m_base->write(token) : ov::genai::StreamingStatus::RUNNING;
        ++m_total_len;
        m_last_write_stopped_by_us = false;
        // total_len == threshold + 1  <=>  the just-appended token's 0-based position
        // (total_len - 1) equals `threshold`.
        if (status == ov::genai::StreamingStatus::RUNNING && m_total_len == m_threshold + 1) {
            m_threshold_reached = true;
            m_last_write_stopped_by_us = true;
            return ov::genai::StreamingStatus::STOP;
        }
        return status;
    }

    // get_lm_encoded_results() calls end() once per call. If OUR own threshold check is what
    // just stopped generation, a continuation phase follows, so forwarding end() here would
    // close the wrapped streamer prematurely - suppress it; the call site flushes it via
    // flush_end() once it knows no continuation will run after all. Otherwise (generation
    // ended for any other reason - EOS, cancel, etc. - with no continuation to follow),
    // forward immediately as usual.
    void end() override {
        if (!m_last_write_stopped_by_us)
            flush_end();
    }

    void flush_end() {
        if (m_base)
            m_base->end();
    }

    bool threshold_reached() const {
        return m_threshold_reached;
    }

private:
    std::shared_ptr<ov::genai::StreamerBase> m_base;
    size_t m_total_len;
    size_t m_threshold;
    bool m_threshold_reached = false;
    bool m_last_write_stopped_by_us = false;
};

}  // namespace

namespace ov::genai {

StatefulLLMPipeline::StatefulLLMPipeline(
    const ov::InferRequest& request,
    const ov::genai::Tokenizer& tokenizer,
    OptionalGenerationConfig generation_config)
    : LLMPipelineImplBase(tokenizer, generation_config.value_or(GenerationConfig())),
    m_model_runner(request) {
    auto compiled_model = m_model_runner.get_compiled_model();
    auto execution_devices = compiled_model.get_property(ov::execution_devices);
    if (execution_devices[0].find("NPU") != std::string::npos) {
        OPENVINO_ASSERT(execution_devices.size() == 1u);
        m_is_npu = true;
        m_max_prompt_len = compiled_model.get_property("NPUW_LLM_MAX_PROMPT_LEN").as<uint32_t>();
        const auto min_response_len = compiled_model.get_property("NPUW_LLM_MIN_RESPONSE_LEN").as<uint32_t>();
        m_max_kv_cache_size = m_max_prompt_len + min_response_len;
    }
}

StatefulLLMPipeline::StatefulLLMPipeline(
    const std::filesystem::path& models_path,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& properties)
    : StatefulLLMPipeline{
        utils::read_model(models_path, properties),
        tokenizer,
        device,
        properties,
        utils::from_config_json_if_exists(models_path)
    } {
    set_longrope_threshold(models_path);
}

void StatefulLLMPipeline::set_longrope_threshold(const std::filesystem::path& models_path) {
    if (models_path.empty()) {
        m_longrope_threshold = std::nullopt;
        m_force_longrope_reprefill = false;
        m_longrope_reprefill_pending = false;
        return;
    }
    m_longrope_threshold = detect_longrope_threshold(models_path);
}

bool StatefulLLMPipeline::should_force_longrope_reprefill(size_t new_total_len) {
    if (!m_is_npu || !m_longrope_threshold.has_value())
        return false;
    // A previous call's mid-generation crossing landed on the very last allowed token, so no
    // continuation re-prefill ran then - force one now, unconditionally, regardless of the
    // transition check below (the cache is known-inconsistent until this fires once).
    if (m_longrope_reprefill_pending) {
        m_longrope_reprefill_pending = false;
        return true;
    }
    const size_t prev_total_len = m_cache_state.get_state().size();
    return prev_total_len < *m_longrope_threshold && new_total_len >= *m_longrope_threshold;
}

StatefulLLMPipeline::StatefulLLMPipeline(
    const std::shared_ptr<ov::Model>& model,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& properties,
    const ov::genai::GenerationConfig& generation_config)
    : LLMPipelineImplBase(tokenizer, generation_config), m_sampler(m_tokenizer), m_cache_state(model) {
    if (device.find("NPU") != std::string::npos) {
        m_is_npu = true;
        m_use_full_chat_history = true;
    }

    // FIXME: slicing produces incorrect results for some models on NPU.
    // On NPU, applying slice the safe way is done by the underlying plugin
    if (!m_is_npu) {
        utils::apply_slice_before_matmul_transformation(model);
    }

    auto kv_pos = ov::genai::utils::get_kv_axes_pos(model);

    if (!m_use_full_chat_history)
        m_cache_state.seq_length_axis = kv_pos.seq_len;

    auto [filtered_properties_without_gguf, enable_save_ov_model] = utils::extract_gguf_properties(properties);
    auto filtered_properties = extract_adapters_from_properties(filtered_properties_without_gguf, &m_generation_config.adapters);
    if (m_generation_config.adapters) {
        m_generation_config.adapters->set_tensor_name_prefix("base_model.model.");
        m_adapter_controller = AdapterController(model, *m_generation_config.adapters, device);   // TODO: Make the prefix name configurable
    }
    ov::CompiledModel compiled_model;
    if (m_is_npu) {
        utils::KVDesc kv_desc;
        std::tie(compiled_model, kv_desc) = utils::compile_decoder_for_npu(model, *filtered_properties, kv_pos);
        m_max_prompt_len = kv_desc.max_prompt_len;
        m_max_kv_cache_size = kv_desc.max_prompt_len + kv_desc.min_response_len;
    } else {
       compiled_model = utils::singleton_core().compile_model(model, device, *filtered_properties);
    }
    m_model_runner = compiled_model.create_infer_request();
    ov::genai::utils::print_compiled_model_properties(compiled_model, "Stateful LLM model");

    // If eos_token_id was not provided, take value
    if (m_generation_config.eos_token_id == -1)
        m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());

    m_sampler.set_seed(m_generation_config.rng_seed);
}

StatefulLLMPipeline::StatefulLLMPipeline(
    const std::filesystem::path& models_path,
    const std::string& device,
    const ov::AnyMap& plugin_config)
    : StatefulLLMPipeline{models_path, Tokenizer(models_path, plugin_config), device, plugin_config} {}

GenerationConfig StatefulLLMPipeline::resolve_generation_config(OptionalGenerationConfig generation_config) const {
    GenerationConfig config = generation_config.value_or(m_generation_config);
    // If stop_token_ids were not provided, take value from default m_generation_config
    if (config.stop_token_ids.empty())
        config.stop_token_ids = m_generation_config.stop_token_ids;
    // If eos_token_id was not provided, take value from default m_generation_config
    if (config.eos_token_id == -1)
        config.set_eos_token_id(m_generation_config.eos_token_id);
    config.validate();
    return config;
}

DecodedResults StatefulLLMPipeline::get_decoded_results(
    TokenizedInputs encoded_input,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer,
    std::chrono::steady_clock::time_point start_time,
    std::chrono::steady_clock::time_point tokenization_start_time,
    std::optional<float> chat_template_duration_us
) {
    auto encode_stop_time =  std::chrono::steady_clock::now();
    auto encoded_results = generate(encoded_input, generation_config, streamer);

    auto decode_start_time =  std::chrono::steady_clock::now();
    DecodedResults decoded_results;
    decoded_results.texts = m_tokenizer.decode(encoded_results.tokens);
    decoded_results.scores = encoded_results.scores;
    decoded_results.finish_reasons = encoded_results.finish_reasons;
    auto decode_stop_time =  std::chrono::steady_clock::now();

    // generate_durations
    decoded_results.perf_metrics = encoded_results.perf_metrics;

    auto& raw_counters = decoded_results.perf_metrics.raw_metrics;
    auto stop_time = std::chrono::steady_clock::now();
    raw_counters.generate_durations.clear();
    raw_counters.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
    raw_counters.tokenization_durations.emplace_back(PerfMetrics::get_microsec(encode_stop_time - tokenization_start_time));
    if (chat_template_duration_us.has_value()) {
        raw_counters.chat_template_durations.emplace_back(*chat_template_duration_us);
    }
    raw_counters.detokenization_durations.emplace_back(PerfMetrics::get_microsec(decode_stop_time - decode_start_time));

    // Added tokenization/detokenization times, and updated generate duration, need to reevaluate statistics.
    decoded_results.perf_metrics.m_evaluated = false;
    decoded_results.perf_metrics.evaluate_statistics(start_time);
    return decoded_results;
}

DecodedResults StatefulLLMPipeline::generate(
    StringInputs inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer) {
    if (is_chat_conversation && m_chat_input_type == ov::genai::utils::GenerationChatInputsType::UNDEF)
        m_chat_input_type = ov::genai::utils::GenerationChatInputsType::STRING;

    if (is_chat_conversation)
        OPENVINO_ASSERT(m_chat_input_type == ov::genai::utils::GenerationChatInputsType::STRING,
                        "Chat doesn't support switching between input types. Please, continue using StringInputs or restart the chat.");

    auto start_time = std::chrono::steady_clock::now();

    GenerationConfig config = resolve_generation_config(generation_config);

    TokenizedInputs encoded_input;
    auto tokenization_start_time = start_time;
    std::optional<float> chat_template_duration_us;

    if (auto input_vector = std::get_if<std::vector<std::string>>(&inputs)) {
        if (is_chat_conversation) {
            OPENVINO_ASSERT(input_vector->size() == 1, "Can't chat with multiple prompts");
            m_history.push_back({{"role", "user"}, {"content", (*input_vector)[0]}});
            constexpr bool add_generation_prompt = true;
            const auto template_start_time = std::chrono::steady_clock::now();
            auto new_templated_chat_history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
            chat_template_duration_us = PerfMetrics::get_microsec(std::chrono::steady_clock::now() - template_start_time);
            tokenization_start_time = std::chrono::steady_clock::now();
            auto new_chat_tokens = m_tokenizer.encode(new_templated_chat_history, ov::genai::add_special_tokens(false));

            m_force_longrope_reprefill = should_force_longrope_reprefill(new_chat_tokens.input_ids.get_size());
            if (m_use_full_chat_history || m_force_longrope_reprefill) {
                encoded_input = new_chat_tokens;
            } else {
                ov::genai::align_cache_and_history(new_chat_tokens.input_ids, m_cache_state);
                encoded_input = get_chat_encoded_input(new_chat_tokens.input_ids, m_cache_state);
            }
        } else if (config.apply_chat_template && !m_tokenizer.get_chat_template().empty()) {
            std::vector<std::string> templated_input_vector;
            for (auto& input : *input_vector) {
                ChatHistory history({{{"role", "user"}, {"content", input}}});
                constexpr bool add_generation_prompt = true;
                const auto template_start_time = std::chrono::steady_clock::now();
                auto templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
                chat_template_duration_us = chat_template_duration_us.value_or(0.0f) +
                                            PerfMetrics::get_microsec(std::chrono::steady_clock::now() - template_start_time);
                templated_input_vector.push_back(templated_prompt);
            }
            tokenization_start_time = std::chrono::steady_clock::now();
            encoded_input = m_tokenizer.encode(templated_input_vector, ov::genai::add_special_tokens(false));
        } else {
            tokenization_start_time = std::chrono::steady_clock::now();
            encoded_input = m_tokenizer.encode(*input_vector, ov::genai::add_special_tokens(true));
        }
    } else if (auto input_prompt = std::get_if<std::string>(&inputs)) {
        std::string& prompt = *input_prompt;

        if (is_chat_conversation) {
            m_history.push_back({{"role", "user"}, {"content", prompt}});
            constexpr bool add_generation_prompt = true;
            const auto template_start_time = std::chrono::steady_clock::now();
            auto new_templated_chat_history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
            chat_template_duration_us = PerfMetrics::get_microsec(std::chrono::steady_clock::now() - template_start_time);
            tokenization_start_time = std::chrono::steady_clock::now();
            // Do not add special tokens in chat scenario to be aligned with HF.
            auto new_chat_tokens = m_tokenizer.encode(new_templated_chat_history, ov::genai::add_special_tokens(false));

            m_force_longrope_reprefill = should_force_longrope_reprefill(new_chat_tokens.input_ids.get_size());
            if (m_use_full_chat_history || m_force_longrope_reprefill) {
                encoded_input = new_chat_tokens;
            } else {
                ov::genai::align_cache_and_history(new_chat_tokens.input_ids, m_cache_state);
                encoded_input = get_chat_encoded_input(new_chat_tokens.input_ids, m_cache_state);
            }
            // TODO: Forbid LoRA config change if we are in the chat mode, because it requires regenerating the history with LoRA applied
        } else {
            if (config.apply_chat_template && !m_tokenizer.get_chat_template().empty()) {
                ChatHistory history({{{"role", "user"}, {"content", prompt}}});
                constexpr bool add_generation_prompt = true;
                const auto template_start_time = std::chrono::steady_clock::now();
                auto templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
                chat_template_duration_us = PerfMetrics::get_microsec(std::chrono::steady_clock::now() - template_start_time);
                tokenization_start_time = std::chrono::steady_clock::now();
                encoded_input = m_tokenizer.encode(templated_prompt, ov::genai::add_special_tokens(false));
            } else {
                // in case when chat_template was not found in tokenizer_config.json or set
                tokenization_start_time = std::chrono::steady_clock::now();
                encoded_input = m_tokenizer.encode(prompt, ov::genai::add_special_tokens(true));
            }
        }
    }

    DecodedResults decoded_results = get_decoded_results(
        encoded_input,
        config,
        streamer,
        start_time,
        tokenization_start_time,
        chat_template_duration_us
    );

    if (is_chat_conversation) {
        if (m_chat_generation_finish_status == ov::genai::GenerationStatus::CANCEL) {
            // If chat generation process was cancelled by user, let's rollback to previous state of history
            m_history.pop_back();
        } else {
            // Tail of chat template is missing in KV cache.
            // Find the tail to concatenate it with the next input prompt.
            auto answer = decoded_results.texts[0];
            m_history.push_back({{"role", "assistant"}, {"content", std::move(answer)}});
        }
    }

    return decoded_results;
}

DecodedResults StatefulLLMPipeline::generate(
    const ChatHistory& history,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {
    is_chat_conversation = true;

    if (is_chat_conversation && m_chat_input_type == ov::genai::utils::GenerationChatInputsType::UNDEF)
        m_chat_input_type = ov::genai::utils::GenerationChatInputsType::CHAT_HISTORY;

    if (is_chat_conversation)
        OPENVINO_ASSERT(m_chat_input_type == ov::genai::utils::GenerationChatInputsType::CHAT_HISTORY,
                        "Chat doesn't support switching between input types. Please, continue using ChatHistory or restart the chat.");

    auto start_time = std::chrono::steady_clock::now();

    GenerationConfig config = resolve_generation_config(generation_config);

    OPENVINO_ASSERT(config.apply_chat_template, "Chat template must be applied when using ChatHistory in generate method.");
    OPENVINO_ASSERT(!m_tokenizer.get_chat_template().empty(), "Chat template must not be empty when using ChatHistory in generate method.");
    OPENVINO_ASSERT(!history.empty(), "Chat history must not be empty when using ChatHistory in generate method.");

    bool is_history_continuation = history.size() > m_history.size();
    if (is_history_continuation) {
        for (size_t i = 0; i < m_history.size(); ++i) {
            if (history[i] != m_history[i]) {
                is_history_continuation = false;
                break;
            }
        }
    }

    if (!is_history_continuation) {
        reset_state();
        m_model_runner.get_tensor("attention_mask").set_shape({1, 0});
        m_cache_state.reset_state();
        // A brand new/unrelated conversation - any pending LongRoPE reprefill obligation from
        // whatever conversation was previously in the cache no longer applies.
        m_longrope_reprefill_pending = false;
    }

    m_history = history;

    constexpr bool add_generation_prompt = true;
    const auto template_start_time = std::chrono::steady_clock::now();
    auto new_templated_chat_history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
    const auto tokenization_start_time = std::chrono::steady_clock::now();
    auto new_chat_tokens = m_tokenizer.encode(new_templated_chat_history, ov::genai::add_special_tokens(false));

    TokenizedInputs encoded_input;
    m_force_longrope_reprefill = should_force_longrope_reprefill(new_chat_tokens.input_ids.get_size());
    if (m_use_full_chat_history || m_force_longrope_reprefill) {
        encoded_input = new_chat_tokens;
    } else {
        ov::genai::align_cache_and_history(new_chat_tokens.input_ids, m_cache_state);
        encoded_input = get_chat_encoded_input(new_chat_tokens.input_ids, m_cache_state);
    }
    return get_decoded_results(
        encoded_input,
        config,
        streamer,
        start_time,
        tokenization_start_time,
        PerfMetrics::get_microsec(tokenization_start_time - template_start_time)
    );
}

EncodedResults StatefulLLMPipeline::generate(
    const EncodedInputs& inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer) {
    if (is_chat_conversation && m_chat_input_type == ov::genai::utils::GenerationChatInputsType::UNDEF)
        m_chat_input_type = ov::genai::utils::GenerationChatInputsType::ENCODED_INPUTS;

    if (is_chat_conversation)
        // if chat was run in StringInputs mode, but it was called EncodedInputs generate, last m_history entry will be with assistant role
        OPENVINO_ASSERT(m_chat_input_type == ov::genai::utils::GenerationChatInputsType::ENCODED_INPUTS || m_history.last()["role"] == "user",
                        "Chat doesn't support switching between input types. Please, continue using StringInputs or restart the chat.");

    // Set by the StringInputs/ChatHistory overloads before delegating here; consumed once so
    // it doesn't leak into unrelated future calls.
    bool force_longrope_reprefill = m_force_longrope_reprefill;
    m_force_longrope_reprefill = false;

    if (!is_chat_conversation) {
        reset_state();
        m_model_runner.get_tensor("attention_mask").set_shape({1, 0});
        m_cache_state.reset_state();
        m_longrope_reprefill_pending = false;
    }

    auto start_time = std::chrono::steady_clock::now();
    ov::Tensor input_ids;
    ov::Tensor attention_mask;
    if (auto data = std::get_if<ov::Tensor>(&inputs)) {
        if (m_is_npu) {
            // Prefill model in NPU is reshaped to NPUW_LLM_MAX_PROMPT_LEN x NPUW_LLM_MAX_PROMPT_LEN
            OPENVINO_ASSERT(data->get_size() <= m_max_prompt_len,
                "Stateful LLM pipeline on NPU may only process prompts or hold chat history up to ",
                m_max_prompt_len,
                " tokens. ",
                data->get_size(),
                " is passed.\n Set the \"MAX_PROMPT_LEN\" config option to increase the limit.");
        }
        input_ids = ov::Tensor(data->get_element_type(), data->get_shape());
        data->copy_to(input_ids);
        attention_mask = ov::genai::utils::init_attention_mask(input_ids);
    } else if (auto data = std::get_if<TokenizedInputs>(&inputs)) {
        if (m_is_npu) {
            // Prefill model in NPU is reshaped to NPUW_LLM_MAX_PROMPT_LEN x NPUW_LLM_MAX_PROMPT_LEN
            OPENVINO_ASSERT(data->input_ids.get_size() <= m_max_prompt_len,
                "Stateful LLM pipeline on NPU may only process prompts or hold chat history up to ",
                m_max_prompt_len,
                " tokens. ",
                data->input_ids.get_size(),
                " is passed.\n Set the \"MAX_PROMPT_LEN\" config option to increase the limit.");
        }
        input_ids = ov::Tensor(data->input_ids.get_element_type(), data->input_ids.get_shape());
        data->input_ids.copy_to(input_ids);

        attention_mask = ov::Tensor{data->attention_mask.get_element_type(), data->attention_mask.get_shape()};
        data->attention_mask.copy_to(attention_mask);
    }

    if (is_chat_conversation && m_chat_input_type == ov::genai::utils::GenerationChatInputsType::ENCODED_INPUTS) {
        std::copy(input_ids.data<int64_t>(), input_ids.data<int64_t>() + input_ids.get_size(), std::back_inserter(m_tokenized_chat_history));
        // Called directly with EncodedInputs (not via StringInputs/ChatHistory), so compute
        // the crossing check here instead of relying on the caller to have set it.
        force_longrope_reprefill = force_longrope_reprefill || should_force_longrope_reprefill(m_tokenized_chat_history.size());
    }

    size_t real_input_ids_size = input_ids.get_shape().at(1);

    if (is_chat_conversation && (m_use_full_chat_history || force_longrope_reprefill))
        m_cache_state.reset_state();

    // Tail of previous output in chat mode is missing in KV cache.
    if (is_chat_conversation && m_chat_input_type == ov::genai::utils::GenerationChatInputsType::ENCODED_INPUTS) {
        ov::Tensor new_chat_tokens = ov::Tensor{ov::element::i64, {1, m_tokenized_chat_history.size()}, m_tokenized_chat_history.data()};
        ov::genai::align_cache_and_history(new_chat_tokens, m_cache_state);

        auto encoded_input = get_chat_encoded_input(new_chat_tokens, m_cache_state);
        input_ids = encoded_input.input_ids;
        attention_mask = encoded_input.attention_mask;
    }

    GenerationConfig config = resolve_generation_config(generation_config);

    auto batch_size = input_ids.get_shape().at(0);

    if (m_is_npu) {
        OPENVINO_ASSERT(batch_size == 1u, "Currently only batch size equal to 1 is supported for NPU device!");
        OPENVINO_ASSERT(config.is_greedy_decoding() || config.is_multinomial(),
            "Currently only greedy and multinomial decoding are supported for NPU device!");
        OPENVINO_ASSERT(config.num_return_sequences == 1u,
            "Currently only \"num_return_sequences\" equal to 1 is supported for NPU device!");
    }

    // Stateful pipeline does not provide logprobs for prompt tokens
    OPENVINO_ASSERT(config.echo == false, "Echo is not supported in the stateful pipeline");

    std::shared_ptr<StreamerBase> streamer_ptr = ov::genai::utils::create_streamer(streamer, m_tokenizer);

    OPENVINO_ASSERT(streamer_ptr == nullptr || batch_size == 1 && config.num_return_sequences == 1 &&
        (config.is_greedy_decoding() || config.is_multinomial()),
        "Currently streaming is possible only with batch size=1 and only for greedy or multinomial decoding");

    auto num_inputs = m_model_runner.get_compiled_model().inputs().size();
    OPENVINO_ASSERT(num_inputs == 4 || num_inputs == 3, "Model should have 3 or 4 inputs: "
                    "either (input_ids, attention_mask, beam_idx) or "
                    "(input_ids, attention_mask, position_ids, beam_idx) "
                    "but you have '" + std::to_string(num_inputs) + "' inputs");

    if (is_chat_conversation) {
        if (m_use_full_chat_history || force_longrope_reprefill)
            reset_state();
        else
            ov::genai::utils::trim_kv_cache(m_model_runner, m_cache_state, m_adapter_controller);
    }

    size_t cache_len = 0;
    ov::Tensor concatenated_attention_mask;
    if (is_chat_conversation && !m_cache_state.get_state().empty() && !(m_use_full_chat_history || force_longrope_reprefill)) {
        OPENVINO_ASSERT(batch_size == 1, "continuation of generation is possible only for batch 1");
        // If history is saved in KV cache, concatenate new attention_mask with the already existing.
        // Between subsequent runs attention_mask should not be modified.
        auto atten_mask_history = m_model_runner.get_tensor("attention_mask");
        auto prompt_len = attention_mask.get_shape()[1];

        cache_len = m_cache_state.get_state().size();

        ov::Tensor new_atten_mask = ov::Tensor{ov::element::i64, {batch_size, cache_len + prompt_len}};
        auto start_atten_hst = atten_mask_history.data<int64_t>();

        std::copy(start_atten_hst, start_atten_hst + cache_len,
                new_atten_mask.data<int64_t>());
        std::copy(attention_mask.data<int64_t>(), attention_mask.data<int64_t>() + prompt_len,
                new_atten_mask.data<int64_t>() + cache_len);
        concatenated_attention_mask = new_atten_mask;
    } else {
        concatenated_attention_mask = attention_mask;
    }

    size_t prev_attn_mask_size = concatenated_attention_mask.get_shape()[1];

    bool position_ids_available = (num_inputs == 4);
    std::optional<ov::Tensor> position_ids = std::nullopt;
    if (position_ids_available) {
        position_ids = ov::Tensor{ov::element::i64, input_ids.get_shape()};
        utils::initialize_position_ids(*position_ids, attention_mask, cache_len);
    }

    if(m_adapter_controller) {
        m_adapter_controller->apply(m_model_runner, config.adapters);
    }

    std::vector<SequenceGroup::Ptr> requests;

    for (size_t request_id = 0; request_id < batch_size; request_id++) {
        SequenceGroup::Ptr sequence_group;
        if (is_chat_conversation) {
            std::vector<int64_t>& state = m_cache_state.get_state();
            std::vector<int64_t> tokenized_chat_hist;
            tokenized_chat_hist.reserve(state.size() + input_ids.get_size());
            std::copy(state.begin(), state.end(), std::back_inserter(tokenized_chat_hist));
            std::copy(input_ids.data<int64_t>(), input_ids.data<int64_t>() + input_ids.get_size(), std::back_inserter(tokenized_chat_hist));
            sequence_group = std::make_shared<SequenceGroup>(request_id,  ov::Tensor(ov::element::i64, {1, tokenized_chat_hist.size()}, tokenized_chat_hist.data()), config);
        } else {
            size_t seq_len = input_ids.get_shape().at(1);
            size_t batch_offset = request_id * seq_len;
            const int64_t* prompt_start = input_ids.data<const int64_t>() + batch_offset;
            std::vector<int64_t> tokenized_prompt(prompt_start, prompt_start + seq_len);

            sequence_group = std::make_shared<SequenceGroup>(request_id, tokenized_prompt, config);
        }

        requests.push_back(sequence_group);
    }

    if (m_sampler.get_seed() != config.rng_seed) {
        m_sampler.set_seed(config.rng_seed);
    }

    std::shared_ptr<LongRopeThresholdStreamer> threshold_streamer;
    std::shared_ptr<StreamerBase> effective_streamer_ptr = streamer_ptr;
    if (m_is_npu && m_longrope_threshold.has_value()) {
        threshold_streamer = std::make_shared<LongRopeThresholdStreamer>(
            streamer_ptr, concatenated_attention_mask.get_shape().at(1), *m_longrope_threshold);
        effective_streamer_ptr = threshold_streamer;
    }

    ov::genai::utils::GenerationFinishInfo finish_info = get_lm_encoded_results(m_model_runner, input_ids, concatenated_attention_mask, effective_streamer_ptr, m_sampler,
                                                                                requests, position_ids, std::nullopt, m_cache_state, nullptr, std::nullopt, m_max_kv_cache_size);

    if (threshold_streamer && threshold_streamer->threshold_reached()) {
        // Generation was stopped right after the token at the LongRoPE threshold position
        // (that token was already sampled - keep it). Reset the KV cache and re-run a full
        // prefill over the whole history (original prompt + everything generated so far) so
        // all keys are consistently re-encoded under the "long" RoPE factor, then continue
        // generating the remaining token budget.
        TokenIds generated_so_far = finish_info.results.tokens.at(0);
        TokenIds full_ids = requests.at(0)->get_prompt_ids();
        full_ids.insert(full_ids.end(), generated_so_far.begin(), generated_so_far.end());

        size_t total_max_new_tokens = requests.at(0)->get_max_new_tokens();
        size_t remaining_new_tokens = total_max_new_tokens > generated_so_far.size() ?
            total_max_new_tokens - generated_so_far.size() : 0;

        if (remaining_new_tokens == 0) {
            // The threshold-crossing token was also the last one allowed by max_new_tokens, so
            // no continuation phase runs below - the KV cache is left holding that token still
            // encoded under the pre-crossing RoPE factor. Nothing else will call end() on the
            // wrapped streamer, so flush it now. Record that a reprefill is still owed so the
            // very next call for this conversation forces one (see should_force_longrope_reprefill()).
            threshold_streamer->flush_end();
            m_longrope_reprefill_pending = true;
        } else {
            PerfMetrics first_phase_metrics = finish_info.results.perf_metrics;

            reset_state();
            m_cache_state.reset_state();

            ov::Tensor continuation_input_ids(ov::element::i64, {1, full_ids.size()}, full_ids.data());
            ov::Tensor continuation_attention_mask = utils::init_attention_mask(continuation_input_ids);
            std::optional<ov::Tensor> continuation_position_ids = std::nullopt;
            if (position_ids_available) {
                continuation_position_ids = ov::Tensor{ov::element::i64, continuation_input_ids.get_shape()};
                utils::initialize_position_ids(*continuation_position_ids, continuation_attention_mask);
            }

            GenerationConfig continuation_config = config;
            continuation_config.max_new_tokens = remaining_new_tokens;
            std::vector<SequenceGroup::Ptr> continuation_requests{
                std::make_shared<SequenceGroup>(0, full_ids, continuation_config)
            };

            finish_info = get_lm_encoded_results(m_model_runner, continuation_input_ids, continuation_attention_mask, effective_streamer_ptr, m_sampler,
                                                  continuation_requests, continuation_position_ids, std::nullopt, m_cache_state, nullptr, std::nullopt, m_max_kv_cache_size);

            // PerfMetrics::operator+ does not merge m_new_token_times, so capture the
            // continuation's entries before they're overwritten by the merge below and
            // re-append them afterwards.
            auto continuation_new_token_times = finish_info.results.perf_metrics.raw_metrics.m_new_token_times;

            // Stitch tokens/perf-metrics generated before the reset with the post-reset
            // continuation.
            finish_info.results.tokens[0].insert(finish_info.results.tokens[0].begin(), generated_so_far.begin(), generated_so_far.end());
            finish_info.results.perf_metrics = first_phase_metrics + finish_info.results.perf_metrics;

            auto& merged_new_token_times = finish_info.results.perf_metrics.raw_metrics.m_new_token_times;
            merged_new_token_times.insert(merged_new_token_times.end(),
                continuation_new_token_times.begin(), continuation_new_token_times.end());

            // m_inference_durations is a single running total per get_lm_encoded_results()
            // call, so operator+ above left it holding both phases' totals as two separate
            // entries, which calc_mean_and_std() would average instead of sum. Collapse it
            // back into one summed entry.
            auto& merged_inference_durations = finish_info.results.perf_metrics.raw_metrics.m_inference_durations;
            if (merged_inference_durations.size() > 1) {
                MicroSeconds total_inference_duration{0.0f};
                for (const auto& duration : merged_inference_durations)
                    total_inference_duration += duration;
                merged_inference_durations = {total_inference_duration};
            }
        }
    }

    ov::genai::EncodedResults& result = finish_info.results;
    m_chat_generation_finish_status = finish_info.streaming_finish_status;

    if (is_chat_conversation) {
        m_cache_state.num_tokens_to_trim = 0;

        if (m_chat_input_type == ov::genai::utils::GenerationChatInputsType::ENCODED_INPUTS) {
            if (m_chat_generation_finish_status == ov::genai::GenerationStatus::CANCEL) {
                m_tokenized_chat_history.resize(m_tokenized_chat_history.size() - real_input_ids_size);
            } else {
                std::copy(result.tokens[0].begin(), result.tokens[0].end(), std::back_inserter(m_tokenized_chat_history));
            }
        }
        if (config.is_beam_search()) {
            m_cache_state.num_tokens_to_trim = m_model_runner.get_tensor("attention_mask").get_shape()[1] - prev_attn_mask_size;
        }
    }

    auto stop_time = std::chrono::steady_clock::now();

    // If is called without tokenization then that stat will not be reported.
    auto& metrics = result.perf_metrics;
    metrics.num_input_tokens = batch_size * input_ids.get_shape().at(1);
    metrics.load_time = m_load_time_ms;
    metrics.raw_metrics.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
    metrics.evaluate_statistics(start_time);
    return result;
}

void StatefulLLMPipeline::start_chat(const std::string& system_message) {
    finish_chat();
    is_chat_conversation = true;

    if (system_message.empty())
        return;

    m_history.push_back({{"role", "system"}, {"content", system_message}});
}

void StatefulLLMPipeline::reset_state() {
    if(m_adapter_controller) {
        for(auto& state: m_model_runner.query_state()) {
            if(!m_adapter_controller->has_state_name(state.get_name())) {
                state.reset();
            }
        }
    } else {
        m_model_runner.reset_state();
    }
}

void StatefulLLMPipeline::finish_chat() {
    is_chat_conversation = false;
    m_chat_input_type = ov::genai::utils::GenerationChatInputsType::UNDEF;
    bool have_state = 0 != m_model_runner.get_tensor("attention_mask").get_size();
    if (!m_cache_state.get_state().empty() || have_state) {
        reset_state();
        m_model_runner.get_tensor("attention_mask").set_shape({1, 0});
        m_history.clear();
        m_tokenized_chat_history.clear();
        m_cache_state.reset_state();
        m_longrope_reprefill_pending = false;
    }
}

StatefulLLMPipeline::~StatefulLLMPipeline() {
    m_model_runner.get_compiled_model().release_memory();
}

} // namespace ov::genai
