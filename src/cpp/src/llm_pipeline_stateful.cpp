
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "llm_pipeline_stateful.hpp"

#include "lora_helper.hpp"
#include "lm_encoding.hpp"
#include "openvino/genai/text_streamer.hpp"

#include "utils.hpp"

namespace ov::genai {

StatefulLLMPipeline::StatefulLLMPipeline(
    const ov::InferRequest& request,
    const ov::genai::Tokenizer& tokenizer,
    OptionalGenerationConfig generation_config)
    : LLMPipelineImplBase(tokenizer, generation_config.has_value() ? *generation_config : GenerationConfig()),
    m_model_runner(request) {}

StatefulLLMPipeline::StatefulLLMPipeline(
    const std::filesystem::path& models_path,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& properties)
    : StatefulLLMPipeline{
        utils::singleton_core().read_model(models_path / "openvino_model.xml", {}, properties),
        tokenizer,
        device,
        properties,
        utils::from_config_json_if_exists(models_path)
    } {}

StatefulLLMPipeline::StatefulLLMPipeline(
    const std::shared_ptr<ov::Model>& model,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& properties,
    const ov::genai::GenerationConfig& generation_config)
    : LLMPipelineImplBase(tokenizer, generation_config), m_sampler(m_tokenizer) {
    utils::apply_slice_before_matmul_transformation(model);
    m_kv_history_manager.kv_cache_seq_length_axis = ov::genai::utils::get_kv_axes_pos(model).seq_len;

    auto filtered_properties = extract_adapters_from_properties(properties, &m_generation_config.adapters);
    if (m_generation_config.adapters) {
        m_generation_config.adapters->set_tensor_name_prefix("base_model.model.");
        m_adapter_controller = AdapterController(model, *m_generation_config.adapters, device);   // TODO: Make the prefix name configurable
    }
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(model, device, *filtered_properties);
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
    : StatefulLLMPipeline{models_path, Tokenizer(models_path), device, plugin_config} {}

DecodedResults StatefulLLMPipeline::generate(
    StringInputs inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer) {
    if (is_chat_conversation && m_chat_input_type == ov::genai::utils::GenerationChatInputsType::UNDEF)
        m_chat_input_type = ov::genai::utils::GenerationChatInputsType::STRING;

    if (is_chat_conversation)
        OPENVINO_ASSERT(m_chat_input_type != ov::genai::utils::GenerationChatInputsType::ENCODED_INPUTS,
                        "Chat doesn't support switching between input types. Please, continue using EncodedInputs or restart the chat.");

    auto start_time = std::chrono::steady_clock::now();
    GenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;
    // If stop_token_ids were not provided, take value from default m_generation_config
    if (config.stop_token_ids.empty())
        config.stop_token_ids = m_generation_config.stop_token_ids;
    // If eos_token_id was not provided, take value from default m_generation_config
    if (config.eos_token_id == -1)
        config.set_eos_token_id(m_generation_config.eos_token_id);
    config.validate();

    TokenizedInputs encoded_input;

    std::string prev_templated_chat_history(m_templated_chat_history);
    std::vector<int64_t> prev_tokenized_chat_history(m_tokenized_chat_history);

    if (auto input_vector = std::get_if<std::vector<std::string>>(&inputs)) {
        OPENVINO_ASSERT(!is_chat_conversation, "Can't chat with multiple prompts");
        if (config.apply_chat_template && !m_tokenizer.get_chat_template().empty()) {
            std::vector<std::string> templated_input_vector;
            for (auto& input : *input_vector) {
                ChatHistory history({{{"role", "user"}, {"content", input}}});
                constexpr bool add_generation_prompt = true;
                auto templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
                templated_input_vector.push_back(templated_prompt);
            }
            encoded_input = m_tokenizer.encode(templated_input_vector, ov::genai::add_special_tokens(false));
        } else {
            encoded_input = m_tokenizer.encode(*input_vector, ov::genai::add_special_tokens(true));
        }
    } else if (auto input_prompt = std::get_if<std::string>(&inputs)) {
        std::string& prompt = *input_prompt;

        if (is_chat_conversation) {
            // KV cache in model already contains prompts and answers from previous iterations.
            // So only new prompt wrapped into chat template to be sent into model. Tokenizer always returns
            // token_ids = {<bos token>, ...<valuable tokens>}. So if tokenizer applies only to the new prompt,
            // <bos token> will be inserted on every iteration.
            // So actual pipeline calculates input_ids for whole chat history + for whole chat history without the new prompt
            // and takes only the difference between them.
            // The chat history cannot be saved as already encoded tokens because generate call doesn't return <eos> token, but
            // KV cache contains it. So we have to add it manually or get it by tokenization all chat history.

            m_history.push_back({{"role", "user"}, {"content", prompt}});
            constexpr bool add_generation_prompt = true;
            auto new_templated_chat_history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
            // Do not add special tokens in chat scenario to be aligned with HF.
            auto new_chat_tokens = m_tokenizer.encode(new_templated_chat_history, ov::genai::add_special_tokens(false));
            auto prev_chat_tokens = m_tokenizer.encode(m_templated_chat_history, ov::genai::add_special_tokens(false));

            // some symbols combinations can be encoded by the tokenizer in different ways
            // if we met sequence with such combination of symbols, we cannot correctly subtract the new history from the old history
            // so let's check it out, find the trusted part and use it in on the next step
            size_t trusted_history_length = 0;
            if (!m_tokenized_chat_history.empty()) {
                std::set<int64_t> stop_tokens = config.stop_token_ids;
                trusted_history_length = ov::genai::utils::get_first_history_difference(prev_chat_tokens.input_ids, m_tokenized_chat_history, stop_tokens);
            }

            if (m_tokenized_chat_history.empty()) {
                encoded_input = new_chat_tokens;
            } else if (trusted_history_length != SIZE_MAX || m_kv_history_manager.does_history_cache_need_to_update()) {
                // does_history_cache_need_to_update will be true here if beam search is activated
                // in beam search mode we want to remove all history about last model answer from kv cache and add the best answer directly
                // if we have difference in model answer and decoded answer it anyway will be less then entire history, so let's use data from m_kv_history_manager
                if (m_kv_history_manager.does_history_cache_need_to_update()) {
                    trusted_history_length = m_kv_history_manager.trusted_history_length;
                } else {
                    size_t num_tokens_to_remove_from_kv_cache = m_tokenized_chat_history.size() - trusted_history_length;
                    // last generated token is present in tokenized_history, but not included to attention mask, let's keep it in historyt
                    num_tokens_to_remove_from_kv_cache -= 1;

                    // if streaming was used and cancelled on prev step, m_kv_history_manager.num_tokens_to_remove_from_kv_cache could be already set
                    // and it would be bigger as it includes answer + prompt
                    m_kv_history_manager.num_tokens_to_remove_from_kv_cache = m_kv_history_manager.num_tokens_to_remove_from_kv_cache > num_tokens_to_remove_from_kv_cache ?
                                                                              m_kv_history_manager.num_tokens_to_remove_from_kv_cache : num_tokens_to_remove_from_kv_cache;
                }

                ov::Tensor new_tensor = ov::Tensor(new_chat_tokens.input_ids.get_element_type(),
                                                    {1, new_chat_tokens.input_ids.get_shape().at(1) - trusted_history_length},
                                                    new_chat_tokens.input_ids.data<int64_t>() + trusted_history_length);

                ov::Tensor new_attention_mask(ov::element::i64, new_tensor.get_shape());
                std::fill_n(new_attention_mask.data<int64_t>(), new_tensor.get_shape()[1], 1);

                encoded_input.input_ids = ov::Tensor(new_chat_tokens.input_ids.get_element_type(),
                                                    {1, new_chat_tokens.input_ids.get_shape().at(1) - trusted_history_length});
                new_tensor.copy_to(encoded_input.input_ids);
                encoded_input.attention_mask = new_attention_mask;
                m_last_disappeared_token = std::nullopt;
                m_kv_history_manager.reset_kv_cache = (trusted_history_length == 0);
            } else {
                encoded_input = utils::subtract_chat_tokenized_inputs(new_chat_tokens, prev_chat_tokens);
            }
            m_templated_chat_history = new_templated_chat_history;

            m_tokenized_chat_history.clear();
            m_tokenized_chat_history.reserve(new_chat_tokens.input_ids.get_size());
            std::copy_n(new_chat_tokens.input_ids.data<int64_t>(), new_chat_tokens.input_ids.get_size(),
                        std::back_inserter(m_tokenized_chat_history));

            // TODO: Forbid LoRA config change if we are in the chat mode, because it requires regenerating the history with LoRA applied
        } else {
            std::string& prompt = *input_prompt;
            if (config.apply_chat_template && !m_tokenizer.get_chat_template().empty()) {
                ChatHistory history({{{"role", "user"}, {"content", prompt}}});
                constexpr bool add_generation_prompt = true;
                auto templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
                encoded_input = m_tokenizer.encode(templated_prompt, ov::genai::add_special_tokens(false));
            } else {
                // in case when chat_template was not found in tokenizer_config.json or set
                encoded_input = m_tokenizer.encode(prompt, ov::genai::add_special_tokens(true));
            }
        }
    }

    auto encode_stop_time =  std::chrono::steady_clock::now();
    auto encoded_results = generate(encoded_input, config, streamer);

    auto decode_start_time =  std::chrono::steady_clock::now();
    DecodedResults decoded_results = {m_tokenizer.decode(encoded_results.tokens), encoded_results.scores};
    auto decode_stop_time =  std::chrono::steady_clock::now();

    if (is_chat_conversation) {
        if (m_chat_generation_finish_status == ov::genai::GenerationStatus::CANCEL) {
            // If chat generation process was cancelled by user, let's rollback to previous state of history
            m_history.pop_back();
            m_kv_history_manager.num_tokens_to_remove_from_kv_cache += m_tokenized_chat_history.size() - prev_tokenized_chat_history.size();
            m_templated_chat_history = std::move(prev_templated_chat_history);
            m_tokenized_chat_history = std::move(prev_tokenized_chat_history);
            m_kv_history_manager.reset_kv_cache = m_tokenized_chat_history.empty();
        } else {
            // Tail of chat template is missing in KV cache.
            // Find the tail to concatenate it with the next input prompt.
            auto answer = decoded_results.texts[0];
            m_templated_chat_history.append(answer);
            m_history.push_back({{"role", "assistant"}, {"content", std::move(answer)}});
        }
    }

    // generate_durations
    decoded_results.perf_metrics = encoded_results.perf_metrics;

    auto& raw_counters = decoded_results.perf_metrics.raw_metrics;
    auto stop_time = std::chrono::steady_clock::now();
    raw_counters.generate_durations.clear();
    raw_counters.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
    raw_counters.tokenization_durations.emplace_back(PerfMetrics::get_microsec(encode_stop_time - start_time));
    raw_counters.detokenization_durations.emplace_back(PerfMetrics::get_microsec(decode_stop_time - decode_start_time));

    // Added tokenization/detokenization times, and updated generate duration, need to reevaluate statistics.
    decoded_results.perf_metrics.m_evaluated = false;
    decoded_results.perf_metrics.evaluate_statistics(start_time);
    return decoded_results;
}

EncodedResults StatefulLLMPipeline::generate(
    const EncodedInputs& inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer) {
    if (is_chat_conversation && m_chat_input_type == ov::genai::utils::GenerationChatInputsType::UNDEF)
        m_chat_input_type = ov::genai::utils::GenerationChatInputsType::ENCODED_INPUTS;

    if (is_chat_conversation)
        // if chat was run in StringInputs mode, but it was called EncodedInputs generate, last m_history entry will be with assistant role
        OPENVINO_ASSERT(m_chat_input_type == ov::genai::utils::GenerationChatInputsType::ENCODED_INPUTS || m_history.back()["role"] == "user",
                        "Chat doesn't support switching between input types. Please, continue using StringInputs or restart the chat.");

    if (!is_chat_conversation) {
        reset_kv_state();
        m_model_runner.get_tensor("attention_mask").set_shape({1, 0});
    }

    auto start_time = std::chrono::steady_clock::now();
    ov::Tensor input_ids;
    ov::Tensor attention_mask;
    if (auto data = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *data;
        attention_mask = ov::genai::utils::init_attention_mask(input_ids);
    } else if (auto data = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = data->input_ids;
        attention_mask = data->attention_mask;
    }

    if (is_chat_conversation && m_chat_input_type == ov::genai::utils::GenerationChatInputsType::ENCODED_INPUTS)
        std::copy(input_ids.data<int64_t>(), input_ids.data<int64_t>() + input_ids.get_size(), std::back_inserter(m_tokenized_chat_history));

    size_t real_input_ids_size = input_ids.get_shape().at(1);

    // Tail of previous output in chat mode is missing in KV cache.
    if (m_last_disappeared_token.has_value()) {
        attention_mask = ov::genai::utils::push_front_inputs(attention_mask, 1);
        input_ids = ov::genai::utils::push_front_inputs(input_ids, *m_last_disappeared_token);
    }

    GenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;

    // If stop_token_ids were not provided, take value from default m_generation_config
    if (config.stop_token_ids.empty())
        config.stop_token_ids = m_generation_config.stop_token_ids;
    // If eos_token_id was not provided, take value from default m_generation_config
    if (config.eos_token_id == -1)
        config.set_eos_token_id(m_generation_config.eos_token_id);
    config.validate();

    // Stateful pipeline does not provide logprobs for prompt tokens
    OPENVINO_ASSERT(config.echo == false, "Echo is not supported in the stateful pipeline");

    std::shared_ptr<StreamerBase> streamer_ptr = ov::genai::utils::create_streamer(streamer, m_tokenizer);

    auto batch_size = input_ids.get_shape().at(0);
    OPENVINO_ASSERT(streamer_ptr == nullptr || batch_size == 1 && config.num_return_sequences == 1 &&
        (config.is_greedy_decoding() || config.is_multinomial()),
        "Currently streaming is possible only with batch size=1 and only for greedy or multinomial decoding");

    auto num_inputs = m_model_runner.get_compiled_model().inputs().size();
    OPENVINO_ASSERT(num_inputs == 4 || num_inputs == 3, "Model should have 3 or 4 inputs: "
                    "either (input_ids, attention_mask, beam_idx) or "
                    "(input_ids, attention_mask, position_ids, beam_idx) "
                    "but you have '" + std::to_string(num_inputs) + "' inputs");

    if (m_kv_history_manager.reset_kv_cache)
        reset_kv_state();
    else
        ov::genai::utils::trim_kv_cache(m_model_runner, m_kv_history_manager.num_tokens_to_remove_from_kv_cache,
                                        m_kv_history_manager.kv_cache_seq_length_axis, m_adapter_controller);

    size_t kv_cache_len = 0;
    ov::Tensor concatenated_attention_mask;
    if (is_chat_conversation && !m_tokenized_chat_history.empty()) {
        OPENVINO_ASSERT(batch_size == 1, "continuation of generation is possible only for batch 1");
        // If history is saved in KV cache, concatenate new attention_mask with the already existing.
        // Between subsequent runs attention_mask should not be modified.
        auto atten_mask_history = m_model_runner.get_tensor("attention_mask");
        auto prompt_len = attention_mask.get_shape()[1];

        kv_cache_len = atten_mask_history.get_shape()[1] - m_kv_history_manager.num_tokens_to_remove_from_kv_cache;

        ov::Tensor new_atten_mask = ov::Tensor{ov::element::i64, {batch_size, kv_cache_len + prompt_len}};
        auto start_atten_hst = atten_mask_history.data<int64_t>();

        std::copy(start_atten_hst, start_atten_hst + kv_cache_len,
                new_atten_mask.data<int64_t>());
        std::copy(attention_mask.data<int64_t>(), attention_mask.data<int64_t>() + prompt_len,
                new_atten_mask.data<int64_t>() + kv_cache_len);
        concatenated_attention_mask = new_atten_mask;
    } else {
        concatenated_attention_mask = attention_mask;
    }

    size_t prev_attn_mask_size = concatenated_attention_mask.get_shape()[1];

    bool position_ids_available = (num_inputs == 4);
    std::optional<ov::Tensor> position_ids = std::nullopt;
    if (position_ids_available) {
        position_ids = ov::Tensor{ov::element::i64, input_ids.get_shape()};
        utils::initialize_position_ids(*position_ids, attention_mask, kv_cache_len);
    }

    if(m_adapter_controller) {
        m_adapter_controller->apply(m_model_runner, config.adapters);
    }

    std::vector<SequenceGroup::Ptr> requests;
    size_t block_size = 1;

    for (size_t request_id = 0; request_id < batch_size; request_id++) {
        SequenceGroup::Ptr sequence_group;
        if (is_chat_conversation) {
            ov::Tensor tokenized_chat_history = ov::Tensor(ov::element::i64, {1, m_tokenized_chat_history.size()}, m_tokenized_chat_history.data());
            sequence_group = std::make_shared<SequenceGroup>(request_id, tokenized_chat_history, config, block_size);
        } else {
            size_t seq_len = input_ids.get_shape().at(1);
            size_t batch_offset = request_id * seq_len;
            const int64_t* prompt_start = input_ids.data<const int64_t>() + batch_offset;
            std::vector<int64_t> tokenized_prompt(prompt_start, prompt_start + seq_len);

            sequence_group = std::make_shared<SequenceGroup>(request_id, tokenized_prompt, config, block_size);
        }

        requests.push_back(sequence_group);
    }

    if (m_sampler.get_seed() != config.rng_seed) {
        m_sampler.set_seed(config.rng_seed);
    }

    ov::genai::utils::GenerationFinishInfo finish_info = get_lm_encoded_results(m_model_runner, input_ids, concatenated_attention_mask,
                                                                        streamer_ptr, m_sampler, requests, position_ids, std::nullopt);
    ov::genai::EncodedResults& result = finish_info.results;
    m_last_disappeared_token = finish_info.probably_disappeared_token;
    m_chat_generation_finish_status = finish_info.streaming_finish_status;

    if (is_chat_conversation) {
        m_kv_history_manager.reset();

        // force remove from kv_cache last answer
        if (config.is_beam_search() && m_chat_input_type != ov::genai::utils::GenerationChatInputsType::ENCODED_INPUTS) {
            m_kv_history_manager.trusted_history_length = m_tokenized_chat_history.size();
            m_kv_history_manager.num_tokens_to_remove_from_kv_cache = m_model_runner.get_tensor("attention_mask").get_shape()[1] - prev_attn_mask_size;
        }

        if (m_chat_generation_finish_status == ov::genai::GenerationStatus::CANCEL) {
            m_kv_history_manager.num_tokens_to_remove_from_kv_cache = m_model_runner.get_tensor("attention_mask").get_shape()[1] - prev_attn_mask_size;

            if (m_chat_input_type == ov::genai::utils::GenerationChatInputsType::ENCODED_INPUTS) {
                m_tokenized_chat_history.resize(m_tokenized_chat_history.size() - real_input_ids_size);
                m_kv_history_manager.num_tokens_to_remove_from_kv_cache += real_input_ids_size;
            }
        } else {
            std::copy(result.tokens[0].begin(), result.tokens[0].end(), std::back_inserter(m_tokenized_chat_history));
        }
    } else {
        m_last_disappeared_token = std::nullopt;
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
    constexpr bool add_generation_prompt = false;

    m_templated_chat_history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
}

void StatefulLLMPipeline::reset_kv_state() {
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
    m_kv_history_manager.reset();
    m_chat_input_type = ov::genai::utils::GenerationChatInputsType::UNDEF;
    m_last_disappeared_token = std::nullopt;
    bool have_state = 0 != m_model_runner.get_tensor("attention_mask").get_size();
    if (!m_tokenized_chat_history.empty() || have_state) {
        reset_kv_state();
        m_model_runner.get_tensor("attention_mask").set_shape({1, 0});
        m_history.clear();
        m_templated_chat_history.clear();
        m_tokenized_chat_history.clear();
    }
}

} // namespace ov::genai
