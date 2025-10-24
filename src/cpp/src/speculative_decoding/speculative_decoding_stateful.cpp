// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speculative_decoding_stateful.hpp"
#include "continuous_batching/timer.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/genai/text_streamer.hpp"

#include <algorithm>

namespace ov::genai {
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

bool are_tokenizers_equal(ov::genai::Tokenizer& lhs, ov::genai::Tokenizer& rhs);
} // ov::genai

namespace {
ov::genai::StreamingStatus stream_generated_tokens(std::shared_ptr<ov::genai::StreamerBase> streamer_ptr,
                                                   const std::vector<int64_t>& tokens) {
    if (streamer_ptr) {
        return streamer_ptr->write(tokens);
    }
    return ov::genai::StreamingStatus{};
}

void update_perf_stat_by_token_time(ov::genai::RawPerfMetrics& raw_perf_counters, const float duration_microsec,
                                    const ov::genai::TimePoint new_token_time, const std::size_t num_generated_tokens) {
    raw_perf_counters.m_token_infer_durations.emplace_back(duration_microsec);
    raw_perf_counters.m_inference_durations[0] += ov::genai::MicroSeconds(duration_microsec);
    raw_perf_counters.m_new_token_times.emplace_back(new_token_time);
    raw_perf_counters.m_batch_sizes.emplace_back(num_generated_tokens);
}

void update_perf_stat_by_infer_duration(ov::genai::RawPerfMetrics& raw_perf_counters,
                                        const float inference_duration,
                                        const float token_duration,
                                        const std::size_t num_generated_tokens) {
    raw_perf_counters.m_durations.emplace_back(token_duration);
    raw_perf_counters.m_inference_durations[0] += ov::genai::MicroSeconds(inference_duration);
    raw_perf_counters.m_batch_sizes.emplace_back(num_generated_tokens);
}

void ensure_num_assistant_tokens_is_set(ov::genai::GenerationConfig& generation_config) {
    auto assistant_confidence_threshold = generation_config.assistant_confidence_threshold;
    OPENVINO_ASSERT(assistant_confidence_threshold == 0.f,
        "Stateful (non Continuous Batching) Speculative Decoding pipeline only supports `num_assistant_tokens` "
        "as parameter in GenerationConfig and doesn't work with `assistant_confidence_threshold`.\nPlease "
        "remove its specification or set it to 0.f.");

    constexpr std::size_t default_num_assistant_tokens = 5;
    if (generation_config.num_assistant_tokens == 0) {
        generation_config.num_assistant_tokens = default_num_assistant_tokens;
    }
}
}// anonymous namespace

namespace ov {
namespace genai {
    LLMInferWrapper::LLMInferWrapper(
    const ov::genai::ModelDesc& model_desc
) : m_device(model_desc.device),
    m_properties(model_desc.properties),
    m_generation_config(model_desc.generation_config),
    m_tokenizer(model_desc.tokenizer) {
    m_kv_pos = ov::genai::utils::get_kv_axes_pos(model_desc.model);
    if (m_device == "NPU") {
        auto [compiled, kv_desc] = utils::compile_decoder_for_npu(model_desc.model, m_properties, m_kv_pos);
        m_max_prompt_len = kv_desc.max_prompt_len;
        m_kvcache_total = kv_desc.max_prompt_len + kv_desc.min_response_len;
        m_request = compiled.create_infer_request();
    } else {
        // TODO: We might need it for manipulations with indices
        // utils::apply_gather_before_matmul_transformation(model_desc.model);
        m_request = ov::genai::utils::singleton_core().compile_model(model_desc.model, m_device, m_properties).create_infer_request();
    }
    raw_perf_metrics.m_inference_durations =  {{ ov::genai::MicroSeconds(0.0f) }};
}

std::string LLMInferWrapper::device() const {
    return m_device;
}

ov::genai::GenerationConfig LLMInferWrapper::get_generation_config() const {
    return m_generation_config;
}

void LLMInferWrapper::set_generation_config(ov::genai::GenerationConfig config) {
    m_generation_config = config;
}

int64_t LLMInferWrapper::get_kvcache_capacity() const {
    if (m_device == "NPU") {
        return m_kvcache_total - m_num_processed_tokens;
    }
    return std::numeric_limits<int64_t>::max();
}

int64_t LLMInferWrapper::get_generation_capacity() const {
    int64_t max_new_tokens = static_cast<int64_t>(m_generation_config.get_max_new_tokens());
    if (m_first_prompt_len > 0) {
        int64_t generated_new_tokens = static_cast<int64_t>(m_num_processed_tokens - m_first_prompt_len) + 1;
        return max_new_tokens - generated_new_tokens;
    } else {
        return max_new_tokens;
    }
}

int64_t LLMInferWrapper::infer_first(const ov::Tensor &input_ids,
                                     const ov::Tensor &attention_mask,
                                     const ov::Tensor &position_ids) {
    ManualTimer infer_first_timer("infer_first()");
    infer_first_timer.start();

    if (m_device == "NPU") {
        // NB: Check if there is enough space in KV-cache to process input prompt
        auto prompt_len = input_ids.get_shape()[1];
        if (prompt_len > m_max_prompt_len) {
            OPENVINO_THROW("LLM model on NPU may only process prompts up to "
                           + std::to_string(m_max_prompt_len) + " tokens. "
                           + "Set the \"MAX_PROMPT_LEN\" config option to "
                           + "increase the limit.");
        }
    }

    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);
    if (m_device != "NPU") {
        // set beam_idx for stateful model: no beam search is used and BATCH_SIZE = 1
        m_request.get_tensor("beam_idx").set_shape({BATCH_SIZE});
        m_request.get_tensor("beam_idx").data<int32_t>()[0] = 0;
    }

    const auto infer_start = std::chrono::steady_clock::now();
    m_request.infer();
    const auto infer_end = std::chrono::steady_clock::now();

    m_num_processed_tokens = input_ids.get_shape()[1];
    m_first_prompt_len = m_num_processed_tokens;

    // Initialize placeholder data for next inferences on input_ids of size 1 (if any)
    // with values of previous iteration for simple increment on next iteration:
    m_new_input_token = -1;
    m_new_position_id = m_num_processed_tokens - 1;
    m_new_atten_mask_data = std::vector<int64_t>(m_num_processed_tokens, 1);
    set_already_allocated_input_for_1_token();

    // Update last_token variable for `can_infer()` logic:
    last_token = std::get<int64_t>(sample_tokens(get_logits(), 1u));

    infer_first_timer.end();
    update_perf_stat_by_infer_duration(raw_perf_metrics,
         ov::genai::PerfMetrics::get_microsec(infer_end - infer_start),
         infer_first_timer.get_duration_microsec(), BATCH_SIZE);
    return last_token;
}

bool LLMInferWrapper::can_infer(const std::size_t prompt_len) {
    OPENVINO_ASSERT(m_num_processed_tokens > 0, "can_infer() can be called only after infer_first()!");

    if (m_device == "NPU") {
        if (prompt_len > get_kvcache_capacity()) {
            // Not enough room in KVCache to process prompt_len tokens.
            return false;
        }
    }

    if (!m_generation_config.ignore_eos && (last_token == m_generation_config.eos_token_id)) {
        return false;
    }
    auto stop_token_ids = m_generation_config.stop_token_ids;
    if (std::find(stop_token_ids.begin(), stop_token_ids.end(), last_token) != stop_token_ids.end()) {
        return false;
    }
    if (get_generation_capacity() <= 0) {
        return false;
    }
    return true;
}

int64_t LLMInferWrapper::infer_next(int64_t token, bool append_perf_stat) {
    OPENVINO_ASSERT(m_num_processed_tokens > 0, "infer_next() can be called only after infer_first()!");

    ManualTimer infer_next_timer("infer_next()");
    infer_next_timer.start();

    // Just change the variables here, as pointers to them are already set to corresponding tensors
    m_new_input_token = token;
    ++m_new_position_id;
    // However, attention_mask changes its shape on each iteration, it should be re-set explicitly
    m_new_atten_mask_data.push_back(1);
    m_request.set_tensor("attention_mask", ov::Tensor(ov::element::i64, ov::Shape{1,m_new_atten_mask_data.size()}, (void*)&m_new_atten_mask_data[0]));

    const auto infer_start = std::chrono::steady_clock::now();
    m_request.infer();
    const auto infer_end = std::chrono::steady_clock::now();

    m_num_processed_tokens += 1u;

    // Update last_token variable for `can_infer()` logic:
    last_token = std::get<int64_t>(sample_tokens(get_logits(), 1u));

    infer_next_timer.end();
    // prepend perf stat
    if (!append_perf_stat) {
        update_perf_stat_by_infer_duration(
            raw_perf_metrics,
            ov::genai::PerfMetrics::get_microsec(infer_end - infer_start),
            infer_next_timer.get_duration_microsec(),
            BATCH_SIZE);
    } else {
        raw_perf_metrics.m_durations.back() +=
            ov::genai::MicroSeconds(infer_next_timer.get_duration_microsec());
        raw_perf_metrics.m_inference_durations[0] += 
            ov::genai::MicroSeconds(ov::genai::PerfMetrics::get_microsec(infer_end - infer_start));
    }

    return last_token;
}

std::vector<int64_t> LLMInferWrapper::infer_next_return_all(const std::vector<int64_t>& tokens) {
    OPENVINO_ASSERT(m_num_processed_tokens > 0, "infer_next_return_all() can be called only after infer_first()!");

    ManualTimer infer_next_return_all_timer("infer_next_return_all()");
    infer_next_return_all_timer.start();

    size_t tokens_size = tokens.size();
    auto input_ids = m_request.get_tensor("input_ids");
    ov::Tensor new_input_ids(input_ids.get_element_type(), ov::Shape{BATCH_SIZE, tokens_size});
    std::copy_n(tokens.begin(), tokens_size, new_input_ids.data<int64_t>());
    m_request.set_tensor("input_ids", new_input_ids);

    auto attention_mask = m_request.get_tensor("attention_mask");
    ov::Tensor new_attention_mask(attention_mask.get_element_type(), ov::Shape{BATCH_SIZE, m_num_processed_tokens + tokens_size});
    std::copy_n(attention_mask.data<int64_t>(), m_num_processed_tokens, new_attention_mask.data<int64_t>());
    std::fill_n(new_attention_mask.data<int64_t>() + m_num_processed_tokens, tokens_size, 1);
    m_request.set_tensor("attention_mask", new_attention_mask);

    auto position_ids = m_request.get_tensor("position_ids");
    ov::Tensor new_position_ids(position_ids.get_element_type(), ov::Shape{BATCH_SIZE, tokens_size});
    std::iota(new_position_ids.data<int64_t>(),
              new_position_ids.data<int64_t>() + new_position_ids.get_size(),
              m_num_processed_tokens);
    m_request.set_tensor("position_ids", new_position_ids);

    const auto infer_start = std::chrono::steady_clock::now();
    m_request.infer();
    const auto infer_end = std::chrono::steady_clock::now();

    m_num_processed_tokens += tokens_size;

    // Update pre-allocated inputs for 1 token and return back to use it
    // in case if next infer will be called on input_ids of size 1
    // (most frequent case).
    m_new_input_token = -1;
    m_new_position_id = m_num_processed_tokens - 1;
    for (std::size_t i = 0; i < tokens_size; ++i) {
        m_new_atten_mask_data.push_back(1);
    }
    set_already_allocated_input_for_1_token();

    auto logits = get_logits();
    auto sampled_tokens = std::get<std::vector<int64_t>>(sample_tokens(logits, tokens_size));
    // Update last_token variable for `can_infer()` logic:
    last_token = sampled_tokens.back();

    infer_next_return_all_timer.end();
    update_perf_stat_by_infer_duration(
        raw_perf_metrics, ov::genai::PerfMetrics::get_microsec(infer_end - infer_start),
        infer_next_return_all_timer.get_duration_microsec(), tokens_size);
    return sampled_tokens;
}

ov::Tensor LLMInferWrapper::get_logits() {
    return m_request.get_tensor("logits");
}

std::size_t LLMInferWrapper::get_num_processed_tokens() const {
    return m_num_processed_tokens;
}

void LLMInferWrapper::trim_kv_cache(const size_t tokens_to_remove) {
    OPENVINO_ASSERT(m_num_processed_tokens > 0, "trim_kv_cache() can be called only after infer_first()!");

    OPENVINO_ASSERT(tokens_to_remove < m_num_processed_tokens);
    // For NPU "trim" is done by position ids on NPUW side.
    if (m_device != "NPU") {
        // Trim kv_cache values on tokens_to_remove
        ov::genai::utils::KVCacheState to_trim_state;
        to_trim_state.num_tokens_to_trim = tokens_to_remove;
        to_trim_state.seq_length_axis =  m_kv_pos.seq_len;
        to_trim_state.reset_mem_state = false;
        ov::genai::utils::trim_kv_cache(m_request, to_trim_state, {});
    }
    m_num_processed_tokens -= tokens_to_remove;

    // Update pre-allocated inputs for 1 token and return back to use it
    // in case if next infer will be called on input_ids of size 1
    // (most frequent case).
    m_new_input_token = -1;
    m_new_position_id = m_num_processed_tokens - 1;
    for (std::size_t i = 0; i < tokens_to_remove; ++i) {
        m_new_atten_mask_data.pop_back();
    }
    set_already_allocated_input_for_1_token();
}

void LLMInferWrapper::reset_state() {
    raw_perf_metrics.m_inference_durations =  {{ ov::genai::MicroSeconds(0.0f) }};
    return m_request.reset_state();
}

void LLMInferWrapper::release_memory() {
    m_request.get_compiled_model().release_memory();
}

void LLMInferWrapper::set_already_allocated_input_for_1_token() {
    m_request.set_tensor("input_ids", ov::Tensor(ov::element::i64, ov::Shape{1,1},  reinterpret_cast<void*>(&m_new_input_token)));
    m_request.set_tensor("position_ids", ov::Tensor(ov::element::i64, ov::Shape{1,1}, reinterpret_cast<void*>(&m_new_position_id)));
}

// TODO: Use already provided Sampler API, that will support both greedy and
//       multinomial decoding.
std::variant<int64_t, std::vector<int64_t>>
    LLMInferWrapper::sample_tokens(const ov::Tensor& logits, std::size_t num_tokens_to_sample) {
    OPENVINO_ASSERT(m_num_processed_tokens > 0, "sample_tokens() can be called only after infer_first()!");

    // logits.shape = [1, seq_len, vocab_size].
    auto logits_shape = logits.get_shape();
    OPENVINO_ASSERT(logits_shape.size() == 3);
    std::size_t batch_size = logits_shape[0];
    OPENVINO_ASSERT(batch_size == 1);
    std::size_t seq_len = logits_shape[1];
    OPENVINO_ASSERT(num_tokens_to_sample <= seq_len);
    std::size_t vocab_size = logits_shape[2];

    auto sample_token = [&](const ov::Tensor& logits, std::size_t idx) {
        size_t sequence_offset = idx * vocab_size;
        float* logits_data = logits.data<float>() + sequence_offset;
        return std::max_element(logits_data, logits_data + vocab_size) - logits_data;
    };

    if (num_tokens_to_sample == 1) {
        // Sample last logit:
        return sample_token(logits, seq_len - 1);
    } else {
        // Sample last num_tokens_to_sample logits:
        std::vector<int64_t> sampled_tokens;
        for (std::size_t i = 0; i < num_tokens_to_sample; i++) {
            sampled_tokens.push_back(sample_token(logits, seq_len - num_tokens_to_sample + i));
        }
        return sampled_tokens;
    }
}

StatefulSpeculativeLLMPipeline::StatefulSpeculativeLLMPipeline(
    const ov::genai::ModelDesc& main_model_desc, 
    const ov::genai::ModelDesc& draft_model_desc
) : LLMPipelineImplBase(main_model_desc.tokenizer, main_model_desc.generation_config) {
    auto draft_model = draft_model_desc.model;

    // FIXME: slicing produces incorrect results for some models on NPU.
    // On NPU, applying slice the safe way is done by the underlying plugin
    if (draft_model_desc.device != "NPU") {
        // As draft_model_desc contains std::shared_ptr<ov::Model>,
        // this model update will be reflected in draft_model_desc
        utils::apply_slice_before_matmul_transformation(draft_model);
    }

    // Main and Draft model can have different tokenizers
    // to do: support retokenization: 154103
    ov::genai::Tokenizer main_model_tokenizer = main_model_desc.tokenizer;
    ov::genai::Tokenizer draft_model_tokenizer = draft_model_desc.tokenizer;
    // todo: remove this condition after support of CVS-154103
    OPENVINO_ASSERT(are_tokenizers_equal(main_model_tokenizer, draft_model_tokenizer), "Tokenizers for draft and main models are different!");
    m_tokenizer = main_model_tokenizer;
    
    // Draft model (which is smaller, less accurate but faster)
    auto draft_model_desc_copy = draft_model_desc;
    if (draft_model_desc_copy.device.empty()) {
        draft_model_desc_copy.device = main_model_desc.device;
    }
    if (draft_model_desc_copy.properties.empty() && (draft_model_desc_copy.device == main_model_desc.device)) {
        draft_model_desc_copy.properties = main_model_desc.properties;
    }
    m_draft_request = std::make_unique<LLMInferWrapper>(draft_model_desc_copy);
    OPENVINO_ASSERT(m_draft_request != nullptr, "Failed to create draft model inference wrapper");

    // Specifying number candidates to generate
    ensure_num_assistant_tokens_is_set(m_generation_config);
    m_candidates_num = m_generation_config.num_assistant_tokens;
    // We set the upper limit for candidates number as two times the number requested
    // by user.
    m_max_candidates_num = m_candidates_num * 2;

    // Main model (which is bigger, more accurate but slower)
    auto main_model_desc_copy = main_model_desc;
    if (main_model_desc_copy.device == "NPU") {
        main_model_desc_copy.properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = m_max_candidates_num + 1;
    }
    m_main_request = std::make_unique<LLMInferWrapper>(main_model_desc_copy);
    OPENVINO_ASSERT(m_main_request != nullptr, "Failed to create main model inference wrapper");
   
    m_sd_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
}

GenerationConfig StatefulSpeculativeLLMPipeline::resolve_generation_config(OptionalGenerationConfig generation_config) {
    GenerationConfig config = generation_config.value_or(m_generation_config);
    
    ensure_num_assistant_tokens_is_set(config);
    m_candidates_num = config.num_assistant_tokens;
    // We set the upper limit for candidates number as two times the number
    // requested by user.
    m_max_candidates_num = m_candidates_num * 2;
    
    // If stop_token_ids were not provided, take value from default m_generation_config
    if (config.stop_token_ids.empty())
        config.stop_token_ids = m_generation_config.stop_token_ids;
    // If eos_token_id was not provided, take value from default m_generation_config
    if (config.eos_token_id == -1)
        config.set_eos_token_id(m_generation_config.eos_token_id);
    config.validate();
    return config;
}

DecodedResults StatefulSpeculativeLLMPipeline::generate(
    StringInputs inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {
    ManualTimer generate_timer("StatefulSpeculativeLLMPipeline::generate()");
    generate_timer.start();
    ManualTimer encode_timer("Encode");
    encode_timer.start();

    std::string prompt = std::visit(overloaded{
        [](const std::string& prompt_str) {
            return prompt_str;
        },
        [](std::vector<std::string>& prompts) {
            OPENVINO_ASSERT(prompts.size() == 1u, "Currently only batch size=1 is supported");
            return prompts.front();
        }
    }, inputs);

    GenerationConfig config = resolve_generation_config(generation_config);

    ov::genai::TokenizedInputs tokenized_input;
    if (m_is_chat_conversation) {
        m_history.push_back({{"role", "user"}, {"content", prompt}});
        constexpr bool add_generation_prompt = true;
        prompt = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
        // for chat ov::genai::add_special_tokens(false) is aligned with stateful pipeline and HF
        tokenized_input = m_tokenizer.encode(prompt, ov::genai::add_special_tokens(false));
    } else {
        if (config.apply_chat_template && !m_tokenizer.get_chat_template().empty()) {
            ChatHistory history({{{"role", "user"}, {"content", prompt}}});
            constexpr bool add_generation_prompt = true;
            auto templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
            tokenized_input = m_tokenizer.encode(templated_prompt, ov::genai::add_special_tokens(false));
        } else {
            // in case when chat_template was not found in tokenizer_config.json or set
            tokenized_input = m_tokenizer.encode(prompt, ov::genai::add_special_tokens(true));
        }
    }

    encode_timer.end();
    auto encoded_results = generate(tokenized_input, config, streamer);

    ManualTimer decode_timer("Decode");
    decode_timer.start();
    DecodedResults decoded_results = {m_tokenizer.decode(encoded_results.tokens), encoded_results.scores};
    decode_timer.end();

    if (m_is_chat_conversation) {
        auto answer = decoded_results.texts[0];
        if (m_streaming_was_cancelled)
            // If generation process was cancelled by user, let's rollback to previous state of history
            m_history.pop_back();
        else
            m_history.push_back({{"role", "assistant"}, {"content", answer}});
    }

    // Update perf metrics
    decoded_results.perf_metrics = encoded_results.perf_metrics;
    decoded_results.extended_perf_metrics = encoded_results.extended_perf_metrics;
    generate_timer.end();
    auto& raw_counters = decoded_results.perf_metrics.raw_metrics;
    raw_counters.generate_durations.clear();
    raw_counters.generate_durations.emplace_back(generate_timer.get_duration_microsec());
    raw_counters.tokenization_durations.emplace_back(encode_timer.get_duration_microsec());
    raw_counters.detokenization_durations.emplace_back(decode_timer.get_duration_microsec());
    decoded_results.perf_metrics.m_evaluated = false;
    decoded_results.perf_metrics.evaluate_statistics(generate_timer.get_start_time());
    return decoded_results;
}

DecodedResults StatefulSpeculativeLLMPipeline::generate(
    const ChatHistory& history,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {
    ManualTimer generate_timer("StatefulSpeculativeLLMPipeline::generate()");
    generate_timer.start();
    ManualTimer encode_timer("Encode");
    encode_timer.start();

    GenerationConfig config = resolve_generation_config(generation_config);

    OPENVINO_ASSERT(config.apply_chat_template, "Chat template must be applied when using ChatHistory in generate method.");
    OPENVINO_ASSERT(!m_tokenizer.get_chat_template().empty(), "Chat template must not be empty when using ChatHistory in generate method.");
    OPENVINO_ASSERT(!history.empty(), "Chat history must not be empty when using ChatHistory in generate method.");

    constexpr bool add_generation_prompt = true;
    auto templated_chat_history = m_tokenizer.apply_chat_template(history, add_generation_prompt);
    // for chat ov::genai::add_special_tokens(false) is aligned with stateful pipeline and HF
    auto tokenized_inputs = m_tokenizer.encode(templated_chat_history, ov::genai::add_special_tokens(false));
    encode_timer.end();
    auto encoded_results = generate(tokenized_inputs, config, streamer);

    ManualTimer decode_timer("Decode");
    decode_timer.start();
    DecodedResults decoded_results = {m_tokenizer.decode(encoded_results.tokens), encoded_results.scores};
    decode_timer.end();
    
    // Update perf metrics
    decoded_results.perf_metrics = encoded_results.perf_metrics;
    decoded_results.extended_perf_metrics = encoded_results.extended_perf_metrics;
    auto& raw_counters = decoded_results.perf_metrics.raw_metrics;
    generate_timer.end();
    raw_counters.generate_durations.clear();
    raw_counters.generate_durations.emplace_back(generate_timer.get_duration_microsec());
    raw_counters.tokenization_durations.emplace_back(encode_timer.get_duration_microsec());
    raw_counters.detokenization_durations.emplace_back(decode_timer.get_duration_microsec());
    decoded_results.perf_metrics.m_evaluated = false;
    decoded_results.perf_metrics.evaluate_statistics(generate_timer.get_start_time());

    return decoded_results;
}

EncodedResults StatefulSpeculativeLLMPipeline::generate(
    const EncodedInputs& inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer) {
    ManualTimer generate_timer("StatefulSpeculativeLLMPipeline::generate()");
    generate_timer.start();

    ov::Tensor input_ids;
    ov::Tensor attention_mask;

    if (auto data = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *data;
        attention_mask = ov::genai::utils::init_attention_mask(input_ids);
    } else if (auto data = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = data->input_ids;
        attention_mask = data->attention_mask;
    }

    ov::Shape prompt_shape = input_ids.get_shape();
    const size_t batch_size = prompt_shape[0];
    OPENVINO_ASSERT(batch_size == 1u, "Currently only batch size=1 is supported");

    GenerationConfig config = resolve_generation_config(generation_config);

    OPENVINO_ASSERT(config.is_greedy_decoding(),
        "Currently only greedy decoding is supported");

    OPENVINO_ASSERT(config.num_return_sequences == 1u,
        "Currently only \"num_return_sequences\" equal to 1 is supported!");

    m_main_request->set_generation_config(config);

    // Config draft model to not stop on EOS and remove stop strings:
    ov::genai::GenerationConfig draft_config = m_draft_request->get_generation_config();
    draft_config.ignore_eos = true;
    draft_config.stop_strings = {};
    // Need to set `max_new_tokens` as GenerationConfig requires it if `ignore_eos` is true.
    // However, this parameter won't be utilized in pipeline, only main's `max_new_tokens`
    // will be utilized.
    draft_config.max_new_tokens = config.get_max_new_tokens();
    draft_config.validate();
    m_draft_request->set_generation_config(draft_config);

    std::shared_ptr<StreamerBase> streamer_ptr = ov::genai::utils::create_streamer(streamer, m_tokenizer);
    ov::genai::EncodedResults results;
    auto& raw_perf_counters = m_sd_perf_metrics.raw_metrics;
    // NB: Only batch=1 is supported now.
    // NB: In the case of greedy decoding scores are filled with zeros.
    results.scores.resize(1u);
    results.scores[0] = 0u;
    results.tokens.resize(1u);

    ov::Tensor position_ids{ov::element::i64, input_ids.get_shape()};
    utils::initialize_position_ids(position_ids, attention_mask);

    // To collect KV-cache for the prompt and to get the next token, run the very first infer request
    // for main and draft models:
    ManualTimer first_token_timer("Speculative decode: first token timer");
    first_token_timer.start();

    auto out_token = m_main_request->infer_first(input_ids, attention_mask, position_ids);

    first_token_timer.end();
    update_perf_stat_by_token_time(raw_perf_counters, first_token_timer.get_duration_microsec(),
                                   first_token_timer.get_end_time(), 1u);

    m_draft_request->infer_first(input_ids, attention_mask, position_ids);

    // logits shape is [BATCH_SIZE, seq_len, vocab_size]
    auto draft_logits = m_draft_request->get_logits();
    auto main_logits = m_main_request->get_logits();
    size_t draft_vocab_size = draft_logits.get_shape().back();
    size_t main_vocab_size = main_logits.get_shape().back();
    OPENVINO_ASSERT(draft_vocab_size == main_vocab_size,
                    "Vocab sizes should be the same for the both: main and draft models!");

    OPENVINO_ASSERT(draft_logits.get_shape().at(1) <= main_logits.get_shape().at(1),
                    "Num of generated useful logits from draft models should be less"
                    " or equal than ones from main model.");

    auto streaming_status = stream_generated_tokens(streamer_ptr, std::vector<int64_t> {out_token});
    results.tokens[0].push_back(out_token);

    // Creating timers for performance metrics calculation:
    ManualTimer iteration_timer("Speculative decode: infer iteration");
    ManualTimer candidates_timer("Draft model: candidates generation");
    ManualTimer main_timer("Main model");

    // Speculative decoding works the following way. The draft model predicts the next K
    // tokens one by one in an autoregressive manner, while the main model validates these
    // predictions and corrects them if necessary. We go through each predicted token, and
    // if a difference is detected between the draft and main model, we stop and keep the
    // last token predicted by the main model. Then the draft model gets the latest main
    // prediction and again tries to predict the next K tokens, repeating the cycle.

    // This approach reduces the need for multiple infer requests to the main model,
    // enhancing performance. For instance, in more predictable parts of text generation,
    // the draft model can, in best-case scenarios, generate the next K tokens that exactly
    // match the target. In that case they are validated in a single inference call to
    // the main model instead of running K subsequent requests.

    // Last generated token by draft model needs to be prepended before next run if it is accepted
    // by the main model! So it will get into the kvcache of the draft model.
    int64_t draft_prefix_token = -1;
    while (m_main_request->can_infer() && (streaming_status == ov::genai::StreamingStatus::RUNNING)) {
        iteration_timer.start();

        // Phase 1: Generation of candidates with the draft model.
        candidates_timer.start();

        std::vector<int64_t> candidates;
        int64_t kvcache_room_for_candidates = std::min(
            // Take into the account the draft prefix token, described above
            // (before the while loop). If it is needed to be prepended to kvcache,
            // then we can generate candidates as number of left kvcache space of
            // draft model but minus 1:
            m_draft_request->get_kvcache_capacity() - ((draft_prefix_token == -1) ? 1 : 0),
            // Take into the account reference token that is prefixed to candidates.
            // We can generate candidates as number of left kvcache space of main
            // model, but as main model will consume candidates + its previous output
            // then we need to preserve this one spot in main kvcache for previous
            // output.
            m_main_request->get_kvcache_capacity() - 1);
        int64_t generation_room_for_candidates = 
            // Take into the account output token, generated on candidates.
            // If we accept all candidates by the main model, then we will generate
            // output of length equal to number of candidates + one output token from
            // the main model.
            // As output token number is limited we can generate candidates of only
            // remained output tokens number - 1 (for output token).
            m_main_request->get_generation_capacity() - 1;
        int64_t candidates_can_be_generated = std::min(
            kvcache_room_for_candidates, generation_room_for_candidates);
        if (candidates_can_be_generated <= 0) {
            auto remainder = m_main_request->get_generation_capacity();
            // If user asked for more tokens in answer and we have
            // KVCache capacity to sequentially infer them:
            if (remainder > 0 && m_main_request->can_infer(remainder)) {
                for (std::size_t i = 0; i < remainder; ++i) {
                    main_timer.start();
                    out_token = m_main_request->infer_next(out_token);
                    main_timer.end();

                    streaming_status = stream_generated_tokens(streamer_ptr, {out_token});
                    results.tokens[0].push_back(out_token);

                    iteration_timer.end();
                    auto iteration_duration = iteration_timer.get_duration_microsec();
                    update_perf_stat_by_token_time(raw_perf_counters, iteration_duration, main_timer.get_end_time(), 1u);

                    main_timer.clear();
                    iteration_timer.clear();
                    iteration_timer.start();
                }
            }
            break;
        }
        auto candidates_to_generate = std::min(static_cast<int64_t>(m_candidates_num),
            candidates_can_be_generated);
        candidates.reserve(candidates_to_generate);

        // If draft_prefix_token is present, run an infer on it to collect KV cache for it
        const bool draft_prefix_exists = (draft_prefix_token != -1);
        if (draft_prefix_exists) {
            m_draft_request->infer_next(draft_prefix_token);
        }
        // Note: If `draft_prefix_exists == true`, then we append performance metrics of
        // newly generated candidate to the previously generated token on draft prefix prompt,
        // as we are only interested in one output from these two inference operations.
        int64_t candidate = m_draft_request->infer_next(out_token, draft_prefix_exists); 
        candidates.push_back(candidate);

        for (size_t i = 1; i < candidates_to_generate; i++) {
            candidate = m_draft_request->infer_next(candidate);
            candidates.push_back(candidate);
        }
        draft_prefix_token = candidates.back();

        candidates_timer.end();
        m_sd_metrics.draft_duration += candidates_timer.get_duration();
        candidates_timer.clear();

        // Phase 2. Main inference.
        // For the main network, candidates_size + 1 tokens will be fed at once in a
        // single infer request: last token from previous main inference + all candidates
        // from the draft stage.
        //
        // Note on model's return variable: If model isn't sliced to return only
        // certain logits, then it returns logits for all elements of the input
        // prompt. In that tensor, for each token `t` of the input prompt it contains
        // distribution (over the vocabulary) for the next possible token that is
        // generated based on subsequence [first token,...,`t`] of the input prompt.
        main_timer.start();

        std::vector<int64_t> input_for_main(candidates.begin(), candidates.end());
        input_for_main.insert(input_for_main.begin(), {out_token});
        auto ref_tokens = m_main_request->infer_next_return_all(input_for_main);

        main_timer.end();
        m_sd_metrics.main_duration += main_timer.get_duration();

        // Phase 3. Validation of candidates by output of main model:
        size_t accepted_tokens_number = 0u;
        // Last token is a new token from the main model, skip it:
        for (size_t i = 0; i < ref_tokens.size() - 1; ++i) {
            if (ref_tokens[i] != candidates[i]) {
                break;
            }
            accepted_tokens_number++;
        }

        auto mismatched_candidates = candidates.size() - accepted_tokens_number;
        std::vector<int64_t> validated_tokens(ref_tokens.begin(), ref_tokens.end() - mismatched_candidates);
        out_token = validated_tokens.back();
    
        // Phase 4: Update inference wrappers based on found matches and mismatches
        if (mismatched_candidates > 0) {
            m_draft_request->trim_kv_cache(mismatched_candidates - 1);
            m_main_request->trim_kv_cache(mismatched_candidates);
            // We don't need last candidate in KVCache of draft model, as
            // it fails validation.
            draft_prefix_token = -1;
        }
        update_candidate_strategy(accepted_tokens_number);

        auto& main_perf_generated_tokens = m_main_request->raw_perf_metrics.m_batch_sizes.back();
        main_perf_generated_tokens -= mismatched_candidates;
        m_sd_metrics.update_draft_generated_len(0 /* request_id */, candidates_to_generate);
        m_sd_metrics.update_acceptance_rate(0 /* request_id */, (accepted_tokens_number * 100.f) / candidates_to_generate);
        m_sd_metrics.update_draft_accepted_tokens(0 /* request_id */, accepted_tokens_number);
        m_sd_metrics.update_generated_len(validated_tokens.size());
        if (utils::env_setup_for_print_debug_info()) {
            m_sd_metrics.print(true);
            m_sd_metrics.clean_up();
        }

        streaming_status = stream_generated_tokens(streamer_ptr, validated_tokens);
        results.tokens[0].insert(results.tokens[0].end(), validated_tokens.begin(), validated_tokens.end());

        iteration_timer.end();
        auto iteration_duration = iteration_timer.get_duration_microsec();
        update_perf_stat_by_token_time(raw_perf_counters, iteration_duration, main_timer.get_end_time(), validated_tokens.size());
        iteration_timer.clear();
        main_timer.clear();
    }

    m_streaming_was_cancelled = (streaming_status == ov::genai::StreamingStatus::CANCEL);
    if (streamer_ptr) { // push streamer's cache
        streamer_ptr->end();
    }

    // If not chat conversation, then reset all states.
    if (!m_is_chat_conversation) {
        m_candidates_num = config.num_assistant_tokens;
        m_draft_request->reset_state();
        m_main_request->reset_state();
    }

    generate_timer.end();

    // Update perf metrics
    // If is called without tokenization then that stat will not be reported.
    m_sd_perf_metrics.num_input_tokens = input_ids.get_shape().at(1);
    m_sd_perf_metrics.load_time = this->m_load_time_ms;
    m_sd_perf_metrics.raw_metrics.generate_durations.clear();
    m_sd_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_timer.get_duration_microsec());

    m_sd_perf_metrics.draft_model_metrics.raw_metrics = m_draft_request->raw_perf_metrics;
    m_sd_perf_metrics.main_model_metrics.raw_metrics = m_main_request->raw_perf_metrics;

    m_sd_perf_metrics.evaluate_statistics(generate_timer.get_start_time());

    results.perf_metrics = m_sd_perf_metrics;
    results.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_sd_perf_metrics);

    // Reset all timers.
    generate_timer.clear();
    iteration_timer.clear();
    candidates_timer.clear();
    main_timer.clear();
    return results;
}

ov::genai::SpeculativeDecodingMetrics
StatefulSpeculativeLLMPipeline::get_speculative_decoding_metrics() const {
    return m_sd_metrics;
};

void StatefulSpeculativeLLMPipeline::start_chat(const std::string& system_message) {
    if (!system_message.empty()) {
        m_history.push_back({{"role", "system"}, {"content", system_message}});
    }
    m_is_chat_conversation = true;
};

void StatefulSpeculativeLLMPipeline::finish_chat() {
    m_is_chat_conversation = false;
    m_history.clear();
    m_draft_request->reset_state();
    m_main_request->reset_state();
};

StatefulSpeculativeLLMPipeline::~StatefulSpeculativeLLMPipeline() {
    m_main_request->release_memory();
    m_draft_request->release_memory();
}

void StatefulSpeculativeLLMPipeline::update_candidate_strategy(const std::size_t matches_num) {
    // Dynamically adjust number of generated candidates based on number of matches,
    // we want to balance the benefits of getting candidates tokens correct with the
    // cost of forecasting incorrect candidates tokens.
    if (matches_num == m_candidates_num) {
        m_candidates_num = std::min(m_candidates_num + 2, m_max_candidates_num);
    } else {
        m_candidates_num = static_cast<std::size_t>(std::max(static_cast<int64_t>(m_candidates_num) - 1, int64_t(1)));
    }
}
}  // namespace genai
}  // namespace ov
