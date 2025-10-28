// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "llm/pipeline_static.hpp"

#include "sampling/sampler.hpp"
#include "utils.hpp"

#include <fstream>

#include "openvino/runtime/core.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/genai/text_streamer.hpp"

namespace {

template <typename T>
void fill_tensor(ov::Tensor tensor, T fill_val, size_t offset = 0u) {
    T* tensor_data = tensor.data<T>();
    std::fill(tensor_data + offset, tensor_data + tensor.get_size(), fill_val);
}

void copy_with_offset(const ov::Tensor& orig, const std::size_t offset, ov::Tensor& padded) {
    auto orig_data = orig.data<int64_t>();
    int64_t* padded_data = padded.data<int64_t>();
    std::copy(orig_data, orig_data + orig.get_size(), padded_data + offset);
}

ov::Tensor make_tensor_slice(ov::Tensor tensor, size_t dim, size_t start_pos, size_t end_pos) {
    ov::Shape start_shape(std::vector<size_t>(tensor.get_shape().size(), 0u));
    start_shape[dim] = start_pos;
    ov::Shape end_shape = tensor.get_shape();
    end_shape[dim] = end_pos;
    return ov::Tensor(tensor, start_shape, end_shape);
}

void copy_columns_by_row_chunks(const ov::Tensor& src, ov::Tensor& dst) {
    const auto src_shape = src.get_shape();

    OPENVINO_ASSERT(src_shape.size() == 4u);
    OPENVINO_ASSERT(src_shape == dst.get_shape());
    OPENVINO_ASSERT(src.get_byte_size() == dst.get_byte_size());

    const auto src_strides = src.get_strides();
    const auto dst_strides = dst.get_strides();
    const auto elem_size   = src.get_byte_size() / src.get_size();

    const auto C = src_shape[1];
    const auto H = src_shape[2];
    const auto W = src_shape[3];

    const auto IS_H = src_strides[2];
    const auto OS_H = dst_strides[2];

    const size_t chunk_byte_size = W * elem_size;

    const auto* src_p = static_cast<const uint8_t*>(src.data());
    auto* dst_p = static_cast<uint8_t*>(dst.data());

    for (size_t i = 0; i < C*H; ++i) {
        const size_t src_offset = i * IS_H;
        const size_t dst_offset = i * OS_H;
        std::copy_n(src_p + src_offset, chunk_byte_size, dst_p + dst_offset);
    }
}

void stream_generated_tokens(std::shared_ptr<ov::genai::StreamerBase> streamer_ptr,
                             ov::genai::GenerationHandle& handle) {
    if (streamer_ptr && handle->can_read()) {
        std::unordered_map<uint64_t, ov::genai::GenerationOutput> token = handle->read();
        auto streaming_status = streamer_ptr->write(token.begin()->second.generated_ids);
        if (streaming_status != ov::genai::StreamingStatus::RUNNING) {
            streaming_status == ov::genai::StreamingStatus::CANCEL ? handle->cancel() : handle->stop();
        }
    }
}

enum StaticPipelineKind {
    STATEFUL
};

StaticPipelineKind str_to_pipeline(const std::string& str) {
    if (str == "STATEFUL") {
        return StaticPipelineKind::STATEFUL;
    }
    OPENVINO_THROW("Unsupported \"PIPELINE\" provided: ",
                   str, ". Please select \"STATEFUL\".");
}
} // anonymous namespace

namespace ov {
namespace genai {
namespace static_llm {

StatefulLLMPipeline::StatefulLLMPipeline(
    const std::filesystem::path& models_path,
    const ov::genai::Tokenizer& tokenizer,
    const ov::AnyMap& config
): StatefulLLMPipeline(
       genai::utils::singleton_core().read_model(models_path / "openvino_model.xml", {}, config),
       tokenizer, config,
       utils::from_config_json_if_exists(models_path)
   ) {
}

StatefulLLMPipeline::StatefulLLMPipeline(
    const std::shared_ptr<ov::Model>& model,
    const ov::genai::Tokenizer& tokenizer,
    const ov::AnyMap& properties,
    const ov::genai::GenerationConfig& generation_config
) : LLMPipelineImplBase(tokenizer, generation_config),
    m_sampler(m_tokenizer) {
    auto kv_pos = ov::genai::utils::get_kv_axes_pos(model);
    auto [compiled, kv_desc] = utils::compile_decoder_for_npu(model, properties, kv_pos);
    m_max_prompt_len = kv_desc.max_prompt_len;
    m_kvcache_total = kv_desc.max_prompt_len + kv_desc.min_response_len;
    m_request = compiled.create_infer_request();
    m_sampler.set_seed(m_generation_config.rng_seed);
}

DecodedResults StatefulLLMPipeline::generate(
    StringInputs inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {
    auto start_time = std::chrono::steady_clock::now();

    GenerationConfig config = generation_config.value_or(m_generation_config);
    std::string prompt;
    if (auto input_vector = std::get_if<std::vector<std::string>>(&inputs)) {
        OPENVINO_ASSERT(input_vector->size() == 1u, "Currently only batch size=1 is supported");
        prompt = std::move(input_vector->front());
    } else {
        OPENVINO_ASSERT(std::holds_alternative<std::string>(inputs));
        prompt = std::get<std::string>(inputs);
    }

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

    auto encode_stop_time =  std::chrono::steady_clock::now();
    auto encoded_results = generate(tokenized_input, config, streamer);

    auto decode_start_time =  std::chrono::steady_clock::now();
    DecodedResults decoded_results = {m_tokenizer.decode(encoded_results.tokens), encoded_results.scores};
    auto decode_stop_time =  std::chrono::steady_clock::now();

    if (m_is_chat_conversation) {
        auto answer = decoded_results.texts[0];
        if (m_chat_generation_finish_status == GenerationStatus::CANCEL)
            // If chat generation process was cancelled by user, let's rollback to previous state of history
            m_history.pop_back();
        else
            m_history.push_back({{"role", "assistant"}, {"content", answer}});
    }

    // Update perf metrics
    decoded_results.perf_metrics = encoded_results.perf_metrics;
    auto& raw_counters = decoded_results.perf_metrics.raw_metrics;
    auto stop_time = std::chrono::steady_clock::now();
    raw_counters.generate_durations.clear();
    raw_counters.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
    raw_counters.tokenization_durations.emplace_back(PerfMetrics::get_microsec(encode_stop_time - start_time));
    raw_counters.detokenization_durations.emplace_back(PerfMetrics::get_microsec(decode_stop_time - decode_start_time));
    decoded_results.perf_metrics.m_evaluated = false;
    decoded_results.perf_metrics.evaluate_statistics(start_time);
    return decoded_results;
}

DecodedResults StatefulLLMPipeline::generate(
    const ChatHistory& history,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {
    auto start_time = std::chrono::steady_clock::now();

    GenerationConfig config = generation_config.value_or(m_generation_config);

    OPENVINO_ASSERT(config.apply_chat_template, "Chat template must be applied when using ChatHistory in generate method.");
    OPENVINO_ASSERT(!m_tokenizer.get_chat_template().empty(), "Chat template must not be empty when using ChatHistory in generate method.");
    OPENVINO_ASSERT(!history.empty(), "Chat history must not be empty when using ChatHistory in generate method.");
    
    constexpr bool add_generation_prompt = true;
    auto templated_chat_history = m_tokenizer.apply_chat_template(history, add_generation_prompt);
    // for chat ov::genai::add_special_tokens(false) is aligned with stateful pipeline and HF
    auto tokenized_inputs = m_tokenizer.encode(templated_chat_history, ov::genai::add_special_tokens(false));

    auto encode_stop_time =  std::chrono::steady_clock::now();
    auto encoded_results = generate(tokenized_inputs, config, streamer);

    auto decode_start_time =  std::chrono::steady_clock::now();
    DecodedResults decoded_results = {m_tokenizer.decode(encoded_results.tokens), encoded_results.scores};
    auto decode_stop_time =  std::chrono::steady_clock::now();
    
    // Update perf metrics
    decoded_results.perf_metrics = encoded_results.perf_metrics;
    auto& raw_counters = decoded_results.perf_metrics.raw_metrics;
    auto stop_time = std::chrono::steady_clock::now();
    raw_counters.generate_durations.clear();
    raw_counters.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
    raw_counters.tokenization_durations.emplace_back(PerfMetrics::get_microsec(encode_stop_time - start_time));
    raw_counters.detokenization_durations.emplace_back(PerfMetrics::get_microsec(decode_stop_time - decode_start_time));
    decoded_results.perf_metrics.m_evaluated = false;
    decoded_results.perf_metrics.evaluate_statistics(start_time);
    
    return decoded_results;
}

EncodedResults StatefulLLMPipeline::generate(
    const EncodedInputs& inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {
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

    ov::Shape prompts_shape = input_ids.get_shape();
    const size_t batch_size = prompts_shape[0];
    OPENVINO_ASSERT(batch_size == 1u, "Currently only batch size=1 is supported");

    GenerationConfig config = generation_config.value_or(m_generation_config);
    // If stop_token_ids were not provided, take value from default m_generation_config
    if (config.stop_token_ids.empty())
        config.stop_token_ids = m_generation_config.stop_token_ids;
    // If eos_token_id was not provided, take value from default m_generation_config
    if (config.eos_token_id == -1)
        config.set_eos_token_id(m_generation_config.eos_token_id);
    config.validate();

    std::shared_ptr<StreamerBase> streamer_ptr = ov::genai::utils::create_streamer(streamer, m_tokenizer);

    OPENVINO_ASSERT(config.is_greedy_decoding() || config.is_multinomial(),
        "Currently only greedy and multinomial decoding are supported");

    OPENVINO_ASSERT(config.num_return_sequences == 1u,
        "Currently only \"num_return_sequences\" equal to 1 is supported!");

    ov::genai::EncodedResults results;
    auto& raw_perf_counters = results.perf_metrics.raw_metrics;
    // NB: Only batch=1 is supported now
    results.scores.resize(1u);
    results.scores[0] = 0u;
    results.tokens.resize(1u);

    // NB: Check if there is enough space in KV-cache to process input prompt
    auto prompt_len = input_ids.get_size();
    if (prompt_len > m_max_prompt_len) {
        OPENVINO_THROW("Static Stateful LLM pipeline may only process prompts up to "
                       + std::to_string(m_max_prompt_len) + " tokens. "
                       + "Set the \"MAX_PROMPT_LEN\" config option to increase the limit.");
    }

    ov::Tensor position_ids{ov::element::i64, input_ids.get_shape()};
    utils::initialize_position_ids(position_ids, attention_mask);

    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);

    m_request.infer();

    auto padded_logits = m_request.get_tensor("logits");
    // FIXME: Here is workaround to get only useful units of returned logits.
    //        If SliceOut is applied, there will be only 1 useful logit returned,
    //        nothing is required here.
    //        Other way, model will return logits of full context length,
    //        as internally prefill model is specially reshaped to return them.
    //        Fix should be done on OpenVINO side, so the model should return only
    //        useful logits of input prompt length, dropping the implementation-related
    //        padding ones.
    auto logits = padded_logits;
    auto padded_sequence_len = padded_logits.get_shape()[1];
    if (padded_sequence_len > 1) {
        // If SliceOut is not applied:
        logits = make_tensor_slice(padded_logits, 1, padded_sequence_len - prompt_len, padded_sequence_len);
    }
    int64_t output_sequence_len = logits.get_shape().at(1);

    auto sequence_group = std::make_shared<SequenceGroup>(
        0 /* request_id */, input_ids, config, 1 /* block_size */);
    sequence_group->schedule_tokens(sequence_group->get_prompt_len());
    sequence_group->set_output_seq_len(output_sequence_len);

    // NB: Controls what tokens are ready to be pushed into the streamer
    GenerationHandle handle = std::make_shared<GenerationHandleImpl>(
        sequence_group->get_generation_stream(), sequence_group->get_sampling_parameters());

    SamplerOutput sampler_output = m_sampler.sample({sequence_group}, logits);
    stream_generated_tokens(streamer_ptr, handle);

    int64_t input_ids_data = -1;
    int64_t position_ids_data = prompt_len - 1;
    std::vector<int64_t> attention_mask_data(prompt_len, 1);
    m_request.set_tensor("input_ids", ov::Tensor(ov::element::i64, ov::Shape{1,1},  reinterpret_cast<void*>(&input_ids_data)));
    m_request.set_tensor("position_ids", ov::Tensor(ov::element::i64, ov::Shape{1,1}, reinterpret_cast<void*>(&position_ids_data)));

    while (sequence_group->is_running() && !sequence_group->handle_stopped() && !sequence_group->handle_cancelled()) {
        // KV Cache is full, no further generation is possible
        if (position_ids_data + 1 == m_kvcache_total) {
            sequence_group->set_out_of_memory();
            break;
        }

        sequence_group->schedule_tokens(1);
        const auto running_sequences = sequence_group->get_running_sequences();
        OPENVINO_ASSERT(running_sequences.size() == 1u);
        auto last_token = running_sequences.front()->get_generated_ids().back();

        // Just change the variables here, as pointers to them are already set to corresponding tensors
        input_ids_data = last_token;
        ++position_ids_data;
        // However, attention_mask changes its shape on each iteration, it should be re-set explicitly
        attention_mask_data.push_back(1);
        m_request.set_tensor("attention_mask", ov::Tensor(ov::element::i64, ov::Shape{1,attention_mask_data.size()}, (void*)&attention_mask_data[0]));

        m_request.infer();

        raw_perf_counters.m_new_token_times.emplace_back(std::chrono::steady_clock::now());
        raw_perf_counters.m_batch_sizes.emplace_back(batch_size);

        SamplerOutput sampler_output = m_sampler.sample({sequence_group}, m_request.get_tensor("logits"));
        stream_generated_tokens(streamer_ptr, handle);
    }

    if (streamer_ptr) { // push streamer's cache
        streamer_ptr->end();
    }

    OPENVINO_ASSERT(sequence_group->get_finished_sequences().size() == 1u);
    auto sequence = sequence_group->get_finished_sequences().front();
    results.tokens[0] = sequence->get_generated_ids();
    results.scores[0] = sequence->get_cumulative_log_prob();
    m_chat_generation_finish_status = sequence_group->get_generation_stream()->get_status();
    m_sampler.clear_request_info(sequence_group->get_request_id());

    auto stop_time = std::chrono::steady_clock::now();

    // Update perf metrics
    // If is called without tokenization then that stat will not be reported.
    auto& metrics = results.perf_metrics;
    metrics.num_input_tokens = batch_size * input_ids.get_shape().at(1);
    metrics.load_time = this->m_load_time_ms;
    metrics.raw_metrics.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
    metrics.evaluate_statistics(start_time);
    return results;
}

void StatefulLLMPipeline::start_chat(const std::string& system_message) {
    if (!system_message.empty()) {
        m_history.push_back({{"role", "system"}, {"content", system_message}});
    }
    m_is_chat_conversation = true;
};

void StatefulLLMPipeline::finish_chat() {
    m_is_chat_conversation = false;
    m_history.clear();
};

StatefulLLMPipeline::~StatefulLLMPipeline() {
    m_request.get_compiled_model().release_memory();
}

std::unique_ptr<LLMPipelineImplBase>
LLMPipelineFactory::create(const std::filesystem::path& models_path,
                           const ov::AnyMap& config) {
    return create(models_path, Tokenizer(models_path), config);
}

std::unique_ptr<LLMPipelineImplBase> LLMPipelineFactory::create(const std::shared_ptr<ov::Model>& model,
                                                                const ov::genai::Tokenizer& tokenizer,
                                                                const ov::AnyMap& properties,
                                                                const ov::genai::GenerationConfig& generation_config) {
    auto properties_copy = properties;
    const auto pipeline_mode = str_to_pipeline(utils::pop_or_default(properties_copy, "STATIC_PIPELINE", std::string("STATEFUL")));
    if (pipeline_mode == StaticPipelineKind::STATEFUL) {
        return std::make_unique<ov::genai::static_llm::StatefulLLMPipeline>(model,
                                                                            tokenizer,
                                                                            properties_copy,
                                                                            generation_config);
    }
    OPENVINO_ASSERT(false);
}

std::unique_ptr<LLMPipelineImplBase>
LLMPipelineFactory::create(const std::filesystem::path& models_path,
                           const ov::genai::Tokenizer& tokenizer,
                           const ov::AnyMap& config) {
    auto properties = config;
    const auto pipeline_mode = str_to_pipeline(utils::pop_or_default(properties, "STATIC_PIPELINE", std::string("STATEFUL")));
    if (pipeline_mode == StaticPipelineKind::STATEFUL) {
        return std::make_unique<ov::genai::static_llm::StatefulLLMPipeline>(models_path, tokenizer, properties);
    }
    OPENVINO_ASSERT(false);
}


}  // namespace static_llm
}  // namespace genai
}  // namespace ov
