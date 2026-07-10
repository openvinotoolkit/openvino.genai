// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gemma4_mtp_strategy.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <unordered_set>

#include "continuous_batching/timer.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "openvino/op/result.hpp"

namespace {

bool model_has_ports(const std::shared_ptr<ov::Model>& model,
                     const std::unordered_set<std::string>& required_names,
                     bool check_inputs) {
    if (!model) {
        return false;
    }

    std::unordered_set<std::string> actual_names;
    const auto ports = check_inputs ? model->inputs() : model->outputs();
    for (const auto& port : ports) {
        for (const auto& name : port.get_names()) {
            actual_names.insert(name);
        }
    }

    return std::all_of(required_names.begin(), required_names.end(), [&](const std::string& name) {
        return actual_names.count(name) > 0;
    });
}

ov::genai::StreamingStatus stream_generated_tokens(const std::shared_ptr<ov::genai::StreamerBase>& streamer,
                                                   const std::vector<int64_t>& tokens) {
    if (streamer) {
        return streamer->write(tokens);
    }
    return ov::genai::StreamingStatus{};
}

}  // namespace

namespace ov::genai {

bool is_gemma4_mtp_model_pair(const std::shared_ptr<ov::Model>& target_model,
                              const std::shared_ptr<ov::Model>& draft_model) {
    static const std::unordered_set<std::string> target_outputs = {
        "logits",
        "mtp_last_hidden_state",
        "mtp_full_attention_key",
        "mtp_full_attention_value",
        "mtp_sliding_attention_key",
        "mtp_sliding_attention_value",
    };
    static const std::unordered_set<std::string> draft_inputs = {
        "inputs_embeds",
        "position_ids",
        "attention_mask",
        "full_attention_key",
        "full_attention_value",
        "sliding_attention_key",
        "sliding_attention_value",
    };
    static const std::unordered_set<std::string> draft_outputs = {"logits", "last_hidden_state"};

    return model_has_ports(target_model, target_outputs, false) &&
           model_has_ports(draft_model, draft_inputs, true) &&
           model_has_ports(draft_model, draft_outputs, false);
}

Gemma4MTPTargetWrapper::Gemma4MTPTargetWrapper(const ModelDesc& model_desc)
    : m_device(model_desc.device),
      m_properties(model_desc.properties) {
    OPENVINO_ASSERT(model_desc.model, "Target model must not be null");
    m_kv_axes_pos = utils::get_kv_axes_pos(model_desc.model);
    m_cache_types = utils::get_cache_types(*model_desc.model);
    OPENVINO_ASSERT(!m_cache_types.has_linear(),
                    "Gemma4 MTP stateful speculative decoding does not support linear attention states.");

    auto embedding_model = create_embedding_model(model_desc.model);
    m_request = utils::singleton_core().compile_model(model_desc.model, m_device, m_properties).create_infer_request();
    m_embedding_request = utils::singleton_core().compile_model(embedding_model, m_device, m_properties).create_infer_request();
    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.tokenization_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.detokenization_durations = {MicroSeconds(0.0f)};
}

std::shared_ptr<ov::Model> Gemma4MTPTargetWrapper::create_embedding_model(const std::shared_ptr<ov::Model>& model) const {
    ov::Output<ov::Node> embedding_output;
    for (const auto& node : model->get_ordered_ops()) {
        const std::string name = node->get_friendly_name();
        if (name.find("embed_tokens/aten::mul/Multiply") != std::string::npos &&
            name.find("embed_tokens_per_layer") == std::string::npos && node->get_output_size() > 0) {
            embedding_output = node->output(0);
            break;
        }
    }
    if (!embedding_output.get_node_shared_ptr()) {
        for (const auto& node : model->get_ordered_ops()) {
            const std::string name = node->get_friendly_name();
            if (name.find("embed_tokens/aten::embedding/Gather") != std::string::npos &&
                name.find("embed_tokens_per_layer") == std::string::npos && node->get_output_size() > 0) {
                embedding_output = node->output(0);
                break;
            }
        }
    }
    OPENVINO_ASSERT(embedding_output.get_node_shared_ptr(),
                    "Cannot find Gemma4 token embedding path in target model.");

    ov::ParameterVector parameters;
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == "input_ids") {
            parameters.push_back(parameter);
            break;
        }
    }
    OPENVINO_ASSERT(parameters.size() == 1, "Cannot find input_ids parameter for Gemma4 embedding model.");
    auto result = std::make_shared<ov::op::v0::Result>(embedding_output);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, parameters, "gemma4_mtp_embeddings");
}

uint64_t Gemma4MTPTargetWrapper::execute_inference() {
    const auto start = std::chrono::steady_clock::now();
    m_request.infer();
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
}

void Gemma4MTPTargetWrapper::update_inference_time(uint64_t inference_time_us) {
    m_raw_perf_metrics.m_durations.emplace_back(static_cast<float>(inference_time_us));
    m_raw_perf_metrics.m_inference_durations[0] += MicroSeconds(static_cast<float>(inference_time_us));
    m_raw_perf_metrics.m_batch_sizes.emplace_back(0u);
}

void Gemma4MTPTargetWrapper::reset_state() {
    m_request.reset_state();
    m_processed_tokens = 0;
    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.m_durations.clear();
    m_raw_perf_metrics.m_batch_sizes.clear();
}

void Gemma4MTPTargetWrapper::release_memory() {
    m_request.get_compiled_model().release_memory();
    m_embedding_request.get_compiled_model().release_memory();
}

Gemma4MTPOutput Gemma4MTPTargetWrapper::infer(const ov::Tensor& input_ids,
                                              const ov::Tensor& attention_mask,
                                              const ov::Tensor& position_ids) {
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);
    m_request.get_tensor("beam_idx").set_shape({BATCH_SIZE});
    m_request.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    const uint64_t inference_time_us = execute_inference();
    update_inference_time(inference_time_us);
    m_processed_tokens += input_ids.get_shape().at(1);

    Gemma4MTPOutput output;
    output.logits = m_request.get_tensor("logits");
    output.hidden_states = m_request.get_tensor("mtp_last_hidden_state");
    output.shared_kv.full_key = m_request.get_tensor("mtp_full_attention_key");
    output.shared_kv.full_value = m_request.get_tensor("mtp_full_attention_value");
    output.shared_kv.sliding_key = m_request.get_tensor("mtp_sliding_attention_key");
    output.shared_kv.sliding_value = m_request.get_tensor("mtp_sliding_attention_value");
    return output;
}

ov::Tensor Gemma4MTPTargetWrapper::embed_token(int64_t token_id) {
    ov::Tensor input_ids(ov::element::i64, {BATCH_SIZE, 1});
    input_ids.data<int64_t>()[0] = token_id;
    m_embedding_request.set_tensor("input_ids", input_ids);
    m_embedding_request.infer();
    return m_embedding_request.get_output_tensor(0);
}

void Gemma4MTPTargetWrapper::crop_state_to_length(size_t target_length) {
    if (target_length >= m_processed_tokens) {
        return;
    }
    utils::CacheState state(m_cache_types);
    state.num_tokens_to_trim = m_processed_tokens - target_length;
    state.seq_length_axis = m_kv_axes_pos.seq_len;
    state.reset_mem_state = false;
    utils::trim_kv_cache(m_request, state, {});
    m_processed_tokens = target_length;
}

Gemma4MTPAssistantWrapper::Gemma4MTPAssistantWrapper(const ModelDesc& model_desc)
    : m_device(model_desc.device),
      m_properties(model_desc.properties) {
    
    OPENVINO_ASSERT(model_desc.model, "Assistant model must not be null");
    OPENVINO_ASSERT(!m_device.empty(), "Assistant device must not be empty.");
    m_request = utils::singleton_core().compile_model(model_desc.model, m_device, m_properties).create_infer_request();
    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.tokenization_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.detokenization_durations = {MicroSeconds(0.0f)};
}

uint64_t Gemma4MTPAssistantWrapper::execute_inference() {
    const auto start = std::chrono::steady_clock::now();
    m_request.infer();
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
}

void Gemma4MTPAssistantWrapper::update_inference_time(uint64_t inference_time_us) {
    m_raw_perf_metrics.m_durations.emplace_back(static_cast<float>(inference_time_us));
    m_raw_perf_metrics.m_inference_durations[0] += MicroSeconds(static_cast<float>(inference_time_us));
    m_raw_perf_metrics.m_batch_sizes.emplace_back(1u);
}

void Gemma4MTPAssistantWrapper::reset_state() {
    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.m_durations.clear();
    m_raw_perf_metrics.m_batch_sizes.clear();
}

void Gemma4MTPAssistantWrapper::release_memory() {
    m_request.get_compiled_model().release_memory();
}

Gemma4MTPOutput Gemma4MTPAssistantWrapper::infer(const ov::Tensor& inputs_embeds,
                                                 const ov::Tensor& attention_mask,
                                                 const ov::Tensor& position_ids,
                                                 const Gemma4MTPSharedKV& shared_kv) {
    m_request.set_tensor("inputs_embeds", inputs_embeds);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);
    m_request.set_tensor("full_attention_key", shared_kv.full_key);
    m_request.set_tensor("full_attention_value", shared_kv.full_value);
    m_request.set_tensor("sliding_attention_key", shared_kv.sliding_key);
    m_request.set_tensor("sliding_attention_value", shared_kv.sliding_value);
    const uint64_t inference_time_us = execute_inference();
    update_inference_time(inference_time_us);

    Gemma4MTPOutput output;
    output.logits = m_request.get_tensor("logits");
    output.hidden_states = m_request.get_tensor("last_hidden_state");
    return output;
}

StatefulGemma4MTPLLMPipeline::StatefulGemma4MTPLLMPipeline(const ModelDesc& target_model_desc,
                                                           const ModelDesc& draft_model_desc)
    : StatefulSpeculativePipelineBase(target_model_desc.tokenizer, target_model_desc.generation_config) {
    OPENVINO_ASSERT(is_gemma4_mtp_model_pair(target_model_desc.model, draft_model_desc.model),
                    "Target and draft models do not match the Gemma4 MTP OpenVINO IR contract.");
    OPENVINO_ASSERT(target_model_desc.device != "NPU" && draft_model_desc.device != "NPU",
                    "Gemma4 MTP stateful speculative decoding currently supports CPU/GPU only.");

    if (m_generation_config.num_assistant_tokens == 0 && draft_model_desc.generation_config.num_assistant_tokens > 0) {
        m_generation_config.num_assistant_tokens = draft_model_desc.generation_config.num_assistant_tokens;
    }
    ensure_num_assistant_tokens_is_set(m_generation_config);
    m_num_assistant_tokens = m_generation_config.num_assistant_tokens;

    auto assistant_desc = draft_model_desc;
    if (assistant_desc.device.empty()) {
        assistant_desc.device = target_model_desc.device;
    }
    if (assistant_desc.properties.empty() && assistant_desc.device == target_model_desc.device) {
        assistant_desc.properties = target_model_desc.properties;
    }
    m_target = std::make_unique<Gemma4MTPTargetWrapper>(target_model_desc);
    m_assistant = std::make_unique<Gemma4MTPAssistantWrapper>(assistant_desc);
}

StatefulGemma4MTPLLMPipeline::~StatefulGemma4MTPLLMPipeline() {
    m_target->release_memory();
    m_assistant->release_memory();
}

GenerationConfig StatefulGemma4MTPLLMPipeline::resolve_generation_config(OptionalGenerationConfig generation_config) {
    GenerationConfig config = StatefulSpeculativePipelineBase::resolve_generation_config(generation_config);
    if (config.num_assistant_tokens == 0) {
        config.num_assistant_tokens = m_generation_config.num_assistant_tokens;
    }
    ensure_num_assistant_tokens_is_set(config);
    m_num_assistant_tokens = config.num_assistant_tokens;
    return config;
}

Gemma4MTPSharedKV StatefulGemma4MTPLLMPipeline::crop_shared_kv(const Gemma4MTPSharedKV& shared_kv,
                                                               size_t accepted_length) const {
    const size_t sequence_axis = m_target->get_kv_sequence_axis();
    auto crop = [accepted_length, sequence_axis](const ov::Tensor& tensor) {
        const ov::Shape shape = tensor.get_shape();
        OPENVINO_ASSERT(shape.size() == 4 && sequence_axis < shape.size() && shape[sequence_axis] >= accepted_length,
                        "Invalid Gemma4 shared KV shape. Expected rank-4 tensor with sequence axis ",
                        sequence_axis,
                        ": ",
                        shape,
                        ", accepted length: ",
                        accepted_length);
        ov::Coordinate end = shape;
        end[sequence_axis] = accepted_length;
        return ov::Tensor(tensor, ov::Coordinate(shape.size(), 0), end);
    };
    return Gemma4MTPSharedKV{crop(shared_kv.full_key), crop(shared_kv.full_value), crop(shared_kv.sliding_key), crop(shared_kv.sliding_value)};
}

ov::Tensor StatefulGemma4MTPLLMPipeline::select_hidden_state(const ov::Tensor& hidden_states, size_t position) const {
    const ov::Shape shape = hidden_states.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && position < shape[1], "Invalid Gemma4 hidden state shape.");
    return ov::Tensor(hidden_states, ov::Coordinate{0, position, 0}, ov::Coordinate{shape[0], position + 1, shape[2]});
}

ov::Tensor StatefulGemma4MTPLLMPipeline::concatenate_embedding_and_hidden(const ov::Tensor& embedding,
                                                                          const ov::Tensor& hidden_state) {
    const ov::Shape embedding_shape = embedding.get_shape();
    const ov::Shape hidden_shape = hidden_state.get_shape();
    OPENVINO_ASSERT(embedding_shape.size() == 3 && hidden_shape.size() == 3,
                    "Gemma4 assistant input tensors must be rank 3.");
    OPENVINO_ASSERT(embedding_shape[0] == hidden_shape[0] && embedding_shape[1] == hidden_shape[1],
                    "Gemma4 embedding and hidden state shape mismatch.");
    OPENVINO_ASSERT(embedding.get_element_type() == ov::element::f32 && hidden_state.get_element_type() == ov::element::f32,
                    "Gemma4 assistant input tensors must be f32, got embedding=",
                    embedding.get_element_type(), ", hidden_state=", hidden_state.get_element_type(), ".");

    const ov::Shape inputs_embeds_shape = {embedding_shape[0], embedding_shape[1], embedding_shape[2] + hidden_shape[2]};
    if (!m_inputs_embeds_buffer || m_inputs_embeds_buffer.get_shape() != inputs_embeds_shape) {
        m_inputs_embeds_buffer = ov::Tensor(ov::element::f32, inputs_embeds_shape);
    }
    float* destination = m_inputs_embeds_buffer.data<float>();
    std::copy_n(embedding.data<const float>(), embedding_shape[2], destination);
    std::copy_n(hidden_state.data<const float>(), hidden_shape[2], destination + embedding_shape[2]);
    return m_inputs_embeds_buffer;
}

std::vector<int64_t> StatefulGemma4MTPLLMPipeline::sample_greedy_tokens(const ov::Tensor& logits, size_t token_count) const {
    const ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && token_count <= shape[1], "Invalid Gemma4 logits shape.");
    OPENVINO_ASSERT(logits.get_element_type() == ov::element::f32,
                    "Gemma4 logits must be f32, got ", logits.get_element_type(), ".");
    const size_t vocab_size = shape[2];
    const float* data = logits.data<const float>();
    std::vector<int64_t> tokens;
    tokens.reserve(token_count);
    for (size_t position = shape[1] - token_count; position < shape[1]; ++position) {
        const float* logits_begin = data + position * vocab_size;
        tokens.push_back(static_cast<int64_t>(std::max_element(logits_begin, logits_begin + vocab_size) - logits_begin));
    }
    return tokens;
}

bool StatefulGemma4MTPLLMPipeline::is_stop_token(int64_t token, const GenerationConfig& config) const {
    if (!config.ignore_eos && token == config.eos_token_id) {
        return true;
    }
    return std::find(config.stop_token_ids.begin(), config.stop_token_ids.end(), token) != config.stop_token_ids.end();
}

StatefulGemma4MTPLLMPipeline::DraftResult StatefulGemma4MTPLLMPipeline::draft_tokens(
    const GenerationConfig& config,
    const Gemma4MTPOutput& previous_target_output,
    const std::vector<int64_t>& accepted_tokens,
    size_t n_last_matches,
    size_t remaining_tokens) {
    DraftResult result;
    const size_t draft_limit = std::min(m_num_assistant_tokens, remaining_tokens > 0 ? remaining_tokens - 1 : 0);
    if (draft_limit == 0) {
        return result;
    }

    const size_t cur_len = accepted_tokens.size();
    Gemma4MTPSharedKV shared_kv = crop_shared_kv(previous_target_output.shared_kv, cur_len - 1);
    ov::Tensor last_hidden_state = select_hidden_state(previous_target_output.hidden_states, n_last_matches);
    ov::Tensor attention_mask(ov::element::i64, {1, cur_len});
    std::fill_n(attention_mask.data<int64_t>(), cur_len, 1);
    ov::Tensor position_ids(ov::element::i64, {1, 1});
    position_ids.data<int64_t>()[0] = static_cast<int64_t>(cur_len - 1);
    int64_t last_token_id = accepted_tokens.back();

    result.tokens.reserve(draft_limit);
    for (size_t i = 0; i < draft_limit; ++i) {
        ov::Tensor inputs_embeds = concatenate_embedding_and_hidden(m_target->embed_token(last_token_id), last_hidden_state);
        Gemma4MTPOutput assistant_output = m_assistant->infer(inputs_embeds, attention_mask, position_ids, shared_kv);
        last_token_id = sample_greedy_tokens(assistant_output.logits, 1).front();
        result.tokens.push_back(last_token_id);
        last_hidden_state = assistant_output.hidden_states;
        if (is_stop_token(last_token_id, config)) {
            break;
        }
    }
    return result;
}

EncodedResults StatefulGemma4MTPLLMPipeline::generate_tokens(const EncodedInputs& inputs,
                                                             const GenerationConfig& config,
                                                             StreamerVariant streamer) {
    ManualTimer generate_timer("StatefulGemma4MTPLLMPipeline::generate_tokens");
    generate_timer.start();
    OPENVINO_ASSERT(config.is_greedy_decoding(), "Gemma4 MTP stateful speculative decoding supports greedy decoding only.");
    OPENVINO_ASSERT(config.num_return_sequences == 1u,
                    "Gemma4 MTP stateful speculative decoding supports num_return_sequences=1 only.");

    ov::Tensor input_ids;
    ov::Tensor attention_mask;
    if (const auto* tensor_input = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *tensor_input;
        attention_mask = utils::init_attention_mask(input_ids);
    } else if (const auto* tokenized_input = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = tokenized_input->input_ids;
        attention_mask = tokenized_input->attention_mask;
    }

    const ov::Shape input_shape = input_ids.get_shape();
    OPENVINO_ASSERT(input_shape.size() == 2 && input_shape[0] == 1 && input_shape[1] > 0,
                    "Gemma4 MTP supports batch size 1 and non-empty prompts only.");
    m_prompt_length = input_shape[1];

    std::vector<int64_t> accepted_tokens(input_ids.data<const int64_t>(), input_ids.data<const int64_t>() + input_ids.get_size());
    std::vector<int64_t> generated_tokens;
    generated_tokens.reserve(config.get_max_new_tokens());
    std::shared_ptr<StreamerBase> streamer_ptr = utils::create_streamer(streamer, m_tokenizer);
    StreamingStatus streaming_status = StreamingStatus::RUNNING;

    m_target->reset_state();
    m_assistant->reset_state();
    ov::Tensor position_ids(ov::element::i64, input_shape);
    utils::initialize_position_ids(position_ids, attention_mask);
    Gemma4MTPOutput target_output = m_target->infer(input_ids, attention_mask, position_ids);

    int64_t first_token = sample_greedy_tokens(target_output.logits, 1).front();
    accepted_tokens.push_back(first_token);
    generated_tokens.push_back(first_token);
    m_target->get_raw_perf_metrics().m_batch_sizes.back() = 1u;
    m_target->crop_state_to_length(accepted_tokens.size() - 1);
    streaming_status = stream_generated_tokens(streamer_ptr, {first_token});

    size_t n_last_matches = 0;
    size_t total_draft_generated = 0;
    size_t total_draft_accepted = 0;
    bool eos_reached = is_stop_token(first_token, config);

    while (!eos_reached && generated_tokens.size() < config.get_max_new_tokens() && streaming_status == StreamingStatus::RUNNING) {
        const size_t remaining_tokens = config.get_max_new_tokens() - generated_tokens.size();
        DraftResult draft_result = draft_tokens(config, target_output, accepted_tokens, n_last_matches, remaining_tokens);
        total_draft_generated += draft_result.tokens.size();

        std::vector<int64_t> candidate_tokens;
        candidate_tokens.reserve(draft_result.tokens.size() + 1);
        candidate_tokens.push_back(accepted_tokens.back());
        candidate_tokens.insert(candidate_tokens.end(), draft_result.tokens.begin(), draft_result.tokens.end());

        ov::Tensor candidate_input_ids(ov::element::i64, {1, candidate_tokens.size()});
        std::copy(candidate_tokens.begin(), candidate_tokens.end(), candidate_input_ids.data<int64_t>());
        const size_t previous_length = accepted_tokens.size();
        ov::Tensor candidate_attention_mask(ov::element::i64, {1, previous_length + draft_result.tokens.size()});
        std::fill_n(candidate_attention_mask.data<int64_t>(), candidate_attention_mask.get_size(), 1);
        ov::Tensor candidate_position_ids(ov::element::i64, {1, candidate_tokens.size()});
        std::iota(candidate_position_ids.data<int64_t>(),
              candidate_position_ids.data<int64_t>() + candidate_tokens.size(),
              static_cast<int64_t>(previous_length - 1));

        target_output = m_target->infer(candidate_input_ids, candidate_attention_mask, candidate_position_ids);
        std::vector<int64_t> target_tokens = sample_greedy_tokens(target_output.logits, draft_result.tokens.size() + 1);

        n_last_matches = 0;
        while (n_last_matches < draft_result.tokens.size() && draft_result.tokens[n_last_matches] == target_tokens[n_last_matches]) {
            ++n_last_matches;
        }
        total_draft_accepted += n_last_matches;

        std::vector<int64_t> validated_tokens(target_tokens.begin(), target_tokens.begin() + n_last_matches + 1);
        if (generated_tokens.size() + validated_tokens.size() > config.get_max_new_tokens()) {
            validated_tokens.resize(config.get_max_new_tokens() - generated_tokens.size());
        }
        accepted_tokens.insert(accepted_tokens.end(), validated_tokens.begin(), validated_tokens.end());
        generated_tokens.insert(generated_tokens.end(), validated_tokens.begin(), validated_tokens.end());
        m_target->get_raw_perf_metrics().m_batch_sizes.back() = validated_tokens.size();
        m_target->crop_state_to_length(accepted_tokens.size() - 1);

        streaming_status = stream_generated_tokens(streamer_ptr, validated_tokens);
        eos_reached = std::any_of(validated_tokens.begin(), validated_tokens.end(), [&](int64_t token) {
            return is_stop_token(token, config);
        });
    }

    m_streaming_was_cancelled = streaming_status == StreamingStatus::CANCEL;
    if (streamer_ptr) {
        streamer_ptr->end();
    }

    EncodedResults results;
    results.tokens = {generated_tokens};
    results.scores = {0.0f};
    results.finish_reasons = {GenerationFinishReason::NONE};
    if (streaming_status == StreamingStatus::TOOL_CALL_STOP) {
        results.finish_reasons[0] = GenerationFinishReason::TOOL_CALL;
    } else if (streaming_status == StreamingStatus::STOP || eos_reached) {
        results.finish_reasons[0] = GenerationFinishReason::STOP;
    } else if (generated_tokens.size() >= config.get_max_new_tokens()) {
        results.finish_reasons[0] = GenerationFinishReason::LENGTH;
    }

    generate_timer.end();
    m_sd_perf_metrics.num_input_tokens = m_prompt_length;
    m_sd_perf_metrics.load_time = m_load_time_ms;
    m_sd_perf_metrics.num_accepted_tokens = total_draft_accepted;
    m_sd_perf_metrics.raw_metrics.generate_durations.clear();
    m_sd_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_timer.get_duration_microsec());

    m_sd_perf_metrics.m_evaluated = false;
    m_sd_perf_metrics.main_model_metrics.m_evaluated = false;
    m_sd_perf_metrics.draft_model_metrics.m_evaluated = false;

    m_sd_perf_metrics.main_model_metrics.raw_metrics = m_target->get_raw_perf_metrics();
    m_sd_perf_metrics.draft_model_metrics.raw_metrics = m_assistant->get_raw_perf_metrics();
    if (total_draft_generated > 0) {
        m_sd_metrics.update_acceptance_rate(0, static_cast<float>(total_draft_accepted) * 100.0f / total_draft_generated);
        m_sd_metrics.update_draft_accepted_tokens(0, total_draft_accepted);
        m_sd_metrics.update_draft_generated_len(0, total_draft_generated);
        m_sd_metrics.update_generated_len(generated_tokens.size());
    }
    m_sd_perf_metrics.evaluate_statistics(generate_timer.get_start_time());
    results.perf_metrics = m_sd_perf_metrics;
    results.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_sd_perf_metrics);

    if (!m_is_chat_active) {
        m_target->reset_state();
    }
    return results;
}

void StatefulGemma4MTPLLMPipeline::finish_chat() {
    StatefulSpeculativePipelineBase::finish_chat();
    m_target->reset_state();
}

SpeculativeDecodingMetrics StatefulGemma4MTPLLMPipeline::get_speculative_decoding_metrics() const {
    return m_sd_metrics;
}

}  // namespace ov::genai
