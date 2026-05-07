// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_strategy.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>

#include "continuous_batching/timer.hpp"
#include "generation_stream.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

namespace {

void copy_bytes(const ov::Tensor& src, ov::Tensor& dst) {
    std::memcpy(dst.data(), src.data(), src.get_byte_size());
}

bool has_compiled_input(const ov::CompiledModel& model, const std::string& name) {
    const auto inputs = model.inputs();
    return std::find_if(inputs.begin(), inputs.end(), [&](const ov::Output<const ov::Node>& port) {
        return port.get_names().count(name) != 0;
    }) != inputs.end();
}

}  // namespace

void DFlashHiddenStateProvider::reset() {
    m_context = ov::Tensor();
}

size_t DFlashHiddenStateProvider::context_length() const {
    if (!m_context || m_context.get_size() == 0) {
        return 0;
    }
    const auto shape = m_context.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "DFlash hidden context must have rank 3.");
    return shape[1];
}

void DFlashHiddenStateProvider::append(const ov::Tensor& hidden_states, size_t token_count) {
    OPENVINO_ASSERT(hidden_states && hidden_states.get_size() > 0, "Cannot append empty DFlash hidden states.");
    const auto shape = hidden_states.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "DFlash hidden states must have shape [1, seq_len, hidden].");
    token_count = std::min(token_count, shape[1]);
    if (token_count == 0) {
        return;
    }

    ov::Tensor source = hidden_states;
    ov::Shape source_shape = shape;
    source_shape[1] = token_count;
    if (token_count != shape[1]) {
        // Keep only the hidden states for tokens that were accepted/finalized.
        source = ov::Tensor(hidden_states, ov::Coordinate{0, 0, 0}, ov::Coordinate{1, token_count, shape[2]});
    }

    if (!m_context || m_context.get_size() == 0) {
        m_context = ov::Tensor(source.get_element_type(), source_shape);
        copy_bytes(source, m_context);
        return;
    }

    const auto old_shape = m_context.get_shape();
    OPENVINO_ASSERT(old_shape[0] == 1 && old_shape[2] == source_shape[2],
                    "Cannot append DFlash hidden states with incompatible hidden size.");
    ov::Shape new_shape = old_shape;
    new_shape[1] += token_count;
    // Stage 1 keeps a full-context target_hidden tensor for the stateless DFlash draft.
    // Reallocate here to keep the storage contiguous for the draft model input.
    ov::Tensor merged(m_context.get_element_type(), new_shape);
    ov::Tensor old_dst(merged, ov::Coordinate{0, 0, 0}, ov::Coordinate{1, old_shape[1], old_shape[2]});
    ov::Tensor new_dst(merged, ov::Coordinate{0, old_shape[1], 0}, ov::Coordinate{1, new_shape[1], new_shape[2]});
    copy_bytes(m_context, old_dst);
    copy_bytes(source, new_dst);
    m_context = merged;
}

void DFlashHiddenStateProvider::truncate(size_t context_length) {
    if (context_length >= this->context_length()) {
        return;
    }
    const auto shape = m_context.get_shape();
    // Materialize the ROI so later appends do not keep a view into a larger discarded buffer.
    ov::Tensor view(m_context, ov::Coordinate{0, 0, 0}, ov::Coordinate{1, context_length, shape[2]});
    ov::Tensor copied(m_context.get_element_type(), view.get_shape());
    copy_bytes(view, copied);
    m_context = copied;
}

DFlashSamplerAdapter::DFlashSamplerAdapter(const Tokenizer& tokenizer) : m_sampler(tokenizer) {}

void DFlashSamplerAdapter::clear(uint64_t request_id) {
    m_sampler.clear_request_info(request_id);
}

std::vector<int64_t> DFlashSamplerAdapter::sample(SequenceGroup::Ptr sequence_group,
                                                  const ov::Tensor& logits,
                                                  size_t input_token_count,
                                                  size_t sample_count,
                                                  size_t num_tokens_to_validate,
                                                  bool validation_mode) {
    if (sample_count == 0) {
        return {};
    }

    const auto sequence = (*sequence_group)[0];
    const size_t generated_before = sequence->get_generated_len();
    ov::Tensor sliced_logits = logits;
    const auto shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "DFlash sampler expects logits [1, seq, vocab].");
    if (sample_count < shape[1]) {
        sliced_logits = ov::Tensor(logits,
                                  ov::Coordinate{0, shape[1] - sample_count, 0},
                                  ov::Coordinate{1, shape[1], shape[2]});
    }

    sequence_group->schedule_tokens(input_token_count);
    sequence_group->set_output_seq_len(sample_count);
    sequence_group->set_num_validated_tokens(num_tokens_to_validate);
    m_sampler.sample({sequence_group}, sliced_logits, validation_mode);
    sequence_group->finish_iteration();

    const auto& generated = sequence->get_generated_ids();
    const size_t generated_after = generated.size();
    // In validation mode the sampler may remove rejected draft tokens before appending the fallback token.
    const size_t result_count = validation_mode
                                    ? generated_after - generated_before + num_tokens_to_validate
                                    : (generated_after > generated_before ? generated_after - generated_before : 0);
    return std::vector<int64_t>(generated.end() - result_count, generated.end());
}

DFlashTargetWrapper::DFlashTargetWrapper(const ModelDesc& model_desc)
    : m_device(model_desc.device),
      m_properties(model_desc.properties),
      m_tokenizer(model_desc.tokenizer),
      m_sampler(model_desc.tokenizer) {
    m_kv_axes_pos = utils::get_kv_axes_pos(model_desc.model);
    m_cache_types = utils::get_cache_types(*model_desc.model);
    OPENVINO_ASSERT(!m_cache_types.has_linear(),
                    "DFlash stateful speculative decoding does not support linear attention states.");

    if (m_device == "NPU") {
        auto [compiled, kv_desc] = utils::compile_decoder_for_npu(model_desc.model, m_properties, m_kv_axes_pos);
        m_request = compiled.create_infer_request();
    } else {
        m_request = utils::singleton_core().compile_model(model_desc.model, m_device, m_properties).create_infer_request();
    }

    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.tokenization_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.detokenization_durations = {MicroSeconds(0.0f)};
}

void DFlashTargetWrapper::initialize_sequence(const ov::Tensor& input_ids, const GenerationConfig& config) {
    const auto shape = input_ids.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] == 1 && shape[1] > 0, "Expected input_ids shape [1, seq_len]");
    const int64_t* ids_data = input_ids.data<const int64_t>();
    TokenIds prompt_ids(ids_data, ids_data + shape[1]);
    m_sequence_group = std::make_shared<SequenceGroup>(0, prompt_ids, config, 0);
}

Sequence::Ptr DFlashTargetWrapper::get_current_sequence() const {
    if (!m_sequence_group || m_sequence_group->num_total_seqs() == 0) {
        return nullptr;
    }
    return (*m_sequence_group)[0];
}

const std::vector<int64_t>& DFlashTargetWrapper::get_generated_tokens() const {
    static const std::vector<int64_t> empty;
    auto seq = get_current_sequence();
    return seq ? seq->get_generated_ids() : empty;
}

size_t DFlashTargetWrapper::get_sequence_length() const {
    auto seq = get_current_sequence();
    return seq ? m_sequence_group->get_prompt_len() + seq->get_generated_len() : 0;
}

void DFlashTargetWrapper::append_tokens(const std::vector<int64_t>& tokens) {
    auto seq = get_current_sequence();
    OPENVINO_ASSERT(seq, "DFlash target sequence is not initialized.");
    for (auto token : tokens) {
        seq->append_token(token, 0.0f);
    }
}

void DFlashTargetWrapper::truncate_sequence(size_t size) {
    auto seq = get_current_sequence();
    OPENVINO_ASSERT(seq, "DFlash target sequence is not initialized.");
    const size_t prompt_len = m_sequence_group->get_prompt_len();
    const size_t current_len = prompt_len + seq->get_generated_len();
    if (size < current_len) {
        OPENVINO_ASSERT(size >= prompt_len, "Cannot truncate prompt tokens.");
        seq->remove_last_tokens(current_len - size);
    }
}

void DFlashTargetWrapper::trim_kv_cache(size_t tokens_to_remove) {
    const size_t current_len = get_sequence_length();
    if (tokens_to_remove == 0 || current_len == 0 || m_device == "NPU") {
        return;
    }
    utils::CacheState state(m_cache_types);
    state.num_tokens_to_trim = tokens_to_remove;
    state.seq_length_axis = m_kv_axes_pos.seq_len;
    utils::trim_kv_cache(m_request, state, {});
}

void DFlashTargetWrapper::reset_state() {
    m_sequence_group = nullptr;
    m_request.reset_state();
    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.m_durations.clear();
    m_raw_perf_metrics.m_batch_sizes.clear();
}

void DFlashTargetWrapper::release_memory() {
    m_request.get_compiled_model().release_memory();
}

void DFlashTargetWrapper::build_model_inputs(size_t input_token_count,
                                             ov::Tensor& input_ids,
                                             ov::Tensor& attention_mask,
                                             ov::Tensor& position_ids) {
    auto seq = get_current_sequence();
    OPENVINO_ASSERT(seq, "DFlash target sequence is not initialized.");
    const auto& prompt_ids = m_sequence_group->get_prompt_ids();
    const auto& generated_ids = seq->get_generated_ids();
    const size_t prompt_len = prompt_ids.size();
    const size_t total_len = prompt_len + generated_ids.size();
    OPENVINO_ASSERT(input_token_count > 0 && input_token_count <= total_len, "Invalid DFlash input token count.");
    const size_t start_pos = total_len - input_token_count;

    input_ids = ov::Tensor(ov::element::i64, {BATCH_SIZE, input_token_count});
    position_ids = ov::Tensor(ov::element::i64, {BATCH_SIZE, input_token_count});
    auto* ids = input_ids.data<int64_t>();
    auto* pos = position_ids.data<int64_t>();

    // Reconstruct the requested suffix from prompt + generated sequence state.
    for (size_t idx = 0; idx < input_token_count; ++idx) {
        const size_t absolute_pos = start_pos + idx;
        ids[idx] = absolute_pos < prompt_len ? prompt_ids[absolute_pos] : generated_ids[absolute_pos - prompt_len];
        pos[idx] = static_cast<int64_t>(absolute_pos);
    }

    attention_mask = ov::Tensor(ov::element::i64, {BATCH_SIZE, total_len});
    std::fill_n(attention_mask.data<int64_t>(), total_len, 1);
}

ov::Tensor DFlashTargetWrapper::get_logits() const {
    return m_request.get_tensor("logits");
}

ov::Tensor DFlashTargetWrapper::get_hidden_features() const {
    auto hidden = m_request.get_tensor("last_hidden_state");
    const auto shape = hidden.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Invalid DFlash hidden state shape.");
    const size_t input_len = m_request.get_tensor("input_ids").get_shape()[1];
    if (shape[1] == input_len) {
        return hidden;
    }
    // Some transformed models may return the whole prefix; DFlash only needs this inference suffix.
    return ov::Tensor(hidden, ov::Coordinate{0, shape[1] - input_len, 0}, ov::Coordinate{1, shape[1], shape[2]});
}

uint64_t DFlashTargetWrapper::execute_inference() {
    auto start = std::chrono::steady_clock::now();
    m_request.infer();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
}

void DFlashTargetWrapper::update_inference_time(uint64_t inference_time_us) {
    m_raw_perf_metrics.m_durations.emplace_back(static_cast<float>(inference_time_us));
    m_raw_perf_metrics.m_inference_durations[0] += MicroSeconds(static_cast<float>(inference_time_us));
}

DFlashInferenceOutput DFlashTargetWrapper::infer(const ov::Tensor& input_ids,
                                                 const ov::Tensor& attention_mask,
                                                 const ov::Tensor& position_ids) {
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);
    // CPU KV-cache attention uses beam_idx to map each batch item to its cache state.
    // This DFlash POC currently supports a single sequence only, so all active rows map to state 0.
    if (has_compiled_input(m_request.get_compiled_model(), "beam_idx")) {
        ov::Tensor beam_idx(ov::element::i32, {input_ids.get_shape()[0]});
        std::fill_n(beam_idx.data<int32_t>(), beam_idx.get_size(), 0);
        m_request.set_tensor("beam_idx", beam_idx);
    }
    update_inference_time(execute_inference());
    return {get_logits(), get_hidden_features()};
}

DFlashInferResult DFlashTargetWrapper::forward(size_t input_token_count,
                                               size_t sample_count,
                                               size_t num_tokens_to_validate) {
    ov::Tensor input_ids, attention_mask, position_ids;
    build_model_inputs(input_token_count, input_ids, attention_mask, position_ids);
    auto output = infer(input_ids, attention_mask, position_ids);
    auto sampled = m_sampler.sample(m_sequence_group,
                                    output.logits,
                                    input_token_count,
                                    sample_count,
                                    num_tokens_to_validate,
                                    num_tokens_to_validate > 0);
    m_raw_perf_metrics.m_batch_sizes.emplace_back(sampled.size());
    return {std::move(output), std::move(sampled)};
}

DFlashDraftWrapper::DFlashDraftWrapper(const ModelDesc& model_desc,
                                       const Tokenizer& tokenizer,
                                       const utils::dflash::DFlashRTInfo& rt_info)
    : m_device(model_desc.device),
      m_properties(model_desc.properties),
      m_tokenizer(tokenizer),
      m_sampler(tokenizer),
      m_block_size(rt_info.block_size),
      m_mask_token_id(rt_info.mask_token_id) {
    OPENVINO_ASSERT(m_block_size > 1, "DFlash block_size must be greater than 1.");
    if (m_device == "NPU") {
        auto kv_axes_pos = utils::get_kv_axes_pos(model_desc.model);
        auto [compiled, kv_desc] = utils::compile_decoder_for_npu(model_desc.model, m_properties, kv_axes_pos);
        m_request = compiled.create_infer_request();
    } else {
        m_request = utils::singleton_core().compile_model(model_desc.model, m_device, m_properties).create_infer_request();
    }
    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.tokenization_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.detokenization_durations = {MicroSeconds(0.0f)};
}

void DFlashDraftWrapper::initialize_sequence(const ov::Tensor& input_ids, const GenerationConfig& config) {
    const auto shape = input_ids.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] == 1 && shape[1] > 0, "Expected input_ids shape [1, seq_len]");
    const int64_t* ids_data = input_ids.data<const int64_t>();
    m_prompt_length = shape[1];
    TokenIds prompt_ids(ids_data, ids_data + shape[1]);
    m_sequence_group = std::make_shared<SequenceGroup>(1, prompt_ids, config, 0);
}

void DFlashDraftWrapper::append_tokens(const std::vector<int64_t>& tokens) {
    auto seq = (*m_sequence_group)[0];
    for (auto token : tokens) {
        seq->append_token(token, 0.0f);
    }
}

void DFlashDraftWrapper::sync_generated_tokens(const std::vector<int64_t>& target_generated_tokens) {
    auto seq = (*m_sequence_group)[0];
    if (seq->get_generated_len() > 0) {
        seq->remove_last_tokens(seq->get_generated_len());
    }
    append_tokens(target_generated_tokens);
    seq->set_status(SequenceStatus::RUNNING);
}

void DFlashDraftWrapper::reset_state() {
    m_sequence_group = nullptr;
    m_request.reset_state();
    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.m_durations.clear();
    m_raw_perf_metrics.m_batch_sizes.clear();
}

void DFlashDraftWrapper::release_memory() {
    m_request.get_compiled_model().release_memory();
}

ov::Tensor DFlashDraftWrapper::build_input_ids(int64_t seed_token) const {
    ov::Tensor input_ids(ov::element::i64, {BATCH_SIZE, m_block_size});
    auto* data = input_ids.data<int64_t>();
    data[0] = seed_token;
    // DFlash predicts the rest of the block from mask placeholders in one draft inference.
    std::fill(data + 1, data + m_block_size, m_mask_token_id);
    return input_ids;
}

ov::Tensor DFlashDraftWrapper::build_position_ids(size_t context_length) const {
    ov::Tensor position_ids(ov::element::i64, {BATCH_SIZE, context_length + m_block_size});
    auto* data = position_ids.data<int64_t>();
    std::iota(data, data + position_ids.get_size(), 0);
    return position_ids;
}

ov::Tensor DFlashDraftWrapper::get_logits() const {
    return m_request.get_tensor("logits");
}

uint64_t DFlashDraftWrapper::execute_inference() {
    auto start = std::chrono::steady_clock::now();
    m_request.infer();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
}

void DFlashDraftWrapper::update_inference_time(uint64_t inference_time_us) {
    m_raw_perf_metrics.m_durations.emplace_back(static_cast<float>(inference_time_us));
    m_raw_perf_metrics.m_inference_durations[0] += MicroSeconds(static_cast<float>(inference_time_us));
}

DFlashInferenceOutput DFlashDraftWrapper::infer(int64_t seed_token, const ov::Tensor& target_hidden) {
    OPENVINO_ASSERT(target_hidden && target_hidden.get_size() > 0, "DFlash target_hidden must be provided.");
    auto input_ids = build_input_ids(seed_token);
    auto position_ids = build_position_ids(target_hidden.get_shape()[1]);
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("target_hidden", target_hidden);
    m_request.set_tensor("position_ids", position_ids);
    update_inference_time(execute_inference());
    return {get_logits(), ov::Tensor()};
}

std::vector<int64_t> DFlashDraftWrapper::sample_candidates(const ov::Tensor& logits, size_t candidate_count) {
    candidate_count = std::min(candidate_count, m_block_size - 1);
    std::vector<int64_t> candidates;
    candidates.reserve(candidate_count);
    const auto shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[1] >= candidate_count, "Invalid DFlash draft logits shape.");
    for (size_t idx = 0; idx < candidate_count; ++idx) {
        // Reuse the regular sampler one DFlash position at a time to keep sampling behavior consistent.
        ov::Tensor one_position(logits,
                                ov::Coordinate{0, idx, 0},
                                ov::Coordinate{1, idx + 1, shape[2]});
        auto sampled = m_sampler.sample(m_sequence_group, one_position, 1, 1);
        candidates.insert(candidates.end(), sampled.begin(), sampled.end());
    }
    m_raw_perf_metrics.m_batch_sizes.emplace_back(candidates.size());
    return candidates;
}

StatefulDFlashLLMPipeline::StatefulDFlashLLMPipeline(const ModelDesc& target_model_desc,
                                                     const ModelDesc& draft_model_desc,
                                                     const utils::dflash::DFlashRTInfo& rt_info)
    : StatefulSpeculativePipelineBase(target_model_desc.tokenizer, target_model_desc.generation_config),
      m_rt_info(rt_info) {
    OPENVINO_ASSERT(m_rt_info.dflash_mode, "DFlash pipeline requires dflash_mode=true.");
    OPENVINO_ASSERT(!m_rt_info.target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");

    auto target_desc = target_model_desc;
    utils::dflash::expose_target_hidden_states(target_desc.model, m_rt_info.target_layer_ids);
    m_target = std::make_unique<DFlashTargetWrapper>(target_desc);
    m_draft = std::make_unique<DFlashDraftWrapper>(draft_model_desc, target_model_desc.tokenizer, m_rt_info);
}

StatefulDFlashLLMPipeline::~StatefulDFlashLLMPipeline() {
    m_target->release_memory();
    m_draft->release_memory();
}

GenerationConfig StatefulDFlashLLMPipeline::resolve_generation_config(OptionalGenerationConfig generation_config) {
    GenerationConfig config = StatefulSpeculativePipelineBase::resolve_generation_config(generation_config);
    OPENVINO_ASSERT(config.is_greedy_decoding(),
                    "DFlash stateful POC supports greedy sampling only.");
    OPENVINO_ASSERT(config.num_beams == 1, "DFlash stateful POC does not support beam search.");
    return config;
}

void StatefulDFlashLLMPipeline::finish_chat() {
    StatefulSpeculativePipelineBase::finish_chat();
    m_hidden_state_provider.reset();
}

SpeculativeDecodingMetrics StatefulDFlashLLMPipeline::get_speculative_decoding_metrics() const {
    return m_sd_metrics;
}

EncodedResults StatefulDFlashLLMPipeline::generate_tokens(const EncodedInputs& inputs,
                                                          const GenerationConfig& config,
                                                          StreamerVariant streamer) {
    ManualTimer generate_timer("StatefulDFlashLLMPipeline::generate(EncodedInputs)");
    generate_timer.start();
    auto streamer_ptr = utils::create_streamer(streamer, m_tokenizer);

    ov::Tensor input_ids;
    ov::Tensor attention_mask;
    if (auto* tensor_input = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *tensor_input;
        attention_mask = utils::init_attention_mask(input_ids);
    } else if (auto* tokenized_input = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = tokenized_input->input_ids;
        attention_mask = tokenized_input->attention_mask;
    }

    OPENVINO_ASSERT(input_ids && input_ids.get_shape().size() == 2 && input_ids.get_shape()[0] == 1,
                    "DFlash stateful POC supports batch size 1 only.");
    m_prompt_length = input_ids.get_shape()[1];

    m_target->reset_state();
    m_draft->reset_state();
    m_hidden_state_provider.reset();

    auto sampling_config = config;
    // The sampler sees speculative candidates too; length is enforced by the outer DFlash loop.
    sampling_config.max_new_tokens = config.max_new_tokens + m_rt_info.block_size;
    m_target->initialize_sequence(input_ids, sampling_config);
    m_draft->initialize_sequence(input_ids, sampling_config);

    auto prefill_result = m_target->forward(m_prompt_length, 1);
    OPENVINO_ASSERT(prefill_result.sampled_tokens.size() == 1, "Expected one DFlash target seed token.");
    // Stage 1 keeps the full target hidden prefix for each draft block.
    m_hidden_state_provider.append(prefill_result.output.hidden_features, m_prompt_length);

    int64_t seed_token = prefill_result.sampled_tokens.front();
    m_draft->append_tokens({seed_token});
    auto streaming_status = streamer_ptr ? streamer_ptr->write(std::vector<int64_t>{seed_token}) : StreamingStatus::RUNNING;

    size_t generated_tokens = 1;
    size_t total_draft_generated = 0;
    size_t total_draft_accepted = 0;
    bool eos_reached = (seed_token == static_cast<int64_t>(config.eos_token_id));

    while (!eos_reached && generated_tokens < config.max_new_tokens &&
           streaming_status == StreamingStatus::RUNNING) {
        auto result = run_speculative_iteration(generated_tokens,
                                                config.max_new_tokens,
                                                static_cast<int64_t>(config.eos_token_id));
        if (streamer_ptr && !result.validated_tokens.empty()) {
            streaming_status = streamer_ptr->write(result.validated_tokens);
        }
        generated_tokens += result.validated_tokens.size();
        total_draft_generated += result.validated_tokens.empty() ? 0 : std::min(m_rt_info.block_size - 1, config.max_new_tokens - (generated_tokens - result.validated_tokens.size()));
        total_draft_accepted += result.accepted_tokens_count;
        eos_reached = result.eos_reached;
    }

    m_streaming_was_cancelled = (streaming_status == StreamingStatus::CANCEL);
    if (streamer_ptr) {
        streamer_ptr->end();
    }

    EncodedResults results;
    results.tokens = {m_target->get_generated_tokens()};
    results.scores = {0.0f};
    auto seq = m_target->get_current_sequence();
    auto finish_reason = seq ? seq->get_finish_reason() : GenerationFinishReason::NONE;
    if (finish_reason == GenerationFinishReason::NONE && generated_tokens >= config.max_new_tokens) {
        finish_reason = GenerationFinishReason::LENGTH;
    }
    results.finish_reasons = {finish_reason};

    generate_timer.end();
    const size_t actual_generated_tokens = results.tokens.empty() ? 0 : results.tokens.front().size();
    m_sd_perf_metrics.num_input_tokens = m_prompt_length;
    m_sd_perf_metrics.num_generated_tokens = actual_generated_tokens;
    m_sd_perf_metrics.load_time = m_load_time_ms;
    m_sd_perf_metrics.num_accepted_tokens = total_draft_accepted;
    m_sd_perf_metrics.raw_metrics.m_durations.clear();
    m_sd_perf_metrics.raw_metrics.m_durations.emplace_back(generate_timer.get_duration_microsec());
    m_sd_perf_metrics.raw_metrics.m_batch_sizes.clear();
    m_sd_perf_metrics.raw_metrics.m_batch_sizes.emplace_back(actual_generated_tokens);
    m_sd_perf_metrics.raw_metrics.generate_durations.clear();
    m_sd_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_timer.get_duration_microsec());
    m_sd_perf_metrics.m_evaluated = false;
    m_sd_perf_metrics.main_model_metrics.m_evaluated = false;
    m_sd_perf_metrics.draft_model_metrics.m_evaluated = false;
    m_sd_perf_metrics.main_model_metrics.raw_metrics = m_target->get_raw_perf_metrics();
    m_sd_perf_metrics.draft_model_metrics.raw_metrics = m_draft->get_raw_perf_metrics();
    if (total_draft_generated > 0) {
        m_sd_metrics.update_acceptance_rate(0, static_cast<float>(total_draft_accepted) / total_draft_generated * 100.0f);
        m_sd_metrics.update_draft_accepted_tokens(0, total_draft_accepted);
        m_sd_metrics.update_draft_generated_len(0, total_draft_generated);
        m_sd_metrics.update_generated_len(generated_tokens);
    }
    m_sd_perf_metrics.evaluate_statistics(generate_timer.get_start_time());
    results.perf_metrics = m_sd_perf_metrics;
    results.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_sd_perf_metrics);
    return results;
}

StatefulDFlashLLMPipeline::SpeculativeResult StatefulDFlashLLMPipeline::run_speculative_iteration(
    size_t current_generated_tokens,
    size_t max_new_tokens,
    int64_t eos_token_id) {
    SpeculativeResult result;
    OPENVINO_ASSERT(m_hidden_state_provider.tensor() && m_hidden_state_provider.tensor().get_size() > 0,
                    "DFlash hidden-state context is empty.");

    const auto& generated = m_target->get_generated_tokens();
    OPENVINO_ASSERT(!generated.empty(), "DFlash needs a target seed token before draft inference.");
    const int64_t seed_token = generated.back();
    const size_t remaining = max_new_tokens - current_generated_tokens;
    const size_t candidate_count = remaining > 0 ? std::min(m_rt_info.block_size - 1, remaining - 1) : 0;

    std::vector<int64_t> candidates;
    if (candidate_count > 0) {
        auto draft_output = m_draft->infer(seed_token, m_hidden_state_provider.tensor());
        candidates = m_draft->sample_candidates(draft_output.logits, candidate_count);
        m_target->append_tokens(candidates);
    }

    const size_t validation_window = candidates.size() + 1;
    auto validation_result = m_target->forward(validation_window, validation_window, candidates.size());
    auto validated_tokens = validation_result.sampled_tokens;
    OPENVINO_ASSERT(!validated_tokens.empty(), "DFlash validation must return at least the target fallback token.");

    const size_t accepted_candidates = validated_tokens.size() - 1;
    const size_t rejected_candidates = candidates.size() - accepted_candidates;
    if (rejected_candidates > 0) {
        m_target->trim_kv_cache(rejected_candidates);
    }

    const size_t hidden_tokens_to_append = validated_tokens.size();
    m_hidden_state_provider.append(validation_result.output.hidden_features, hidden_tokens_to_append);
    // Draft is stateless today, but keeping its SequenceGroup aligned lets the sampler apply normal penalties.
    m_draft->sync_generated_tokens(m_target->get_generated_tokens());

    result.accepted_tokens_count = accepted_candidates;
    result.validated_tokens = std::move(validated_tokens);
    result.eos_reached = std::find(result.validated_tokens.begin(), result.validated_tokens.end(), eos_token_id) != result.validated_tokens.end();
    return result;
}

}  // namespace genai
}  // namespace ov
