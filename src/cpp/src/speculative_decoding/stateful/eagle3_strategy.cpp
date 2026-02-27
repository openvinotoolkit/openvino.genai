// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eagle3_strategy.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>

#include "continuous_batching/timer.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "speculative_decoding/eagle3_model_transforms.hpp"
#include "utils.hpp"

namespace ov::genai {
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;
}  // namespace ov::genai

namespace {

ov::genai::StreamingStatus stream_generated_tokens(std::shared_ptr<ov::genai::StreamerBase> streamer_ptr,
                                                   const std::vector<int64_t>& tokens) {
    if (streamer_ptr) {
        return streamer_ptr->write(tokens);
    }
    return ov::genai::StreamingStatus{};
}

}  // anonymous namespace

namespace ov::genai {

Eagle3InferWrapperBase::Eagle3InferWrapperBase(const ModelDesc& model_desc)
    : m_device(model_desc.device),
      m_properties(model_desc.properties),
      m_tokenizer(model_desc.tokenizer),
      m_sampler(model_desc.tokenizer) {
    m_kv_axes_pos = utils::get_kv_axes_pos(model_desc.model);

    if (m_device == "NPU") {
        auto [compiled, kv_desc] = utils::compile_decoder_for_npu(model_desc.model, m_properties, m_kv_axes_pos);
        m_max_prompt_len = kv_desc.max_prompt_len;
        m_request = compiled.create_infer_request();
    } else {
        m_request =
            utils::singleton_core().compile_model(model_desc.model, m_device, m_properties).create_infer_request();
    }

    // Initialize performance metrics
    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.tokenization_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.detokenization_durations = {MicroSeconds(0.0f)};

    m_sequence_group = nullptr;
}

void Eagle3InferWrapperBase::append_tokens(const std::vector<int64_t>& tokens) {
    if (tokens.empty())
        return;

    auto current_sequence = get_current_sequence();
    OPENVINO_ASSERT(current_sequence, "SequenceGroup not initialized");

    for (auto token : tokens) {
        current_sequence->append_token(token, 0.0f);
    }
}

void Eagle3InferWrapperBase::truncate_sequence(size_t size) {
    auto current_sequence = get_current_sequence();
    OPENVINO_ASSERT(current_sequence, "SequenceGroup not initialized");

    const size_t prompt_len = m_sequence_group->get_prompt_len();
    const size_t current_len = prompt_len + current_sequence->get_generated_len();

    if (size < current_len) {
        OPENVINO_ASSERT(size >= prompt_len, "Cannot truncate prompt tokens");
        const size_t tokens_to_remove = current_len - size;
        current_sequence->remove_last_tokens(tokens_to_remove);
    }
}

void Eagle3InferWrapperBase::trim_kv_cache(size_t tokens_to_remove) {
    const size_t current_len = get_sequence_length();
    if (tokens_to_remove == 0 || current_len == 0) {
        return;
    }

    OPENVINO_ASSERT(tokens_to_remove > 0 && tokens_to_remove < current_len,
                    "Cannot trim ",
                    tokens_to_remove,
                    " tokens from ",
                    current_len,
                    " tokens. Valid range: 0 < tokens_to_remove < current_len");

    if (m_device != "NPU") {
        utils::KVCacheState state;
        state.num_tokens_to_trim = tokens_to_remove;
        state.seq_length_axis = m_kv_axes_pos.seq_len;
        state.reset_mem_state = false;
        utils::trim_kv_cache(m_request, state, {});
    }
}

void Eagle3InferWrapperBase::reset_state() {
    m_sequence_group = nullptr;

    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.m_durations.clear();
    m_raw_perf_metrics.m_batch_sizes.clear();
}

void Eagle3InferWrapperBase::release_memory() {
    m_request.get_compiled_model().release_memory();
}

void Eagle3InferWrapperBase::build_model_inputs(const size_t input_token_count,
                                                ov::Tensor& input_ids,
                                                ov::Tensor& attention_mask,
                                                ov::Tensor& position_ids) {
    auto current_sequence = get_current_sequence();
    OPENVINO_ASSERT(current_sequence, "SequenceGroup not initialized");

    const auto& prompt_ids = m_sequence_group->get_prompt_ids();
    const auto& generated_ids = current_sequence->get_generated_ids();

    const size_t prompt_len = prompt_ids.size();
    const size_t generated_len = generated_ids.size();
    const size_t total_len = prompt_len + generated_len;
    const size_t start_pos = total_len - input_token_count;

    OPENVINO_ASSERT(input_token_count > 0 && input_token_count <= total_len,
                    "Invalid input_token_count: ",
                    input_token_count,
                    ", total_len: ",
                    total_len);

    // Allocate tensors
    input_ids = ov::Tensor(ov::element::i64, {1, input_token_count});
    position_ids = ov::Tensor(ov::element::i64, {1, input_token_count});

    int64_t* input_ids_ptr = input_ids.data<int64_t>();
    int64_t* position_ids_ptr = position_ids.data<int64_t>();

    // Fill input_ids and position_ids from sequence
    if (start_pos < prompt_len) {
        // Part from prompt
        const size_t prompt_count = std::min(input_token_count, prompt_len - start_pos);
        std::copy_n(prompt_ids.data() + start_pos, prompt_count, input_ids_ptr);
        std::iota(position_ids_ptr, position_ids_ptr + prompt_count, static_cast<int64_t>(start_pos));

        // Part from generated (if any)
        if (input_token_count > prompt_count) {
            const size_t generated_count = input_token_count - prompt_count;
            std::copy_n(generated_ids.data(), generated_count, input_ids_ptr + prompt_count);
            std::iota(position_ids_ptr + prompt_count,
                      position_ids_ptr + prompt_count + generated_count,
                      static_cast<int64_t>(prompt_len));
        }
    } else {
        // All from generated
        const size_t generated_start = start_pos - prompt_len;
        std::copy_n(generated_ids.data() + generated_start, input_token_count, input_ids_ptr);
        std::iota(position_ids_ptr,
                  position_ids_ptr + input_token_count,
                  static_cast<int64_t>(prompt_len + generated_start));
    }

    // Build attention mask
    const size_t attention_mask_len = static_cast<size_t>(position_ids_ptr[input_token_count - 1] + 1);
    attention_mask = ov::Tensor(ov::element::i64, {1, attention_mask_len});
    std::fill_n(attention_mask.data<int64_t>(), attention_mask_len, 1);
}

std::vector<int64_t> Eagle3InferWrapperBase::sample_tokens(const ov::Tensor& logits,
                                                           size_t input_token_count,
                                                           size_t sample_count,
                                                           size_t num_tokens_to_validate) {
    const ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Invalid logits shape");
    OPENVINO_ASSERT(sample_count > 0 && sample_count <= shape[1],
                    "Invalid sample_count: ",
                    sample_count,
                    ", logits seq_len: ",
                    shape[1]);
    OPENVINO_ASSERT(input_token_count > 0, "Invalid input_token_count");

    const bool is_validation_mode = num_tokens_to_validate > 0;

    auto sequence_group = get_sequence_group();
    OPENVINO_ASSERT(sequence_group, "SequenceGroup not initialized");

    OPENVINO_ASSERT(get_running_sequence_count() == 1,
                    "Eagle3 currently only supports single sequence, got ",
                    get_running_sequence_count(),
                    " sequences");

    auto current_seq = get_current_sequence();
    OPENVINO_ASSERT(current_seq, "No running sequence at index 0");

    const size_t prev_generated_len = current_seq->get_generated_len();
    const size_t logits_seq_len = shape[1];
    const size_t vocab_size = shape[2];

    // Slice logits to last 'sample_count' positions if needed
    ov::Tensor sliced_logits = logits;
    if (sample_count < logits_seq_len) {
        auto [start_coord, end_coord] =
            ov::genai::utils::make_roi(shape, 1, logits_seq_len - sample_count, logits_seq_len);
        sliced_logits = ov::Tensor(logits, start_coord, end_coord);
    }

    // Configure sequence group for sampling
    sequence_group->schedule_tokens(input_token_count);
    sequence_group->set_output_seq_len(sample_count);
    sequence_group->set_num_validated_tokens(num_tokens_to_validate);

    // Execute sampling
    m_sampler.sample({sequence_group}, sliced_logits, is_validation_mode);
    sequence_group->finish_iteration();

    // Extract results based on mode
    const auto& generated_ids = current_seq->get_generated_ids();
    const size_t new_generated_len = generated_ids.size();

    if (!is_validation_mode) {
        OPENVINO_ASSERT(new_generated_len - prev_generated_len == sample_count,
                        "Sampled token count mismatch: expected ",
                        sample_count,
                        ", got ",
                        new_generated_len - prev_generated_len);

        std::vector<int64_t> result_tokens(generated_ids.end() - sample_count, generated_ids.end());

        record_generated_tokens(sample_count);

        return result_tokens;
    } else {
        // Validation mode: Sampler validates draft tokens and removes rejected ones
        // Calculate result_count to extract the final validated sequence:
        //   - Result contains: accepted_draft_tokens + new_token_from_target (total = num_accepted + 1)
        //   - prev_generated_len = previously validated tokens + unvalidated draft tokens (num_tokens_to_validate)
        //   - new_generated_len = prev_generated_len - num_tokens_to_validate + (num_accepted + 1)
        //   - Therefore: result_count = new_generated_len - prev_generated_len + num_tokens_to_validate
        const size_t result_count = new_generated_len - prev_generated_len + num_tokens_to_validate;
        std::vector<int64_t> result_tokens(generated_ids.end() - result_count, generated_ids.end());

        record_generated_tokens(result_tokens.size());

        return result_tokens;
    }
}

ov::Tensor Eagle3InferWrapperBase::get_logits() const {
    return m_request.get_tensor("logits");
}

ov::Tensor Eagle3InferWrapperBase::get_hidden_features() const {
    auto hidden_state = m_request.get_tensor("last_hidden_state");
    const auto shape = hidden_state.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Invalid hidden state shape");

    const size_t output_seq_len = shape[1];
    const size_t hidden_size = shape[2];
    const size_t actual_seq_len = m_request.get_tensor("input_ids").get_shape()[1];

    if (output_seq_len == actual_seq_len)
        return hidden_state;

    OPENVINO_ASSERT(actual_seq_len <= output_seq_len,
                    "Sequence length mismatch: actual=",
                    actual_seq_len,
                    ", output=",
                    output_seq_len);
    auto [start_coord, end_coord] =
        ov::genai::utils::make_roi(shape, 1, output_seq_len - actual_seq_len, output_seq_len);
    return ov::Tensor(hidden_state, start_coord, end_coord);
}

uint64_t Eagle3InferWrapperBase::execute_inference() {
    auto start = std::chrono::steady_clock::now();
    m_request.infer();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    return duration_us;
}

void Eagle3InferWrapperBase::update_inference_time(uint64_t inference_time_us) {
    m_raw_perf_metrics.m_durations.emplace_back(static_cast<float>(inference_time_us));
    m_raw_perf_metrics.m_inference_durations[0] += MicroSeconds(static_cast<float>(inference_time_us));
}

void Eagle3InferWrapperBase::record_generated_tokens(size_t actual_generated_count) {
    m_raw_perf_metrics.m_batch_sizes.emplace_back(actual_generated_count);
}

Eagle3TargetWrapper::Eagle3TargetWrapper(const ov::genai::ModelDesc& model_desc) : Eagle3InferWrapperBase(model_desc) {}

void Eagle3TargetWrapper::initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config) {
    const auto shape = input_ids.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] == 1, "Expected input_ids shape [1, seq_len]");

    const int64_t* ids_data = input_ids.data<const int64_t>();
    const size_t seq_len = shape[1];
    OPENVINO_ASSERT(seq_len > 0, "Empty prompt");

    TokenIds prompt_ids(ids_data, ids_data + seq_len);
    m_sequence_group = std::make_shared<SequenceGroup>(0, prompt_ids, config, 0);

    OPENVINO_ASSERT(get_running_sequence_count() == 1,
                    "Expected single sequence after initialization, got ",
                    get_running_sequence_count());
}

InferenceOutput Eagle3TargetWrapper::infer(const ov::Tensor& input_ids,
                                           const ov::Tensor& attention_mask,
                                           const ov::Tensor& position_ids) {
    const size_t prompt_len = input_ids.get_shape()[1];

    if (m_device == "NPU") {
        OPENVINO_ASSERT(prompt_len <= m_max_prompt_len,
                        "NPU prompt length ",
                        prompt_len,
                        " exceeds max ",
                        m_max_prompt_len);
    }

    // Set model inputs
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);

    // Execute inference
    uint64_t time_us = execute_inference();
    update_inference_time(time_us);

    // Collect outputs
    InferenceOutput output;
    output.logits = get_logits();
    output.hidden_features = get_hidden_features();

    return output;
}

InferResult Eagle3TargetWrapper::forward(const InferContext& ctx) {
    // 1. Prepare inputs from sequence state
    ov::Tensor input_ids, attention_mask, position_ids;
    build_model_inputs(ctx.input_token_count, input_ids, attention_mask, position_ids);

    // 2. Infer
    auto output = infer(input_ids, attention_mask, position_ids);

    // 3. Sample (use sample_count for number of positions to sample from)
    auto sampled = sample_tokens(output.logits, ctx.input_token_count, ctx.sample_count, ctx.num_tokens_to_validate);

    // 4. Store hidden states to sequence for draft model to use
    get_current_sequence()->update_hidden_state(output.hidden_features);

    return InferResult{std::move(output), std::move(sampled)};
}

Eagle3DraftWrapper::Eagle3DraftWrapper(const ov::genai::ModelDesc& model_desc) : Eagle3InferWrapperBase(model_desc) {}

void Eagle3DraftWrapper::initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config) {
    const auto shape = input_ids.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] == 1, "Expected input_ids shape [1, seq_len]");

    const int64_t* ids_data = input_ids.data<const int64_t>();
    const size_t total_len = shape[1];
    OPENVINO_ASSERT(total_len >= 2, "Draft model requires at least 2 tokens");

    // Draft model uses tokens[1:] (Eagle3 specific behavior)
    TokenIds draft_prompt_ids(ids_data + 1, ids_data + total_len);
    m_sequence_group = std::make_shared<SequenceGroup>(1, draft_prompt_ids, config, 0);

    OPENVINO_ASSERT(get_running_sequence_count() == 1,
                    "Expected single sequence after initialization, got ",
                    get_running_sequence_count());
}

InferenceOutput Eagle3DraftWrapper::infer(const ov::Tensor& input_ids,
                                          const ov::Tensor& attention_mask,
                                          const ov::Tensor& position_ids,
                                          const ov::Tensor& hidden_states) {
    const size_t input_token_count = input_ids.get_shape()[1];

    // Set standard inputs
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);

    // Set hidden states (either from target model or internal)
    OPENVINO_ASSERT(hidden_states && hidden_states.get_size() > 0, "hidden_states must be provided");
    auto shape = hidden_states.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "Invalid hidden states shape");

    m_request.set_tensor("hidden_states", hidden_states);

    // Execute inference
    uint64_t time_us = execute_inference();
    update_inference_time(time_us);

    // Collect outputs
    InferenceOutput output;
    output.logits = get_logits();
    output.hidden_features = get_hidden_features();

    return output;
}

InferResult Eagle3DraftWrapper::forward(const InferContext& ctx) {
    // 1. Prepare inputs
    ov::Tensor input_ids, attention_mask, position_ids;
    build_model_inputs(ctx.input_token_count, input_ids, attention_mask, position_ids);

    // 2. Get hidden states from appropriate source
    ov::Tensor hidden_states;
    if (ctx.use_target_hidden) {
        OPENVINO_ASSERT(ctx.target_sequence, "target_sequence required when use_target_hidden=true");
        hidden_states = ctx.target_sequence->get_hidden_state();
        OPENVINO_ASSERT(hidden_states && hidden_states.get_size() > 0, "Source sequence contains invalid hidden state");
    } else {
        hidden_states = get_current_sequence()->get_hidden_state();
        OPENVINO_ASSERT(hidden_states && hidden_states.get_size() > 0, "Own sequence contains invalid hidden state");
    }

    // 3. Infer
    auto output = infer(input_ids, attention_mask, position_ids, hidden_states);

    // 4. Sample
    auto sampled = sample_tokens(output.logits, ctx.input_token_count, 1);

    // 5. Store internal hidden state (last position) for next iteration
    auto next_hidden = utils::eagle3::slice_hidden_state_for_last_token(output.hidden_features);
    get_current_sequence()->update_hidden_state(next_hidden);

    return InferResult{std::move(output), std::move(sampled)};
}

StatefulEagle3LLMPipeline::StatefulEagle3LLMPipeline(const ov::genai::ModelDesc& target_model_desc,
                                                     const ov::genai::ModelDesc& draft_model_desc)
    : StatefulSpeculativePipelineBase(target_model_desc.tokenizer, target_model_desc.generation_config) {
    // Initialize draft iterations from generation config
    ensure_num_assistant_tokens_is_set(m_generation_config);
    m_draft_iterations = m_generation_config.num_assistant_tokens;

    // Extract hidden_layers_list from draft model properties
    OPENVINO_ASSERT(draft_model_desc.properties.find("hidden_layers_list") != draft_model_desc.properties.end(),
                    "hidden_layers_list must be present in draft model properties");

    m_hidden_layers_to_abstract = draft_model_desc.properties.at("hidden_layers_list").as<std::vector<int32_t>>();

    OPENVINO_ASSERT(m_hidden_layers_to_abstract.size() == 3,
                    "Eagle3 requires exactly three layers for feature extraction, got: " +
                        std::to_string(m_hidden_layers_to_abstract.size()) +
                        ". Please ensure 'hidden_layers_list' is properly configured in draft model properties.");

    auto target_model = target_model_desc.model;
    auto draft_model = draft_model_desc.model;

    // Model preparation
    utils::eagle3::share_vocabulary(target_model, draft_model);

    auto d2t_mapping = utils::eagle3::extract_d2t_mapping_table(draft_model);
    OPENVINO_ASSERT(d2t_mapping && d2t_mapping->get_element_type() == ov::element::i64, "Invalid d2t mapping tensor");

    utils::eagle3::transform_hidden_state(target_model, m_hidden_layers_to_abstract);
    utils::eagle3::move_fc_from_draft_to_main(draft_model, target_model);
    utils::eagle3::transform_hidden_state(draft_model, {-1});

    const size_t validation_window = m_draft_iterations + 1;

    // Configure and create draft model
    auto draft_desc = draft_model_desc;
    if (draft_desc.device == "NPU") {
        draft_desc.properties["NPUW_EAGLE"] = "TRUE";
        draft_desc.properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = validation_window;
        draft_desc.properties["NPUW_ONLINE_PIPELINE"] = "NONE";
    }
    m_draft = std::make_unique<Eagle3DraftWrapper>(draft_desc);

    m_draft->set_draft_target_mapping(d2t_mapping);

    // Configure and create target model
    auto target_desc = target_model_desc;
    if (target_desc.device == "NPU") {
        target_desc.properties["NPUW_EAGLE"] = "TRUE";
        target_desc.properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = validation_window;
        target_desc.properties["NPUW_SLICE_OUT"] = "NO";
    }
    m_target = std::make_unique<Eagle3TargetWrapper>(target_desc);
}

StatefulEagle3LLMPipeline::~StatefulEagle3LLMPipeline() {
    m_target->release_memory();
    m_draft->release_memory();
}

GenerationConfig StatefulEagle3LLMPipeline::resolve_generation_config(OptionalGenerationConfig generation_config) {
    // Call base class implementation to handle common defaults
    GenerationConfig config = StatefulSpeculativePipelineBase::resolve_generation_config(generation_config);

    // Apply Eagle3 specific validations
    const size_t prev_draft_iterations = m_draft_iterations;
    ensure_num_assistant_tokens_is_set(config);
    m_draft_iterations = config.num_assistant_tokens;

    return config;
}

EncodedResults StatefulEagle3LLMPipeline::generate_tokens(const EncodedInputs& inputs,
                                                          const GenerationConfig& config,
                                                          StreamerVariant streamer) {
    ManualTimer generate_timer("StatefulEagle3LLMPipeline::generate(EncodedInputs)");
    generate_timer.start();

    std::shared_ptr<StreamerBase> streamer_ptr = ov::genai::utils::create_streamer(streamer, m_tokenizer);

    // Extract input tensors
    ov::Tensor input_ids, attention_mask;
    if (auto* tensor_input = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *tensor_input;
        attention_mask = ov::genai::utils::init_attention_mask(input_ids);
    } else if (auto* tokenized_input = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = tokenized_input->input_ids;
        attention_mask = tokenized_input->attention_mask;
    }

    OPENVINO_ASSERT(input_ids.get_shape()[0] == 1, "Only batch size 1 supported");
    m_prompt_length = input_ids.get_shape()[1];

    // Initialize position IDs
    ov::Tensor position_ids{ov::element::i64, input_ids.get_shape()};
    utils::initialize_position_ids(position_ids, attention_mask);

    // Reset model states
    m_target->reset_state();
    m_draft->reset_state();

    // Prepare sampling config with extended max_new_tokens to prevent premature termination
    // during draft generation. Actual length control is in the generation loop.
    auto sampling_config = config;
    sampling_config.max_new_tokens = config.max_new_tokens + m_draft_iterations + 1;

    // Initialize sequences with sampling config
    m_target->initialize_sequence(input_ids, sampling_config);
    m_draft->initialize_sequence(input_ids, sampling_config);

    // Phase 1: Initial Prompt Processing (Prefill)

    // Prefill: process all prompt tokens from sequence
    InferContext prefill_ctx;
    prefill_ctx.input_token_count = m_prompt_length;
    auto prefill_result = m_target->forward(prefill_ctx);
    OPENVINO_ASSERT(prefill_result.sampled_tokens.size() == 1, "Expected single token from prefill");
    auto initial_token = prefill_result.sampled_tokens[0];

    // Append initial token to draft model
    m_draft->append_tokens({initial_token});

    auto streaming_status = stream_generated_tokens(streamer_ptr, {initial_token});

    // Phase 2: Speculative Decoding Loop
    size_t generated_tokens = 1;
    size_t total_draft_accepted = 0;
    size_t total_draft_generated = 0;
    bool eos_reached = false;

    size_t input_token_count = m_draft->get_sequence_length();

    while (!eos_reached && generated_tokens < config.max_new_tokens &&
           m_target->get_sequence_length() < m_prompt_length + config.max_new_tokens &&
           streaming_status == ov::genai::StreamingStatus::RUNNING) {
        auto result = run_speculative_iteration(input_token_count, static_cast<int64_t>(config.eos_token_id));

        streaming_status = stream_generated_tokens(streamer_ptr, result.validated_tokens);

        // Update statistics
        total_draft_generated += m_draft_iterations;
        total_draft_accepted += result.accepted_tokens_count;
        eos_reached = result.eos_reached;
        generated_tokens++;

        // Prepare for next iteration (hidden states are stored in sequence)
        input_token_count = result.next_window_size;
    }

    // Phase 3: Finalization
    m_streaming_was_cancelled = (streaming_status == ov::genai::StreamingStatus::CANCEL);
    if (streamer_ptr)
        streamer_ptr->end();

    // Collect results
    EncodedResults results;
    results.tokens = {m_target->get_generated_tokens()};
    results.scores = {0.0f};

    generate_timer.end();

    // Update performance metrics
    m_sd_perf_metrics.num_input_tokens = m_prompt_length;
    m_sd_perf_metrics.load_time = m_load_time_ms;
    m_sd_perf_metrics.num_accepted_tokens = total_draft_accepted;
    m_sd_perf_metrics.raw_metrics.generate_durations.clear();
    m_sd_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_timer.get_duration_microsec());

    // Reset evaluated flags before updating raw_metrics to ensure statistics are recalculated
    m_sd_perf_metrics.m_evaluated = false;
    m_sd_perf_metrics.main_model_metrics.m_evaluated = false;
    m_sd_perf_metrics.draft_model_metrics.m_evaluated = false;

    m_sd_perf_metrics.main_model_metrics.raw_metrics = m_target->get_raw_perf_metrics();
    m_sd_perf_metrics.draft_model_metrics.raw_metrics = m_draft->get_raw_perf_metrics();

    if (total_draft_generated > 0) {
        float acceptance_rate = static_cast<float>(total_draft_accepted) / total_draft_generated * 100.0f;
        m_sd_metrics.update_acceptance_rate(0, acceptance_rate);
        m_sd_metrics.update_draft_accepted_tokens(0, total_draft_accepted);
        m_sd_metrics.update_draft_generated_len(0, total_draft_generated);
        m_sd_metrics.update_generated_len(generated_tokens);
    }

    m_sd_perf_metrics.evaluate_statistics(generate_timer.get_start_time());
    results.perf_metrics = m_sd_perf_metrics;
    results.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_sd_perf_metrics);

    // Reset timer
    generate_timer.clear();

    return results;
}

StatefulEagle3LLMPipeline::SpeculativeResult StatefulEagle3LLMPipeline::run_speculative_iteration(
    size_t input_token_count,
    int64_t eos_token_id) {
    SpeculativeResult result;

    OPENVINO_ASSERT(m_target->get_running_sequence_count() == 1 && m_draft->get_running_sequence_count() == 1,
                    "Eagle3 speculative iteration requires single sequence per model");

    auto target_hidden_states = m_target->get_current_sequence()->get_hidden_state();
    OPENVINO_ASSERT(target_hidden_states && target_hidden_states.get_size() > 0,
                    "Target model contains invalid hidden state for speculation");

    // Record pre-draft sequence lengths for potential rollback
    const size_t pre_draft_token_len = m_draft->get_sequence_length();

    // Step 1: Generate first draft token using target hidden states
    InferContext first_ctx;
    first_ctx.input_token_count = input_token_count;
    first_ctx.use_target_hidden = true;
    first_ctx.target_sequence = m_target->get_current_sequence();
    auto first_result = m_draft->forward(first_ctx);

    OPENVINO_ASSERT(first_result.sampled_tokens.size() == 1, "Expected single token from first draft");
    int64_t first_draft_token = first_result.sampled_tokens[0];

    // Collect draft candidates
    std::vector<int64_t> draft_candidates;
    draft_candidates.reserve(m_draft_iterations);
    draft_candidates.push_back(first_draft_token);

    // Append first token to target model (draft model already has it from sampler)
    m_target->append_tokens({first_draft_token});

    // Step 2: Generate additional draft tokens using internal hidden states
    for (size_t i = 1; i < m_draft_iterations; ++i) {
        InferContext more_ctx;
        more_ctx.input_token_count = 1;
        more_ctx.use_target_hidden = false;
        auto more_result = m_draft->forward(more_ctx);

        OPENVINO_ASSERT(more_result.sampled_tokens.size() == 1, "Expected single token from draft iteration");
        int64_t draft_token = more_result.sampled_tokens[0];
        draft_candidates.push_back(draft_token);

        // Append draft token to target sequence for validation phase
        // During validation, target model will retrieve tokens from its own sequence
        // so we need to speculatively add draft predictions here
        m_target->append_tokens({draft_token});
    }

    // Step 3: Validate draft tokens with target model

    const size_t validation_window_size = m_draft_iterations + 1;

    InferContext val_ctx;
    val_ctx.input_token_count = validation_window_size;
    val_ctx.sample_count = validation_window_size;
    val_ctx.num_tokens_to_validate = m_draft_iterations;
    auto val_result = m_target->forward(val_ctx);

    // Sampler validates draft tokens and returns accepted + new sampled token
    auto validated_tokens = val_result.sampled_tokens;

    // Result: [accepted_draft_tokens..., new_sampled_token]
    const size_t accepted_count = validated_tokens.size() - 1;
    const int64_t target_predicted_token = validated_tokens.back();
    const size_t tokens_to_remove = m_draft_iterations - accepted_count;
    const size_t total_accepted_tokens = validated_tokens.size();

    // Step 4: Synchronize sequences and KV cache
    // Target model's sequence is already updated by Sampler
    // Sync draft model's sequence
    m_draft->truncate_sequence(pre_draft_token_len);
    m_draft->append_tokens(validated_tokens);

    // Trim KV cache for rejected tokens
    if (tokens_to_remove > 0) {
        m_target->trim_kv_cache(tokens_to_remove);
        m_draft->trim_kv_cache(tokens_to_remove);
    }

    // Step 5: Update hidden states for next iteration
    // Note: forward() already stored hidden_features to sequence, but we need to slice it
    auto current_hidden = val_result.output.hidden_features;
    OPENVINO_ASSERT(current_hidden && current_hidden.get_size() > 0, "Missing hidden features");

    const auto h_shape = current_hidden.get_shape();
    OPENVINO_ASSERT(h_shape.size() == 3 && h_shape[0] == 1 && h_shape[1] >= total_accepted_tokens,
                    "Invalid hidden state shape");

    // Store sliced hidden states (only accepted tokens) for next iteration
    auto [start_coord, end_coord] = ov::genai::utils::make_roi(h_shape, 1, 0, total_accepted_tokens);
    auto next_hidden = ov::Tensor(current_hidden, start_coord, end_coord);
    m_target->get_current_sequence()->update_hidden_state(next_hidden);

    result.accepted_tokens_count = accepted_count;
    result.next_window_size = accepted_count + 1;
    result.validated_tokens = std::move(validated_tokens);
    result.eos_reached = (target_predicted_token == eos_token_id);

    return result;
}

void StatefulEagle3LLMPipeline::finish_chat() {
    // Eagle3 uses base class implementation directly (no model state reset needed)
    StatefulSpeculativePipelineBase::finish_chat();
}

SpeculativeDecodingMetrics StatefulEagle3LLMPipeline::get_speculative_decoding_metrics() const {
    return m_sd_metrics;
}

}  // namespace ov::genai
