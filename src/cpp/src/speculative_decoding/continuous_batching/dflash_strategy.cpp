// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_strategy.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include <openvino/pass/sdpa_to_paged_attention.hpp>

#include "continuous_batching/paged_attention_transformations.hpp"
#include "sampling/sampler.hpp"
#include "sequence_group.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

struct DraftCandidateToken {
    int64_t token_id;
    float log_prob;
};

std::vector<float> zero_log_probs(size_t count) {
    return std::vector<float>(count, 0.0f);
}

bool has_compiled_input(const ov::CompiledModel& model, const std::string& name) {
    const auto inputs = model.inputs();
    return std::find_if(inputs.begin(), inputs.end(), [&](const ov::Output<const ov::Node>& port) {
        return port.get_names().count(name) != 0;
    }) != inputs.end();
}

bool model_has_output(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    const auto outputs = model->outputs();
    return std::find_if(outputs.begin(), outputs.end(), [&](const ov::Output<ov::Node>& port) {
        return port.get_names().count(name) != 0;
    }) != outputs.end();
}

void validate_target_has_no_unmanaged_state(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(model, "DFlash target model cannot be null.");
    std::vector<std::string> unmanaged_state_ops;
    for (const auto& op : model->get_ordered_ops()) {
        const std::string type_name = op->get_type_name();
        if (type_name == "ReadValue" || type_name == "Assign") {
            unmanaged_state_ops.push_back(op->get_friendly_name());
        }
    }
    OPENVINO_ASSERT(unmanaged_state_ops.empty(),
                    "DFlash CB/PA target model contains unmanaged state op(s) after paging conversion. ",
                    "First unmanaged op: ", unmanaged_state_ops.empty() ? std::string{} : unmanaged_state_ops.front(),
                    ". Convert KV, conv, and GatedDeltaNet state to managed paging before enabling DFlash.");
}

}  // namespace

class ContinuousBatchingPipeline::DFlashDecodingImpl::DFlashCBDraftRunner {
public:
    DFlashCBDraftRunner(const ov::genai::ModelDesc& model_desc,
                        const Tokenizer& tokenizer,
                        const ov::genai::utils::dflash::DFlashRTInfo& rt_info)
        : m_tokenizer(tokenizer),
          m_request(create_draft_infer_request(model_desc)),
          m_sampler(tokenizer),
          m_mask_token_id(rt_info.mask_token_id) {
        m_has_beam_idx = has_compiled_input(m_request.get_compiled_model(), "beam_idx");
        if (m_has_beam_idx) {
            m_beam_idx = ov::Tensor(ov::element::i32, {BATCH_SIZE});
            std::fill_n(m_beam_idx.data<int32_t>(), m_beam_idx.get_size(), 0);
        }
        m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
        m_raw_perf_metrics.tokenization_durations = {MicroSeconds(0.0f)};
        m_raw_perf_metrics.detokenization_durations = {MicroSeconds(0.0f)};
    }

    void initialize_sequence(const ov::Tensor& input_ids, const GenerationConfig& config) {
        const auto shape = input_ids.get_shape();
        OPENVINO_ASSERT(shape.size() == 2 && shape[0] == BATCH_SIZE && shape[1] > 0,
                        "Expected DFlash input_ids shape [1, seq_len].");
        const int64_t* ids_data = input_ids.data<const int64_t>();
        TokenIds prompt_ids(ids_data, ids_data + shape[1]);
        m_prompt_length = shape[1];
        // Sampler state is keyed by request_id; we reuse request_id=1, so clear per-request context.
        m_sampler.clear_request_info(1);
        m_sequence_group = std::make_shared<SequenceGroup>(1, prompt_ids, config);
        m_sequence_group->update_processed_tokens_num(m_prompt_length);
        m_committed_context_length = 0;
        m_request.reset_state();
        if (m_has_beam_idx) {
            m_request.set_tensor("beam_idx", m_beam_idx);
        }
        m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
        m_raw_perf_metrics.m_durations.clear();
        m_raw_perf_metrics.m_batch_sizes.clear();
    }

    void sync_generated_tokens(const std::vector<int64_t>& target_generated_tokens) {
        auto seq = (*m_sequence_group)[0];
        if (seq->get_generated_len() > 0) {
            seq->remove_last_tokens(seq->get_generated_len());
        }
        for (auto token : target_generated_tokens) {
            seq->append_token(token, 0.0f);
        }
        m_sequence_group->update_processed_tokens_num(m_prompt_length + target_generated_tokens.size());
        seq->set_status(SequenceStatus::RUNNING);
    }

    ov::Tensor infer(int64_t seed_token, const ov::Tensor& hidden_delta, size_t candidate_count) {
        OPENVINO_ASSERT(hidden_delta && hidden_delta.get_size() > 0, "DFlash hidden delta must be provided.");
        const auto hidden_delta_shape = hidden_delta.get_shape();
        OPENVINO_ASSERT(hidden_delta_shape.size() == 3 && hidden_delta_shape[1] == BATCH_SIZE,
                        "DFlash draft hidden_states input must have shape [seq_len, 1, hidden].");
        const size_t hidden_delta_length = hidden_delta_shape[0];

        auto input_ids = build_input_ids(seed_token, candidate_count);
        auto position_ids = build_position_ids(hidden_delta_length, candidate_count);
        auto attention_mask = build_attention_mask(hidden_delta_length, candidate_count);
        m_request.set_tensor("hidden_states", hidden_delta);
        m_request.set_tensor("position_ids", position_ids);
        m_request.set_tensor("attention_mask", attention_mask);
        // After load-time transforms the draft is always input_ids-native (target embedding
        // attached in-graph when needed), so feed token ids directly.
        m_request.set_tensor("input_ids", input_ids);
        update_inference_time(execute_inference());
        m_committed_context_length += hidden_delta_length;
        return m_request.get_tensor("logits");
    }

    std::vector<DraftCandidateToken> sample_candidates(const ov::Tensor& logits, size_t candidate_count) {
        std::vector<DraftCandidateToken> candidates;
        candidates.reserve(candidate_count);
        const auto shape = logits.get_shape();
        for (size_t idx = 0; idx < candidate_count; ++idx) {
            ov::Tensor one_position(logits,
                                    ov::Coordinate{0, idx, 0},
                                    ov::Coordinate{1, idx + 1, shape[2]});
            auto sampled = sample_one_candidate(one_position);
            candidates.insert(candidates.end(), sampled.begin(), sampled.end());
        }
        m_raw_perf_metrics.m_batch_sizes.emplace_back(candidates.size());
        return candidates;
    }

    ov::genai::RawPerfMetrics& get_raw_perf_metrics() {
        return m_raw_perf_metrics;
    }

    size_t get_consumed_hidden_states() const {
        return m_committed_context_length;
    }

private:
    static ov::InferRequest create_draft_infer_request(const ov::genai::ModelDesc& model_desc) {
        OPENVINO_ASSERT(model_desc.model, "DFlash draft model cannot be null.");
        OPENVINO_ASSERT(utils::has_input(model_desc.model, "hidden_states"),
                        "DFlash CB/PA draft model must have 'hidden_states' input.");
        OPENVINO_ASSERT(utils::has_input(model_desc.model, "input_ids"),
                        "DFlash CB/PA draft model must have an 'input_ids' input after load-time transforms.");
        OPENVINO_ASSERT(utils::has_input(model_desc.model, "position_ids"),
                        "DFlash CB/PA draft model must have a 'position_ids' input.");
        OPENVINO_ASSERT(utils::has_input(model_desc.model, "attention_mask"),
                        "DFlash CB/PA draft model must have an 'attention_mask' input.");
        if (model_desc.device == "NPU") {
            auto kv_axes_pos = utils::get_kv_axes_pos(model_desc.model);
            auto npu_compile_result = utils::compile_decoder_for_npu(model_desc.model, model_desc.properties, kv_axes_pos);
            return npu_compile_result.first.create_infer_request();
        }
        return utils::singleton_core()
            .compile_model(model_desc.model, model_desc.device, model_desc.properties)
            .create_infer_request();
    }

    ov::Tensor build_input_ids(int64_t seed_token, size_t candidate_count) const {
        return dflash_cb::build_draft_input_ids(seed_token, m_mask_token_id, candidate_count);
    }

    ov::Tensor build_position_ids(size_t hidden_delta_length, size_t candidate_count) const {
        return dflash_cb::build_draft_position_ids(m_committed_context_length, hidden_delta_length, candidate_count);
    }

    ov::Tensor build_attention_mask(size_t hidden_delta_length, size_t candidate_count) const {
        return dflash_cb::build_draft_attention_mask(m_committed_context_length, hidden_delta_length, candidate_count);
    }

    std::vector<DraftCandidateToken> sample_one_candidate(const ov::Tensor& logits) {
        const auto sequence = (*m_sequence_group)[0];
        const size_t generated_before = sequence->get_generated_len();
        m_sequence_group->schedule_tokens(1);
        m_sequence_group->set_output_seq_len(1);
        m_sequence_group->set_num_validated_tokens(0);
        m_sampler.sample({m_sequence_group}, logits, false);
        m_sequence_group->finish_iteration();

        const auto& generated = sequence->get_generated_ids();
        if (generated.size() <= generated_before) {
            return {};
        }
        const auto& log_probs = sequence->get_generated_log_probs();
        OPENVINO_ASSERT(log_probs.size() >= generated.size(), "Generated token log-probs are out of sync.");
        std::vector<DraftCandidateToken> candidates;
        candidates.reserve(generated.size() - generated_before);
        for (size_t idx = generated_before; idx < generated.size(); ++idx) {
            candidates.push_back({generated[idx], log_probs[idx]});
        }
        return candidates;
    }

    uint64_t execute_inference() {
        auto start = std::chrono::steady_clock::now();
        m_request.infer();
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
    }

    void update_inference_time(uint64_t inference_time_us) {
        m_raw_perf_metrics.m_durations.emplace_back(static_cast<float>(inference_time_us));
        m_raw_perf_metrics.m_inference_durations[0] += MicroSeconds(static_cast<float>(inference_time_us));
    }

    static constexpr size_t BATCH_SIZE = 1;
    Tokenizer m_tokenizer;
    mutable ov::InferRequest m_request;
    SequenceGroup::Ptr m_sequence_group;
    Sampler m_sampler;
    ov::genai::RawPerfMetrics m_raw_perf_metrics;
    bool m_has_beam_idx = false;
    ov::Tensor m_beam_idx;
    size_t m_prompt_length = 0;
    size_t m_committed_context_length = 0;
    int64_t m_mask_token_id = -1;
};

ContinuousBatchingPipeline::DFlashDecodingImpl::DFlashDecodingImpl(
    const ov::genai::ModelDesc& main_model_desc,
    const ov::genai::ModelDesc& draft_model_desc,
    const ov::genai::utils::dflash::DFlashRTInfo& rt_info)
    : m_rt_info(rt_info) {
    OPENVINO_ASSERT(m_rt_info.dflash_mode, "DFlash continuous batching requires dflash_mode=true.");
    OPENVINO_ASSERT(!m_rt_info.target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");

    auto main_model = main_model_desc.model;
    OPENVINO_ASSERT(main_model && draft_model_desc.model, "DFlash requires both target and draft models.");

    // Detect each draft capability independently from its I/O signature (no RT-info markers) and
    // apply only the missing transform(s); a well-formed draft has exactly one input/output per pair.
    const bool needs_embedding_attach = utils::has_input(draft_model_desc.model, "inputs_embeds");
    const bool needs_lm_head_graft = model_has_output(draft_model_desc.model, "last_hidden_state");
    OPENVINO_ASSERT(needs_embedding_attach != utils::has_input(draft_model_desc.model, "input_ids"),
                    "DFlash draft model must have exactly one of 'inputs_embeds' or 'input_ids' input.");
    OPENVINO_ASSERT(needs_lm_head_graft != model_has_output(draft_model_desc.model, "logits"),
                    "DFlash draft model must have exactly one of 'last_hidden_state' or 'logits' output.");

    // Resolve authoritative metadata before mutating either model so a malformed target leaves the draft reusable.
    auto retained_hidden_state_locators =
        utils::dflash::resolve_target_hidden_state_locators(main_model, m_rt_info.target_layer_ids);

    // Transform while the target is pristine so the compiled draft is self-contained (input_ids -> logits).
    if (needs_embedding_attach) {
        utils::dflash::attach_target_embedding_to_draft(main_model, draft_model_desc.model);
    }
    if (needs_lm_head_graft) {
        utils::dflash::attach_target_lm_head_to_draft(main_model, draft_model_desc.model);
    }

    const bool target_has_linear_attention = utils::get_cache_types(*main_model).has_linear();

    const bool allow_score_aggregation = true;
    const bool allow_cache_rotation = false;
    const bool allow_xattention = false;
    const bool allow_adaptive_rkv = false;
    const bool allow_qq_bias = main_model_desc.properties.count("query_to_query_bias") > 0 &&
                               main_model_desc.properties.at("query_to_query_bias").as<bool>();
    ov::pass::SDPAToPagedAttention(main_model_desc.scheduler_config.use_cache_eviction,
                                   main_model_desc.scheduler_config.use_cache_eviction,
                                   allow_score_aggregation,
                                   allow_cache_rotation,
                                   allow_xattention,
                                   allow_adaptive_rkv,
                                   allow_qq_bias)
        .run_on_model(main_model);
    utils::apply_gather_before_matmul_transformation(main_model);
    validate_target_has_no_unmanaged_state(main_model);
    utils::dflash::expose_target_hidden_states(
        main_model,
        retained_hidden_state_locators,
        m_rt_info.target_layer_ids);

    m_tokenizer = main_model_desc.tokenizer;
    auto main_generation_config = main_model_desc.generation_config;
    dflash_cb::ensure_num_assistant_tokens_is_set(main_generation_config);
    OPENVINO_ASSERT(main_model_desc.scheduler_config.max_num_batched_tokens >= main_generation_config.num_assistant_tokens + 1,
                    "DFlash CB/PA requires max_num_batched_tokens >= num_assistant_tokens + 1 while it is limited ",
                    "to one active request and one running sequence.");
    m_generation_config = main_generation_config;
    auto target_scheduler_config = main_model_desc.scheduler_config;
    target_scheduler_config.num_linear_attention_blocks =
        dflash_cb::adjusted_linear_attention_block_count(target_scheduler_config.num_linear_attention_blocks,
                                                         main_generation_config.num_assistant_tokens,
                                                         target_has_linear_attention);
    auto draft_model_desc_for_runner = draft_model_desc;
    if (draft_model_desc_for_runner.device.empty()) {
        draft_model_desc_for_runner.device = main_model_desc.device;
    }
    utils::dflash::reshape_draft_hidden_states_input_for_cb(draft_model_desc_for_runner.model);

    m_draft = std::make_shared<DFlashCBDraftRunner>(draft_model_desc_for_runner,
                                                    m_tokenizer,
                                                    m_rt_info);

    m_main_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(
        main_model,
        main_model_desc.tokenizer,
        main_generation_config,
        target_scheduler_config,
        main_model_desc.device,
        main_model_desc.properties,
        true);
    m_main_pipeline->set_hidden_state_export_needed(true);

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_perf_metrics.main_model_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_perf_metrics.draft_model_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
}

GenerationConfig ContinuousBatchingPipeline::DFlashDecodingImpl::make_draft_generation_config(
    const GenerationConfig& config) const {
    auto draft_config = config;
    draft_config.ignore_eos = true;
    draft_config.stop_strings = {};
    draft_config.max_new_tokens = config.max_new_tokens + config.num_assistant_tokens;
    draft_config.num_assistant_tokens = 0;
    return draft_config;
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::append_pending_hidden_delta(RequestState& state,
                                                                                const ov::Tensor& hidden_delta,
                                                                                bool copy_data) {
    state.pending_hidden_deltas.append(hidden_delta, copy_data);
}

bool ContinuousBatchingPipeline::DFlashDecodingImpl::has_pending_hidden_delta(const RequestState& state) {
    return !state.pending_hidden_deltas.empty();
}

ov::Tensor ContinuousBatchingPipeline::DFlashDecodingImpl::materialize_pending_hidden_delta(
    const RequestState& state) {
    return state.pending_hidden_deltas.materialize();
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::clear_pending_hidden_delta(RequestState& state) {
    state.pending_hidden_deltas.clear();
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::validate_hidden_prefix_length(const RequestState& state) const {
    OPENVINO_ASSERT(!state.generated_tokens.empty(),
                    "DFlash hidden prefix can only be validated after target generated a seed token.");
    const size_t expected = state.prompt_len + state.generated_tokens.size() - 1;
    const size_t actual = m_draft->get_consumed_hidden_states() + state.pending_hidden_deltas.token_count();
    OPENVINO_ASSERT(actual == expected, "DFlash hidden prefix length mismatch before draft inference.");
}

bool ContinuousBatchingPipeline::DFlashDecodingImpl::has_active_request_state() const {
    for (const auto& [_, state] : m_request_states) {
        if (!state.finished) {
            return true;
        }
    }
    return false;
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::drop_finished_request_states() {
    for (auto state_it = m_request_states.begin(); state_it != m_request_states.end();) {
        if (state_it->second.finished) {
            state_it = m_request_states.erase(state_it);
        } else {
            ++state_it;
        }
    }
}

GenerationHandle ContinuousBatchingPipeline::DFlashDecodingImpl::add_request(
    uint64_t request_id,
    const ov::Tensor& input_ids,
    const ov::genai::GenerationConfig& sampling_params,
    std::optional<ov::Tensor> token_type_ids,
    std::optional<ov::Tensor> prompt_ids,
    std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    OPENVINO_ASSERT(sampling_params.is_greedy_decoding(), "DFlash CB/PA currently supports greedy decoding only.");
    OPENVINO_ASSERT(sampling_params.num_beams == 1, "DFlash CB/PA does not support beam search.");
    OPENVINO_ASSERT(sampling_params.num_return_sequences == 1, "DFlash CB/PA supports one sequence per request.");
    drop_finished_request_states();
    OPENVINO_ASSERT(!has_active_request_state() && !m_main_pipeline->has_non_finished_requests(),
                    "DFlash CB/PA POC supports only one active request. Wait for the current request to finish before adding another.");

    const auto input_shape = input_ids.get_shape();
    OPENVINO_ASSERT(input_shape.size() == 2 && input_shape[0] == 1 && input_shape[1] > 0,
                    "Expected DFlash input_ids shape [1, seq_len].");
    auto sampling_params_copy = sampling_params;
    dflash_cb::ensure_num_assistant_tokens_is_set(sampling_params_copy);
    RequestState state;
    state.generation_config = sampling_params_copy;
    state.prompt_len = input_shape[1];
    m_draft->initialize_sequence(input_ids, make_draft_generation_config(sampling_params_copy));
    m_request_states[request_id] = std::move(state);

    return m_main_pipeline->add_request(request_id,
                                        input_ids,
                                        sampling_params_copy,
                                        token_type_ids,
                                        prompt_ids,
                                        lm_extra_inputs);
}

GenerationHandle ContinuousBatchingPipeline::DFlashDecodingImpl::add_request(
    uint64_t request_id,
    const std::string& prompt,
    const ov::genai::GenerationConfig& sampling_params) {
    auto input_ids = m_tokenizer.encode(prompt).input_ids;
    return add_request(request_id, input_ids, sampling_params);
}

bool ContinuousBatchingPipeline::DFlashDecodingImpl::has_non_finished_requests() {
    return m_main_pipeline->has_non_finished_requests();
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::step() {
    std::lock_guard<std::mutex> lock{m_draft_generations_mutex};

    auto& raw_perf_counters = m_perf_metrics.raw_metrics;
    auto& main_raw_perf_counters = m_perf_metrics.main_model_metrics.raw_metrics;
    const auto step_start = std::chrono::steady_clock::now();

    m_main_pipeline->pull_awaiting_requests();

    std::map<uint64_t, size_t> draft_generated_by_request;
    const auto draft_start = std::chrono::steady_clock::now();
    for (auto& [request_id, state] : m_request_states) {
        if (state.finished) {
            clear_pending_hidden_delta(state);
            state.draft_generated = 0;
            continue;
        }
        if (!has_pending_hidden_delta(state) || state.generated_tokens.empty()) {
            state.draft_generated = 0;
            continue;
        }

        const size_t generated_len = state.generated_tokens.size();
        if (generated_len >= state.generation_config.max_new_tokens) {
            clear_pending_hidden_delta(state);
            state.draft_generated = 0;
            continue;
        }

        const size_t draft_count =
            dflash_cb::draft_candidate_count(state.generation_config.num_assistant_tokens,
                                            generated_len,
                                            state.generation_config.max_new_tokens);
        const size_t validation_count =
            dflash_cb::validation_candidate_count(draft_count, generated_len, state.generation_config.max_new_tokens);
        if (validation_count == 0) {
            clear_pending_hidden_delta(state);
            state.draft_generated = 0;
            continue;
        }

        const int64_t seed_token = state.generated_tokens.back();
        validate_hidden_prefix_length(state);
        auto hidden_delta = materialize_pending_hidden_delta(state);
        auto draft_logits = m_draft->infer(seed_token, hidden_delta, draft_count);
        clear_pending_hidden_delta(state);
        auto candidates = m_draft->sample_candidates(draft_logits, validation_count);

        state.generated_before_draft = state.generated_tokens.size();
        state.draft_generated = candidates.size();
        OPENVINO_ASSERT(!candidates.empty(),
                        "DFlash draft sampler produced no candidates despite requested validation candidates. ",
                        "request_id=", request_id,
                        ", generated_before_draft=", state.generated_before_draft,
                        ", draft_count=", draft_count,
                        ", validation_count=", validation_count);
        draft_generated_by_request[request_id] = candidates.size();

        auto candidate_tokens = state.generated_tokens;
        auto candidate_log_probs = zero_log_probs(candidate_tokens.size());
        for (const auto& candidate : candidates) {
            candidate_tokens.push_back(candidate.token_id);
            candidate_log_probs.push_back(candidate.log_prob);
        }
        OPENVINO_ASSERT(candidate_tokens.size() == candidate_log_probs.size(),
                        "DFlash draft candidate tokens and log-probs must stay aligned.");
        GeneratedSequences candidate_sequences;
        candidate_sequences.emplace(0, GeneratedSequence(candidate_tokens, candidate_log_probs));
        m_main_pipeline->update_request(request_id, candidate_sequences, false);
        state.target_la_checkpoint_sequence_id =
            m_main_pipeline->reserve_linear_attention_checkpoints_for_next_step(request_id, candidates.size() + 1);
    }
    const auto draft_end = std::chrono::steady_clock::now();
    m_sd_metrics.draft_duration += PerfMetrics::get_microsec(draft_end - draft_start) / 1e6;

    const auto main_start = std::chrono::steady_clock::now();
    try {
        m_main_pipeline->step();
    } catch (...) {
        for (auto& [_, state] : m_request_states) {
            m_main_pipeline->release_linear_attention_checkpoints_for_sequence(state.target_la_checkpoint_sequence_id);
            state.target_la_checkpoint_sequence_id.reset();
        }
        throw;
    }
    const auto main_end = std::chrono::steady_clock::now();
    const auto main_duration = PerfMetrics::get_microsec(main_end - main_start);
    m_sd_metrics.main_duration += main_duration / 1e6;
    m_pipeline_metrics = m_main_pipeline->get_metrics();

    auto main_generated_requests = m_main_pipeline->get_generated_requests();
    update_draft_states_from_main(main_generated_requests);
    for (auto& [request_id, state] : m_request_states) {
        if (main_generated_requests.find(request_id) == main_generated_requests.end()) {
            m_main_pipeline->release_linear_attention_checkpoints_for_sequence(state.target_la_checkpoint_sequence_id);
            state.target_la_checkpoint_sequence_id.reset();
            state.finished = true;
        }
    }

    for (const auto& [request_id, draft_generated] : draft_generated_by_request) {
        auto state_it = m_request_states.find(request_id);
        if (state_it == m_request_states.end()) {
            continue;
        }
        auto& state = state_it->second;
        if (draft_generated == 0 || state.generated_tokens.size() <= state.generated_before_draft) {
            m_main_pipeline->release_linear_attention_checkpoints_for_sequence(state.target_la_checkpoint_sequence_id);
            state.target_la_checkpoint_sequence_id.reset();
            continue;
        }
        const auto accounting =
            dflash_cb::validation_accounting(draft_generated, state.generated_before_draft, state.generated_tokens.size());
        if (!accounting.target_extended) {
            m_main_pipeline->release_linear_attention_checkpoints_for_sequence(state.target_la_checkpoint_sequence_id);
            state.target_la_checkpoint_sequence_id.reset();
            continue;
        }
        const size_t checkpoint_slot =
            dflash_cb::linear_attention_checkpoint_slot_for_validation(
                accounting,
                /*validation_input_includes_seed_token=*/true);
        m_main_pipeline->promote_linear_attention_checkpoint_for_sequence(state.target_la_checkpoint_sequence_id,
                                                                         checkpoint_slot);
        state.target_la_checkpoint_sequence_id.reset();
        const float acceptance_rate =
            draft_generated > 0 ? static_cast<float>(accounting.accepted) / draft_generated * 100.0f : 0.0f;
        m_sd_metrics.update_draft_generated_len(request_id, draft_generated);
        m_sd_metrics.update_draft_accepted_tokens(request_id, accounting.accepted);
        m_sd_metrics.update_acceptance_rate(request_id, acceptance_rate);
    }

    const auto step_end = std::chrono::steady_clock::now();
    const auto step_microsec_duration = PerfMetrics::get_microsec(step_end - step_start);
    const auto num_generated_tokens = m_main_pipeline->get_processed_tokens_per_iteration();
    if (num_generated_tokens > 0) {
        raw_perf_counters.m_token_infer_durations.emplace_back(step_microsec_duration);
        raw_perf_counters.m_inference_durations[0] += MicroSeconds(step_microsec_duration);
        raw_perf_counters.m_new_token_times.emplace_back(main_end);
        raw_perf_counters.m_batch_sizes.emplace_back(num_generated_tokens);

        auto main_pipeline_metrics = m_main_pipeline->get_metrics();
        main_raw_perf_counters.m_durations.push_back(MicroSeconds(main_duration));
        main_raw_perf_counters.m_inference_durations[0] += MicroSeconds(main_pipeline_metrics.inference_duration);
        main_raw_perf_counters.m_batch_sizes.push_back(num_generated_tokens);
        m_sd_metrics.update_generated_len(num_generated_tokens);
    }

}

void ContinuousBatchingPipeline::DFlashDecodingImpl::update_draft_states_from_main(
    const GeneratedRequests& main_generated_requests) {
    for (const auto& [request_id, generated_sequences] : main_generated_requests) {
        auto state_it = m_request_states.find(request_id);
        if (state_it == m_request_states.end() || generated_sequences.empty()) {
            continue;
        }

        auto& state = state_it->second;
        const auto& generated_sequence = generated_sequences.begin()->second;
        const auto accounting =
            dflash_cb::validation_accounting(state.draft_generated,
                                             state.generated_before_draft,
                                             generated_sequence.token_ids.size());

        auto hidden_delta = dflash_cb::truncate_normalized_hidden_state_from_end(generated_sequence.hidden_states,
                                                                                 accounting.rejected);
        append_pending_hidden_delta(state, hidden_delta, generated_sequence.token_ids.empty());
        state.generated_tokens = generated_sequence.token_ids;
        m_draft->sync_generated_tokens(state.generated_tokens);
        state.draft_generated = 0;
    }
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::drop_requests() {
    for (auto& [_, state] : m_request_states) {
        if (m_main_pipeline) {
            m_main_pipeline->release_linear_attention_checkpoints_for_sequence(state.target_la_checkpoint_sequence_id);
            state.target_la_checkpoint_sequence_id.reset();
        }
    }
    if (m_main_pipeline) {
        m_main_pipeline->finish_request();
    }
    m_request_states.clear();
}

ov::genai::RawPerfMetrics ContinuousBatchingPipeline::DFlashDecodingImpl::collect_draft_raw_metrics() {
    ov::genai::RawPerfMetrics raw_metrics;
    raw_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    if (m_draft) {
        auto& draft_metrics = m_draft->get_raw_perf_metrics();
        raw_metrics.m_durations.insert(raw_metrics.m_durations.end(),
                                       draft_metrics.m_durations.begin(),
                                       draft_metrics.m_durations.end());
        raw_metrics.m_batch_sizes.insert(raw_metrics.m_batch_sizes.end(),
                                         draft_metrics.m_batch_sizes.begin(),
                                         draft_metrics.m_batch_sizes.end());
        if (!draft_metrics.m_inference_durations.empty()) {
            raw_metrics.m_inference_durations[0] += draft_metrics.m_inference_durations[0];
        }
    }
    return raw_metrics;
}

std::vector<EncodedGenerationResult> ContinuousBatchingPipeline::DFlashDecodingImpl::generate(
    const std::vector<ov::Tensor>& input_ids,
    const std::vector<GenerationConfig>& sampling_params,
    const StreamerVariant& streamer,
    const std::optional<std::vector<ov::Tensor>>& token_type_ids,
    const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids,
    const std::optional<std::vector<ov::Tensor>>& prompt_ids,
    const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list) {
    OPENVINO_ASSERT(!position_ids.has_value(), "DFlash CB/PA does not support explicit position_ids yet.");
    OPENVINO_ASSERT(!has_non_finished_requests(),
                    "Generate cannot be called while ContinuousBatchingPipeline is already running");
    OPENVINO_ASSERT(input_ids.size() == sampling_params.size());
    OPENVINO_ASSERT(input_ids.size() == 1, "DFlash CB/PA POC supports batch size 1 only.");

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_perf_metrics.main_model_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_perf_metrics.draft_model_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_sd_metrics.clean_up();
    m_request_states.clear();
    auto start_time = std::chrono::steady_clock::now();

    for (size_t i = 1; i < sampling_params.size(); ++i) {
        OPENVINO_ASSERT(sampling_params[i - 1].adapters == sampling_params[i].adapters,
                        "LoRA adapters must be same for all requests");
    }
    m_main_pipeline->set_adapters(sampling_params[0].adapters);

    auto streamer_ptr = std::make_shared<ThreadedStreamerWrapper>(streamer, m_tokenizer);
    OPENVINO_ASSERT(!streamer_ptr->has_callback() ||
                        (input_ids.size() == 1 && sampling_params[0].is_greedy_decoding()),
                    "DFlash CB/PA streaming only supports batch size=1 with greedy decoding.");

    std::vector<GenerationHandle> main_generations;
    for (size_t request_id = 0; request_id < input_ids.size(); ++request_id) {
        OPENVINO_ASSERT(input_ids[request_id].get_shape().at(0) == 1, "Use multiple tensors to pass a batch.");
        const bool has_valid_token_type_ids = token_type_ids.has_value() && request_id < token_type_ids->size();
        const bool has_valid_prompt_ids = prompt_ids.has_value() && request_id < prompt_ids->size();
        const bool has_valid_lm_extra_inputs = lm_extra_inputs_list.has_value() && request_id < lm_extra_inputs_list->size();
        main_generations.push_back(add_request(
            request_id,
            input_ids[request_id],
            sampling_params[request_id],
            has_valid_token_type_ids ? std::make_optional((*token_type_ids)[request_id]) : std::nullopt,
            has_valid_prompt_ids ? std::make_optional((*prompt_ids)[request_id]) : std::nullopt,
            has_valid_lm_extra_inputs ? std::make_optional((*lm_extra_inputs_list)[request_id]) : std::nullopt));
    }

    auto all_requests = m_main_pipeline->get_awaiting_requests();
    GenerationHandle& generation = main_generations.at(0);

    streamer_ptr->start();
    while (has_non_finished_requests()) {
        try {
            step();
        } catch (...) {
            drop_requests();
            streamer_ptr->end();
            std::rethrow_exception(std::current_exception());
        }
        stream_tokens(streamer_ptr, generation);
    }
    streamer_ptr->end();

    OPENVINO_ASSERT(m_main_pipeline->is_requests_empty(),
                    "Internal error: current request is supposed to be dropped within step() function as completed");

    m_perf_metrics.draft_model_metrics.raw_metrics = collect_draft_raw_metrics();
    uint64_t generate_duration_us = PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start_time);

    std::vector<EncodedGenerationResult> results;
    results.reserve(all_requests.size());

    for (size_t request_id = 0; request_id < all_requests.size(); ++request_id) {
        const auto& request = all_requests[request_id];
        auto cfg = request->get_sampling_parameters();
        const auto& seqs = request->get_finished_sequences();
        size_t num_out = std::min(cfg.num_return_sequences, seqs.size());

        EncodedGenerationResult result;
        result.m_request_id = request_id;
        result.m_generation_ids.resize(num_out);
        result.m_scores.resize(num_out);
        result.m_finish_reasons.resize(num_out, GenerationFinishReason::NONE);
        result.m_status = main_generations[request_id]->get_status();

        for (size_t i = 0; i < num_out; ++i) {
            const auto& seq = seqs[i];
            float score = cfg.is_beam_search() ? seq->get_beam_search_score(cfg) : seq->get_cumulative_log_prob();
            const auto& gen_ids = seq->get_generated_ids();
            if (cfg.echo) {
                result.m_generation_ids[i] = request->get_prompt_ids();
            }
            std::copy(gen_ids.begin(), gen_ids.end(), std::back_inserter(result.m_generation_ids[i]));
            result.m_scores[i] = score;
            result.m_finish_reasons[i] = seq->get_finish_reason();
            if (result.m_finish_reasons[i] == GenerationFinishReason::NONE && request->handle_stopped()) {
                result.m_finish_reasons[i] = request->get_generation_stream()->get_finish_reason();
            }
        }

        m_perf_metrics.raw_metrics.generate_durations.clear();
        m_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_duration_us);
        m_perf_metrics.num_input_tokens = request->get_prompt_len();
        m_perf_metrics.evaluate_statistics(start_time);

        result.perf_metrics = m_perf_metrics;
        result.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_perf_metrics);
        results.push_back(std::move(result));
    }

    m_request_states.clear();
    OPENVINO_ASSERT(results.size() == input_ids.size());
    return results;
}

}  // namespace ov::genai
