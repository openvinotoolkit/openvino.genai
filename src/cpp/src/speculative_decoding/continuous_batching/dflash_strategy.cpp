// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_strategy.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <string>
#include <vector>

#include "sequence_group.hpp"
#include "speculative_decoding/continuous_batching/dflash_candidate_selector.hpp"
#include "utils.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

namespace {

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

// The target decodes a request alone with num_assistant_tokens == 0, and an echo-only request
// (max_new_tokens == 0) generates nothing to draft.
bool has_draft_request(const GenerationConfig& config) {
    return config.num_assistant_tokens.value_or(0) > 0 && config.max_new_tokens > 0;
}

}  // namespace

ContinuousBatchingPipeline::DFlashDecodingImpl::DFlashDecodingImpl(
    const ov::genai::ModelDesc& main_model_desc,
    const ov::genai::ModelDesc& draft_model_desc,
    const ov::genai::utils::dflash::DFlashRTInfo& rt_info)
    : m_rt_info(rt_info) {
    OPENVINO_ASSERT(m_rt_info.dflash_mode, "DFlash continuous batching requires dflash_mode=true.");
    OPENVINO_ASSERT(!m_rt_info.target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");
    OPENVINO_ASSERT(m_rt_info.input_embedding_scale > 0.0f,
                    "DFlash input_embedding_scale must be positive, got ",
                    m_rt_info.input_embedding_scale,
                    ".");
    OPENVINO_ASSERT(m_rt_info.output_multiplier > 0.0f,
                    "DFlash output_multiplier must be positive, got ",
                    m_rt_info.output_multiplier,
                    ".");
    OPENVINO_ASSERT(m_rt_info.final_logit_softcapping >= 0.0f,
                    "DFlash final_logit_softcapping cannot be negative, got ",
                    m_rt_info.final_logit_softcapping,
                    ".");
    auto main_model = main_model_desc.model;
    auto draft_model = draft_model_desc.model;
    OPENVINO_ASSERT(main_model && draft_model, "DFlash requires both target and draft models.");
    OPENVINO_ASSERT(utils::has_input(draft_model, "token_type_ids"),
                    "DFlash draft model has no 'token_type_ids' input. It was exported for the removed stateful "
                    "DFlash draft runner; re-export the draft with the current optimum-intel.");
    // TODO: support cache eviction. The draft runs with an unlimited max_num_batched_tokens, which sizes the cache
    // rotation table, and the draft windows were not validated with evicted draft KV.
    OPENVINO_ASSERT(!main_model_desc.scheduler_config.use_cache_eviction &&
                        !draft_model_desc.scheduler_config.use_cache_eviction,
                    "DFlash speculative decoding does not support cache eviction.");
    const std::string draft_device = draft_model_desc.device.empty() ? main_model_desc.device : draft_model_desc.device;
    const bool draft_inherits_target_properties = draft_model_desc.properties.empty();
    // PagedAttention is the only DFlash draft backend.
    auto [draft_properties, draft_attention_backend] = utils::extract_attention_backend(
        draft_inherits_target_properties ? main_model_desc.properties : draft_model_desc.properties);
    OPENVINO_ASSERT(draft_attention_backend == PA_BACKEND,
                    "DFlash drafts run on PagedAttention only; the draft ATTENTION_BACKEND must be '",
                    PA_BACKEND,
                    "', got '",
                    draft_attention_backend,
                    "'.");
    if (main_model_desc.inputs_embedder) {
        m_inputs_embedder = main_model_desc.inputs_embedder;
        m_model_input_type = ModelInputType::EMBEDDINGS;
        m_embeddings_registry = std::make_shared<EmbeddingsRegistry>();
    }

    // Detect each draft capability independently from its I/O signature (no RT-info markers) and
    // apply only the missing transform(s); a well-formed draft has exactly one input/output per pair.
    const bool needs_embedding_attach = utils::has_input(draft_model, "inputs_embeds");
    const bool needs_lm_head_graft = model_has_output(draft_model, "last_hidden_state");
    OPENVINO_ASSERT(needs_embedding_attach != utils::has_input(draft_model, "input_ids"),
                    "DFlash draft model must have exactly one of 'inputs_embeds' or 'input_ids' input.");
    OPENVINO_ASSERT(needs_lm_head_graft != model_has_output(draft_model, "logits"),
                    "DFlash draft model must have exactly one of 'last_hidden_state' or 'logits' output.");

    // Resolve authoritative metadata before mutating either model so a malformed target leaves the draft reusable.
    auto retained_hidden_state_locators =
        utils::dflash::resolve_target_hidden_state_locators(main_model, m_rt_info.target_layer_ids);

    // The VLM draft keeps its external inputs_embeds; the LLM draft gets the target embedding grafted in.
    if (m_inputs_embedder) {
        OPENVINO_ASSERT(needs_embedding_attach, "DFlash VLM draft model must be inputs_embeds-native.");
        utils::dflash::scale_draft_inputs_embeds(draft_model, m_rt_info.input_embedding_scale);
    } else if (needs_embedding_attach) {
        utils::dflash::attach_target_embedding_to_draft(main_model, draft_model, m_rt_info.input_embedding_scale);
    }
    // A gated Markov head scores the candidates from the draft hidden states as well as from its logits.
    const bool markov_head_reads_hidden_states =
        m_rt_info.markov_head_computation && utils::has_input(m_rt_info.markov_head_computation, "draft_hidden_states");
    if (needs_lm_head_graft) {
        utils::dflash::attach_target_lm_head_to_draft(main_model, draft_model, markov_head_reads_hidden_states);
    }
    OPENVINO_ASSERT(!markov_head_reads_hidden_states || model_has_output(draft_model, "last_hidden_state"),
                    "A DFlash draft with a gated Markov head must expose 'last_hidden_state'.");

    auto [main_scheduler_config, draft_scheduler_config] = init_speculative_models(main_model_desc, draft_model_desc);
    OPENVINO_ASSERT(main_scheduler_config.enable_prefix_caching == draft_scheduler_config.enable_prefix_caching,
                    "DFlash main and draft pipelines must use the same enable_prefix_caching setting.");
    // The draft candidates are read at fixed rows of the gathered logits.
    OPENVINO_ASSERT(utils::has_input(draft_model, "sampled_tokens_indices"),
                    "DFlash draft model has no 'sampled_tokens_indices' input, so its candidate logits cannot be gathered.");
    if (markov_head_reads_hidden_states) {
        utils::dflash::gather_last_hidden_state_with_logits(draft_model);
    }
    validate_target_has_no_unmanaged_state(main_model);
    utils::dflash::expose_target_hidden_states(main_model, retained_hidden_state_locators, m_rt_info.target_layer_ids);
    utils::dflash::move_fc_from_draft_to_main(draft_model, main_model);

    // The budget never limits a draft window, so the scheduler never splits a bidirectional block. A draft
    // step is still bounded by the target's previous step plus one block per request.
    draft_scheduler_config.num_linear_attention_blocks = 0;
    draft_scheduler_config.max_num_batched_tokens = std::numeric_limits<size_t>::max();

    m_tokenizer = main_model_desc.tokenizer;
    m_generation_config = main_model_desc.generation_config;
    auto candidate_selector = std::make_shared<dflash_cb::CandidateSelector>(m_rt_info, draft_device, draft_properties);
    m_rt_info.markov_head_computation.reset();
    utils::dflash::apply_dflash_gpu_compile_properties(m_rt_info,
                                                       draft_device,
                                                       draft_inherits_target_properties,
                                                       draft_properties);

    if (m_inputs_embedder) {
        m_main_pipeline = std::make_shared<ContinuousBatchingForDFlashDecodingImpl>(main_model,
                                                                                   m_inputs_embedder,
                                                                                   m_tokenizer,
                                                                                   m_generation_config,
                                                                                   main_scheduler_config,
                                                                                   main_model_desc.device,
                                                                                   main_model_desc.properties,
                                                                                   true);
        m_draft_pipeline =
            std::make_shared<ContinuousBatchingForDFlashDecodingImpl>(draft_model,
                                                                      m_inputs_embedder,
                                                                      m_tokenizer,
                                                                      m_generation_config,
                                                                      draft_scheduler_config,
                                                                      draft_device,
                                                                      draft_properties,
                                                                      false,
                                                                      m_rt_info.mask_token_id,
                                                                      m_rt_info.candidate_position_offset,
                                                                      candidate_selector);
    } else {
        m_main_pipeline = std::make_shared<ContinuousBatchingForDFlashDecodingImpl>(main_model,
                                                                                   m_tokenizer,
                                                                                   m_generation_config,
                                                                                   main_scheduler_config,
                                                                                   main_model_desc.device,
                                                                                   main_model_desc.properties,
                                                                                   true);
        m_draft_pipeline =
            std::make_shared<ContinuousBatchingForDFlashDecodingImpl>(draft_model,
                                                                      m_tokenizer,
                                                                      m_generation_config,
                                                                      draft_scheduler_config,
                                                                      draft_device,
                                                                      draft_properties,
                                                                      false,
                                                                      m_rt_info.mask_token_id,
                                                                      m_rt_info.candidate_position_offset,
                                                                      candidate_selector);
    }
    m_main_pipeline->set_hidden_state_export_needed(true);

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_draft_pipeline->reset_draft_perf_metrics();
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::validate_request_config(const GenerationConfig& config) const {
    OPENVINO_ASSERT(config.is_greedy_decoding() || config.is_multinomial(),
                    "DFlash supports greedy and multinomial decoding only.");
    OPENVINO_ASSERT(config.num_return_sequences == 1,
                    "DFlash supports one sequence per request; num_return_sequences must be 1, got ",
                    config.num_return_sequences,
                    ".");
    OPENVINO_ASSERT(!config.adapters.has_value(),
                    "DFlash does not support adapters until target and draft adapter parity is validated.");
    OPENVINO_ASSERT(config.assistant_confidence_threshold == 0.f,
                    "DFlash supports num_assistant_tokens only; assistant_confidence_threshold must be 0.f.");
    OPENVINO_ASSERT(config.max_ngram_size == 0, "DFlash does not support prompt lookup decoding; max_ngram_size must be 0.");
    if (m_model_input_type == ModelInputType::EMBEDDINGS) {
        // Speculative decoding results carry no Omni intermediate hidden states or full token IDs.
        OPENVINO_ASSERT(!config.return_omni_outputs,
                        "DFlash VLM does not support Omni intermediate outputs; return_omni_outputs must be false.");
    }
}

GenerationConfig ContinuousBatchingPipeline::DFlashDecodingImpl::make_draft_generation_config(
    const GenerationConfig& config) {
    // The target decides when to stop and what to return; the draft only proposes candidates, drawn independently
    // of the target's draws. ignore_eos disables the stop tokens; they stay because validate() requires
    // eos_token_id among them.
    GenerationConfig draft_config = make_draft_sampling_params(config);
    draft_config.include_stop_str_in_output = false;
    draft_config.structured_output_config.reset();
    draft_config.parsers.clear();
    draft_config.echo = false;
    draft_config.logprobs = 0;
    if (draft_config.max_new_tokens == SIZE_MAX && draft_config.max_length == SIZE_MAX) {
        // ignore_eos requires a finite generation limit.
        draft_config.max_new_tokens = SIZE_MAX - 1;
    }
    return draft_config;
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::align_request_pair_processed_prefix(uint64_t request_id) {
    if (!m_main_pipeline->is_prefix_caching_enabled()) {
        return;
    }
    const auto main_group = m_main_pipeline->find_awaiting_request(request_id);
    OPENVINO_ASSERT(main_group, "DFlash prefix alignment requires the awaiting target request: ", request_id);
    // The target recomputes at least the last prompt token, which yields the first generated token.
    size_t ceiling = main_group->get_prompt_len() - 1;
    if (!m_draft_pipeline->find_awaiting_request(request_id)) {
        m_main_pipeline->restore_awaiting_prefix(request_id, ceiling);
        return;
    }
    // The draft needs target hidden states from its restored length on, so both restore the same prefix.
    while (true) {
        const size_t main_processed = m_main_pipeline->restore_awaiting_prefix(request_id, ceiling);
        const size_t draft_processed = m_draft_pipeline->restore_awaiting_prefix(request_id, main_processed);
        if (main_processed == draft_processed) {
            return;
        }
        OPENVINO_ASSERT(draft_processed < ceiling, "DFlash prefix alignment must select an earlier prefix.");
        ceiling = draft_processed;
    }
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::validate_awaiting_requests(
    const std::vector<SequenceGroup::Ptr>& main_awaiting_requests,
    const std::vector<SequenceGroup::Ptr>& draft_awaiting_requests) const {
    const size_t expected_draft_requests =
        std::count_if(main_awaiting_requests.begin(), main_awaiting_requests.end(), [](const SequenceGroup::Ptr& request) {
            return has_draft_request(request->get_sampling_parameters());
        });
    OPENVINO_ASSERT(draft_awaiting_requests.size() == expected_draft_requests,
                    "DFlash awaiting request mismatch: draft queue size is ", draft_awaiting_requests.size(),
                    ", but expected ", expected_draft_requests, " (main requests with drafting enabled).");
}

GenerationHandle ContinuousBatchingPipeline::DFlashDecodingImpl::add_request(
    uint64_t request_id,
    const ov::Tensor& input_ids,
    const ov::genai::GenerationConfig& sampling_params,
    std::optional<ov::Tensor> prompt_ids,
    std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    validate_request_config(sampling_params);
    OPENVINO_ASSERT(m_model_input_type != ModelInputType::EMBEDDINGS || !m_main_pipeline->is_prefix_caching_enabled() ||
                        (prompt_ids.has_value() && prompt_ids->get_element_type() == ov::element::i64 &&
                         prompt_ids->get_size() == input_ids.get_shape()[1]),
                    "DFlash prefix caching requires built-in prompt preparation with one token ID per embedding");
    auto main_sampling_params = sampling_params;
    if (!main_sampling_params.num_assistant_tokens.has_value()) {
        main_sampling_params.num_assistant_tokens = m_main_pipeline->default_num_assistant_tokens;
    }
    OPENVINO_ASSERT(!has_draft_request(main_sampling_params) || main_sampling_params.is_greedy_decoding() ||
                        !m_main_pipeline->is_prefix_caching_enabled() || !m_main_pipeline->has_linear_attention_cache(),
                    "DFlash speculative decoding of models with linear-attention state supports greedy decoding only "
                    "when prefix caching is enabled; set enable_prefix_caching to false to use multinomial sampling.");

    // The draft prompt is the target prompt: draft position t pairs with target hidden state t.
    // insert_or_assign drops a stale handle a finished request may leave under the same request_id.
    try {
        if (has_draft_request(main_sampling_params)) {
            m_draft_generations.insert_or_assign(
                request_id,
                m_draft_pipeline->add_request(request_id, input_ids, make_draft_generation_config(main_sampling_params), prompt_ids));
        }
        auto main_handle =
            m_main_pipeline->add_request(request_id, input_ids, main_sampling_params, prompt_ids, lm_extra_inputs);
        align_request_pair_processed_prefix(request_id);
        return main_handle;
    } catch (...) {
        m_main_pipeline->discard_awaiting_request(request_id);
        m_draft_pipeline->discard_awaiting_request(request_id);
        m_draft_generations.erase(request_id);
        throw;
    }
}

GenerationHandle ContinuousBatchingPipeline::DFlashDecodingImpl::add_request(
    uint64_t request_id,
    const std::string& prompt,
    const ov::genai::GenerationConfig& sampling_params) {
    if (m_model_input_type == ModelInputType::EMBEDDINGS) {
        return ContinuousBatchingPipeline::IContinuousBatchingPipeline::add_request(request_id, prompt, {}, sampling_params);
    }
    return add_request(request_id, m_tokenizer.encode(prompt).input_ids, sampling_params);
}

std::vector<EncodedGenerationResult> ContinuousBatchingPipeline::DFlashDecodingImpl::generate(
    const std::vector<ov::Tensor>& input_ids,
    const std::vector<GenerationConfig>& sampling_params,
    const StreamerVariant& streamer,
    const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids,
    const std::optional<std::vector<ov::Tensor>>& prompt_ids,
    const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list) {
    GenerateStrategy strategy;
    // The configs pass through unchanged: add_request fills an unset num_assistant_tokens with the default and
    // keeps an explicit 0, which runs the request on the target alone.
    strategy.prepare_request = [](size_t, const ov::Tensor& in_ids, GenerationConfig&, ov::Tensor& main_in) {
        main_in = in_ids;
    };
    strategy.check_streaming = [](const std::shared_ptr<ThreadedStreamerWrapper>& streamer_ptr,
                                  const std::vector<ov::Tensor>& input_ids,
                                  const std::vector<GenerationConfig>&) {
        OPENVINO_ASSERT(!streamer_ptr->has_callback() || input_ids.size() == 1,
                        "DFlash streaming only supports batch size=1.");
    };
    strategy.start_timer = []() { return std::chrono::steady_clock::now(); };
    strategy.stop_timer = [](const TimePoint& start) {
        return PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start);
    };

    return generate_common(this, input_ids, sampling_params, streamer, position_ids, prompt_ids, lm_extra_inputs_list,
                           strategy);
}

}  // namespace ov::genai
