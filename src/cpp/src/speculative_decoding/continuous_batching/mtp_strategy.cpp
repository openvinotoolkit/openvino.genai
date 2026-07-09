// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mtp_strategy.hpp"

#include "openvino/pass/sdpa_to_paged_attention.hpp"

#include "continuous_batching/paged_attention_transformations.hpp"
#include "logger.hpp"
#include "speculative_decoding/mtp_model_transforms.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

// Align hidden[t] with embed(token[t+1]).
ov::Tensor create_draft_input_embeds(const ov::Tensor& input_embeds) {
    const auto shape = input_embeds.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && shape[1] > 1,
                    "MTP draft input embeds expect shape [1, seq_len>1, hidden_size], got ", shape);

    auto [start_coord, end_coord] = ov::genai::utils::make_roi(shape, 1, 1, shape[1]);
    ov::Tensor shifted(input_embeds.get_element_type(), {shape[0], shape[1] - 1, shape[2]});
    ov::Tensor(input_embeds, start_coord, end_coord).copy_to(shifted);
    return shifted;
}

}  // namespace

ContinuousBatchingPipeline::MtpDecodingImpl::MtpDecodingImpl(const ov::genai::ModelDesc& main_model_desc,
                                                            const ov::genai::ModelDesc& draft_model_desc,
                                                            const std::shared_ptr<InputsEmbedder>& inputs_embedder) {
    auto main_model = main_model_desc.model;
    auto draft_model = draft_model_desc.model;
    OPENVINO_ASSERT(main_model && draft_model, "MTP requires both a main and a draft (MTP) model.");
    OPENVINO_ASSERT(inputs_embedder, "MTP requires a shared InputsEmbedder for the text embeddings model.");

    auto main_device = main_model_desc.device;
    std::string draft_device = draft_model_desc.device.empty() ? main_model_desc.device : draft_model_desc.device;
    ov::AnyMap draft_properties =
        draft_model_desc.properties.empty() ? main_model_desc.properties : draft_model_desc.properties;

    const Tokenizer& main_model_tokenizer = main_model_desc.tokenizer;
    const Tokenizer& draft_model_tokenizer = draft_model_desc.tokenizer;
    m_tokenizer = main_model_tokenizer;
    m_inputs_embedder = inputs_embedder;
    m_model_input_type = ModelInputType::EMBEDDINGS;

    // Strip exporter f32->bf16->f32 KV-cache round trips before PA conversion.
    utils::mtp::remove_roundtrip_converts(draft_model);

    // PA conversion must precede the MTP lm_head graft.
    bool allow_score_aggregation = true;
    bool allow_xattention = false;
    ov::pass::SDPAToPagedAttention(main_model_desc.scheduler_config.use_cache_eviction,
                                   main_model_desc.scheduler_config.use_cache_eviction,
                                   allow_score_aggregation,
                                   allow_xattention).run_on_model(main_model);
    ov::pass::SDPAToPagedAttention(false, false, allow_score_aggregation, allow_xattention).run_on_model(draft_model);

    utils::mtp::graft_lm_head_on_mtp(draft_model, main_model);
    utils::mtp::expose_last_hidden_state(main_model);

    utils::apply_gather_before_matmul_transformation(main_model);
    utils::apply_gather_before_matmul_transformation(draft_model);

    // Bypass default KV-ratio split: MTP draft needs only a small cache slice.
    auto main_scheduler_config = main_model_desc.scheduler_config;
    auto draft_scheduler_config = main_scheduler_config;
    if (draft_model_desc.scheduler_config == SchedulerConfig()) {
        constexpr size_t MTP_DRAFT_CACHE_SIZE_GB = 1;
        if (main_scheduler_config.cache_size > MTP_DRAFT_CACHE_SIZE_GB) {
            draft_scheduler_config.cache_size = MTP_DRAFT_CACHE_SIZE_GB;
            main_scheduler_config.cache_size -= MTP_DRAFT_CACHE_SIZE_GB;
        }
    } else {
        draft_scheduler_config = draft_model_desc.scheduler_config;
        draft_scheduler_config.dynamic_split_fuse = main_scheduler_config.dynamic_split_fuse;
        draft_scheduler_config.max_num_batched_tokens = main_scheduler_config.max_num_batched_tokens;
    }

    m_main_pipeline = std::make_shared<ContinuousBatchingForMtpDecodingImpl>(main_model,
                                                                            inputs_embedder,
                                                                            main_model_tokenizer,
                                                                            main_model_desc.generation_config,
                                                                            main_scheduler_config,
                                                                            main_device,
                                                                            main_model_desc.properties,
                                                                            true);
    m_draft_pipeline = std::make_shared<ContinuousBatchingForMtpDecodingImpl>(draft_model,
                                                                             inputs_embedder,
                                                                             draft_model_tokenizer,
                                                                             draft_model_desc.generation_config,
                                                                             draft_scheduler_config,
                                                                             draft_device,
                                                                             draft_properties,
                                                                             false);

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_draft_pipeline->raw_perf_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};

    enable_mtp_hidden_state_pairing();
}

void ContinuousBatchingPipeline::MtpDecodingImpl::enable_mtp_hidden_state_pairing() {
    auto main_mtp_pipeline = std::dynamic_pointer_cast<ContinuousBatchingForMtpDecodingImpl>(m_main_pipeline);
    auto draft_mtp_pipeline = std::dynamic_pointer_cast<ContinuousBatchingForMtpDecodingImpl>(m_draft_pipeline);
    main_mtp_pipeline->set_hidden_state_export_needed(true);
    draft_mtp_pipeline->set_hidden_state_export_needed(true);
    draft_mtp_pipeline->set_hidden_state_import_needed(true);
    draft_mtp_pipeline->set_hidden_state_internal_needed(true);
    draft_mtp_pipeline->set_mtp_draft_positions_needed(true);
}

GenerationHandle ContinuousBatchingPipeline::MtpDecodingImpl::add_request(
    uint64_t request_id,
    const ov::Tensor& input_ids,
    const ov::genai::GenerationConfig& sampling_params,
    std::optional<ov::Tensor> token_type_ids,
    std::optional<ov::Tensor> prompt_ids,
    std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};
    // Draft gets shifted embeds only; VLM extras belong to the main model.
    ov::Tensor draft_input_embeds = create_draft_input_embeds(input_ids);
    // Use insert_or_assign, not insert: a finished prior request may leave a stale (stopped) handle
    // under the same request_id. insert() would be a no-op there, dropping the new handle as a
    // temporary whose destructor stops the freshly added draft request before it can draft.
    m_draft_generations.insert_or_assign(request_id, m_draft_pipeline->add_request(request_id, draft_input_embeds, draft_sampling_params));
    return m_main_pipeline->add_request(request_id, input_ids, sampling_params, token_type_ids, prompt_ids, lm_extra_inputs);
}

GenerationHandle ContinuousBatchingPipeline::MtpDecodingImpl::add_request(
    uint64_t request_id,
    const std::string& prompt,
    const ov::genai::GenerationConfig& sampling_params) {
    // Text-only serving path.
    ov::genai::VLMPerfMetrics metrics;
    ov::Tensor inputs_embeds;
    std::optional<ov::Tensor> token_type_ids;
    {
        std::lock_guard<std::mutex> lock(m_embeddings_mutex);
        m_inputs_embedder->set_apply_chat_template_status(sampling_params.apply_chat_template);
        const std::vector<ov::genai::EncodedImage> no_images;
        const auto [unified_prompt, image_sequence, video_sequence] =
            m_inputs_embedder->normalize_prompt(prompt, 0, no_images);
        if (m_inputs_embedder->has_token_type_ids()) {
            std::tie(inputs_embeds, token_type_ids) =
                m_inputs_embedder->get_inputs_embeds_with_token_type_ids(unified_prompt, no_images, metrics, true, image_sequence);
        } else {
            inputs_embeds = m_inputs_embedder->get_inputs_embeds(unified_prompt, no_images, metrics, true, image_sequence);
        }
        const auto [position_ids, rope_delta] = m_inputs_embedder->get_position_ids(inputs_embeds.get_shape()[1], 0);
        m_inputs_embedder->set_position_ids(position_ids);
        if (rope_delta.has_value()) {
            m_inputs_embedder->set_rope_delta(*rope_delta);
        }
    }
    return add_request(request_id, inputs_embeds, sampling_params, token_type_ids, std::nullopt,
                       m_inputs_embedder->get_lm_extra_inputs());
}

std::vector<EncodedGenerationResult> ContinuousBatchingPipeline::MtpDecodingImpl::generate(
    const std::vector<ov::Tensor>& input_ids,
    const std::vector<GenerationConfig>& sampling_params,
    const StreamerVariant& streamer,
    const std::optional<std::vector<ov::Tensor>>& token_type_ids,
    const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids,
    const std::optional<std::vector<ov::Tensor>>& prompt_ids,
    const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list) {
    GenerateStrategy strategy;
    strategy.prepare_request = [this](size_t,
                                      const ov::Tensor& in_embeds,
                                      GenerationConfig& main_cfg,
                                      GenerationConfig& draft_cfg,
                                      ov::Tensor& main_in,
                                      ov::Tensor& draft_in) {
        OPENVINO_ASSERT(main_cfg.assistant_confidence_threshold == 0.f,
                        "MTP only supports num_assistant_tokens (assistant_confidence_threshold must be 0.f).");
        OPENVINO_ASSERT(main_cfg.is_greedy_decoding() && main_cfg.num_return_sequences == 1,
                        "MTP speculative decoding currently supports only greedy, batch-1 generation.");
        OPENVINO_ASSERT(main_cfg.num_assistant_tokens > 0,
                        "MTP speculative decoding requires num_assistant_tokens > 0.");
        draft_cfg.num_assistant_tokens = main_cfg.num_assistant_tokens;
        draft_cfg.ignore_eos = true;
        draft_cfg.stop_strings = {};
        main_in = in_embeds;
        draft_in = create_draft_input_embeds(in_embeds);
    };

    strategy.check_streaming = [](const std::shared_ptr<ThreadedStreamerWrapper>& streamer_ptr,
                                  const std::vector<ov::Tensor>& input_ids,
                                  const std::vector<GenerationConfig>& sampling_params) {
        OPENVINO_ASSERT(!streamer_ptr->has_callback() ||
                            (input_ids.size() == 1 && sampling_params[0].is_greedy_decoding()),
                        "MTP streaming only supports batch size=1 with greedy decoding.");
    };
    strategy.start_timer = []() { return std::chrono::steady_clock::now(); };
    strategy.stop_timer = [](const TimePoint& start) {
        return PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start);
    };

    // generate_common threads position_ids / token_type_ids / lm_extra_inputs to the main pipeline
    // (priming the shared embedder's M-RoPE positions) via self->add_request(); the MTP draft ignores
    // them (sequential positions, no VLM inputs) inside MtpDecodingImpl::add_request.
    return generate_common(this, input_ids, sampling_params, streamer, token_type_ids, position_ids,
                           prompt_ids, lm_extra_inputs_list, strategy);
}
}  // namespace ov::genai
