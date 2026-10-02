// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mtp_strategy.hpp"

#include <array>
#include <charconv>
#include <chrono>
#include <limits>
#include <map>

#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"

#include "continuous_batching/paged_attention_transformations.hpp"
#include "logger.hpp"
#include "speculative_decoding/mtp_model_transforms.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

constexpr std::array<const char*, 4> gemma4_shared_kv_names = {
    "full_attention_key", "full_attention_value", "sliding_attention_key", "sliding_attention_value"
};

std::array<size_t, 2> find_gemma4_pa_cache_layers(const std::shared_ptr<ov::Model>& model,
                                                  const std::array<size_t, 2>& widths) {
    std::map<size_t, size_t> last_layer_by_width;
    constexpr char cache_prefix[] = "key_cache.";
    for (const auto& node : model->get_ordered_ops()) {
        if (std::string(node->get_type_name()) != "PagedAttentionExtension") {
            continue;
        }
        const auto& info = node->get_rt_info();
        const size_t width = static_cast<size_t>(info.at("num_k_heads").as<int64_t>() *
                                                 info.at("k_head_size").as<int64_t>());
        const std::string name = node->input_value(3).get_any_name();
        OPENVINO_ASSERT(name.compare(0, sizeof(cache_prefix) - 1, cache_prefix) == 0,
                        "Gemma4 MTP target PA key cache has an unexpected name: ", name);
        size_t layer = 0;
        const char* first = name.data() + sizeof(cache_prefix) - 1;
        const auto parsed = std::from_chars(first, name.data() + name.size(), layer);
        OPENVINO_ASSERT(parsed.ec == std::errc{} && parsed.ptr == name.data() + name.size() &&
                            node->input_value(4).get_any_name() == "value_cache." + std::to_string(layer),
                        "Gemma4 MTP target PA cache layer names do not match: ", name);
        last_layer_by_width[width] = layer;
    }
    OPENVINO_ASSERT(widths[0] != widths[1] && last_layer_by_width.size() == 2 &&
                        last_layer_by_width.count(widths[0]) && last_layer_by_width.count(widths[1]),
                    "Gemma4 MTP target PA must have matching full and sliding cache widths.");
    return {last_layer_by_width.at(widths[0]), last_layer_by_width.at(widths[1])};
}

void validate_mtp_generation_config(const GenerationConfig& config) {
    OPENVINO_ASSERT(config.assistant_confidence_threshold == 0.f,
                    "MTP speculative decoding supports static candidate counts only; "
                    "assistant_confidence_threshold must be 0.f.");
    OPENVINO_ASSERT(!config.is_tree_search(),
                    "MTP speculative decoding does not support tree search.");
    OPENVINO_ASSERT(config.is_greedy_decoding(),
                    "MTP speculative decoding supports greedy decoding only.");
    OPENVINO_ASSERT(config.num_return_sequences == 1,
                    "MTP speculative decoding does not support parallel sampling; "
                    "num_return_sequences must be 1.");
    OPENVINO_ASSERT(config.num_assistant_tokens > 0,
                    "MTP speculative decoding requires num_assistant_tokens > 0.");
}

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
    m_vision_registry = std::make_shared<VisionRegistry>();

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
        draft_scheduler_config.num_linear_attention_blocks = 0;
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
    OPENVINO_ASSERT(main_scheduler_config.enable_prefix_caching == draft_scheduler_config.enable_prefix_caching,
                    "MTP main and draft pipelines must use the same enable_prefix_caching setting");

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
    OPENVINO_ASSERT(main_mtp_pipeline && draft_mtp_pipeline,
                    "Internal error: expected MTP pipelines to be ContinuousBatchingForMtpDecodingImpl.");
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
    std::optional<ov::Tensor> prompt_ids,
    std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs) {
    validate_mtp_generation_config(sampling_params);

    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};
    // Draft gets shifted embeds only; VLM extras belong to the main model.
    ov::Tensor draft_input_embeds = create_draft_input_embeds(input_ids);
    OPENVINO_ASSERT(!std::static_pointer_cast<ContinuousBatchingForMtpDecodingImpl>(m_main_pipeline)->is_prefix_caching_enabled() ||
                        (prompt_ids.has_value() && prompt_ids->get_element_type() == ov::element::i64 &&
                         prompt_ids->get_size() == input_ids.get_shape()[1]),
                    "MTP prefix caching requires built-in prompt preparation with one token ID per embedding");
    // Use insert_or_assign, not insert: a finished prior request may leave a stale (stopped) handle
    // under the same request_id. insert() would be a no-op there, dropping the new handle as a
    // temporary whose destructor stops the freshly added draft request before it can draft.
    try {
        m_draft_generations.insert_or_assign(request_id, m_draft_pipeline->add_request(request_id, draft_input_embeds, draft_sampling_params));
        auto main_handle = m_main_pipeline->add_request(request_id, input_ids, sampling_params, prompt_ids, lm_extra_inputs);
        align_request_pair_processed_prefix(request_id);
        return main_handle;
    } catch (...) {
        std::static_pointer_cast<ContinuousBatchingForMtpDecodingImpl>(m_main_pipeline)->discard_awaiting_request(request_id);
        std::static_pointer_cast<ContinuousBatchingForMtpDecodingImpl>(m_draft_pipeline)->discard_awaiting_request(request_id);
        m_draft_generations.erase(request_id);
        throw;
    }
}

void ContinuousBatchingPipeline::MtpDecodingImpl::align_request_pair_processed_prefix(uint64_t request_id) {
    auto find_request = [request_id](const std::vector<SequenceGroup::Ptr>& requests) {
        const auto found = std::find_if(requests.begin(), requests.end(), [request_id](const auto& group) {
            return group->get_request_id() == request_id;
        });
        OPENVINO_ASSERT(found != requests.end(), "MTP alignment requires both awaiting requests: ", request_id);
        return *found;
    };
    const auto main_group = find_request(m_main_pipeline->get_awaiting_requests());
    const auto draft_group = find_request(m_draft_pipeline->get_awaiting_requests());
    draft_group->get_sequences().front()->set_prefix_cache_policy(draft_group->get_prompt_len(),
        [main_group](size_t length, size_t block_size) {
            return main_group->get_sequences().front()->get_hash(length + 1, block_size);
        });
    auto main_pipeline = std::static_pointer_cast<ContinuousBatchingForMtpDecodingImpl>(m_main_pipeline);
    auto draft_pipeline = std::static_pointer_cast<ContinuousBatchingForMtpDecodingImpl>(m_draft_pipeline);
    size_t ceiling = draft_group->get_prompt_len() - 1;
    while (true) {
        const size_t main_processed = main_pipeline->restore_awaiting_prefix(request_id, ceiling);
        const size_t draft_processed = draft_pipeline->restore_awaiting_prefix(request_id, main_processed);
        if (main_processed == draft_processed) {
            if (main_processed > 0) {
                GENAI_DEBUG("MTP paired prefix replay: request=%llu main=%zu draft=%zu",
                            static_cast<unsigned long long>(request_id), main_processed, draft_processed);
            }
            return;
        }
        OPENVINO_ASSERT(draft_processed < ceiling, "MTP prefix alignment must select an earlier checkpoint");
        ceiling = draft_processed;
    }
}

GenerationHandle ContinuousBatchingPipeline::MtpDecodingImpl::add_request(
    uint64_t request_id,
    const std::string& prompt,
    const ov::genai::GenerationConfig& sampling_params) {
    validate_mtp_generation_config(sampling_params);

    // Text-only serving path.
    ov::genai::VLMPerfMetrics metrics;
    ov::Tensor inputs_embeds;
    ov::Tensor prompt_ids;
    std::unordered_map<std::string, ov::Tensor> lm_extra_inputs;
    {
        std::lock_guard<std::mutex> lock(m_embeddings_mutex);
        m_inputs_embedder->set_apply_chat_template_status(sampling_params.apply_chat_template);
        const size_t previous_tokens = m_inputs_embedder->get_cache_state().get_state().size();
        const std::vector<ov::genai::EncodedImage> no_images;
        const auto [unified_prompt, image_sequence, video_sequence] =
            m_inputs_embedder->normalize_prompt(prompt, 0, no_images);
        inputs_embeds = m_inputs_embedder->get_inputs_embeds(unified_prompt, no_images, metrics, true, image_sequence);
        const auto& cached_ids = m_inputs_embedder->get_cache_state().get_state();
        OPENVINO_ASSERT(cached_ids.size() >= previous_tokens, "MTP prompt preparation cannot shrink token history");
        prompt_ids = ov::Tensor(ov::element::i64, {1, cached_ids.size() - previous_tokens});
        std::copy(cached_ids.begin() + previous_tokens, cached_ids.end(), prompt_ids.data<int64_t>());
        const auto [position_ids, rope_delta] = m_inputs_embedder->get_position_ids(inputs_embeds.get_shape()[1], 0);
        m_inputs_embedder->set_position_ids(position_ids);
        if (rope_delta.has_value()) {
            m_inputs_embedder->set_rope_delta(*rope_delta);
        }
        lm_extra_inputs = m_inputs_embedder->get_lm_extra_inputs();
        return add_request(request_id, inputs_embeds, sampling_params, prompt_ids, std::move(lm_extra_inputs));
    }
}

std::vector<EncodedGenerationResult> ContinuousBatchingPipeline::MtpDecodingImpl::generate(
    const std::vector<ov::Tensor>& input_ids,
    const std::vector<GenerationConfig>& sampling_params,
    const StreamerVariant& streamer,
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
        (void)main_cfg;
        (void)draft_cfg;
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

    // generate_common threads position_ids / lm_extra_inputs to the main pipeline
    // (priming the shared embedder's M-RoPE positions) via self->add_request(); the MTP draft ignores
    // them (sequential positions, no VLM inputs) inside MtpDecodingImpl::add_request.
    return generate_common(this, input_ids, sampling_params, streamer, position_ids,
                           prompt_ids, lm_extra_inputs_list, strategy);
}
}  // namespace ov::genai

namespace ov::genai {

ContinuousBatchingPipeline::Gemma4MtpDecodingImpl::Gemma4MtpDecodingImpl(
    const ModelDesc& main_desc, const ModelDesc& draft_desc,
    const std::shared_ptr<InputsEmbedder>& embedder)
    : m_default_num_assistant_tokens(draft_desc.generation_config.num_assistant_tokens.value_or(6)),
      m_max_num_batched_tokens(main_desc.scheduler_config.max_num_batched_tokens) {
    OPENVINO_ASSERT(main_desc.model && draft_desc.model && embedder,
                    "Gemma4 MTP requires a decomposed VLM and an assistant model.");
    OPENVINO_ASSERT(!main_desc.scheduler_config.enable_prefix_caching &&
                    !main_desc.scheduler_config.use_cache_eviction,
                    "Gemma4 MTP shared KV requires prefix caching and cache eviction to be disabled.");
    OPENVINO_ASSERT(m_default_num_assistant_tokens > 0, "Gemma4 MTP needs at least one assistant token.");
    m_tokenizer = main_desc.tokenizer;
    m_generation_config = main_desc.generation_config;
    if (!m_generation_config.num_assistant_tokens) {
        m_generation_config.num_assistant_tokens = m_default_num_assistant_tokens;
    }
    m_inputs_embedder = embedder;
    m_model_input_type = ModelInputType::EMBEDDINGS;
    m_vision_registry = std::make_shared<VisionRegistry>();
    m_embedding = embedder->get_embedding_model();
    OPENVINO_ASSERT(m_embedding, "Gemma4 MTP requires a text embedding model.");

    utils::mtp::expose_last_hidden_state(main_desc.model);
    ov::pass::SDPAToPagedAttention(false, false, true, false).run_on_model(main_desc.model);
    std::array<size_t, 4> shared_kv_widths;
    for (size_t i = 0; i < gemma4_shared_kv_names.size(); ++i) {
        shared_kv_widths[i] = draft_desc.model->input(gemma4_shared_kv_names[i])
                                  .get_partial_shape()[3].get_length();
    }
    m_main_cache_layers = find_gemma4_pa_cache_layers(main_desc.model,
                                                      {shared_kv_widths[0], shared_kv_widths[2]});
    utils::apply_gather_before_matmul_transformation(main_desc.model);

    m_main_pipeline = std::make_shared<ContinuousBatchingForMtpDecodingImpl>(
        main_desc.model, embedder, main_desc.tokenizer, main_desc.generation_config,
        main_desc.scheduler_config, main_desc.device, main_desc.properties, true);
    m_main_pipeline->set_hidden_state_export_needed(true);

    const std::string draft_device = draft_desc.device.empty() ? main_desc.device : draft_desc.device;
    ov::AnyMap draft_properties = draft_desc.properties.empty() ? main_desc.properties : draft_desc.properties;
    draft_properties.erase("sampler_num_threads");
    draft_properties = utils::get_model_properties(draft_properties, "language_model", draft_device);
    const auto main = std::static_pointer_cast<ContinuousBatchingForMtpDecodingImpl>(m_main_pipeline);
    const ov::CompiledModel compiled_main = main->get_compiled_model();
    const auto main_devices = compiled_main.get_property(ov::execution_devices);
    if (main_devices.size() == 1 && main_devices.front().find("CPU") != std::string::npos) {
        const std::string main_key = "key_cache." + std::to_string(m_main_cache_layers[0]);
        const std::string main_value = "value_cache." + std::to_string(m_main_cache_layers[0]);
        const ov::element::Type key_type = compiled_main.input(main_key).get_element_type();
        const ov::element::Type value_type = compiled_main.input(main_value).get_element_type();
        for (const auto& [name, expected] : std::array<std::pair<std::string, ov::element::Type>, 3>{
                 {{ov::hint::kv_cache_precision.name(), key_type},
                  {ov::key_cache_precision.name(), key_type},
                  {ov::value_cache_precision.name(), value_type}}}) {
            const auto property = draft_properties.find(name);
            OPENVINO_ASSERT(property == draft_properties.end() ||
                                property->second.as<ov::element::Type>() == expected,
                            "Gemma4 MTP draft property ", name,
                            " must match the target PA cache precision ", expected, ".");
        }
        draft_properties[ov::key_cache_precision.name()] = key_type;
        draft_properties[ov::value_cache_precision.name()] = value_type;
    }
    OPENVINO_ASSERT(shared_kv_widths[0] != shared_kv_widths[2],
                    "Gemma4 MTP full and sliding shared KV must have distinct widths.");
    ov::pass::SDPAToPagedAttention(false, false, true, false, false, false, false, true)
        .run_on_model(draft_desc.model);
    for (const auto& node : draft_desc.model->get_ordered_ops()) {
        OPENVINO_ASSERT(!ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node),
                        "Gemma4 MTP draft still contains SDPA after the PA transformation.");
    }
    const auto draft_layers = find_gemma4_pa_cache_layers(draft_desc.model,
                                                           {shared_kv_widths[0], shared_kv_widths[2]});
    for (size_t i = 0; i < m_draft_cache_names.size(); ++i) {
        m_draft_cache_names[i] = (i % 2 == 0 ? "key_cache." : "value_cache.") +
                                  std::to_string(draft_layers[i / 2]);
    }
    ov::CompiledModel compiled_draft =
        utils::singleton_core().compile_model(draft_desc.model, draft_device, draft_properties);
    OPENVINO_ASSERT(main_devices == compiled_draft.get_property(ov::execution_devices),
                    "Gemma4 MTP direct PA cache sharing requires target and draft on the same execution devices.");
    for (size_t i = 0; i < m_draft_cache_names.size(); ++i) {
        const std::string main_name = (i % 2 == 0 ? "key_cache." : "value_cache.") +
                                      std::to_string(m_main_cache_layers[i / 2]);
        const auto main_port = compiled_main.input(main_name);
        const auto draft_port = compiled_draft.input(m_draft_cache_names[i]);
        for (size_t dim = 1; dim < 4; ++dim) {
            OPENVINO_ASSERT(main_port.get_partial_shape()[dim] == draft_port.get_partial_shape()[dim],
                            "Gemma4 MTP target and draft PA cache layouts differ for ", main_name,
                            ": target ", main_port.get_partial_shape(), ", draft ",
                            draft_port.get_partial_shape(), ".");
        }
        OPENVINO_ASSERT(main_port.get_element_type() == draft_port.get_element_type(),
                        "Gemma4 MTP target and draft PA cache precisions differ for ", main_name, ".");
    }
    m_draft_request = compiled_draft.create_infer_request();
    m_perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_perf_metrics.main_model_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_perf_metrics.draft_model_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
}

GenerationHandle ContinuousBatchingPipeline::Gemma4MtpDecodingImpl::add_request(
    uint64_t request_id, const ov::Tensor& embeds, const GenerationConfig& config,
    std::optional<ov::Tensor> prompt_ids,
    std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs) {
    GenerationConfig effective_config = config;
    if (!effective_config.num_assistant_tokens) {
        effective_config.num_assistant_tokens = m_default_num_assistant_tokens;
    }
    validate_mtp_generation_config(effective_config);
    OPENVINO_ASSERT(!config.num_assistant_tokens || *config.num_assistant_tokens > 0,
                    "Gemma4 MTP num_assistant_tokens must be positive.");
    OPENVINO_ASSERT(*effective_config.num_assistant_tokens < m_max_num_batched_tokens,
                    "Gemma4 MTP requires max_num_batched_tokens to fit the entire verification window "
                    "(num_assistant_tokens + 1).");
    OPENVINO_ASSERT(!config.adapters, "Gemma4 MTP does not support adapters.");
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    const auto shape = embeds.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && shape[1] > 0,
                    "Gemma4 MTP requires one non-empty prompt per request.");
    auto handle = m_main_pipeline->add_request(request_id, embeds, effective_config, prompt_ids, lm_extra_inputs);
    m_requests.insert_or_assign(request_id, RequestState{{}, shape[1], 0});
    m_request_configs.insert_or_assign(request_id, effective_config);
    return handle;
}

GenerationHandle ContinuousBatchingPipeline::Gemma4MtpDecodingImpl::add_request(
    uint64_t request_id, const std::string& prompt, const GenerationConfig& config) {
    ov::genai::VLMPerfMetrics metrics;
    std::lock_guard<std::mutex> lock(m_embeddings_mutex);
    m_inputs_embedder->set_apply_chat_template_status(config.apply_chat_template);
    const size_t previous_tokens = m_inputs_embedder->get_cache_state().get_state().size();
    const std::vector<ov::genai::EncodedImage> images;
    const auto [normalized, image_sequence, video_sequence] =
        m_inputs_embedder->normalize_prompt(prompt, 0, images);
    ov::Tensor embeds = m_inputs_embedder->get_inputs_embeds(normalized, images, metrics, true, image_sequence);
    const auto& cached_ids = m_inputs_embedder->get_cache_state().get_state();
    OPENVINO_ASSERT(cached_ids.size() >= previous_tokens, "Gemma4 MTP token history shrank unexpectedly.");
    ov::Tensor prompt_ids(ov::element::i64, {1, cached_ids.size() - previous_tokens});
    std::copy(cached_ids.begin() + previous_tokens, cached_ids.end(), prompt_ids.data<int64_t>());
    auto [positions, rope_delta] = m_inputs_embedder->get_position_ids(embeds.get_shape()[1], 0);
    m_inputs_embedder->set_position_ids(positions);
    if (rope_delta) {
        m_inputs_embedder->set_rope_delta(*rope_delta);
    }
    return add_request(request_id, embeds, config, prompt_ids, m_inputs_embedder->get_lm_extra_inputs());
}

void ContinuousBatchingPipeline::Gemma4MtpDecodingImpl::append_main_outputs(const GeneratedRequests& generated) {
    for (const auto& [request_id, sequences] : generated) {
        OPENVINO_ASSERT(sequences.size() == 1, "Gemma4 MTP supports a single sequence per request.");
        const auto& sequence = sequences.begin()->second;
        auto& state = m_requests.at(request_id);
        if (!sequence.token_ids.empty()) {
            const size_t accepted_length = state.prompt_length + sequence.token_ids.size();
            const ov::Tensor& hidden = sequence.hidden_states;
            OPENVINO_ASSERT(hidden && hidden.get_shape().size() == 3,
                            "Gemma4 MTP target did not provide hidden states.");
            const size_t matches = state.generated_length == 0
                ? hidden.get_shape()[0] - 1
                : sequence.token_ids.size() - 1 - state.generated_length;
            OPENVINO_ASSERT(matches < hidden.get_shape()[0],
                            "Gemma4 MTP target did not provide the accepted token's hidden state: matches=",
                            matches, ", hidden length=", hidden.get_shape()[0],
                            ", generated length=", sequence.token_ids.size(),
                            ", previous generated length=", state.generated_length, ".");
            auto [start, end] = utils::make_roi(hidden.get_shape(), 0, matches, matches + 1);
            state.hidden_state = ov::Tensor(hidden.get_element_type(), {1, 1, hidden.get_shape()[2]});
            ov::Tensor(hidden, start, end).copy_to(state.hidden_state);
            state.generated_length = sequence.token_ids.size();
        }
    }
}

std::vector<int64_t> ContinuousBatchingPipeline::Gemma4MtpDecodingImpl::draft_tokens(
    uint64_t request_id, const GeneratedSequence& sequence, const GenerationConfig& config) {
    auto& state = m_requests.at(request_id);
    if (!state.hidden_state || sequence.token_ids.empty()) {
        return {};
    }
    const size_t length = state.prompt_length + sequence.token_ids.size();
    const size_t remaining = config.get_max_new_tokens() - sequence.token_ids.size();
    const size_t limit = std::min(config.num_assistant_tokens.value_or(m_default_num_assistant_tokens),
                                  remaining > 0 ? remaining - 1 : 0);
    if (limit == 0) {
        return {};
    }
    const size_t shared_length = length - 1;
    OPENVINO_ASSERT(length + limit <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                    "Gemma4 MTP draft requires the accepted prefix KV and an i32-compatible context length.");
    auto main = std::static_pointer_cast<ContinuousBatchingForMtpDecodingImpl>(m_main_pipeline);
    const auto shared_cache = main->get_mtp_pa_cache(request_id, m_main_cache_layers, shared_length);
    const size_t num_blocks = shared_cache.block_indices.size();
    ov::Tensor past_lens(ov::element::i32, {1});
    past_lens.data<int32_t>()[0] = static_cast<int32_t>(shared_length);
    ov::Tensor subsequence_begins(ov::element::i32, {2});
    subsequence_begins.data<int32_t>()[0] = 0;
    subsequence_begins.data<int32_t>()[1] = 1;
    ov::Tensor block_indices_begins(ov::element::i32, {2});
    block_indices_begins.data<int32_t>()[0] = 0;
    block_indices_begins.data<int32_t>()[1] = static_cast<int32_t>(num_blocks);
    ov::Tensor block_indices(ov::element::i32, {num_blocks});
    std::copy(shared_cache.block_indices.begin(), shared_cache.block_indices.end(),
              block_indices.data<int32_t>());
    ov::Tensor max_context_len(ov::element::i32, {});
    max_context_len.data<int32_t>()[0] = static_cast<int32_t>(length + limit);
    m_draft_request.set_tensor("past_lens", past_lens);
    m_draft_request.set_tensor("subsequence_begins", subsequence_begins);
    m_draft_request.set_tensor("block_indices_begins", block_indices_begins);
    m_draft_request.set_tensor("block_indices", block_indices);
    m_draft_request.set_tensor("max_context_len", max_context_len);
    for (size_t k = 0; k < m_draft_cache_names.size(); ++k) {
        m_draft_request.set_tensor(m_draft_cache_names[k], shared_cache.tensors[k]);
    }
    ov::Tensor position(ov::element::i64, {1});
    position.data<int64_t>()[0] = static_cast<int64_t>(length - 1);
    ov::Tensor hidden = state.hidden_state;
    int64_t token = sequence.token_ids.back();
    std::vector<int64_t> result;
    result.reserve(limit);
    for (size_t i = 0; i < limit; ++i) {
        position.data<int64_t>()[0] = static_cast<int64_t>(length - 1 + i);
        ov::Tensor ids(ov::element::i64, {1, 1});
        ids.data<int64_t>()[0] = token;
        CircularBufferQueueElementGuard<EmbeddingsRequest> guard(m_embedding->get_request_queue().get());
        const ov::Tensor embedding = m_embedding->infer(guard.get(), ids);
        const auto& embed_shape = embedding.get_shape();
        const auto& hidden_shape = hidden.get_shape();
        OPENVINO_ASSERT(embedding.get_element_type() == ov::element::f32 && hidden.get_element_type() == ov::element::f32 &&
                        embed_shape == hidden_shape, "Gemma4 MTP embedding and hidden state must be equal-size f32 tensors.");
        ov::Tensor input(ov::element::f32, {1, 2 * embed_shape[2]});
        std::copy_n(embedding.data<const float>(), embed_shape[2], input.data<float>());
        std::copy_n(hidden.data<const float>(), hidden_shape[2], input.data<float>() + embed_shape[2]);
        m_draft_request.set_tensor("inputs_embeds", input);
        m_draft_request.set_tensor("position_ids", position);
        const auto start = std::chrono::steady_clock::now();
        m_draft_request.infer();
        const auto duration = PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start);
        auto& draft_metrics = m_perf_metrics.draft_model_metrics.raw_metrics;
        draft_metrics.m_inference_durations[0] += MicroSeconds(duration);
        draft_metrics.m_durations.emplace_back(duration);
        draft_metrics.m_batch_sizes.push_back(1);
        const ov::Tensor logits = m_draft_request.get_tensor("logits");
        OPENVINO_ASSERT(logits.get_shape().size() == 3 && logits.get_shape()[0] == 1 &&
                        logits.get_shape()[1] == 1, "Gemma4 MTP draft must emit one position of logits.");
        const float* scores = logits.data<const float>();
        token = std::max_element(scores, scores + logits.get_shape().back()) - scores;
        result.push_back(token);
        if ((!config.ignore_eos && token == config.eos_token_id) ||
            std::find(config.stop_token_ids.begin(), config.stop_token_ids.end(), token) != config.stop_token_ids.end()) {
            break;
        }
        hidden = m_draft_request.get_tensor("last_hidden_state");
    }
    return result;
}

void ContinuousBatchingPipeline::Gemma4MtpDecodingImpl::step() {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    try {
        m_main_pipeline->pull_awaiting_requests();
        const auto before = m_main_pipeline->get_generated_requests();
        std::map<uint64_t, size_t> draft_counts;
        for (const auto& [request_id, sequences] : before) {
            if (sequences.empty() || sequences.begin()->second.token_ids.empty()) {
                continue;
            }
            const auto& sequence = sequences.begin()->second;
            const auto& state = m_requests.at(request_id);
            if (!state.hidden_state) {
                continue;
            }
            // The preceding verifier pass may have processed rejected candidates.
            // Only the accepted prefix participates in the next assistant pass.
            auto candidates = sequence;
            const auto& config = m_request_configs.at(request_id);
            auto draft = draft_tokens(request_id, sequence, config);
            candidates.token_ids.insert(candidates.token_ids.end(), draft.begin(), draft.end());
            candidates.log_probs.insert(candidates.log_probs.end(), draft.size(), 0.f);
            const auto updated = m_main_pipeline->update_request(request_id, {{sequences.begin()->first, candidates}}, false);
            draft_counts[request_id] = updated.inserted_tokens_cnt;
            m_perf_metrics.num_draft_tokens += updated.inserted_tokens_cnt;
            m_sd_metrics.update_draft_generated_len(request_id, updated.inserted_tokens_cnt);
        }
        m_main_pipeline->sync_generated_embeddings();
        const auto main_start = std::chrono::steady_clock::now();
        m_main_pipeline->step();
        const auto main_end = std::chrono::steady_clock::now();
        m_pipeline_metrics = m_main_pipeline->get_metrics();
        const auto main_duration = PerfMetrics::get_microsec(main_end - main_start);
        const size_t processed = m_main_pipeline->get_processed_tokens_per_iteration();
        if (processed) {
            auto& raw = m_perf_metrics.raw_metrics;
            raw.m_token_infer_durations.emplace_back(main_duration);
            raw.m_inference_durations[0] += MicroSeconds(main_duration);
            raw.m_new_token_times.emplace_back(main_end);
            raw.m_batch_sizes.push_back(processed);
            auto& main_raw = m_perf_metrics.main_model_metrics.raw_metrics;
            main_raw.m_durations.emplace_back(main_duration);
            main_raw.m_inference_durations[0] += MicroSeconds(m_pipeline_metrics.inference_duration);
            main_raw.m_batch_sizes.push_back(processed);
        }
        const auto after = m_main_pipeline->get_generated_requests();
        append_main_outputs(after);
        for (const auto& [request_id, candidates] : after) {
            const auto before_it = before.find(request_id);
            if (before_it == before.end() || before_it->second.empty() || candidates.empty()) {
                continue;
            }
            const size_t old_length = before_it->second.begin()->second.token_ids.size();
            const size_t new_length = candidates.begin()->second.token_ids.size();
            if (new_length > old_length) {
                const size_t accepted = std::min(draft_counts[request_id], new_length - old_length - 1);
                m_perf_metrics.num_accepted_tokens += accepted;
                if (draft_counts[request_id]) {
                    m_sd_metrics.update_acceptance_rate(
                        request_id, 100.f * accepted / draft_counts[request_id]);
                    m_sd_metrics.update_draft_accepted_tokens(request_id, accepted);
                }
                m_sd_metrics.update_generated_len(new_length - old_length);
            }
        }
        for (auto it = m_requests.begin(); it != m_requests.end();) {
            if (!after.count(it->first)) {
                m_request_configs.erase(it->first);
                it = m_requests.erase(it);
            } else {
                ++it;
            }
        }
    } catch (...) {
        m_main_pipeline->fail_pipeline(std::current_exception());
        m_requests.clear();
        m_request_configs.clear();
        throw;
    }
}

void ContinuousBatchingPipeline::Gemma4MtpDecodingImpl::drop_requests() {
    m_main_pipeline->finish_request();
    m_requests.clear();
    m_request_configs.clear();
}

bool ContinuousBatchingPipeline::Gemma4MtpDecodingImpl::is_requests_empty() {
    return m_main_pipeline->is_requests_empty();
}

std::vector<SequenceGroup::Ptr> ContinuousBatchingPipeline::Gemma4MtpDecodingImpl::get_awaiting_requests() {
    return m_main_pipeline->get_awaiting_requests();
}

std::vector<EncodedGenerationResult> ContinuousBatchingPipeline::Gemma4MtpDecodingImpl::generate(
    const std::vector<ov::Tensor>& embeds, const std::vector<GenerationConfig>& configs,
    const StreamerVariant& streamer,
    const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& positions,
    const std::optional<std::vector<ov::Tensor>>& prompt_ids,
    const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& extra) {
    GenerateStrategy strategy;
    strategy.prepare_request = [](size_t, const ov::Tensor& input, GenerationConfig&,
                                  GenerationConfig&, ov::Tensor& main, ov::Tensor& draft) {
        main = input;
        draft = input;
    };
    strategy.check_streaming = [](const std::shared_ptr<ThreadedStreamerWrapper>& streamer_ptr,
                                  const std::vector<ov::Tensor>& inputs,
                                  const std::vector<GenerationConfig>& configs) {
        OPENVINO_ASSERT(!streamer_ptr->has_callback() || (inputs.size() == 1 && configs[0].is_greedy_decoding()),
                        "Gemma4 MTP streaming requires a single greedy request.");
    };
    strategy.start_timer = [] { return std::chrono::steady_clock::now(); };
    strategy.stop_timer = [](const TimePoint& start) {
        return PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start);
    };
    return generate_common(this, embeds, configs, streamer, positions, prompt_ids, extra, strategy);
}

}  // namespace ov::genai
