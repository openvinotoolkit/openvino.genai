// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "eagle3_strategy.hpp"
#include "speculative_decoding/eagle3_model_transforms.hpp"
#include "logger.hpp"

namespace ov::genai {
KVUpdateWrapper::KVUpdateWrapper(const ov::genai::ModelDesc& kv_model_desc) {
    m_compiled_model =
            utils::singleton_core().compile_model(kv_model_desc.model, kv_model_desc.device, kv_model_desc.properties);
    m_request = m_compiled_model.create_infer_request();
}

ContinuousBatchingPipeline::Eagle3DecodingImpl::Eagle3DecodingImpl(const ov::genai::ModelDesc& main_model_desc,
                                                                 const ov::genai::ModelDesc& draft_model_desc,
                                                                 const std::vector<int32_t>& hidden_layers) {
    if (main_model_desc.inputs_embedder) {
        m_inputs_embedder = main_model_desc.inputs_embedder;
        m_model_input_type = ModelInputType::EMBEDDINGS;
        m_vision_registry = std::make_shared<VisionRegistry>();
    }

    // Enable query-to-query bias for Eagle3 main model only
    ov::genai::ModelDesc main_model_desc_with_qq_bias = main_model_desc;
    main_model_desc_with_qq_bias.properties["query_to_query_bias"] = true;
    auto scheduler_configs = init_speculative_models(main_model_desc_with_qq_bias, draft_model_desc);
    auto main_model = main_model_desc.model;
    auto draft_model = draft_model_desc.model;
    OPENVINO_ASSERT(main_model && draft_model);

    auto main_device = main_model_desc.device;
    std::string draft_device = draft_model_desc.device.empty() ? main_model_desc.device : draft_model_desc.device;

    ov::AnyMap draft_properties =
        draft_model_desc.properties.empty() ? main_model_desc.properties : draft_model_desc.properties;

    // main and draft model use same tokenizer, but could differ in configurations
    // for example, llama3 draft model has different eos_token_id in config.json
    const Tokenizer& main_model_tokenizer = main_model_desc.tokenizer;
    const Tokenizer& draft_model_tokenizer = draft_model_desc.tokenizer;
    m_tokenizer = main_model_tokenizer;
    // for eagle model, we need to obtain hidden layer state as extra output
    // apply transformations needed to run eagle model
    utils::eagle3::share_vocabulary(main_model, draft_model);
    utils::eagle3::transform_hidden_state(main_model, hidden_layers);
    // move the FC layer from draft model to main model
    utils::eagle3::move_fc_from_draft_to_main(draft_model, main_model);
    utils::eagle3::transform_hidden_state(draft_model, {-1});

    // to create kv cache update pipeline for main model post-validation
    auto kv_model = utils::eagle3::modeling_eagle3_kv_update_model(main_model);
    // to create `main_pipeline` with enabled validation_mode and `draft_pipeline` with disabled validation mode
    if (m_model_input_type == ModelInputType::EMBEDDINGS) {
        m_main_pipeline = std::make_shared<ContinuousBatchingForEagle3DecodingImpl>(main_model,
                                                                                    m_inputs_embedder,
                                                                                    main_model_tokenizer,
                                                                                    main_model_desc_with_qq_bias.generation_config,
                                                                                    scheduler_configs.first,
                                                                                    main_device,
                                                                                    main_model_desc_with_qq_bias.properties,
                                                                                    true);
        m_draft_pipeline = std::make_shared<ContinuousBatchingForEagle3DecodingImpl>(draft_model,
                                                                                     m_inputs_embedder,
                                                                                     draft_model_tokenizer,
                                                                                     draft_model_desc.generation_config,
                                                                                     scheduler_configs.second,
                                                                                     draft_device,
                                                                                     draft_properties,
                                                                                     false);
    } else {
        m_main_pipeline = std::make_shared<ContinuousBatchingForEagle3DecodingImpl>(main_model,
                                                                                    main_model_tokenizer,
                                                                                    main_model_desc_with_qq_bias.generation_config,
                                                                                    scheduler_configs.first,
                                                                                    main_device,
                                                                                    main_model_desc_with_qq_bias.properties,
                                                                                    true);
        m_draft_pipeline = std::make_shared<ContinuousBatchingForEagle3DecodingImpl>(draft_model,
                                                                                     draft_model_tokenizer,
                                                                                     draft_model_desc.generation_config,
                                                                                     scheduler_configs.second,
                                                                                     draft_device,
                                                                                     draft_properties,
                                                                                     false);
    }

    ov::genai::ModelDesc kv_model_desc;
    kv_model_desc.model = kv_model;
    kv_model_desc.device = main_device;
    // only kv cache information is needed for kv update model
    if (main_model_desc.properties.count(ov::hint::kv_cache_precision.name()) > 0) {
        kv_model_desc.properties[ov::hint::kv_cache_precision.name()] =
            main_model_desc.properties.at(ov::hint::kv_cache_precision.name());
    } else {
        GENAI_INFO("kv cache precision not specified in main model properties. leave to plugin for default precision.");
    }
    m_kv_update_wrapper = std::make_shared<KVUpdateWrapper>(kv_model_desc);

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_draft_pipeline->raw_perf_metrics.m_inference_durations = {{ MicroSeconds(0.0f) }};

    // specific params update for eagle pipeline
    // check draft_model, retrieve d2t table if exists
    auto d2t_tensor = utils::eagle3::extract_d2t_mapping_table(draft_model);
    update_eagle_pipeline_params(d2t_tensor);
}

ov::Tensor ContinuousBatchingPipeline::Eagle3DecodingImpl::create_draft_input(const ov::Tensor& original_input) {
    if (m_model_input_type == ModelInputType::TOKENS) {
        return create_draft_input_ids(original_input);
    } else {
        return create_draft_input_embeddings(original_input);
    }
}

ov::Tensor ContinuousBatchingPipeline::Eagle3DecodingImpl::create_draft_input_ids(const ov::Tensor& original_input_ids) {
    auto shape = original_input_ids.get_shape();
    OPENVINO_ASSERT(shape.size() != 0u && shape.back() > 1u, "input_ids shape is invalid for creating eagle3 draft input_ids");

    size_t original_length = shape.back();
    size_t new_length = original_length - 1;

    ov::Tensor draft_input_ids(ov::element::i64, {1, new_length});
    // Shift the input ids by one token.
    // E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3]
    const int64_t* src_data = original_input_ids.data<int64_t>();
    int64_t* dst_data = draft_input_ids.data<int64_t>();

    std::copy(src_data + 1, src_data + original_length, dst_data);

    return draft_input_ids;
}

ov::Tensor ContinuousBatchingPipeline::Eagle3DecodingImpl::create_draft_input_embeddings(const ov::Tensor& original_input_embeddings) {
    auto shape = original_input_embeddings.get_shape();
    auto element_type = original_input_embeddings.get_element_type();

    OPENVINO_ASSERT(shape.size() == 3u, "Input embedding tensor shape size should be 3.");
    OPENVINO_ASSERT(shape[0] == 1u, "Input embedding tensor only support batch == 1.");

    size_t original_length = shape[1];

    OPENVINO_ASSERT(original_length > 1u,
                    "Input embedding tensor sequence length must be greater than 1 for creating eagle3 draft input embeddings.");
    size_t new_length = original_length - 1;

    ov::Tensor draft_input_embeddings(element_type, {1, new_length, shape[2]});

    // Copy data using byte-wise copy to handle any element type
    size_t embedding_row_bytes = shape[2] * element_type.size() ;
    const uint8_t* src_data = static_cast<const uint8_t*>(original_input_embeddings.data());
    uint8_t* dst_data = static_cast<uint8_t*>(draft_input_embeddings.data());

    std::copy(src_data + embedding_row_bytes, src_data + (original_length * embedding_row_bytes), dst_data);

    return draft_input_embeddings;
}

void ContinuousBatchingPipeline::Eagle3DecodingImpl::update_eagle_pipeline_params(const std::shared_ptr<ov::op::v0::Constant>& d2t_tensor) {
    auto m_main_eagle_pipeline  = std::dynamic_pointer_cast<ContinuousBatchingForEagle3DecodingImpl>(m_main_pipeline);
    auto m_draft_eagle_pipeline = std::dynamic_pointer_cast<ContinuousBatchingForEagle3DecodingImpl>(m_draft_pipeline);
    m_main_eagle_pipeline->set_hidden_state_export_needed(true);
    m_draft_eagle_pipeline->set_hidden_state_export_needed(true);
    m_draft_eagle_pipeline->set_hidden_state_import_needed(true);
    m_draft_eagle_pipeline->set_hidden_state_internal_needed(true);
    m_draft_eagle_pipeline->set_d2t_for_draft_decoding(d2t_tensor);
}

GenerationHandle
ContinuousBatchingPipeline::Eagle3DecodingImpl::add_request(uint64_t request_id,
                                                                 const ov::Tensor& input_ids,
                                                                 const ov::genai::GenerationConfig& sampling_params,
                                                                 std::optional<ov::Tensor> token_type_ids,
                                                                 std::optional<ov::Tensor> prompt_ids,
                                                                 std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};
    // remove first token from input_ids to create draft_input_ids
    ov::Tensor draft_input_ids = create_draft_input(input_ids);
    // Draft Eagle3 inference only uses the language model path. Multimodal auxiliary inputs such as
    // deepstack visual tensors are consumed only by the main model, so lm_extra_inputs are not forwarded here.
    m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, draft_input_ids, draft_sampling_params, token_type_ids, prompt_ids)});
    return m_main_pipeline->add_request(request_id, input_ids, sampling_params, token_type_ids, prompt_ids, lm_extra_inputs);
}

GenerationHandle
ContinuousBatchingPipeline::Eagle3DecodingImpl::add_request(uint64_t request_id,
                                                                 const std::string& prompt,
                                                                 const ov::genai::GenerationConfig& sampling_params) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};
    // remove first token from input_ids to create draft_input_ids
    // add_special_tokens is false for better compression rate
    auto input_ids = m_tokenizer.encode(prompt, ov::genai::add_special_tokens(false)).input_ids;
    ov::Tensor draft_input_ids = create_draft_input(input_ids);
    m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, draft_input_ids, draft_sampling_params)});
    return m_main_pipeline->add_request(request_id, input_ids, sampling_params);
}

std::vector<EncodedGenerationResult> ContinuousBatchingPipeline::Eagle3DecodingImpl::generate(
    const std::vector<ov::Tensor>& input_ids,
    const std::vector<GenerationConfig>& sampling_params,
    const StreamerVariant& streamer,
    const std::optional<std::vector<ov::Tensor>>& token_type_ids,
    const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids,
    const std::optional<std::vector<ov::Tensor>>& prompt_ids,
    const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list
) {
    GenerateStrategy strategy;
    strategy.prepare_request = [this](size_t,
                                      const ov::Tensor& in_ids,
                                      GenerationConfig& main_cfg,
                                      GenerationConfig& draft_cfg,
                                      ov::Tensor& main_in,
                                      ov::Tensor& draft_in) {
        OPENVINO_ASSERT(main_cfg.assistant_confidence_threshold == 0.f,
                        "Eagle3 only supports num_assistant_tokens (assistant_confidence_threshold must be 0.f)");
        if (main_cfg.num_assistant_tokens == 0) {
            main_cfg.num_assistant_tokens = m_main_pipeline->default_num_assistant_tokens;
            draft_cfg.num_assistant_tokens = main_cfg.num_assistant_tokens;
        }
        draft_cfg.ignore_eos = true;
        draft_cfg.stop_strings = {};
        main_in = in_ids;
        draft_in = create_draft_input(in_ids);
    };

    strategy.check_streaming = [](const std::shared_ptr<ThreadedStreamerWrapper>& streamer_ptr,
                                  const std::vector<ov::Tensor>& input_ids,
                                  const std::vector<GenerationConfig>& sampling_params) {
        OPENVINO_ASSERT(!streamer_ptr->has_callback() ||
                        (input_ids.size() == 1 &&
                         (sampling_params[0].is_greedy_decoding() || sampling_params[0].is_tree_search())),
                        "Eagle3 streaming only supports batch size=1 with greedy or tree search");
    };
    strategy.start_timer = [](){
        return std::chrono::steady_clock::now();
    };
    strategy.stop_timer = [](const TimePoint& start){
        return PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start);
    };

    return generate_common(this, input_ids, sampling_params, streamer, token_type_ids, position_ids, prompt_ids, lm_extra_inputs_list, strategy);
}

void ContinuousBatchingPipeline::Eagle3DecodingImpl::step() {
    // general step for speculative decoding
    ContinuousBatchingPipeline::SpeculativeDecodingImpl::step();
    auto main_pipeline = std::dynamic_pointer_cast<ContinuousBatchingForEagle3DecodingImpl>(m_main_pipeline);
    // specific step for eagle3 to update main model kv cache after validation
    {
        // Launch KV update asynchronously
        m_sync_future = std::async(std::launch::async, [wrapper = m_kv_update_wrapper, main_pipeline]() mutable {
            auto main_generated_requests = main_pipeline->get_generated_requests();
            std::vector<int32_t> block_update_indices, block_update_begins;
            main_pipeline->collect_block_update_info(main_generated_requests,
                                                     block_update_indices,
                                                     block_update_begins);

            ov::Tensor block_indices_tensor = main_pipeline->get_tensor_by_name("block_indices");
            ov::Tensor block_indices_begins_tensor = main_pipeline->get_tensor_by_name("block_indices_begins");
            ov::Tensor block_update_indices_tensor(ov::element::i32,
                                                   {block_update_indices.size()},
                                                   block_update_indices.data());
            ov::Tensor block_update_indices_begins_tensor(ov::element::i32,
                                                          {block_update_begins.size()},
                                                          block_update_begins.data());

            // Collect KV caches directly from main model's infer request
            // The infer request already has all KV cache tensors set by cache_manager
            std::vector<ov::Tensor> key_caches, value_caches;
            for (const auto& input : wrapper->get_compiled_model().inputs()) {
                auto input_name = input.get_any_name();
                if (input_name.find("key_cache.") == 0) {
                    // Extract layer_id from "key_cache.N" format (note: uses dot, not underscore)
                    size_t layer_id = std::stoul(input_name.substr(std::string("key_cache.").size()));
                    if (layer_id >= key_caches.size()) {
                        key_caches.resize(layer_id + 1);
                        value_caches.resize(layer_id + 1);
                    }
                    // Get tensors directly from main model's infer request instead of going through scheduler
                    key_caches[layer_id] = main_pipeline->get_tensor_by_name(input_name);
                    value_caches[layer_id] = main_pipeline->get_tensor_by_name("value_cache." + std::to_string(layer_id));
                }
            }

            wrapper->infer(block_indices_tensor,
                          block_indices_begins_tensor,
                          block_update_indices_tensor,
                          block_update_indices_begins_tensor,
                          key_caches,
                          value_caches);
        });
    }
}
}  // namespace ov::genai
