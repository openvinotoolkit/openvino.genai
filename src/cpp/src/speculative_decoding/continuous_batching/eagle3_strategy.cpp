// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eagle3_strategy.hpp"
#include "speculative_decoding/eagle3_model_transforms.hpp"
#include "logger.hpp"

namespace ov::genai {
ContinuousBatchingPipeline::Eagle3DecodingImpl::Eagle3DecodingImpl(const ov::genai::ModelDesc& main_model_desc,
                                                                 const ov::genai::ModelDesc& draft_model_desc,
                                                                 const std::vector<int32_t>& hidden_layers) {
    auto scheduler_configs = init_speculative_models(main_model_desc, draft_model_desc);
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

    // to create `main_pipeline` with enabled validation_mode and `draft_pipeline` with disabled validation mode
    m_main_pipeline = std::make_shared<ContinuousBatchingForEagle3DecodingImpl>(main_model,
                                                                               main_model_tokenizer,
                                                                               main_model_desc.generation_config,
                                                                               scheduler_configs.first,
                                                                               main_device,
                                                                               main_model_desc.properties,
                                                                               true);
    m_draft_pipeline = std::make_shared<ContinuousBatchingForEagle3DecodingImpl>(draft_model,
                                                                                draft_model_tokenizer,
                                                                                draft_model_desc.generation_config,
                                                                                scheduler_configs.second,
                                                                                draft_device,
                                                                                draft_properties,
                                                                                false);
    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_draft_pipeline->raw_perf_metrics.m_inference_durations = {{ MicroSeconds(0.0f) }};

    // specific params update for eagle pipeline
    // check draft_model, retrieve d2t table if exists
    auto d2t_tensor = utils::eagle3::extract_d2t_mapping_table(draft_model);
    update_eagle_pipeline_params(d2t_tensor);
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
                                                                 std::optional<ov::Tensor> token_type_ids) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};
    // remove first token from input_ids to create draft_input_ids
    ov::Tensor draft_input_ids = create_draft_input_ids(input_ids);
    m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, draft_input_ids, draft_sampling_params, token_type_ids)});
    return m_main_pipeline->add_request(request_id, input_ids, sampling_params, token_type_ids);
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
    ov::Tensor draft_input_ids = create_draft_input_ids(input_ids);
    m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, draft_input_ids, draft_sampling_params)});
    return m_main_pipeline->add_request(request_id, input_ids, sampling_params);
}

std::vector<EncodedGenerationResult> ContinuousBatchingPipeline::Eagle3DecodingImpl::generate(
    const std::vector<ov::Tensor>& input_ids,
    const std::vector<GenerationConfig>& sampling_params,
    const StreamerVariant& streamer,
    const std::optional<std::vector<ov::Tensor>>& token_type_ids,
    const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids) {
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
        draft_in = create_draft_input_ids(in_ids);
    };

    strategy.check_streaming = [](const std::shared_ptr<ThreadedStreamerWrapper>& streamer_ptr,
                                  const std::vector<ov::Tensor>& input_ids,
                                  const std::vector<GenerationConfig>& sampling_params) {
        OPENVINO_ASSERT(!streamer_ptr->has_callback() ||
                        (input_ids.size() == 1 &&
                         (sampling_params[0].is_greedy_decoding())),
                        "Eagle3 streaming only supports batch size=1 with greedy");
    };
    strategy.start_timer = [](){
        return std::chrono::steady_clock::now();
    };
    strategy.stop_timer = [](const TimePoint& start){
        return PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start);
    };

    return generate_common(this, input_ids, sampling_params, streamer, token_type_ids, strategy);
}
}  // namespace ov::genai
