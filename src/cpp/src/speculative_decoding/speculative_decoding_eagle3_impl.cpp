// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "speculative_decoding_eagle3_impl.hpp"
#include "logger.hpp"

namespace ov::genai {
void share_embedding_weights(std::shared_ptr<ov::Model>& main_model, std::shared_ptr<ov::Model>& draft_model) {
    // extract embedding weight from main model
    auto find_embedding_gather = [](const std::shared_ptr<ov::Model>& model)
        -> std::shared_ptr<ov::Node> {
        constexpr size_t MIN_VOCAB_SIZE_THRESHOLD = 1000;
        for (const auto& node : model->get_ordered_ops()) {
            auto gather = std::dynamic_pointer_cast<ov::op::util::GatherBase>(node);
            if (!gather) continue;
            // [vocab, hidden_size] * [batch, seq_len] -> [batch, seq_len, hidden_size]
            auto data_node = gather->input_value(0).get_node_shared_ptr();
            auto indices_node = gather->input_value(1).get_node_shared_ptr();
            if (!data_node || !indices_node) continue;
            // indices_node should be on parameter path, maybe this is better rule
            ov::PartialShape ps = data_node->get_output_partial_shape(0);
            if (ps.rank().is_static() && ps.rank().get_length() >= 2) {
                if (ps[0].is_static() && ps[0].get_length() > MIN_VOCAB_SIZE_THRESHOLD) { // Heuristic: vocab size > 1000
                    return gather;
                }
            }
            std::string fname = data_node->get_friendly_name();
            if (fname.find("embed_tokens") != std::string::npos ||
                fname.find("embedding") != std::string::npos) {
                return gather;
            }
        }
        return nullptr;
    };
    auto main_gather  = find_embedding_gather(main_model);
    auto draft_gather = find_embedding_gather(draft_model);
    if (!main_gather || !draft_gather) {
        return;
    }
    auto main_weight_node = main_gather->input_value(0).get_node_shared_ptr();
    auto draft_weight_node = draft_gather->input_value(0).get_node_shared_ptr();

    if (main_weight_node.get() == draft_weight_node.get()) {
        return;
    }

    try {
        draft_weight_node->output(0).replace(main_weight_node->output(0));
    } catch (const std::exception& e) {
        GENAI_WARN(std::string("Error: failed to import embedding weights from main model to draft model. Exception: ") + e.what());
    } catch (...) {
        GENAI_WARN("Error: failed to import embedding weights from main model to draft model due to unknown exception.");
    }
}

void shift_fc_from_draft_to_main(std::shared_ptr<ov::Model>& main_model, std::shared_ptr<ov::Model>& draft_model) {
    // extract the FC transform weight from draft model
    auto remove_fc_and_rewire = [](const std::shared_ptr<ov::Model>& model) -> std::shared_ptr<ov::Node> {
        for (const auto& node : model->get_ordered_ops()) {
            auto matmul_node = ov::as_type_ptr<ov::op::v0::MatMul>(node);
            if (!matmul_node) continue;
            auto input_node = matmul_node->get_input_node_shared_ptr(0);
            auto param_node = ov::as_type_ptr<ov::op::v0::Parameter>(input_node);
            if (!param_node) continue;
            if (input_node->get_friendly_name().find("hidden_states") == std::string::npos) continue;
            // Rewire all outputs of this MatMul to use the input_node directly
            for (auto& output : matmul_node->outputs()) {
                for (auto& target : output.get_target_inputs()) {
                    target.replace_source_output(input_node);
                }
            }
            return matmul_node->input_value(1).get_node_shared_ptr();
        }
        return nullptr;
    };
    auto fc_weights = remove_fc_and_rewire(draft_model);
    if (!fc_weights) {
        GENAI_WARN("Error: failed to retrieve and remove FC matmul from draft model.");
        return;
    }
    // now we create the fc into main model
    for (const auto& result : main_model->get_results()) {
        auto input_node = result->input_value(0).get_node_shared_ptr();
        if (input_node && input_node->get_friendly_name().find("eagle3_hidden_states_concat") != std::string::npos) {
            auto matmul = std::make_shared<ov::op::v0::MatMul>(input_node, fc_weights, false, true);
            matmul->set_friendly_name("eagle3_hidden_state_fc");
            result->input(0).replace_source_output(matmul);
            break;
        }
    }
}

std::shared_ptr<ov::op::v0::Constant> extract_d2t_mapping_table(std::shared_ptr<ov::Model>& model) {
    // extract result nodes from model
    for (const auto& result : model->get_results()) {
        auto input_node = result->input_value(0).get_node_shared_ptr();
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(input_node);
        if (constant && constant->get_friendly_name().find("d2t") != std::string::npos) {
            return constant;
        }
    }
    return nullptr;
}

void hidden_state_transform(std::shared_ptr<ov::Model>& model,
                                  const std::vector<int>& hidden_layers_to_abstract) {
    if (hidden_layers_to_abstract.empty()) {
        return;
    }
    OPENVINO_ASSERT(hidden_layers_to_abstract.size() == 3 || hidden_layers_to_abstract.size() == 1, "invalid hidden layer numbers specified for abstraction.");

    std::vector<std::string> patterns;
    if (hidden_layers_to_abstract.size() > 1) {
        patterns.reserve(hidden_layers_to_abstract.size());
        for (int idx : hidden_layers_to_abstract) {
            patterns.emplace_back("layers." + std::to_string(idx) + "/"); // main description
        }
    } else {
        patterns.emplace_back("midlayer"); // draft description
    }

    // Helper: check if node is a residual Add node with expected structure
    auto is_residual_node = [](const std::shared_ptr<ov::Node>& node) -> bool {
        if (auto add = ov::as_type_ptr<ov::op::v1::Add>(node)) {
            auto input1 = add->get_input_node_shared_ptr(1);
            auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(input1);
            if (!matmul) return false;
            auto matmul_input = matmul->get_input_node_shared_ptr(0);
            return matmul_input && ov::is_type<ov::op::v1::Multiply>(matmul_input);
        }
        return false;
    };

    std::vector<ov::Output<ov::Node>> residual_outputs;
    for (const auto& node : model->get_ordered_ops()) {
        if (!is_residual_node(node)) continue;
        const std::string& name = node->get_friendly_name();
        for (const auto& pattern : patterns) {
            if (name.find(pattern) != std::string::npos) {
                residual_outputs.push_back(node->output(0));
                break;
            }
        }
    }

    if (!residual_outputs.empty()) {
        OPENVINO_ASSERT(residual_outputs.size() == patterns.size(),
                        "Number of extracted hidden states does not match the requested number.");
        std::shared_ptr<ov::Node> node_to_operate;
        if (residual_outputs.size() > 1) {
            auto concat = std::make_shared<ov::op::v0::Concat>(residual_outputs, -1);
            concat->set_friendly_name("eagle3_hidden_states_concat");
            node_to_operate = concat;
        } else {
            node_to_operate = residual_outputs[0].get_node_shared_ptr();
        }
        auto result = std::make_shared<ov::op::v0::Result>(node_to_operate);
        const std::string output_name = "last_hidden_state";
        result->output(0).set_names({output_name});
        result->set_friendly_name(output_name);
        model->add_results({result});
    }
}

ContinuousBatchingPipeline::Eagle3DecodingImpl::Eagle3DecodingImpl(const ov::genai::ModelDesc& main_model_desc,
                                                                 const ov::genai::ModelDesc& draft_model_desc,
                                                                 const std::vector<int>& hidden_layers)
                                                                 : m_hidden_layers_to_abstract(hidden_layers) {
    auto scheduler_configs = init_speculative_models(main_model_desc, draft_model_desc);
    // Eagle speculative decoding does not support dynamic_split_fuse mode
    // because it requires hidden state interaction from main model to draft model
    // to be implemented future
    if (scheduler_configs.first.dynamic_split_fuse) {
        GENAI_WARN(
            "Note: disable dynamic split fuse for eagle3 speculative decoding"
        );
        scheduler_configs.first.dynamic_split_fuse = false;
        scheduler_configs.second.dynamic_split_fuse = false;
    }
    auto main_model = main_model_desc.model;
    auto draft_model = draft_model_desc.model;

    auto main_device = main_model_desc.device;
    std::string draft_device = draft_model_desc.device.empty() ? main_model_desc.device : draft_model_desc.device;

    ov::AnyMap draft_properties =
        draft_model_desc.properties.empty() ? main_model_desc.properties : draft_model_desc.properties;

    // main and draft model can have different tokenizers
    // to do: support retokenization: 154103
    Tokenizer main_model_tokenizer = main_model_desc.tokenizer;
    Tokenizer draft_model_tokenizer = draft_model_desc.tokenizer;
    m_tokenizer = main_model_tokenizer;
    // for eagle model, we need to obtain hidden layer state as extra output
    // apply transformations needed to run eagle model
    share_embedding_weights(main_model, draft_model);
    hidden_state_transform(main_model, hidden_layers);
    // move the FC layer from draft model to main model
    shift_fc_from_draft_to_main(main_model, draft_model);
    hidden_state_transform(draft_model, { -1 });

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
    auto d2t_tensor = extract_d2t_mapping_table(draft_model);
    update_eagle_pipeline_params(d2t_tensor);
}

ov::Tensor ContinuousBatchingPipeline::Eagle3DecodingImpl::create_draft_input_ids(const ov::Tensor& original_input_ids) {
    auto shape = original_input_ids.get_shape();
    if (shape.size() == 0u || shape.back() <= 1u) {
        return ov::Tensor(original_input_ids);
    }

    size_t original_length = shape.back();
    size_t new_length = original_length - 1;

    ov::Tensor draft_input_ids(ov::element::i64, {1, new_length});

    const int64_t* src_data = original_input_ids.data<int64_t>();
    int64_t* dst_data = draft_input_ids.data<int64_t>();

    std::copy(src_data + 1, src_data + original_length, dst_data);

    return draft_input_ids;
}

void ContinuousBatchingPipeline::Eagle3DecodingImpl::update_eagle_pipeline_params(std::shared_ptr<ov::op::v0::Constant>& d2t_tensor) {
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
    strategy.stop_timer = [](TimePoint start){
        return PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start);
    };

    return generate_common(this, input_ids, sampling_params, streamer, token_type_ids, strategy);
}
}  // namespace ov::genai