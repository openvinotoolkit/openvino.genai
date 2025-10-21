// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "speculative_decoding_eagle3_impl.hpp"

namespace ov::genai {
void share_embedding_weights(std::shared_ptr<ov::Model>& main_model, std::shared_ptr<ov::Model>& draft_model) {
    // extract embedding weight from main model
    auto find_embedding_gather = [](const std::shared_ptr<ov::Model>& model)
        -> std::shared_ptr<ov::Node> {
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
                if (ps[0].is_static() && ps[0].get_length() > 1000) { // Heuristic: vocab size > 1000
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
        std::cerr << "Error: failed to import embedding weights from main model to draft model. Exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Error: failed to import embedding weights from main model to draft model due to unknown exception." << std::endl;
    }
}

void extract_hidden_state_generic(std::shared_ptr<ov::Model>& model,
                                  const std::vector<int>& hidden_layers_to_abstract) {
    ov::pass::Manager pm;
    pm.register_pass<EagleModelTransform>(hidden_layers_to_abstract);
    pm.run_passes(model);
}

EagleModelTransform::EagleModelTransform(const std::vector<int>& layers) : m_layer_ids(layers) {
}

bool EagleModelTransform::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // share the embedding weights from main model to draft model
    m_new_parameters.clear();
    m_new_results.clear();
    if (m_layer_ids.size() == 1 && m_layer_ids[0] == -1) {
        ov::pass::Manager manager;
        manager.set_per_pass_validation(false);
        manager.register_pass<EagleBaseTransform>(m_new_results);
        // input transform for draft
        // here we apply a trick for the fc layer in draft model
        manager.register_pass<EagleInputTransform>(m_new_parameters);
        manager.run_passes(model);

        model->add_parameters(m_new_parameters);
        model->add_results(m_new_results);
        return true;
    } else {
        ov::pass::Manager manager;
        manager.set_per_pass_validation(false);
        manager.register_pass<Eagle3Transform>(m_layer_ids, m_hidden_layer_outputs);
        manager.run_passes(model);
        
        if (!m_hidden_layer_outputs.empty()) {
            auto concat = std::make_shared<v0::Concat>(m_hidden_layer_outputs, -1);
            concat->set_friendly_name("eagle3_hidden_states_concat");
            
            auto result = std::make_shared<v0::Result>(concat);
            std::string output_name = "last_hidden_state";
            result->output(0).set_names({output_name});
            result->set_friendly_name(output_name);
            model->add_results({result});
            return true;
        }
    }
    
    return false;
}

EagleInputTransform::EagleInputTransform(std::vector<std::shared_ptr<v0::Parameter>>& params) {
    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::wrap_type<v0::MatMul>(), this->get_type_info().name),
        ([&params, this](ov::pass::pattern::Matcher& m) {
            auto node = m.get_match_root();
            try {
                if (apply(node, params)) {
                    ++applied;
                    return true;
                }
            } catch (...) {
                OPENVINO_ASSERT(false, "EagleTransform failed to apply");
            }
            return false;
        })
    );
}
bool EagleInputTransform::apply(NodePtr node, std::vector<std::shared_ptr<v0::Parameter>>& params) {
    if (ov::is_type<v0::MatMul>(node)) {
        auto matmul_node = ov::as_type_ptr<v0::MatMul>(node);
        // check the input of matmul node, if it is a node with name "hidden_states", then it's the node we want
        auto input_node = matmul_node->get_input_node_shared_ptr(0);
        if (!ov::as_type_ptr<v0::Parameter>(input_node)) {
            return false;
        }

        auto shape = node->get_output_partial_shape(0);
        auto internal_hidden_state = std::make_shared<v0::Parameter>(node->get_element_type(), node->get_output_partial_shape(0));
        internal_hidden_state->output(0).set_names({"internal_hidden_states"});
        internal_hidden_state->set_friendly_name("internal_hidden_states");
        // create new eltwise node to add output of MatMul node and internal hidden state input from last cycle of itself
        auto new_eltwise = std::make_shared<v1::Add>(internal_hidden_state, matmul_node->output(0));
        ov::replace_node(matmul_node, new_eltwise);
        params.push_back(internal_hidden_state);
        return true;
    }
}

EagleBaseTransform::EagleBaseTransform(std::vector<std::shared_ptr<v0::Result>>& results) {
    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::wrap_type<v0::Result>(), this->get_type_info().name),
        ([&results, this](ov::pass::pattern::Matcher& m) {
            auto node = m.get_match_root();
            try {
                if (apply(node, results)) {
                    ++applied;
                    return true;
                }
            } catch (...) {
                OPENVINO_ASSERT(false, "EagleTransform failed to apply");
            }
            return false;
        })
    );
}

std::shared_ptr<ov::Node> EagleBaseTransform::find_last_residual_node(const std::shared_ptr<ov::Node>& start_node, 
                                                               std::set<ov::Node*>& visited_nodes) {
    if (visited_nodes.count(start_node.get())) {
        return nullptr;
    }

    visited_nodes.insert(start_node.get());

    if (ov::is_type<v1::Add>(start_node)) {
        // check the input nodes of MatMul, if found Gather node, return the gather node, otherwise ,retrun the matmul node
        for (size_t i = 0; i < start_node->get_input_size(); ++i) {
            auto input_node = start_node->get_input_node_shared_ptr(i);
            if (!input_node) continue;
            if (ov::as_type_ptr<v0::MatMul>(input_node)) {
                return start_node; // return the Add node itself
            }
        }
    }
    
    for (size_t i = 0; i < start_node->get_input_size(); ++i) {
        auto input_node = start_node->get_input_node_shared_ptr(i);
        if (!input_node) continue;
        
        auto result = find_last_residual_node(input_node, visited_nodes);
        if (result) {
            return result;
        }
    }
    return nullptr;
}

std::shared_ptr<ov::Node> EagleBaseTransform::find_last_residual_node(const std::shared_ptr<ov::Node>& start_node) {
    std::set<ov::Node*> visited_nodes;
    return find_last_residual_node(start_node, visited_nodes);
}

bool EagleBaseTransform::apply(NodePtr node, std::vector<std::shared_ptr<v0::Result>>& results) {
    {
        // 1. without normalization layer 2. add extra input
        if (ov::is_type<v0::Result>(node)) {
            // we are applying transformation to the last hidden state, eagle2 mode
            NodePtr input_node = node->get_input_node_shared_ptr(0);
            if (!input_node) {
                return false;
            }
            auto last_residual_node = find_last_residual_node(input_node);
            if (!last_residual_node) {
                return false;
            }
            // 
            auto result = std::make_shared<v0::Result>(last_residual_node);
            std::string output_name = "last_hidden_state";
            result->output(0).set_names({output_name});
            result->set_friendly_name(output_name);
            results.push_back(result);
            return true;
        }
        return false;
    }
}

Eagle3Transform::Eagle3Transform(const std::vector<int>& layers, std::vector<Output<Node>>& hidden_state_outputs) : m_layers(layers) {
    auto is_target_pattern = [&](const Output<Node>& output) {
        auto add_node = ov::as_type_ptr<v1::Add>(output.get_node_shared_ptr());
        auto add_node_name = add_node->get_friendly_name();
        if (add_node_name.find("self_attn") != std::string::npos)
            return false; // Skip self-attention layers
        bool layer_matched = false;
        for (auto layer_idx : m_layers) {
            if (add_node_name.find("layers." + std::to_string(layer_idx) + "/") != std::string::npos) {
                layer_matched = true;
                break;
            }
        }

        if (!layer_matched) {
            return false; // Skip layers that are not in the specified layers
        }
        auto input0 = add_node->get_input_node_shared_ptr(1);
        if (!input0 || !ov::is_type<v0::MatMul>(input0)) {
            return false;
        }
        auto matmul_node = input0;
        auto matmul_input = matmul_node->get_input_node_shared_ptr(0);
        if (!matmul_input) {
            return false;
        }

        bool has_multiply = ov::is_type<v1::Multiply>(matmul_input); // ACT(up) dot gate
        return has_multiply;    
    };

    auto hidden_layer = ov::pass::pattern::wrap_type<v1::Add>(is_target_pattern);
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(hidden_layer, "Eagle3Transform::hidden_extraction"),
        [&hidden_state_outputs, this](ov::pass::pattern::Matcher& m) {
            auto node = m.get_match_root();
            if (ov::is_type<v1::Add>(node)) {
                hidden_state_outputs.push_back(node->output(0));
                return true;
            }
            return false;
        }
    );
}

ContinuousBatchingPipeline::Eagle3DecodingImpl::Eagle3DecodingImpl(const ov::genai::ModelDesc& main_model_desc,
                                                                 const ov::genai::ModelDesc& draft_model_desc,
                                                                 const std::vector<int>& hidden_layers)
                                                                 : m_hidden_layers_to_abstract(hidden_layers) {
    auto scheduler_configs = init_speculative_models(main_model_desc, draft_model_desc);
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
    // target model: hidden state extraction, draft model: hidden state import , hidden state extraction
    // eagle3 specific : dt importing
    share_embedding_weights(main_model, draft_model);
    extract_hidden_state_generic(main_model, hidden_layers);
    extract_hidden_state_generic(draft_model, { -1 });

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
}

ov::Tensor ContinuousBatchingPipeline::Eagle3DecodingImpl::create_draft_input_ids(const ov::Tensor& original_input_ids) {
    auto shape = original_input_ids.get_shape();
    if (shape.size() == 0 || shape.back() <= 1) {
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

void ContinuousBatchingPipeline::Eagle3DecodingImpl::update_eagle_pipeline_params() {
    auto m_main_eagle_pipeline = std::dynamic_pointer_cast<ContinuousBatchingForEagle3DecodingImpl>(m_main_pipeline);
    auto m_draft_eagle_pipeline = std::dynamic_pointer_cast<ContinuousBatchingForEagle3DecodingImpl>(m_draft_pipeline);
    m_main_eagle_pipeline->set_hidden_state_export_needed(true);
    m_draft_eagle_pipeline->set_hidden_state_export_needed(true);
    m_draft_eagle_pipeline->set_hidden_state_import_needed(true);
    m_draft_eagle_pipeline->set_hidden_state_internal_needed(true);
    m_draft_eagle_pipeline->set_adjust_factor(m_hidden_layers_to_abstract.size() > 0 ? m_hidden_layers_to_abstract.size() : 1);
}

GenerationHandle
ContinuousBatchingPipeline::Eagle3DecodingImpl::add_request(uint64_t request_id,
                                                                 const ov::Tensor& input_ids,
                                                                 ov::genai::GenerationConfig sampling_params,
                                                                 std::optional<ov::Tensor> token_type_ids) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};
    update_eagle_pipeline_params();
    // remove first token from input_ids to create draft_input_ids
    ov::Tensor draft_input_ids = create_draft_input_ids(input_ids);
    m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, draft_input_ids, draft_sampling_params, token_type_ids)});
    return m_main_pipeline->add_request(request_id, input_ids, sampling_params, token_type_ids);
}

GenerationHandle
ContinuousBatchingPipeline::Eagle3DecodingImpl::add_request(uint64_t request_id,
                                                                 const std::string& prompt,
                                                                 ov::genai::GenerationConfig sampling_params) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};
    update_eagle_pipeline_params();
    // remove first token from input_ids to create draft_input_ids
    if (m_model_input_type == ModelInputType::TOKENS) {
        static ManualTimer timer("tokenize");
        timer.start();
        ChatHistory history({{{"role", "user"}, {"content", prompt}}});
        auto templated_prompt = m_tokenizer.apply_chat_template(history, true);
        auto input_ids = m_tokenizer.encode(templated_prompt, ov::genai::add_special_tokens(false)).input_ids;
        timer.end();
        ov::Tensor draft_input_ids = create_draft_input_ids(input_ids);
        m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, draft_input_ids, draft_sampling_params)});
        return m_main_pipeline->add_request(request_id, input_ids, sampling_params);
    } else {
        m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, prompt, draft_sampling_params)});
        return m_main_pipeline->add_request(request_id, prompt, sampling_params);
    }
    
}

std::vector<EncodedGenerationResult> ContinuousBatchingPipeline::Eagle3DecodingImpl::generate(
    const std::vector<ov::Tensor>& input_ids,
    const std::vector<GenerationConfig>& sampling_params,
    const StreamerVariant& streamer,
    std::optional<std::vector<ov::Tensor>> token_type_ids) {
    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_draft_pipeline->raw_perf_metrics.m_inference_durations =  {{ MicroSeconds(0.0f) }};

    OPENVINO_ASSERT(!has_non_finished_requests(),
                    "Generate cannot be called while ContinuousBatchingPipeline is already in running state. Use "
                    "ContinuousBatchingPipeline::add_request");

    OPENVINO_ASSERT(input_ids.size() == sampling_params.size());

    ManualTimer generate_timer("speculative_decoding: generate()");
    generate_timer.start();

    // checks that all requests has the same LoRA adapters property value
    for (size_t i = 1; i < sampling_params.size(); ++i) {
        OPENVINO_ASSERT(sampling_params[i - 1].adapters == sampling_params[i].adapters,
                        "LoRA adapters value must be the same for all requests");
    }
    auto m_main_eagle_pipeline = std::dynamic_pointer_cast<ContinuousBatchingForEagle3DecodingImpl>(m_main_pipeline);
    auto m_draft_eagle_pipeline = std::dynamic_pointer_cast<ContinuousBatchingForEagle3DecodingImpl>(m_draft_pipeline);

    m_main_eagle_pipeline->set_adapters(sampling_params[0].adapters);
    m_draft_eagle_pipeline->set_adapters(sampling_params[0].adapters);
    update_eagle_pipeline_params();

    const auto streamer_ptr = std::make_shared<ThreadedStreamerWrapper>(streamer, m_tokenizer);

    OPENVINO_ASSERT(
        !streamer_ptr->has_callback() || input_ids.size() == 1 && (sampling_params[0].is_greedy_decoding() ||
                                                                   (sampling_params[0].is_multinomial() &&
                                                                    sampling_params[0].num_return_sequences == 1)),
        "Currently eagle streaming is possible only with batch size=1 and only for greedy or multinomial decoding");

    std::vector<GenerationHandle> main_generations;
    ov::Tensor new_input_ids;
    for (size_t request_id = 0; request_id < input_ids.size(); ++request_id) {
        auto new_input_ids = input_ids[request_id];
        OPENVINO_ASSERT(1 == input_ids[request_id].get_shape().at(0), "Use multiple tensors to pass a batch.");
        auto main_sampling_params = sampling_params[request_id];
        OPENVINO_ASSERT(
            main_sampling_params.assistant_confidence_threshold == 0.f,
            "Eagle3 Speculative Decoding pipeline only supports `num_assistant_tokens` "
            "as parameter in GenerationConfig and doesn't work with `assistant_confidence_threshold`.\nPlease "
            "remove its specification or set it to 0.f.");
        if (main_sampling_params.num_assistant_tokens == 0) {
            main_sampling_params.num_assistant_tokens = m_main_pipeline->default_num_assistant_tokens;
        }
        main_generations.push_back(
            m_main_pipeline->add_request(request_id, new_input_ids, main_sampling_params));

        auto draft_sampling_params = main_sampling_params;
        // set the parameters do not stop draft generation without stopping of the same request for main pipeline
        draft_sampling_params.ignore_eos = true;
        draft_sampling_params.stop_strings = {};

        // remove first token from input_ids to create draft_input_ids
        ov::Tensor draft_input_ids = create_draft_input_ids(new_input_ids);

        std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
        m_draft_generations.insert(
            {request_id, m_draft_pipeline->add_request(request_id, draft_input_ids, draft_sampling_params)});
    }
    auto all_requests = get_awaiting_requests();

    GenerationHandle& generation = main_generations.at(0);

    streamer_ptr->start();

    while (has_non_finished_requests()) {
        try {
            step();
        } catch (...) {
            drop_requests();  // remove all requests from pipeline state in case of exception
            streamer_ptr->end();
            std::rethrow_exception(std::current_exception());
        }
        stream_tokens(streamer_ptr, generation);
    }

    // waiting for competion of streaming
    streamer_ptr->end();

    OPENVINO_ASSERT(is_requests_empty(),
                    "Internal error: current request is supposed to be dropped within step() function as completed");

    std::vector<EncodedGenerationResult> results;
    results.reserve(all_requests.size());

    m_perf_metrics.draft_model_metrics.raw_metrics = m_draft_pipeline->raw_perf_metrics;
    generate_timer.end();

    for (size_t request_id = 0; request_id < all_requests.size(); ++request_id) {
        const auto& request = all_requests[request_id];
        auto sampling_params = request->get_sampling_parameters();
        const auto& sequences = request->get_finished_sequences();
        size_t num_outputs = std::min(sampling_params.num_return_sequences, sequences.size());

        EncodedGenerationResult result;
        result.m_request_id = request_id;
        result.m_generation_ids.resize(num_outputs);
        result.m_scores.resize(num_outputs);
        result.m_status = request->get_generation_stream()->get_status();

        for (size_t i = 0; i < num_outputs; ++i) {
            const auto& sequence = sequences[i];
            const float score = sampling_params.is_beam_search() ? sequence->get_beam_search_score(sampling_params)
                                                                 : sequence->get_cumulative_log_prob();
            const auto& generated_ids = sequence->get_generated_ids();

            if (sampling_params.echo) {
                result.m_generation_ids[i] = request->get_prompt_ids();
            }
            std::copy(generated_ids.begin(), generated_ids.end(), std::back_inserter(result.m_generation_ids[i]));
            result.m_scores[i] = score;
        }

        result.m_status = main_generations[request_id]->get_status();

        // The same perf metrics for each sequence, only tokenization/detokenization will differ.
        m_perf_metrics.raw_metrics.generate_durations.clear();
        m_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_timer.get_duration_microsec());
        m_perf_metrics.num_input_tokens = request->get_prompt_len();
        m_perf_metrics.evaluate_statistics(generate_timer.get_start_time());

        result.perf_metrics = m_perf_metrics;
        result.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_perf_metrics);
        results.push_back(std::move(result));
    }

    OPENVINO_ASSERT(results.size() == input_ids.size());
    return results;
}
}  // namespace ov::genai