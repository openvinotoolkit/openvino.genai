// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <thread>

#include "openvino/genai/text_streamer.hpp"
#include "speculative_decoding_impl.hpp"
#include "continuous_batching/paged_attention_transformations.hpp"
#include "utils.hpp"


namespace ov::genai {
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

bool are_tokenizers_equal(Tokenizer& lhs, Tokenizer& rhs) {
    std::string test_string = "Could you please tell me something about OpenVINO.GenAI?";
    ov::Tensor encoded_string_lhs = lhs.encode(test_string).input_ids,
               encoded_string_rhs = rhs.encode(test_string).input_ids;
    
    ov::Shape shape_lhs = encoded_string_lhs.get_shape(),
              shape_rhs = encoded_string_rhs.get_shape();

    return shape_lhs == shape_rhs && lhs.get_eos_token_id() == rhs.get_eos_token_id() &&
           lhs.get_bos_token_id() == rhs.get_bos_token_id() && lhs.get_pad_token_id() == rhs.get_pad_token_id();
}

ContinuousBatchingPipeline::SpeculativeDecodingImpl::SpeculativeDecodingImpl(const ov::genai::ModelDesc& main_model_desc, 
                                                                             const ov::genai::ModelDesc& draft_model_desc) {
    auto main_model = main_model_desc.model;
    auto draft_model = draft_model_desc.model;

    auto main_scheduler_config = main_model_desc.scheduler_config;
    auto main_device = main_model_desc.device;

    utils::apply_paged_attention_transformations(main_model, main_model_desc.scheduler_config.use_cache_eviction);
    utils::apply_paged_attention_transformations(draft_model, main_model_desc.scheduler_config.use_cache_eviction);

    utils::apply_gather_before_matmul_transformation(main_model);
    utils::apply_gather_before_matmul_transformation(draft_model);

    std::string draft_device = draft_model_desc.device.empty() ? main_model_desc.device : draft_model_desc.device;
    bool is_draft_scheduler_undefined = draft_model_desc.scheduler_config == SchedulerConfig();

    ov::genai::SchedulerConfig main_scheduler_config_updated = main_scheduler_config,
                               draft_scheduler_config = is_draft_scheduler_undefined ? main_scheduler_config : draft_model_desc.scheduler_config;

    if (is_draft_scheduler_undefined) {
        // split KV cache to 2 caches for main and draft models
        auto compute_total_hidden_size = [] (const std::shared_ptr<ov::Model>& model) -> size_t {
            size_t total_hidden_size = 0;
            for (const auto& param_ptr : model->get_parameters()) {
                const auto& name = param_ptr->get_friendly_name();
                if (name.find("key_cache.") == 0) {
                    auto pa_op = param_ptr->get_output_target_inputs(0).begin()->get_node();
                    const auto& rt_info = pa_op->get_rt_info();
                    total_hidden_size += rt_info.at("num_k_heads").as<int>() * rt_info.at("k_head_size").as<int>() +
                                         rt_info.at("num_v_heads").as<int>() * rt_info.at("v_head_size").as<int>();
                }
            }
            return total_hidden_size;
        };
        float main_model_hidden_size = compute_total_hidden_size(main_model),
              draft_model_hidden_size = compute_total_hidden_size(draft_model);
        auto k = draft_model_hidden_size / (main_model_hidden_size + draft_model_hidden_size);

        // TODO: work with KV blocks as it will be more precise instead of GBs
        size_t main_cache_size = std::ceil(main_scheduler_config.cache_size * (1.f - k)),
               draft_cache_size = main_scheduler_config.cache_size - main_cache_size;
        if (draft_cache_size == 0 && main_cache_size > 0) {
            main_cache_size -= (main_cache_size > 1 ? 1 : 0);
            draft_cache_size = 1;
        }

        main_scheduler_config_updated.cache_size = main_cache_size;
        draft_scheduler_config.cache_size = draft_cache_size;
    } else {
        draft_scheduler_config.dynamic_split_fuse = main_scheduler_config_updated.dynamic_split_fuse;
        draft_scheduler_config.max_num_batched_tokens = main_scheduler_config_updated.max_num_batched_tokens;
    }

    ov::AnyMap draft_properties = draft_model_desc.properties.empty() ? main_model_desc.properties : draft_model_desc.properties;

    // main and draft model can have different tokenizers
    // to do: support retokenization: 154103
    Tokenizer main_model_tokenizer = main_model_desc.tokenizer;
    Tokenizer draft_model_tokenizer = draft_model_desc.tokenizer;

    // todo: remove this condition after support of CVS-154103
    OPENVINO_ASSERT(are_tokenizers_equal(main_model_tokenizer, draft_model_tokenizer), "Tokenizers for draft and main models are different!");
    
    m_tokenizer = main_model_tokenizer;

    // to create `main_pipeline` with enabled validation_mode and `draft_pipeline` with disabled validation mode
    m_main_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(
        main_model, main_model_tokenizer, main_model_desc.generation_config,
        main_scheduler_config_updated, main_device, main_model_desc.properties, true);
    m_draft_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(
        draft_model, draft_model_tokenizer, draft_model_desc.generation_config,
        draft_scheduler_config, draft_device, draft_properties, false);

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_draft_pipeline->raw_perf_metrics.m_inference_durations =  {{ MicroSeconds(0.0f) }};
}

GenerationHandle
ContinuousBatchingPipeline::SpeculativeDecodingImpl::add_request(uint64_t request_id,
                                                                 const ov::Tensor& input_ids,
                                                                 ov::genai::GenerationConfig sampling_params) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};
    m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, input_ids, draft_sampling_params)});
    return m_main_pipeline->add_request(request_id, input_ids, sampling_params);
};

GenerationHandle
ContinuousBatchingPipeline::SpeculativeDecodingImpl::add_request(uint64_t request_id,
                                                                 const std::string& prompt,
                                                                 ov::genai::GenerationConfig sampling_params) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};
    m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, prompt, draft_sampling_params)});
    return m_main_pipeline->add_request(request_id, prompt, sampling_params);
}

bool ContinuousBatchingPipeline::SpeculativeDecodingImpl::has_non_finished_requests() {
    return m_main_pipeline->has_non_finished_requests();
}

void print_generated_request(const ov::genai::GeneratedRequests& requests) {
    for (const auto& request : requests) {
        for (const auto& sequence : request.second) {
            std::cout << "request_id: " << request.first << " | sequence_id: " << sequence.first << " | ";
            for (const auto& token_id : sequence.second.token_ids) {
                std::cout << token_id << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void ContinuousBatchingPipeline::SpeculativeDecodingImpl::step() {
    // this blocks adding new requests during step as it may break coherence between main and draft models
    std::lock_guard<std::mutex> lock{m_draft_generations_mutex};

    auto& raw_perf_counters = m_perf_metrics.raw_metrics;
    auto& main_raw_perf_counters = m_perf_metrics.main_model_metrics.raw_metrics;

    ManualTimer step_timer("speculative_decoding: step()");
    step_timer.start();

    m_draft_pipeline->pull_awaiting_requests(true);
    m_main_pipeline->pull_awaiting_requests();

    // generate candidates by draft model
    ManualTimer draft_timer("speculative_decoding: draft_model: multistep()");
    draft_timer.start();
    m_draft_pipeline->multistep();
    draft_timer.end();
    m_sd_metrics.draft_duration += draft_timer.get_duration();
    m_pipeline_metrics = m_main_pipeline->get_metrics();

    // to generate num_matches statistic
    std::map<int64_t, UpdateRequestResult> update_sequence_info;
    // put candidates to model KV cache
    auto draft_generated_requests = m_draft_pipeline->get_generated_requests();
    for (const auto& candidate : m_draft_pipeline->get_generated_requests()) {
        auto update_result = m_main_pipeline->update_request(candidate.first, candidate.second, false);
        update_sequence_info.insert({{candidate.first, update_result}});
    }

    ManualTimer main_timer("speculative_decoding: main_model: step()");
    main_timer.start();
    m_main_pipeline->step();
    main_timer.end();
    m_sd_metrics.main_duration += main_timer.get_duration();
    m_pipeline_metrics = m_main_pipeline->get_metrics();

    auto main_generated_requests = m_main_pipeline->get_generated_requests();
    for (const auto& checked_sequence : main_generated_requests) {
        auto update_result = m_draft_pipeline->update_request(checked_sequence.first, checked_sequence.second, true);
        update_sequence_info[checked_sequence.first].removed_tokens_cnt = update_result.removed_tokens_cnt;
    }

    // finish draft request if the generation was completed
    for (const auto& draft_request : draft_generated_requests) {
        auto request_id = draft_request.first;
        if (!main_generated_requests.count(request_id)) {
            m_draft_pipeline->finish_request(request_id);
            // remove draft_generation_handle from queue
            m_draft_generations.erase(request_id);
        }
        auto updated_seq_info = update_sequence_info[request_id];
        m_sd_metrics.update_draft_generated_len(request_id, updated_seq_info.inserted_tokens_cnt);

        // several prompt phase
        if (updated_seq_info.inserted_tokens_cnt == 0 || main_generated_requests.empty()) {
            continue;
        }
        float acceptance_rate = 1 - static_cast<float>(updated_seq_info.removed_tokens_cnt) / updated_seq_info.inserted_tokens_cnt;
        m_sd_metrics.update_acceptance_rate(request_id, acceptance_rate * 100);
        m_sd_metrics.update_draft_accepted_tokens(request_id, (updated_seq_info.inserted_tokens_cnt - updated_seq_info.removed_tokens_cnt));
    }

    step_timer.end();

    // update perf metrics
    const auto num_generated_tokens = m_main_pipeline->get_processed_tokens_per_iteration();
    if (num_generated_tokens > 0) {
        auto infer_duration = step_timer.get_duration_microsec();
        raw_perf_counters.m_token_infer_durations.emplace_back(infer_duration);
        raw_perf_counters.m_inference_durations[0] += MicroSeconds(infer_duration);
        raw_perf_counters.m_new_token_times.emplace_back(main_timer.get_end_time());
        raw_perf_counters.m_batch_sizes.emplace_back(num_generated_tokens);

        auto main_model_gen_duration = main_timer.get_duration_microsec();
        auto m_main_pipeline_metrics = m_main_pipeline->get_metrics();
        main_raw_perf_counters.m_durations.push_back(MicroSeconds(main_model_gen_duration));
        main_raw_perf_counters.m_inference_durations[0] = MicroSeconds(m_main_pipeline_metrics.inference_duration);
        main_raw_perf_counters.m_batch_sizes.push_back(num_generated_tokens); // or should be processed + generated
        m_sd_metrics.update_generated_len(num_generated_tokens);
    }

    if (main_generated_requests.empty() && utils::env_setup_for_print_debug_info()) {
        m_sd_metrics.print(true);
        m_sd_metrics.clean_up();
    }
}



std::vector<EncodedGenerationResult>
ContinuousBatchingPipeline::SpeculativeDecodingImpl::generate(const std::vector<ov::Tensor>& input_ids,
                                                              const std::vector<GenerationConfig>& sampling_params,
                                                              const StreamerVariant& streamer) {
    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_draft_pipeline->raw_perf_metrics.m_inference_durations =  {{ MicroSeconds(0.0f) }};

    OPENVINO_ASSERT(!has_non_finished_requests(), "Generate cannot be called while ContinuousBatchingPipeline is already in running state. Use ContinuousBatchingPipeline::add_request");
    OPENVINO_ASSERT(input_ids.size() == sampling_params.size());

    ManualTimer generate_timer("speculative_decoding: generate()");
    generate_timer.start();

    // checks that all requests has the same LoRA adapters property value
    for (size_t i = 1; i < sampling_params.size(); ++i) {
        OPENVINO_ASSERT(sampling_params[i - 1].adapters == sampling_params[i].adapters,
            "LoRA adapters value must be the same for all requests");
    }
    m_main_pipeline->set_adapters(sampling_params[0].adapters);
    m_draft_pipeline->set_adapters(sampling_params[0].adapters);

    const auto streamer_ptr = std::make_shared<ThreadedStreamerWrapper>(streamer, m_tokenizer);

    OPENVINO_ASSERT(!streamer_ptr->has_callback() || input_ids.size() == 1 && (sampling_params[0].is_greedy_decoding() || sampling_params[0].is_multinomial()),
        "Currently streaming is possible only with batch size=1 and only for greedy or multinomial decoding");

    std::vector<GenerationHandle> main_generations;
    for (size_t request_id = 0; request_id < input_ids.size(); ++request_id) {
        OPENVINO_ASSERT(1 == input_ids[request_id].get_shape().at(0), "Use multiple tensors to pass a batch.");
        main_generations.push_back(m_main_pipeline->add_request(request_id, input_ids[request_id], sampling_params[request_id]));

        auto draft_sampling_params = sampling_params[request_id];
        // set the parameters do not stop draft generation without stopping of the same request for main pipeline
        draft_sampling_params.ignore_eos = true;
        draft_sampling_params.stop_strings = {};
        std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
        m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, input_ids[request_id], draft_sampling_params)});
    }
    auto all_requests = get_awaiting_requests();

    GenerationHandle& generation = main_generations.at(0);

    streamer_ptr->start();

    while (has_non_finished_requests()) {
        try {
            step();
        } catch (...) {
            drop_requests(); // remove all requests from pipeline state in case of exception
            streamer_ptr->end();
            std::rethrow_exception(std::current_exception());
        }
        stream_tokens(streamer_ptr, generation);
    }

    // waiting for competion of streaming
    streamer_ptr->end();

    OPENVINO_ASSERT(is_requests_empty(), "Internal error: current request is supposed to be dropped within step() function as completed");

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
            const auto & sequence = sequences[i];
            const float score = sampling_params.is_beam_search() ? sequence->get_beam_search_score(sampling_params) : sequence->get_cumulative_log_prob();
            const auto & generated_ids = sequence->get_generated_ids();

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

SpeculativeDecodingMetrics
ContinuousBatchingPipeline::SpeculativeDecodingImpl::get_speculative_decoding_metrics() {
    return m_sd_metrics;
};

void ContinuousBatchingPipeline::SpeculativeDecodingImpl::drop_requests() {
    m_draft_pipeline->finish_request();
    m_main_pipeline->finish_request();
}


bool ContinuousBatchingPipeline::SpeculativeDecodingImpl::is_requests_empty() {
    return m_main_pipeline->is_requests_empty() && m_draft_pipeline->is_requests_empty();
}

std::vector<SequenceGroup::Ptr> ContinuousBatchingPipeline::SpeculativeDecodingImpl::get_awaiting_requests() {
    auto main_awaiting_requests = m_main_pipeline->get_awaiting_requests();
    auto draft_awaiting_requests = m_draft_pipeline->get_awaiting_requests();
    OPENVINO_ASSERT(main_awaiting_requests.size() == draft_awaiting_requests.size());
    return main_awaiting_requests;
}
// end of speculative_decoding_impl

void extract_hidden_state_generic(std::shared_ptr<ov::Model>& model,
                                                       const std::string& eagle_version,
                                                       const std::string& model_type,
                                                       const std::string& custom_node_name = "") {
    if (eagle_version == "EAGLE2" || model_type == "draft") { // for draft model, we always only need to extract last hidden state
        std::cout << model_type << " model - last hidden state extraction" << std::endl;
        ov::pass::Manager pm;
        std::vector<int> layers = {-1}; // -1 means last hidden layer
        pm.register_pass<EagleModelTransform>(layers);
        pm.run_passes(model);
    } else if (eagle_version == "EAGLE3") {
        std::cout << model_type << " model - Eagle 3 hidden state extraction" << std::endl;
        ov::pass::Manager pm;
        /*if idx==len(self.layers)-3 or idx==len(self.layers)//2 or idx==2:
            all_hidden_states += (hidden_states,)*/
        std::vector<int> layers = {2, 16, 31}; // need to add check, only support positive values
        pm.register_pass<EagleModelTransform>(layers);
        pm.run_passes(model);
    } else {
        std::cerr << "Error: " << model_type << " model - Unsupported eagle version: " << eagle_version << std::endl;
    }
}

EagleModelTransform::EagleModelTransform(const std::vector<int>& layers) : m_layer_ids(layers) {
}

bool EagleModelTransform::run_on_model(const std::shared_ptr<ov::Model>& model) {
    //m_model = model;
    m_new_results.clear();
    if (m_layer_ids.size() == 1 && m_layer_ids[0] == -1) {
        ov::pass::Manager manager;
        manager.set_per_pass_validation(false);
        manager.register_pass<EagleBaseTransform>(m_layer_ids, m_new_results);
        manager.run_passes(model);
        
        if (!m_new_results.empty()) {
            model->add_results(m_new_results);
            std::cout << "EagleModelTransform - Added last hidden output " << std::endl;
            return true;
        }
    } else {
        ov::pass::Manager manager;
        manager.set_per_pass_validation(false);
        manager.register_pass<Eagle3Transform>(m_layer_ids, m_hidden_layer_outputs);
        manager.run_passes(model);
        
        if (!m_hidden_layer_outputs.empty()) {
            std::cout << "EagleModelTransform - extracted intermediate hidden state outputs " << std::endl;
            auto concat = std::make_shared<ov::op::v0::Concat>(m_hidden_layer_outputs, -1);
            concat->set_friendly_name("eagle3_hidden_states_concat");
            
            auto result = std::make_shared<ov::op::v0::Result>(concat);
            result->set_friendly_name("last_hidden_state");
            model->add_results({result});
            
            std::cout << "EagleModelTransform - Added concated eagle3 hidden state output" << std::endl;
            return true;
        }
    }
    
    return false;
}
EagleBaseTransform::EagleBaseTransform(const std::vector<int>& layers, std::vector<std::shared_ptr<ov::op::v0::Result>>& results) : m_layers(layers) {
    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::wrap_type<v0::Result>(), this->get_type_info().name),
        ([&results, this](ov::pass::pattern::Matcher& m) {
            auto node = m.get_match_root();
            try {
                if (apply(node, results)) {
                    ++applied; // FIXME: For debugging purposes only
                    return true;
                }
            } catch (...) {
                OPENVINO_ASSERT(false, "EagleTransform failed to apply");
            }
            return false;
        })
    );
}

std::shared_ptr<ov::Node> EagleBaseTransform::find_last_hidden_node(const std::shared_ptr<ov::Node>& start_node, 
                                                               std::set<ov::Node*>& visited_nodes) {
    if (visited_nodes.count(start_node.get())) {
        return nullptr;
    }

    visited_nodes.insert(start_node.get());

    if (ov::is_type<ov::op::v0::MatMul>(start_node)) {
        // check the input nodes of MatMul, if found Gather node, return the gather node, otherwise ,retrun the matmul node
        for (size_t i = 0; i < start_node->get_input_size(); ++i) {
            auto input_node = start_node->get_input_node_shared_ptr(i);
            if (!input_node) continue;
            // rule out logit processing node
            if (ov::as_type_ptr<op::util::GatherBase>(input_node)) {
                return input_node;
            }
        }
        return start_node;
    }
    
    for (size_t i = 0; i < start_node->get_input_size(); ++i) {
        auto input_node = start_node->get_input_node_shared_ptr(i);
        if (!input_node) continue;
        
        auto result = find_last_hidden_node(input_node, visited_nodes);
        if (result) {
            return result;
        }
    }
    return nullptr;
}

std::shared_ptr<ov::Node> EagleBaseTransform::find_last_hidden_node(const std::shared_ptr<ov::Node>& start_node) {
    std::set<ov::Node*> visited_nodes;
    return find_last_hidden_node(start_node, visited_nodes);
}

bool EagleBaseTransform::apply(NodePtr node, std::vector<std::shared_ptr<ov::op::v0::Result>>& results) {
    if (ov::is_type<ov::op::v0::Result>(node)) {
        // we are applying transformation to the last hidden state, eagle2 mode
        NodePtr input_node = node->get_input_node_shared_ptr(0);
        if (!input_node) {
            return false;
        }
        auto last_hidden_node = find_last_hidden_node(input_node);
        if (!last_hidden_node) {
            return false;
        }
        // 
        std::shared_ptr<ov::Node> non_const_input = nullptr;
        size_t non_const_input_idx = 0;
        
        for (size_t i = 0; i < last_hidden_node->get_input_size(); ++i) {
            auto input_node = last_hidden_node->get_input_node_shared_ptr(i);
            if (!input_node) continue;
            
            if (!(ov::is_type<ov::op::v0::Constant>(input_node)) && !(ov::is_type<ov::op::v0::Convert>(input_node))) {
                non_const_input = input_node;
                non_const_input_idx = i;
                break;
            }
        }
        
        if (!non_const_input) {
            return false;
        }
        
        auto result = std::make_shared<ov::op::v0::Result>(last_hidden_node->input_value(non_const_input_idx));
        std::string output_name = "last_hidden_state";
        result->output(0).set_names({output_name});
        result->set_friendly_name(output_name);
        results.push_back(result);
        return true;
    }
    return false;
}

Eagle3Transform::Eagle3Transform(const std::vector<int>& layers, std::vector<Output<Node>>& hidden_state_outputs) : m_layers(layers) {
    auto is_target_pattern = [&](const Output<Node>& output) {
        auto add_node = ov::as_type_ptr<ov::op::v1::Add>(output.get_node_shared_ptr());
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
        if (!input0 || !ov::is_type<ov::op::v0::MatMul>(input0)) {
            return false;
        }
        auto matmul_node = input0;
        auto matmul_input = matmul_node->get_input_node_shared_ptr(0);
        if (!matmul_input) {
            return false;
        }

        bool has_multiply = ov::is_type<ov::op::v1::Multiply>(matmul_input); // ACT(up) dot gate
        return has_multiply;    
    };

    auto hidden_layer = ov::pass::pattern::wrap_type<ov::op::v1::Add>(is_target_pattern);
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(hidden_layer, "Eagle3Transform::hidden_extraction"),
        [&hidden_state_outputs, this](ov::pass::pattern::Matcher& m) {
            auto node = m.get_match_root();
            try {
                if (apply(node, hidden_state_outputs)) {
                    ++applied; // FIXME: For debugging purposes only
                    return true;
                }
            } catch (...) {
                OPENVINO_ASSERT(false, "Eagle3Transform failed to apply");
            }
            return false;
        }
    );
}

bool Eagle3Transform::apply(NodePtr node, std::vector<Output<Node>>& hidden_state_outputs) {
    if (ov::is_type<ov::op::v1::Add>(node)) {
        auto add_node = std::dynamic_pointer_cast<ov::op::v1::Add>(node);
        if (!add_node) {
            return false;
        }
        hidden_state_outputs.push_back(add_node->output(0));
        return true;
    }
    return false;
}


ContinuousBatchingPipeline::EagleDecodingImpl::EagleDecodingImpl(const ov::genai::ModelDesc& main_model_desc,
                                                                 const ov::genai::ModelDesc& draft_model_desc,
                                                                 const std::string& eagle_version) : m_eagle_version(eagle_version) {
    auto main_model = main_model_desc.model;
    auto draft_model = draft_model_desc.model;

    auto main_scheduler_config = main_model_desc.scheduler_config;
    auto main_device = main_model_desc.device;

    utils::apply_paged_attention_transformations(main_model, main_model_desc.scheduler_config.use_cache_eviction);
    utils::apply_paged_attention_transformations(draft_model, main_model_desc.scheduler_config.use_cache_eviction);

    utils::apply_gather_before_matmul_transformation(main_model);
    utils::apply_gather_before_matmul_transformation(draft_model);

    std::string draft_device = draft_model_desc.device.empty() ? main_model_desc.device : draft_model_desc.device;
    bool is_draft_scheduler_undefined = draft_model_desc.scheduler_config == SchedulerConfig();

    ov::genai::SchedulerConfig main_scheduler_config_updated = main_scheduler_config,
                               draft_scheduler_config = is_draft_scheduler_undefined
                                                            ? main_scheduler_config
                                                            : draft_model_desc.scheduler_config;

    if (is_draft_scheduler_undefined) {
        // split KV cache to 2 caches for main and draft models
        auto compute_total_hidden_size = [](const std::shared_ptr<ov::Model>& model) -> size_t {
            size_t total_hidden_size = 0;
            for (const auto& param_ptr : model->get_parameters()) {
                const auto& name = param_ptr->get_friendly_name();
                if (name.find("key_cache.") == 0) {
                    auto pa_op = param_ptr->get_output_target_inputs(0).begin()->get_node();
                    const auto& rt_info = pa_op->get_rt_info();
                    total_hidden_size += rt_info.at("num_k_heads").as<int>() * rt_info.at("k_head_size").as<int>() +
                                         rt_info.at("num_v_heads").as<int>() * rt_info.at("v_head_size").as<int>();
                }
            }
            return total_hidden_size;
        };
        float main_model_hidden_size = compute_total_hidden_size(main_model),
              draft_model_hidden_size = compute_total_hidden_size(draft_model);
        auto k = draft_model_hidden_size / (main_model_hidden_size + draft_model_hidden_size);

        // TODO: work with KV blocks as it will be more precise instead of GBs
        size_t main_cache_size = std::ceil(main_scheduler_config.cache_size * (1.f - k)),
               draft_cache_size = main_scheduler_config.cache_size - main_cache_size;
        if (draft_cache_size == 0 && main_cache_size > 0) {
            main_cache_size -= (main_cache_size > 1 ? 1 : 0);
            draft_cache_size = 1;
        }

        main_scheduler_config_updated.cache_size = main_cache_size;
        draft_scheduler_config.cache_size = draft_cache_size;
    }

    ov::AnyMap draft_properties =
        draft_model_desc.properties.empty() ? main_model_desc.properties : draft_model_desc.properties;

    // main and draft model can have different tokenizers
    // to do: support retokenization: 154103
    Tokenizer main_model_tokenizer = main_model_desc.tokenizer;
    Tokenizer draft_model_tokenizer = draft_model_desc.tokenizer;

    // todo: remove this condition after support of CVS-154103
    OPENVINO_ASSERT(are_tokenizers_equal(main_model_tokenizer, draft_model_tokenizer),
                    "Tokenizers for draft and main models are different!");

    m_tokenizer = main_model_tokenizer;
    // for eagle model, we need to obtain hidden layer state as extra output
    // apply transformations needed to run eagle model
    // target model: hidden state extraction, draft model: hidden state import , hidden state extraction
    // eagle3 specific : dt importing
    extract_hidden_state_generic(main_model, m_eagle_version, "main", "");
    extract_hidden_state_generic(draft_model, m_eagle_version, "draft", "");

    // to create `main_pipeline` with enabled validation_mode and `draft_pipeline` with disabled validation mode
    m_main_pipeline = std::make_shared<ContinuousBatchingForEagleDecodingImpl>(main_model,
                                                                               main_model_tokenizer,
                                                                               main_model_desc.generation_config,
                                                                               main_scheduler_config_updated,
                                                                               main_device,
                                                                               main_model_desc.properties,
                                                                               true);
    m_draft_pipeline = std::make_shared<ContinuousBatchingForEagleDecodingImpl>(draft_model,
                                                                                draft_model_tokenizer,
                                                                                draft_model_desc.generation_config,
                                                                                draft_scheduler_config,
                                                                                draft_device,
                                                                                draft_properties,
                                                                                false);

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
}

GenerationHandle ContinuousBatchingPipeline::EagleDecodingImpl::add_request(
    uint64_t request_id,
    const ov::Tensor& input_ids,
    ov::genai::GenerationConfig sampling_params) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};
    m_draft_generations.insert(
        {request_id, m_draft_pipeline->add_request(request_id, input_ids, draft_sampling_params)});
    return m_main_pipeline->add_request(request_id, input_ids, sampling_params);
};

GenerationHandle ContinuousBatchingPipeline::EagleDecodingImpl::add_request(
    uint64_t request_id,
    const std::string& prompt,
    ov::genai::GenerationConfig sampling_params) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};
    m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, prompt, draft_sampling_params)});
    return m_main_pipeline->add_request(request_id, prompt, sampling_params);
}

bool ContinuousBatchingPipeline::EagleDecodingImpl::has_non_finished_requests() {
    return m_main_pipeline->has_non_finished_requests();
}

void ContinuousBatchingPipeline::EagleDecodingImpl::step() {
    // this blocks adding new requests during step as it may break coherence between main and draft models
    std::lock_guard<std::mutex> lock{m_draft_generations_mutex};
    auto& raw_perf_counters = m_perf_metrics.raw_metrics;
    auto& main_raw_perf_counters = m_perf_metrics.main_model_metrics.raw_metrics;

    ManualTimer step_timer("speculative_decoding: step()");
    step_timer.start();

    m_draft_pipeline->pull_awaiting_requests(true);
    m_main_pipeline->pull_awaiting_requests();

    // generate candidates by draft model
    ManualTimer draft_timer("speculative_decoding: draft_model: multistep()");
    draft_timer.start();
    m_draft_pipeline->multistep();
    draft_timer.end();
    m_sd_metrics.draft_duration += draft_timer.get_duration();
    m_pipeline_metrics = m_main_pipeline->get_metrics();

    // to generate num_matches statistic
    std::map<int64_t, UpdateRequestResult> update_sequence_info;
    // put candidates to model KV cache
    auto draft_generated_requests = m_draft_pipeline->get_generated_requests();
    for (const auto& candidate : m_draft_pipeline->get_generated_requests()) {
        auto update_result = m_main_pipeline->update_main_request(candidate.first, candidate.second);
        update_sequence_info.insert({{candidate.first, update_result}});
    }

    ManualTimer main_timer("speculative_decoding: main_model: step()");
    main_timer.start();
    m_main_pipeline->step();
    main_timer.end();
    m_sd_metrics.main_duration += main_timer.get_duration();
    m_pipeline_metrics = m_main_pipeline->get_metrics();
    // get logits and last hidden layer
    auto main_generated_requests =
        m_main_pipeline->get_generated_requests();  // feature extraction is enabled in main pipeline

    for (const auto& checked_sequence : main_generated_requests) {
        auto update_result = m_draft_pipeline->update_draft_request(checked_sequence.first, checked_sequence.second);
        update_sequence_info[checked_sequence.first].removed_tokens_cnt = update_result.removed_tokens_cnt;
    }

    // finish draft request if the generation was completed
    for (const auto& draft_request : draft_generated_requests) {
        auto request_id = draft_request.first;
        if (!main_generated_requests.count(request_id)) {
            m_draft_pipeline->finish_request(request_id);
            // remove draft_generation_handle from queue
            m_draft_generations.erase(request_id);
        }
        auto updated_seq_info = update_sequence_info[request_id];
        // several prompt phase
        if (updated_seq_info.inserted_tokens_cnt == 0) {
            continue;
        }
        float acceptance_rate =
            1 - static_cast<float>(updated_seq_info.removed_tokens_cnt) / updated_seq_info.inserted_tokens_cnt;
        m_sd_metrics.update_acceptance_rate(request_id, acceptance_rate * 100);
        m_sd_metrics.update_draft_accepted_tokens(
            request_id,
            (updated_seq_info.inserted_tokens_cnt - updated_seq_info.removed_tokens_cnt));
    }

    // update perf metrics
    const auto num_generated_tokens = m_main_pipeline->get_processed_tokens_per_iteration();
    if (num_generated_tokens > 0) {
        auto infer_duration = step_timer.get_duration_microsec();

        raw_perf_counters.m_token_infer_durations.emplace_back(infer_duration);
        raw_perf_counters.m_inference_durations[0] += MicroSeconds(infer_duration);
        raw_perf_counters.m_new_token_times.emplace_back(main_timer.get_end_time());

        raw_perf_counters.m_batch_sizes.emplace_back(num_generated_tokens);
    }

    if (main_generated_requests.empty() && utils::env_setup_for_print_debug_info()) {
        m_sd_metrics.print(true);
        m_sd_metrics.clean_up();
    }

    step_timer.end();
}

ov::Tensor ContinuousBatchingPipeline::EagleDecodingImpl::update_main_input_ids(const ov::Tensor& original_input_ids) {
    auto shape = original_input_ids.get_shape();
    if (shape.size() == 0 || shape.back() <= 1) {
        return ov::Tensor(original_input_ids);
    }

    size_t original_length = shape.back();
    size_t new_length = original_length + 1;

    ov::Tensor draft_input_ids(ov::element::i64, {1, new_length});

    const int64_t* src_data = original_input_ids.data<int64_t>();
    int64_t* dst_data = draft_input_ids.data<int64_t>();
    dst_data[0] = m_tokenizer.get_bos_token_id();  // add BOS token at the beginning
    std::copy(src_data, src_data + original_length, dst_data + 1);

    return draft_input_ids;
}

ov::Tensor ContinuousBatchingPipeline::EagleDecodingImpl::create_draft_input_ids(const ov::Tensor& original_input_ids) {
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

std::vector<EncodedGenerationResult> ContinuousBatchingPipeline::EagleDecodingImpl::generate(
    const std::vector<ov::Tensor>& input_ids,
    const std::vector<GenerationConfig>& sampling_params,
    const StreamerVariant& streamer) {
    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};

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
    m_main_pipeline->set_adapters(sampling_params[0].adapters);
    m_main_pipeline->set_hidden_state_export_needed(true);
    m_draft_pipeline->set_adapters(sampling_params[0].adapters);
    m_draft_pipeline->set_hidden_state_export_needed(true);
    m_draft_pipeline->set_hidden_state_import_needed(true);
    m_draft_pipeline->set_hidden_state_internal_needed(true);

    const auto streamer_ptr = std::make_shared<ThreadedStreamerWrapper>(streamer, m_tokenizer);

    OPENVINO_ASSERT(
        !streamer_ptr->has_callback() || input_ids.size() == 1 && (sampling_params[0].is_greedy_decoding() ||
                                                                   (sampling_params[0].is_multinomial() &&
                                                                    sampling_params[0].num_return_sequences == 1) ||
                                                                   sampling_params[0].is_eagle_tree()),
        "Currently eagle streaming is possible only with batch size=1 and only for greedy or multinomial decoding");

    std::vector<GenerationHandle> main_generations;
    ov::Tensor new_input_ids;
    for (size_t request_id = 0; request_id < input_ids.size(); ++request_id) {
        auto new_input_ids = input_ids[request_id]; //update_main_input_ids(input_ids[request_id]);
        OPENVINO_ASSERT(1 == input_ids[request_id].get_shape().at(0), "Use multiple tensors to pass a batch.");
        main_generations.push_back(
            m_main_pipeline->add_request(request_id, new_input_ids, sampling_params[request_id]));

        auto draft_sampling_params = sampling_params[request_id];
        // set the parameters do not stop draft generation without stopping of the same request for main pipeline
        draft_sampling_params.ignore_eos = true;
        draft_sampling_params.stop_strings = {};
        draft_sampling_params.eagle_total_tokens = 5;
        draft_sampling_params.eagle_branching_factor = 3;
        draft_sampling_params.eagle_depth = 2;             // hard code to test now
        draft_sampling_params.eagle_tree_width = 10;       // for eagle model, draft model use beam search for multiple
                                                           // tokens generation for now, will be updated to top-k later
        draft_sampling_params.eagle_final_candidates = 8;  // hard code to test now

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
    //m_perf_metrics.draft_model_metrics.raw_metrics = m_draft_pipeline->raw_perf_metrics;
    generate_timer.end();

    for (size_t request_id = 0; request_id < all_requests.size(); ++request_id) {
        const auto& request = all_requests[request_id];
        auto sampling_params = request->get_sampling_parameters();
        const auto& sequences = request->get_finished_sequences();
        m_draft_pipeline->clear_sampler_top_k_selector(request->get_request_id());
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
        results.push_back(std::move(result));
    }

    OPENVINO_ASSERT(results.size() == input_ids.size());
    generate_timer.end();
    return results;
}

SpeculativeDecodingMetrics ContinuousBatchingPipeline::EagleDecodingImpl::get_speculative_decoding_metrics() {
    return m_sd_metrics;
};

void ContinuousBatchingPipeline::EagleDecodingImpl::drop_requests() {
    m_draft_pipeline->finish_request();
    m_main_pipeline->finish_request();
}

bool ContinuousBatchingPipeline::EagleDecodingImpl::is_requests_empty() {
    return m_main_pipeline->is_requests_empty() && m_draft_pipeline->is_requests_empty();
}

std::vector<SequenceGroup::Ptr> ContinuousBatchingPipeline::EagleDecodingImpl::get_awaiting_requests() {
    auto main_awaiting_requests = m_main_pipeline->get_awaiting_requests();
    auto draft_awaiting_requests = m_draft_pipeline->get_awaiting_requests();
    OPENVINO_ASSERT(main_awaiting_requests.size() == draft_awaiting_requests.size());
    return main_awaiting_requests;
}
}  // namespace ov::genai
