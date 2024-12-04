// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "prompt_lookup_impl.hpp"
#include "text_callback_streamer.hpp"

namespace ov::genai {
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

GenerationHandle
ContinuousBatchingPipeline::PromptLookupImpl::add_request(uint64_t request_id,
                                                          const ov::Tensor& input_ids,
                                                          ov::genai::GenerationConfig sampling_params) {
    return m_pipeline->add_request(request_id, input_ids, sampling_params);
};

GenerationHandle
ContinuousBatchingPipeline::PromptLookupImpl::add_request(uint64_t request_id,
                                                          const std::string& prompt,
                                                          ov::genai::GenerationConfig sampling_params) {
    return m_pipeline->add_request(request_id, prompt, sampling_params);
}

bool ContinuousBatchingPipeline::PromptLookupImpl::has_non_finished_requests() {
    return m_pipeline->has_non_finished_requests();
}

void ContinuousBatchingPipeline::PromptLookupImpl::step() {
    ManualTimer candidates_timer("prompt_lookup_decoding: generate_candidates()");
    candidates_timer.start();
    m_pipeline->generate_candidates();
    candidates_timer.end();
    m_sd_metrics.draft_duration += candidates_timer.get_duration();
    auto generated_len_before = m_pipeline->get_generated_request_len();

    ManualTimer main_timer("prompt_lookup_decoding: step()");
    main_timer.start();
    m_pipeline->step();
    main_timer.end();
    m_sd_metrics.main_duration += main_timer.get_duration();
    m_pipeline_metrics = m_pipeline->get_metrics();
    auto generated_len_after = m_pipeline->get_generated_request_len();

    for (const auto request : generated_len_before) {
        auto request_id = request.first;
        auto prev_validation_len = request.second.second;
        if (prev_validation_len == 0) {
            continue;
        }
        size_t num_matches = prev_validation_len;
        float acceptance_rate = 1.f;
        if (generated_len_after.count(request.first)) {
            auto present_req_len = generated_len_after.at(request.first).first;
            auto prev_full_req_len = request.second.first;

            num_matches = (present_req_len - prev_full_req_len - 1);
            acceptance_rate = static_cast<float>(num_matches) / static_cast<float>(prev_validation_len);
        }        
        m_sd_metrics.update_acceptance_rate(request_id, acceptance_rate * 100);
        m_sd_metrics.update_draft_accepted_tokens(request_id, num_matches);
    }

    if (generated_len_after.empty() && 0) {
        m_sd_metrics.print(true);
        m_sd_metrics.clean_up();
    }
}

std::vector<EncodedGenerationResult>
ContinuousBatchingPipeline::PromptLookupImpl::generate(const std::vector<ov::Tensor>& input_ids,
                                                       const std::vector<GenerationConfig>& sampling_params,
                                                       const StreamerVariant& streamer) {
    ManualTimer generate_timer("speculative_decoding: generate()");
    generate_timer.start();
    OPENVINO_ASSERT(!has_non_finished_requests(), "Generate cannot be called while ContinuousBatchingPipeline is already in running state. Use ContinuousBatchingPipeline::add_request");
    OPENVINO_ASSERT(input_ids.size() == sampling_params.size());
    const std::shared_ptr<StreamerBase>& streamer_ptr = std::visit(overloaded{
        [](std::monostate) -> std::shared_ptr<StreamerBase> {
            return nullptr;
        },
        [](const std::shared_ptr<StreamerBase>& streamer) {
            return streamer;
        },
        [this](const std::function<bool(std::string)>& streamer) -> std::shared_ptr<StreamerBase> {
            return std::make_unique<TextCallbackStreamer>(m_tokenizer, streamer);
        }
    }, streamer);

    OPENVINO_ASSERT(streamer_ptr == nullptr || input_ids.size() == 1 && (sampling_params[0].is_greedy_decoding() || sampling_params[0].is_multinomial()),
        "Currently streaming is possible only with batch size=1 and only for greedy or multinomial decoding");

    std::vector<GenerationHandle> main_generations;
    for (size_t request_id = 0; request_id < input_ids.size(); ++request_id) {
        OPENVINO_ASSERT(1 == input_ids[request_id].get_shape().at(0), "Use multiple tensors to pass a batch.");
        main_generations.push_back(m_pipeline->add_request(request_id, input_ids[request_id], sampling_params[request_id]));
    }

    std::vector<EncodedGenerationResult> results;
    results.reserve(input_ids.size());

    bool continue_generation = true;
    while (has_non_finished_requests() && continue_generation) {
        step();
        if (streamer_ptr) {
            // not generated tokens like several prompt phase
            if (!main_generations.at(0).get()->can_read()) {
                continue;
            }
            std::unordered_map<uint64_t, GenerationOutput> token = main_generations.at(0).get()->back();
            OPENVINO_ASSERT(1 <= token.size());
            OPENVINO_ASSERT(1 <= token.begin()->second.generated_ids.size());
            for (const auto& gen_token : token.begin()->second.generated_ids) {
                continue_generation = !streamer_ptr->put(gen_token);
                if (!continue_generation) {
                    break;
                }
            }
        }
    }
    if (streamer_ptr) {
        streamer_ptr->end();
    }

    for (size_t generation_idx = 0; generation_idx < main_generations.size(); ++generation_idx) {
        const auto& generation = main_generations[generation_idx];
        EncodedGenerationResult result;
        result.m_request_id = 1;
        std::vector<GenerationOutput> generation_outputs = generation->read_all();
        std::sort(generation_outputs.begin(), generation_outputs.end(), [=] (GenerationOutput& r1, GenerationOutput& r2) {
            return r1.score > r2.score;
        });

        auto num_outputs = std::min(sampling_params[generation_idx].num_return_sequences, generation_outputs.size());
        for (size_t generation_output_idx = 0; generation_output_idx < num_outputs; ++generation_output_idx) {
            const auto& generation_output = generation_outputs[generation_output_idx];
            m_sd_metrics.set_generated_len(generation_idx, generation_outputs[generation_output_idx].generated_ids.size());
            result.m_generation_ids.push_back(std::move(generation_output.generated_ids));
            result.m_scores.push_back(generation_output.score);
        }
        result.m_status = generation->get_status();
        results.push_back(std::move(result));
    }

    OPENVINO_ASSERT(results.size() == input_ids.size());
    generate_timer.end();
    m_sd_metrics.total_duration = generate_timer.get_duration();

    return results;
}

SpeculativeDecodingMetrics
ContinuousBatchingPipeline::PromptLookupImpl::get_metrics() {
    return m_sd_metrics;
};
}
