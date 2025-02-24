// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "openvino/genai/generation_handle.hpp"
#include "generation_stream.hpp"

using namespace ov::genai;

GenerationHandleImpl::~GenerationHandleImpl() {
    stop();
}

GenerationStatus GenerationHandleImpl::get_status() {
    return m_generation_stream->get_status();
}

bool GenerationHandleImpl::can_read() {
    return !is_cancelled() && !is_stopped() && m_generation_stream->can_read();
}

bool GenerationHandleImpl::is_stopped() {
    return get_status() == GenerationStatus::STOP;
}

bool GenerationHandleImpl::is_cancelled() {
    return get_status() == GenerationStatus::CANCEL;
}

void GenerationHandleImpl::drop() {
    m_generation_stream->stop();
}

void GenerationHandleImpl::stop() {
    m_generation_stream->stop();
}

void GenerationHandleImpl::cancel() {
    m_generation_stream->cancel();
}

std::unordered_map<uint64_t, GenerationOutput> GenerationHandleImpl::read() {
    OPENVINO_ASSERT(!is_stopped() && !is_cancelled(), "GenerationHandle cannot be used after it is stopped / cancelled.");
    return m_generation_stream->read();
}

void add_partial_result(std::unordered_map<uint64_t, GenerationOutput>& partial_results, std::unordered_map<uint64_t, GenerationOutput>& iteration_results) {
    for (auto& iteration_result: iteration_results) {
        auto partial_result_iter = partial_results.find(iteration_result.first);
        if (partial_result_iter == partial_results.end()) {
            partial_results.emplace(iteration_result.first, iteration_result.second);
        } else {
            auto generated_len = iteration_result.second.generated_ids.size();
            OPENVINO_ASSERT(generated_len == iteration_result.second.generated_log_probs.size());
            for (size_t i = 0; i < generated_len; ++i) {
                partial_result_iter->second.generated_ids.push_back(iteration_result.second.generated_ids[i]);
                partial_result_iter->second.generated_log_probs.push_back(iteration_result.second.generated_log_probs[i]);
            }
            partial_result_iter->second.score = iteration_result.second.score;
            partial_result_iter->second.finish_reason = iteration_result.second.finish_reason;
        }
    }
}

std::vector<GenerationOutput> GenerationHandleImpl::read_all() {
    OPENVINO_ASSERT(!is_stopped() && !is_cancelled(), "GenerationHandle cannot be used after it is stopped / cancelled.");
    std::vector<GenerationOutput> results;
    std::unordered_map<uint64_t, GenerationOutput> partial_results;
    // We iterate until generation is running or there are tokens we haven't read yet
    while (get_status() == GenerationStatus::RUNNING || can_read()) {
        // For unary case there's only one iteration and we get all results in a single read() call
        std::unordered_map<uint64_t, GenerationOutput> iteration_results = read();
        add_partial_result(partial_results, iteration_results);
    }

    for (auto& partial_result : partial_results) {
        results.push_back(partial_result.second);
    }
    std::sort(results.begin(), results.end(), [](const GenerationOutput& lhs, const GenerationOutput& rhs) { return lhs.score > rhs.score; });
    results.resize(std::min(m_sampling_params.num_return_sequences, results.size()));
    return results;
}
