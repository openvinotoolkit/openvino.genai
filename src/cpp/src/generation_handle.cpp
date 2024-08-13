// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "openvino/genai/generation_handle.hpp"
#include "generation_stream.hpp"

using namespace ov::genai;

GenerationHandle::~GenerationHandle() {
    drop();
}

GenerationStatus GenerationHandle::get_status() const {
    return m_generation_stream->get_status();
}

bool GenerationHandle::can_read() const {
    return !is_dropped() &&  m_generation_stream->can_read();
}

bool GenerationHandle::is_dropped() const {
    return get_status() == GenerationStatus::DROPPED_BY_HANDLE;
}

void GenerationHandle::drop() {
    m_generation_stream->drop();
}

std::unordered_map<uint64_t, GenerationOutput> GenerationHandle::back() {
    OPENVINO_ASSERT(!is_dropped(), "GenerationHandle::Ptr cannot be used after it is dropped.");
    return m_generation_stream->back();
}

std::unordered_map<uint64_t, GenerationOutput> GenerationHandle::read() {
    OPENVINO_ASSERT(!is_dropped(), "GenerationHandle::Ptr cannot be used after it is dropped.");
    return m_generation_stream->read();
}

void add_partial_result(std::unordered_map<uint64_t, GenerationOutput>& partial_results, std::unordered_map<uint64_t, GenerationOutput>& iteration_results) {
    for (auto& iteration_result: iteration_results) {
        auto partial_result_iter = partial_results.find(iteration_result.first);
        if (partial_result_iter == partial_results.end()) {
            partial_results.emplace(iteration_result.first, iteration_result.second);
        } else {
            partial_result_iter->second.generated_token_ids.push_back(iteration_result.second.generated_token_ids[0]);
            partial_result_iter->second.score = iteration_result.second.score;
            partial_result_iter->second.finish_reason = iteration_result.second.finish_reason;
        }
    }
}

std::vector<GenerationOutput> GenerationHandle::read_all() {
    OPENVINO_ASSERT(!is_dropped(), "GenerationHandle::Ptr cannot be used after it is dropped.");
    std::vector<GenerationOutput> results;
    std::unordered_map<uint64_t, GenerationOutput> partial_results;
    // We iterate until generation is running or there are tokens we haven't read yet
    while (get_status() == GenerationStatus::RUNNING || can_read()) {
        // For unary case there's only one iteration and we get all results in a single read() call
        std::unordered_map<uint64_t, GenerationOutput> iteration_results = read();
        add_partial_result(partial_results, iteration_results);
    }

    for (auto& partial_result: partial_results) {
        results.push_back(partial_result.second);
    }
    return results;
}
