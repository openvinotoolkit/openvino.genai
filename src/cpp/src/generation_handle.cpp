// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "openvino/genai/generation_handle.hpp"
#include "generation_stream.hpp"

using namespace ov::genai;

GenerationHandleImpl::~GenerationHandleImpl() {
    m_generation_stream->drop();
}

GenerationStatus GenerationHandleImpl::get_status() {
    return m_generation_stream->get_status();
}

bool GenerationHandleImpl::can_read() {
    return m_generation_stream->can_read();
}

std::unordered_map<uint64_t, GenerationOutput> GenerationHandleImpl::read() {
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
        }
    }
}

std::vector<GenerationOutput> GenerationHandleImpl::read_all() {
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
