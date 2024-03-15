
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <limits>

#include "sequence_group.hpp"

class Sampler {
    SamplingParameters m_parameters;

    int64_t _greedy_sample(const float * logits_data, size_t vocab_size) const {
        // currently, greedy search is used
        // TODO: apply m_config
        int64_t out_token = std::max_element(logits_data, logits_data + vocab_size) - logits_data;
        return out_token;
    }

public:
    Sampler(const SamplingParameters & parameters = {}) :
        m_parameters(parameters) { }

    void decode(std::vector<SequenceGroup> & sequence_groups, ov::Tensor logits) const {
        const float * logits_data = logits.data<float>();
        ov::Shape logits_shape = logits.get_shape();
        OPENVINO_ASSERT(logits_shape.size() == 3);
        size_t batch_size = logits_shape[0], seq_len = logits_shape[1], vocab_size = logits_shape[2], logits_stride = seq_len * vocab_size;
        OPENVINO_ASSERT(seq_len == 1);

        for (size_t i = 0, current_token_id = 0; i < sequence_groups.size(); ++i) {
            SequenceGroup sequence_group = sequence_groups[i];
            // TODO: process multuple sequences within a group
            Sequence & sequence = sequence_groups[i][0];

            current_token_id += sequence_group.get_num_scheduled_tokens();

            if (sequence_group.requires_sampling()) {
                int64_t sampled_token_id = _greedy_sample(logits_data + logits_stride * (current_token_id - 1), vocab_size);
                sequence.append_token(sampled_token_id);
            } else {
                // we are in prompt processing phase when prompt is split into chunks and processed step by step
            }

            // update internal state of sequence to reset scheduler tokens and update currently processed onces
            sequence_group.finish_iteration();
        }
    }
};

