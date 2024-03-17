
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <limits>

#include "sequence_group.hpp"

class Sampler {
    SamplingParameters m_parameters;

    bool _is_gready_sampling() const {
        return m_parameters.temperature == 0 && !_is_beam_search();
    }

    bool _is_beam_search() const {
        return m_parameters.n_groups * m_parameters.group_size > 1;
    }

    int64_t _greedy_sample(const float * logits_data, size_t vocab_size) const {
        // currently, greedy search is used
        // TODO: apply m_config
        int64_t out_token = std::max_element(logits_data, logits_data + vocab_size) - logits_data;
        return out_token;
    }

    void _beam_search() {

    }

public:
    struct ChildSequence {
        // parent sequence ID within a group of sequences
        uint64_t m_parent_seq_id;
        // next (child) token ID
        int64_t m_token_id;
    };

    struct Output {
        // a map of sequence group ID => { parent group ID, sampled token ID }
        std::map<uint64_t, std::vector<ChildSequence>> m_child_sequences;
    };

    Sampler(const SamplingParameters & parameters = {}) :
        m_parameters(parameters) { }

    Output sample(std::vector<SequenceGroup> & sequence_groups, ov::Tensor logits) const {
        const float * logits_data = logits.data<float>();
        ov::Shape logits_shape = logits.get_shape();
        OPENVINO_ASSERT(logits_shape.size() == 3);
        size_t batch_size = logits_shape[0], seq_len = logits_shape[1], vocab_size = logits_shape[2], logits_stride = seq_len * vocab_size;
        OPENVINO_ASSERT(seq_len == 1);

        Output sampler_output;

        for (size_t i = 0, current_token_id = 0; i < sequence_groups.size(); ++i) {
            SequenceGroup sequence_group = sequence_groups[i];
            current_token_id += sequence_group.get_num_scheduled_tokens() * sequence_group.num_running_seqs();

            if (sequence_group.requires_sampling()) {
                std::vector<ChildSequence> child_sequences;

                if (_is_gready_sampling()) {
                    int64_t sampled_token_id = _greedy_sample(logits_data + logits_stride * (current_token_id - 1), vocab_size);
                    // in case of greedy search we always have a single parent sequence to sample from
                    child_sequences.push_back(ChildSequence{ .m_parent_seq_id = 0, .m_token_id = sampled_token_id });
                } else if (_is_beam_search()) {
                    // TODO: move implementation of group beam search here
                }

                sampler_output.m_child_sequences[sequence_group.get_request_id()] = std::move(child_sequences);
            } else {
                // we are in prompt processing phase when prompt is split into chunks and processed step by step
            }

            // update internal state of sequence group to reset scheduler tokens and update currently processed onces
            sequence_group.finish_iteration();
        }

        return sampler_output;
    }
};
