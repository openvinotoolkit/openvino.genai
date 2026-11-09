// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speculative_decoding/continuous_batching/dflash_candidate_selector.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <utility>

#include "openvino/genai/perf_metrics.hpp"
#include "speculative_decoding/continuous_batching/dflash_candidate_lattice.hpp"
#include "utils.hpp"

namespace ov::genai::dflash_cb {

namespace {

size_t row_width(const ov::Tensor& rows) {
    OPENVINO_ASSERT(rows && rows.get_element_type() == ov::element::f32 && rows.is_continuous() &&
                        !rows.get_shape().empty() && rows.get_shape().back() > 0,
                    "DFlash candidate selection requires contiguous FP32 draft outputs with nonempty rows.");
    return rows.get_shape().back();
}

bool blocks_are_packed(const std::vector<SampledBlock>& blocks, size_t num_rows) {
    size_t next_row = 0;
    for (const auto& block : blocks) {
        if (block.first_row != next_row) {
            return false;
        }
        next_row += block.num_rows;
    }
    return next_row == num_rows;
}

// [rows of all blocks, width]: a view when the blocks fill the output back to back, else a copy.
ov::Tensor packed_rows(const ov::Tensor& rows, const std::vector<SampledBlock>& blocks, size_t num_packed_rows) {
    const size_t width = row_width(rows);
    if (blocks_are_packed(blocks, rows.get_size() / width)) {
        // const_cast is safe: the view is only set as an input of the candidate scorer, which reads it.
        return ov::Tensor(ov::element::f32, {num_packed_rows, width}, const_cast<void*>(rows.data()));
    }
    ov::Tensor packed(ov::element::f32, {num_packed_rows, width});
    float* destination = packed.data<float>();
    for (const auto& block : blocks) {
        std::copy_n(rows.data<const float>() + block.first_row * width, block.num_rows * width, destination);
        destination += block.num_rows * width;
    }
    return packed;
}

}  // namespace

CandidateSelector::CandidateSelector(const utils::dflash::DFlashRTInfo& rt_info,
                                     const std::string& device,
                                     const ov::AnyMap& properties) {
    const auto& markov_head = rt_info.markov_head_computation;
    OPENVINO_ASSERT(markov_head, "The DFlash candidate selector requires a draft with a Markov head.");
    // A draft without properties takes the target's, which can carry a MODEL_PROPERTIES overlay and the GenAI-only
    // sampler_num_threads that LLMPipeline adds. The head compiles with the language model's properties, as the
    // backbone does.
    ov::AnyMap compile_properties = utils::get_model_properties(properties, "language_model", device);
    compile_properties.erase("sampler_num_threads");
    auto request =
        utils::singleton_core().compile_model(markov_head, device, compile_properties).create_infer_request();
    if (utils::dflash::is_markov_step(markov_head)) {
        m_markov_step = std::move(request);
    } else {
        m_candidate_scorer = std::move(request);
    }
}

void CandidateSelector::prepare(const std::vector<SampledBlock>& blocks, const ov::Tensor& logits) {
    m_vocab_size = row_width(logits);
    const size_t num_rows = logits.get_size() / m_vocab_size;
    for (const auto& block : blocks) {
        OPENVINO_ASSERT(block.num_candidates > 0 && block.num_candidates <= block.num_rows &&
                            block.first_row + block.num_rows <= num_rows,
                        "DFlash draft block rows exceed the ",
                        num_rows,
                        " gathered draft rows.");
    }
    m_blocks = blocks;
    m_logits = logits;
    if (m_candidate_scorer && !m_blocks.empty()) {
        score_candidates();
    }
}

// Scores the rows of all blocks with one candidate scorer inference.
void CandidateSelector::score_candidates() {
    const size_t num_blocks = m_blocks.size();
    ov::Tensor anchor_token_ids(ov::element::i64, {num_blocks});
    ov::Tensor subsequence_begins(ov::element::i64, {num_blocks + 1});
    auto* begins = subsequence_begins.data<int64_t>();
    begins[0] = 0;
    m_lattice_begins.resize(num_blocks);
    for (size_t idx = 0; idx < num_blocks; ++idx) {
        anchor_token_ids.data<int64_t>()[idx] = m_blocks[idx].anchor_token;
        m_lattice_begins[idx] = static_cast<size_t>(begins[idx]);
        begins[idx + 1] = begins[idx] + static_cast<int64_t>(m_blocks[idx].num_rows);
    }
    const size_t num_rows = static_cast<size_t>(begins[num_blocks]);
    validate_subsequence_begins(subsequence_begins, num_blocks, num_rows);

    auto& scorer = *m_candidate_scorer;
    scorer.set_tensor("draft_logits", packed_rows(m_logits, m_blocks, num_rows));
    scorer.set_tensor("anchor_token_ids", anchor_token_ids);
    scorer.set_tensor("subsequence_begins", subsequence_begins);
    infer(scorer);

    m_edge_scores = scorer.get_tensor("edge_scores");
    m_candidate_ids = scorer.get_tensor("candidate_ids");
    validate_candidate_lattice(m_edge_scores, m_candidate_ids, num_rows, m_vocab_size);
}

// Runs the dense Markov step for the tokens the blocks drew at the position before, the anchors at the first position.
void CandidateSelector::step(size_t, const std::vector<size_t>& blocks, const std::vector<int64_t>& previous_tokens) {
    ov::Tensor prev_token_ids(ov::element::i64, {blocks.size()});
    std::copy(previous_tokens.begin(), previous_tokens.end(), prev_token_ids.data<int64_t>());
    m_markov_step->set_tensor("prev_token_ids", prev_token_ids);
    infer(*m_markov_step);
    m_markov_bias = m_markov_step->get_tensor("markov_bias");
    OPENVINO_ASSERT(m_markov_bias.get_element_type() == ov::element::f32 &&
                        m_markov_bias.get_shape() == ov::Shape({blocks.size(), m_vocab_size}),
                    "The Markov bias must be one FP32 row over the draft vocabulary per block.");
    m_markov_bias_rows.resize(m_blocks.size());
    for (size_t row = 0; row < blocks.size(); ++row) {
        m_markov_bias_rows[blocks[row]] = row;
    }
}

std::optional<std::vector<BlockToken>> CandidateSelector::draw(size_t block, const GenerationConfig& config) const {
    if (!m_candidate_scorer || !logit_processors_keep_argmax(config)) {
        return std::nullopt;
    }
    std::vector<BlockToken> tokens;
    for (const int64_t token : greedy_candidate_path(m_edge_scores,
                                                     m_candidate_ids,
                                                     m_lattice_begins[block],
                                                     m_blocks[block].num_candidates)) {
        tokens.push_back({token, 0.0f});
    }
    return tokens;
}

std::optional<BlockToken> CandidateSelector::draw_token(size_t block,
                                                        size_t position,
                                                        const GenerationConfig& config) const {
    if (!logit_processors_keep_argmax(config)) {
        return std::nullopt;
    }
    // the argmax of the row fill_row() writes, the first of equal values as in greedy sampling
    const float* logits = backbone_logits(block, position);
    const float* bias = markov_bias(block);
    int64_t token = 0;
    float best = -std::numeric_limits<float>::infinity();
    for (size_t idx = 0; idx < m_vocab_size; ++idx) {
        const float value = logits[idx] + bias[idx];
        if (value > best) {
            best = value;
            token = static_cast<int64_t>(idx);
        }
    }
    return BlockToken{token, 0.0f};
}

bool CandidateSelector::fill_row(size_t block, size_t position, int64_t previous_token, float* row) const {
    const float* logits = backbone_logits(block, position);
    if (m_markov_step) {
        std::transform(logits, logits + m_vocab_size, markov_bias(block), row, std::plus<float>());
        return true;
    }
    size_t previous_index = 0;
    if (position > 0) {
        // the lattice has no row after a token outside the candidates
        const auto index = find_candidate(m_candidate_ids, m_lattice_begins[block] + position - 1, previous_token);
        if (!index) {
            return false;
        }
        previous_index = *index;
    }
    fill_candidate_row(logits,
                       m_vocab_size,
                       m_edge_scores,
                       m_candidate_ids,
                       m_lattice_begins[block] + position,
                       previous_index,
                       row);
    return true;
}

const float* CandidateSelector::backbone_logits(size_t block, size_t position) const {
    return m_logits.data<const float>() + (m_blocks[block].first_row + position) * m_vocab_size;
}

const float* CandidateSelector::markov_bias(size_t block) const {
    return m_markov_bias.data<const float>() + m_markov_bias_rows[block] * m_vocab_size;
}

void CandidateSelector::infer(ov::InferRequest& request) {
    const auto start = std::chrono::steady_clock::now();
    request.infer();
    m_inference_duration += PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start);
}

float CandidateSelector::take_inference_duration() {
    return std::exchange(m_inference_duration, 0.0f);
}

}  // namespace ov::genai::dflash_cb
