// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "drafter_strategy.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>

#include "continuous_batching/timer.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "sampling/sampler.hpp"
#include "speculative_decoding/eagle3_model_transforms.hpp"
#include "utils.hpp"

namespace ov::genai {

Drafter::~Drafter() = default;

std::pair<std::string, ov::Any> drafter(std::shared_ptr<Drafter> drafter) {
    OPENVINO_ASSERT(drafter, "drafter must not be null");
    return {utils::DRAFTER_ARG_NAME, ov::Any(std::move(drafter))};
}

namespace drafter_utils {

std::vector<size_t> tree_depths(const DraftTree& tree, size_t max_nodes) {
    const size_t nodes = tree.tokens.size();
    OPENVINO_ASSERT(tree.parents.size() == nodes,
                    "Drafter: the DraftTree has ",
                    nodes,
                    " tokens and ",
                    tree.parents.size(),
                    " parents");
    OPENVINO_ASSERT(nodes <= max_nodes, "Drafter: proposed ", nodes, " tokens, at most ", max_nodes, " allowed");
    std::vector<size_t> depths(nodes);
    for (size_t i = 0; i < nodes; ++i) {
        const int64_t parent = tree.parents[i];
        OPENVINO_ASSERT(parent >= -1 && parent < static_cast<int64_t>(i),
                        "Drafter: node ",
                        i,
                        " has parent ",
                        parent,
                        ", expected -1 (the root) or an earlier node");
        depths[i] = parent < 0 ? 1 : depths[static_cast<size_t>(parent)] + 1;
    }
    return depths;
}

ov::Tensor make_tree_mask(const DraftTree& tree, ov::element::Type type) {
    const size_t rows = tree.tokens.size() + 1;
    ov::Tensor mask(type, {1, 1, rows, rows});
    auto fill = [&](auto* data, auto visible, auto hidden) {
        std::fill_n(data, rows * rows, hidden);
        for (size_t row = 0; row < rows; ++row) {
            data[row * rows] = visible;  // the root
            for (int64_t node = static_cast<int64_t>(row) - 1; node >= 0; node = tree.parents[node]) {
                data[row * rows + static_cast<size_t>(node) + 1] = visible;
            }
        }
    };
    if (type == ov::element::f16) {
        fill(mask.data<ov::float16>(), ov::float16(0.0f), ov::float16(-std::numeric_limits<float>::infinity()));
    } else {
        OPENVINO_ASSERT(type == ov::element::f32, "Drafter: eagle_tree_mask must be f32 or f16, got ", type);
        fill(mask.data<float>(), 0.0f, -std::numeric_limits<float>::infinity());
    }
    return mask;
}

}  // namespace drafter_utils

namespace {

using SteadyClock = std::chrono::steady_clock;

double elapsed_ms(SteadyClock::time_point start) {
    return std::chrono::duration<double, std::milli>(SteadyClock::now() - start).count();
}

// NPUW_EAGLE target: the rows the pipeline accepted after a verification step ([num_candidates, num_accepted,
// accepted flag per row]) are read from this state at the next call, which drops the rejected rows from the KV cache.
constexpr const char* SAMPLING_STATE_NAME = "npuw_eagle3_sampling_result";
constexpr size_t SAMPLING_STATE_HEADER = 2;
constexpr const char* FEATURES_OUTPUT = "last_hidden_state";

size_t align_to(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

ov::genai::StreamingStatus stream_tokens(const std::shared_ptr<ov::genai::StreamerBase>& streamer,
                                         const std::vector<int64_t>& tokens) {
    if (streamer && !tokens.empty()) {
        return streamer->write(tokens);
    }
    return ov::genai::StreamingStatus::RUNNING;
}

}  // namespace

struct StatefulDrafterLLMPipeline::Impl {
    std::shared_ptr<Drafter> drafter;
    ov::InferRequest target;
    ov::element::Type tree_mask_type = ov::element::f32;
    size_t max_nodes = 0;
    bool has_features = false;
    // Tokens the KV cache holds before a verification window no longer fits (the generate model keeps room for its
    // whole window).
    size_t kv_capacity = std::numeric_limits<size_t>::max();
    ov::Tensor accepted_features;
    // Passed to the drafter when the main model has no last_hidden_state output.
    const ov::Tensor no_features{ov::element::f32, ov::Shape{0, 0}};

    ov::genai::RawPerfMetrics target_raw, draft_raw;

    void reset_metrics() {
        for (auto* raw : {&target_raw, &draft_raw}) {
            *raw = ov::genai::RawPerfMetrics{};
            raw->m_inference_durations = {MicroSeconds(0.0f)};
            raw->tokenization_durations = {MicroSeconds(0.0f)};
            raw->detokenization_durations = {MicroSeconds(0.0f)};
        }
    }

    static void record(ov::genai::RawPerfMetrics& raw, double ms, size_t batch) {
        raw.m_durations.emplace_back(static_cast<float>(ms * 1000.0));
        raw.m_inference_durations[0] += MicroSeconds(static_cast<float>(ms * 1000.0));
        raw.m_batch_sizes.emplace_back(batch);
    }

    void run_target(const std::vector<int64_t>& ids,
                    const std::vector<int64_t>& positions,
                    size_t attention_len,
                    const ov::Tensor& tree_mask) {
        const size_t n = ids.size();
        ov::Tensor input_ids(ov::element::i64, {1, n});
        ov::Tensor position_ids(ov::element::i64, {1, n});
        ov::Tensor attention_mask(ov::element::i64, {1, attention_len});
        std::copy(ids.begin(), ids.end(), input_ids.data<int64_t>());
        std::copy(positions.begin(), positions.end(), position_ids.data<int64_t>());
        std::fill_n(attention_mask.data<int64_t>(), attention_len, int64_t{1});
        target.set_tensor("input_ids", input_ids);
        target.set_tensor("attention_mask", attention_mask);
        target.set_tensor("position_ids", position_ids);
        target.set_tensor("eagle_tree_mask", tree_mask);
        target.infer();
    }

    /// Greedy token after row `row` of the last `rows` processed rows (NPUW keeps the useful logits rows at the tail).
    int64_t argmax_row(size_t row, size_t rows) {
        const ov::Tensor logits = target.get_tensor("logits");
        const auto& shape = logits.get_shape();
        OPENVINO_ASSERT(shape.size() == 3 && shape[1] >= rows && row < rows,
                        "Drafter: unexpected logits shape ",
                        shape);
        const size_t vocab = shape[2];
        const size_t offset = (shape[1] - rows + row) * vocab;
        if (logits.get_element_type() == ov::element::f16) {
            const ov::float16* data = logits.data<const ov::float16>() + offset;
            return static_cast<int64_t>(std::max_element(data,
                                                         data + vocab,
                                                         [](ov::float16 a, ov::float16 b) {
                                                             return static_cast<float>(a) < static_cast<float>(b);
                                                         }) -
                                        data);
        }
        const float* data = logits.data<const float>() + offset;
        return static_cast<int64_t>(std::max_element(data, data + vocab) - data);
    }

    void set_sampling_result(size_t num_candidates, const std::vector<size_t>& accepted_rows) {
        for (auto& state : target.query_state()) {
            if (state.get_name() != SAMPLING_STATE_NAME) {
                continue;
            }
            ov::Tensor tensor(ov::element::i64, ov::Shape{SAMPLING_STATE_HEADER + num_candidates});
            int64_t* data = tensor.data<int64_t>();
            data[0] = static_cast<int64_t>(num_candidates);
            data[1] = static_cast<int64_t>(accepted_rows.size());
            std::fill_n(data + SAMPLING_STATE_HEADER, num_candidates, int64_t{0});
            for (const size_t row : accepted_rows) {
                data[SAMPLING_STATE_HEADER + row] = 1;
            }
            state.set_state(tensor);
            return;
        }
        OPENVINO_THROW("Drafter: VariableState '",
                       SAMPLING_STATE_NAME,
                       "' not found (the main model must be NPUW_EAGLE)");
    }

    /// Features of the last `rows` processed rows (a view, prompt rows), or an empty tensor.
    ov::Tensor tail_features(size_t rows) {
        if (!has_features) {
            return no_features;
        }
        ov::Tensor all = target.get_tensor(FEATURES_OUTPUT);
        const auto& shape = all.get_shape();
        OPENVINO_ASSERT(shape.size() == 3 && shape[1] >= rows,
                        "Drafter: unexpected ",
                        FEATURES_OUTPUT,
                        " shape ",
                        shape,
                        " for ",
                        rows,
                        " rows");
        const size_t row_bytes = shape[2] * all.get_element_type().size();
        auto* base = static_cast<uint8_t*>(all.data()) + (shape[1] - rows) * row_bytes;
        return ov::Tensor(all.get_element_type(), {rows, shape[2]}, base);
    }

    /// Features of `picked` rows out of the last `rows` processed rows, gathered into one tensor.
    ov::Tensor picked_features(size_t rows, const std::vector<size_t>& picked) {
        if (!has_features) {
            return no_features;
        }
        const ov::Tensor tail = tail_features(rows);
        const size_t width = tail.get_shape()[1];
        if (!accepted_features || accepted_features.get_element_type() != tail.get_element_type() ||
            accepted_features.get_shape()[1] != width || accepted_features.get_shape()[0] < picked.size()) {
            accepted_features = ov::Tensor(tail.get_element_type(), {max_nodes + 1, width});
        }
        const size_t row_bytes = width * tail.get_element_type().size();
        const auto* src = static_cast<const uint8_t*>(tail.data());
        auto* dst = static_cast<uint8_t*>(accepted_features.data());
        for (size_t i = 0; i < picked.size(); ++i) {
            std::memcpy(dst + i * row_bytes, src + picked[i] * row_bytes, row_bytes);
        }
        return ov::Tensor(accepted_features.get_element_type(), {picked.size(), width}, dst);
    }
};

StatefulDrafterLLMPipeline::StatefulDrafterLLMPipeline(const ModelDesc& main_model_desc,
                                                       std::shared_ptr<Drafter> drafter)
    : StatefulSpeculativePipelineBase(main_model_desc.tokenizer, main_model_desc.generation_config),
      m_impl(std::make_unique<Impl>()) {
    auto& impl = *m_impl;
    OPENVINO_ASSERT(drafter, "Drafter: drafter must not be null");
    OPENVINO_ASSERT(main_model_desc.device == "NPU",
                    "Speculative decoding with a drafter runs the main model on NPU, got ",
                    main_model_desc.device);
    impl.drafter = std::move(drafter);
    impl.max_nodes = impl.drafter->max_draft_tokens();

    auto model = main_model_desc.model;
    OPENVINO_ASSERT(model, "Drafter: main model is null");
    utils::eagle3::apply_eagle3_attention_mask_transform(model);
    const size_t window = impl.max_nodes + 1;
    ov::AnyMap properties = main_model_desc.properties;
    properties["NPUW_EAGLE"] = "TRUE";
    properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = window;
    properties["NPUW_SLICE_OUT"] = "NO";
    const auto kv_pos = utils::get_kv_axes_pos(model);
    auto [compiled, kv_desc] = utils::compile_decoder_for_npu(model, properties, kv_pos);
    impl.target = compiled.create_infer_request();
    impl.tree_mask_type = compiled.input("eagle_tree_mask").get_element_type();
    OPENVINO_ASSERT(impl.tree_mask_type == ov::element::f32 || impl.tree_mask_type == ov::element::f16,
                    "Drafter: eagle_tree_mask must be f32 or f16");
    impl.has_features = std::any_of(compiled.outputs().begin(), compiled.outputs().end(), [](const auto& output) {
        return output.get_names().count(FEATURES_OUTPUT) > 0;
    });
    // NPUW aligns a multi-token generate window to 8 rows; its last KV variant holds the prompt and response sizes.
    const size_t generate_rows = window == 1 ? 1 : align_to(window, 8);
    const size_t kv_size = static_cast<size_t>(kv_desc.max_prompt_len) + kv_desc.min_response_len;
    impl.kv_capacity = kv_size > generate_rows ? kv_size - generate_rows : 0;
}

StatefulDrafterLLMPipeline::~StatefulDrafterLLMPipeline() {
    if (m_impl && m_impl->target) {
        m_impl->target.get_compiled_model().release_memory();
    }
}

GenerationConfig StatefulDrafterLLMPipeline::resolve_generation_config(OptionalGenerationConfig generation_config) {
    GenerationConfig config = StatefulSpeculativePipelineBase::resolve_generation_config(std::move(generation_config));
    OPENVINO_ASSERT(!config.do_sample, "Speculative decoding with a drafter supports greedy decoding only");
    // Tokens are picked by the argmax of the main model's logits, so options that change the logits or the output
    // can't be applied.
    const std::pair<bool, const char*> unsupported_options[] = {
        {config.is_beam_search(), "beam search"},
        {config.repetition_penalty != 1.0f, "repetition_penalty"},
        {config.presence_penalty != 0.0f, "presence_penalty"},
        {config.frequency_penalty != 0.0f, "frequency_penalty"},
        {config.min_new_tokens > 0, "min_new_tokens"},
        {!config.stop_strings.empty(), "stop_strings"},
        {config.logprobs > 0, "logprobs"},
        {config.echo, "echo"},
        {config.structured_output_config.has_value(), "structured output"},
        {config.adapters.has_value(), "LoRA adapters"},
    };
    for (const auto& [is_set, option] : unsupported_options) {
        OPENVINO_ASSERT(!is_set, "Speculative decoding with a drafter doesn't support ", option);
    }
    OPENVINO_ASSERT(config.assistant_confidence_threshold == 0.f,
                    "Speculative decoding with a drafter supports num_assistant_tokens only");
    if (!config.num_assistant_tokens.has_value()) {
        config.num_assistant_tokens = m_impl->max_nodes;
    }
    OPENVINO_ASSERT(config.num_assistant_tokens.value() <= m_impl->max_nodes,
                    "num_assistant_tokens (",
                    config.num_assistant_tokens.value(),
                    ") exceeds the drafter's max_draft_tokens() (",
                    m_impl->max_nodes,
                    ")");
    return config;
}

EncodedResults StatefulDrafterLLMPipeline::generate_tokens(const EncodedInputs& inputs,
                                                           const GenerationConfig& config,
                                                           StreamerVariant streamer) {
    auto& impl = *m_impl;
    ManualTimer generate_timer("StatefulDrafterLLMPipeline::generate(EncodedInputs)");
    generate_timer.start();
    const auto streamer_ptr = utils::create_streamer(streamer, m_tokenizer);

    ov::Tensor input_ids;
    if (const auto* tensor_input = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *tensor_input;
    } else if (const auto* tokenized_input = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = tokenized_input->input_ids;
    }
    OPENVINO_ASSERT(input_ids.get_shape().size() == 2 && input_ids.get_shape().at(0) == 1,
                    "Speculative decoding with a drafter supports batch size 1 only");
    const size_t prompt_len = input_ids.get_shape().at(1);
    OPENVINO_ASSERT(prompt_len > 0, "Drafter: empty prompt");
    const int64_t* prompt_data = input_ids.data<const int64_t>();
    const std::vector<int64_t> prompt(prompt_data, prompt_data + prompt_len);

    impl.target.reset_state();
    impl.reset_metrics();
    const size_t node_limit = config.num_assistant_tokens.value_or(impl.max_nodes);
    const size_t max_new_tokens = config.get_max_new_tokens(prompt_len);
    auto is_stop = [&](int64_t token) {
        return !config.ignore_eos && is_stop_token_id_hit(token, config.stop_token_ids);
    };

    // The drafter is told about the end of the generation however the loop ends. A generation that completes calls
    // finish() after the loop, so its errors reach the caller; the guard covers a generation that throws.
    struct FinishGuard {
        Drafter* drafter = nullptr;
        ~FinishGuard() {
            if (drafter) {
                try {
                    drafter->finish();
                } catch (...) {
                    // generate() already throws the error that ended the generation.
                }
            }
        }
    } finish_guard;
    double draft_start_ms = 0.0;
    auto start = SteadyClock::now();
    if (node_limit > 0) {
        impl.drafter->start(prompt, config);
        finish_guard.drafter = impl.drafter.get();
        draft_start_ms = elapsed_ms(start);
    }

    // Prefill.
    ov::Tensor single_row_mask(impl.tree_mask_type, {1, 1, 1, 1});
    std::memset(single_row_mask.data(), 0, single_row_mask.get_byte_size());
    std::vector<int64_t> positions(prompt_len);
    std::iota(positions.begin(), positions.end(), int64_t{0});
    start = SteadyClock::now();
    impl.run_target(prompt, positions, prompt_len, single_row_mask);
    Impl::record(impl.target_raw, elapsed_ms(start), 1);
    std::vector<int64_t> tokens = {impl.argmax_row(0, 1)};
    size_t context = prompt_len;  // tokens in the main model's KV cache

    auto streaming_status = stream_tokens(streamer_ptr, tokens);
    if (finish_guard.drafter) {
        start = SteadyClock::now();
        impl.drafter->accept(prompt, positions, impl.tail_features(prompt_len));
        Impl::record(impl.draft_raw, draft_start_ms + elapsed_ms(start), 0);
    }
    bool stop = is_stop(tokens.back()) || tokens.size() >= max_new_tokens;
    size_t total_accepted = 0, total_proposed = 0;
    std::vector<size_t> accepted_rows;

    while (!stop && streaming_status == ov::genai::StreamingStatus::RUNNING) {
        // The verification window (root + nodes) must fit into the KV cache.
        const size_t room = impl.kv_capacity > context ? impl.kv_capacity - context : 0;
        if (room == 0) {
            break;
        }
        const int64_t root = tokens.back();
        DraftTree tree;
        if (node_limit > 0 && room > 1) {
            start = SteadyClock::now();
            tree = impl.drafter->propose(root, static_cast<int64_t>(context), std::min(node_limit, room - 1));
            Impl::record(impl.draft_raw, elapsed_ms(start), tree.tokens.size());
        }
        const std::vector<size_t> depths = drafter_utils::tree_depths(tree, std::min(node_limit, room - 1));
        const size_t rows = tree.tokens.size() + 1;
        std::vector<int64_t> ids(rows), pos(rows);
        ids[0] = root;
        pos[0] = static_cast<int64_t>(context);
        for (size_t i = 0; i < tree.tokens.size(); ++i) {
            ids[i + 1] = tree.tokens[i];
            pos[i + 1] = static_cast<int64_t>(context + depths[i]);
        }

        // Verify the root and the tree in one call, then walk the tree greedily.
        start = SteadyClock::now();
        impl.run_target(ids, pos, context + rows, drafter_utils::make_tree_mask(tree, impl.tree_mask_type));
        std::vector<int64_t> emitted = drafter_utils::accept_greedy(
            tree,
            [&](size_t row) {
                return impl.argmax_row(row, rows);
            },
            is_stop,
            accepted_rows);
        Impl::record(impl.target_raw, elapsed_ms(start), emitted.size());
        impl.set_sampling_result(rows, accepted_rows);
        if (finish_guard.drafter) {
            std::vector<int64_t> accepted_tokens(accepted_rows.size()), accepted_positions(accepted_rows.size());
            for (size_t i = 0; i < accepted_rows.size(); ++i) {
                accepted_tokens[i] = ids[accepted_rows[i]];
                accepted_positions[i] = pos[accepted_rows[i]];
            }
            impl.drafter->accept(accepted_tokens, accepted_positions, impl.picked_features(rows, accepted_rows));
        }
        context += accepted_rows.size();
        total_accepted += accepted_rows.size() - 1;
        total_proposed += rows - 1;

        const size_t budget = max_new_tokens - tokens.size();
        if (emitted.size() >= budget) {
            emitted.resize(budget);
            stop = true;
        }
        tokens.insert(tokens.end(), emitted.begin(), emitted.end());
        stop = stop || (!emitted.empty() && is_stop(emitted.back()));
        streaming_status = stream_tokens(streamer_ptr, emitted);
    }

    m_streaming_was_cancelled = (streaming_status == ov::genai::StreamingStatus::CANCEL);
    if (streamer_ptr) {
        streamer_ptr->end();
    }
    if (finish_guard.drafter) {
        std::exchange(finish_guard.drafter, nullptr)->finish();
    }

    EncodedResults results;
    results.tokens = {tokens};
    results.scores = {0.0f};
    generate_timer.end();

    m_sd_perf_metrics.num_input_tokens = prompt_len;
    m_sd_perf_metrics.load_time = m_load_time_ms;
    m_sd_perf_metrics.num_accepted_tokens = total_accepted;
    m_sd_perf_metrics.num_draft_tokens = total_proposed;
    m_sd_perf_metrics.raw_metrics.generate_durations.clear();
    m_sd_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_timer.get_duration_microsec());
    m_sd_perf_metrics.m_evaluated = false;
    m_sd_perf_metrics.main_model_metrics.m_evaluated = false;
    m_sd_perf_metrics.draft_model_metrics.m_evaluated = false;
    m_sd_perf_metrics.main_model_metrics.raw_metrics = impl.target_raw;
    m_sd_perf_metrics.draft_model_metrics.raw_metrics = impl.draft_raw;
    if (total_proposed > 0) {
        m_sd_metrics.update_acceptance_rate(0, static_cast<float>(total_accepted) / total_proposed * 100.0f);
        m_sd_metrics.update_draft_accepted_tokens(0, total_accepted);
        m_sd_metrics.update_draft_generated_len(0, total_proposed);
        m_sd_metrics.update_generated_len(tokens.size());
    }
    m_sd_perf_metrics.evaluate_statistics(generate_timer.get_start_time());
    results.perf_metrics = m_sd_perf_metrics;
    results.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_sd_perf_metrics);
    generate_timer.clear();
    return results;
}

}  // namespace ov::genai
