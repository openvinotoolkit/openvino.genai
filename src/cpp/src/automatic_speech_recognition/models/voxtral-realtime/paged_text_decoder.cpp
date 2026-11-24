// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/models/voxtral-realtime/paged_text_decoder.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>

#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/paged_attention_transformations.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "utils.hpp"

namespace ov::genai::voxtral_realtime {
namespace {

constexpr char PAGED_ATTENTION_TYPE_NAME[] = "PagedAttentionExtension";
// Position of sliding_window among PagedAttentionExtension inputs.
constexpr size_t SLIDING_WINDOW_INPUT_INDEX = 10;
constexpr char SAMPLED_TOKENS_INDICES_NAME[] = "sampled_tokens_indices";
constexpr size_t CACHE_GROWTH_FACTOR = 2;
constexpr size_t BYTES_IN_GIB = size_t{1024} * 1024 * 1024;

size_t div_up(const size_t value, const size_t divisor) {
    return (value + divisor - 1) / divisor;
}

}  // namespace

size_t count_blocks_outside_window(const size_t cached_tokens, const size_t sliding_window, const size_t block_size) {
    OPENVINO_ASSERT(sliding_window > 0 && block_size > 0, "Sliding window and KV-cache block size must be positive.");
    const size_t attended_tokens = std::min(cached_tokens, sliding_window - 1);
    return (cached_tokens - attended_tokens) / block_size;
}

size_t set_paged_attention_sliding_window(const std::shared_ptr<ov::Model>& model, const size_t sliding_window) {
    size_t num_updated = 0;
    for (const std::shared_ptr<ov::Node>& node : model->get_ops()) {
        if (std::string{node->get_type_name()} != PAGED_ATTENTION_TYPE_NAME) {
            continue;
        }
        OPENVINO_ASSERT(node->get_input_size() > SLIDING_WINDOW_INPUT_INDEX,
                        "PagedAttention operation '",
                        node->get_friendly_name(),
                        "' has no sliding_window input.");
        ov::Input<ov::Node> input = node->input(SLIDING_WINDOW_INPUT_INDEX);
        input.replace_source_output(
            ov::op::v0::Constant::create(input.get_element_type(),
                                         ov::Shape{},
                                         std::vector<int64_t>{static_cast<int64_t>(sliding_window)}));
        ++num_updated;
    }
    return num_updated;
}

PagedTextDecoder::Stream::Stream(std::shared_ptr<PagedTextDecoder> decoder) : m_decoder{std::move(decoder)} {
    const std::lock_guard<std::mutex> lock{m_decoder->m_blocks_mutex};
    ++m_decoder->m_num_streams;
}

PagedTextDecoder::Stream::~Stream() {
    m_decoder->release_blocks(*this, m_blocks.size());
    const std::lock_guard<std::mutex> lock{m_decoder->m_blocks_mutex};
    --m_decoder->m_num_streams;
}

void PagedTextDecoder::Stream::Deactivator::operator()(Stream* stream) const {
    stream->m_decoder->deactivate(*stream);
}

PagedTextDecoder::Stream::ActiveScope PagedTextDecoder::Stream::activate() {
    const std::lock_guard<std::mutex> lock{m_decoder->m_step_mutex};
    OPENVINO_ASSERT(!m_active, "Voxtral text decoder stream is already active.");
    m_active = true;
    ++m_decoder->m_num_active_streams;
    return ActiveScope{this};
}

TextDecoderStepResult PagedTextDecoder::Stream::decode(const ov::Tensor& inputs_embeds) {
    return m_decoder->decode(*this, inputs_embeds);
}

size_t PagedTextDecoder::Stream::get_kv_cache_size() const {
    return m_num_blocks.load(std::memory_order_relaxed) * m_decoder->m_cache_manager->get_block_size_in_bytes();
}

size_t PagedTextDecoder::Stream::get_window_size() const {
    return m_window_size.load(std::memory_order_relaxed);
}

PagedTextDecoder::PagedTextDecoder(const std::shared_ptr<ov::Model>& model,
                                   const std::string& device,
                                   const ov::AnyMap& properties,
                                   const SchedulerConfig& scheduler_config,
                                   const size_t num_layers,
                                   const size_t sliding_window,
                                   const size_t num_delay_tokens)
    : m_sliding_window{sliding_window} {
    OPENVINO_ASSERT(m_sliding_window > 0, "Voxtral text decoder sliding window must be positive.");
    ov::pass::SDPAToPagedAttention().run_on_model(model);
    const size_t num_paged_layers = set_paged_attention_sliding_window(model, m_sliding_window);
    OPENVINO_ASSERT(num_paged_layers == num_layers,
                    "Voxtral text decoder has ",
                    num_layers,
                    " layers, but ",
                    num_paged_layers,
                    " were converted to PagedAttention.");
    model->validate_nodes_and_infer_types();
    utils::apply_gather_before_matmul_transformation(model);

    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(model, device, properties);
    utils::print_compiled_model_properties(compiled_model, "Voxtral text decoder with Paged Attention");
    m_request = compiled_model.create_infer_request();
    m_cache_manager = std::make_unique<KVCacheManager>(m_request);
    for (const ov::Output<const ov::Node>& input : compiled_model.inputs()) {
        m_has_sampled_tokens_indices |= input.get_names().count(SAMPLED_TOKENS_INDICES_NAME) > 0;
    }

    if (scheduler_config.num_kv_blocks > 0) {
        m_max_num_blocks = scheduler_config.num_kv_blocks;
    } else if (scheduler_config.cache_size > 0) {
        m_max_num_blocks = scheduler_config.cache_size * BYTES_IN_GIB / m_cache_manager->get_block_size_in_bytes();
        OPENVINO_ASSERT(m_max_num_blocks > 0, "SchedulerConfig::cache_size is smaller than one KV-cache block.");
    }

    ov::Tensor num_delay_tokens_tensor{ov::element::f32, ov::Shape{1}};
    num_delay_tokens_tensor.data<float>()[0] = static_cast<float>(num_delay_tokens);
    m_request.set_tensor("num_delay_tokens", num_delay_tokens_tensor);

    m_inputs_embeds = ov::Tensor{ov::element::f32, ov::Shape{0, 0}};
    m_position_ids = ov::Tensor{ov::element::i64, ov::Shape{0}};
    m_past_lens = ov::Tensor{ov::element::i32, ov::Shape{0}};
    m_subsequence_begins = ov::Tensor{ov::element::i32, ov::Shape{0}};
    m_block_indices = ov::Tensor{ov::element::i32, ov::Shape{0}};
    m_block_indices_begins = ov::Tensor{ov::element::i32, ov::Shape{0}};
    m_max_context_len = ov::Tensor{ov::element::i32, ov::Shape{}};
    m_sampled_tokens_indices = ov::Tensor{ov::element::i64, ov::Shape{0}};
}

PagedTextDecoder::~PagedTextDecoder() = default;

std::unique_ptr<PagedTextDecoder::Stream> PagedTextDecoder::create_stream() {
    return std::unique_ptr<Stream>{new Stream{shared_from_this()}};
}

TextDecoderStepResult PagedTextDecoder::decode(Stream& stream, const ov::Tensor& inputs_embeds) {
    OPENVINO_ASSERT(inputs_embeds.get_element_type() == ov::element::f32 && inputs_embeds.get_shape().size() == 2 &&
                        inputs_embeds.get_size() > 0,
                    "Voxtral text decoder expects non-empty f32 [positions, hidden_size] input embeddings.");
    Job job;
    job.stream = &stream;
    job.inputs_embeds = inputs_embeds;

    // Concurrent decode() calls take turns: one runs step() for the submitted jobs, and the others wait for that step
    // to end and then re-check their own job.
    std::unique_lock<std::mutex> lock{m_step_mutex};
    // A stream that didn't announce its positions is active for the duration of the call.
    const bool is_activated_by_call = !stream.m_active;
    if (is_activated_by_call) {
        stream.m_active = true;
        ++m_num_active_streams;
    }
    m_pending_jobs.push_back(&job);
    m_step_cv.notify_all();
    while (!job.done) {
        // Wait for the job of every active stream, so that their positions are decoded in one batch.
        if (m_stepping || m_pending_jobs.size() < m_num_active_streams) {
            m_step_cv.wait(lock);
            continue;
        }
        m_stepping = true;
        const std::vector<Job*> jobs = std::move(m_pending_jobs);
        m_pending_jobs.clear();
        lock.unlock();
        std::exception_ptr step_error;
        try {
            step(jobs);
        } catch (...) {
            step_error = std::current_exception();
        }
        lock.lock();
        for (Job* scheduled_job : jobs) {
            if (step_error) {
                scheduled_job->error = step_error;
            }
            scheduled_job->done = true;
        }
        m_stepping = false;
        m_step_cv.notify_all();
    }
    lock.unlock();
    if (is_activated_by_call) {
        deactivate(stream);
    }
    if (job.error) {
        std::rethrow_exception(job.error);
    }
    return job.result;
}

void PagedTextDecoder::deactivate(Stream& stream) {
    {
        const std::lock_guard<std::mutex> lock{m_step_mutex};
        stream.m_active = false;
        --m_num_active_streams;
    }
    // The jobs submitted so far may be all a step was waiting for.
    m_step_cv.notify_all();
}

void PagedTextDecoder::step(const std::vector<Job*>& jobs) {
    const size_t hidden_size = jobs.front()->inputs_embeds.get_shape()[1];
    std::vector<Job*> scheduled_jobs;
    scheduled_jobs.reserve(jobs.size());
    size_t num_tokens = 0;
    size_t num_blocks = 0;
    size_t max_context_len = 0;
    for (Job* job : jobs) {
        // A stream that can't get KV-cache blocks fails without affecting the other streams of the batch.
        try {
            const ov::Shape& shape = job->inputs_embeds.get_shape();
            OPENVINO_ASSERT(shape[1] == hidden_size, "Voxtral streams must share the embedding size.");
            prepare_blocks(*job->stream, shape[0]);
            num_tokens += shape[0];
            num_blocks += job->stream->m_blocks.size();
            max_context_len = std::max(max_context_len, job->stream->m_cached_tokens + shape[0]);
            scheduled_jobs.push_back(job);
        } catch (...) {
            job->error = std::current_exception();
        }
    }
    if (scheduled_jobs.empty()) {
        return;
    }

    const size_t num_streams = scheduled_jobs.size();
    m_inputs_embeds.set_shape({num_tokens, hidden_size});
    m_position_ids.set_shape({num_tokens});
    m_past_lens.set_shape({num_streams});
    m_subsequence_begins.set_shape({num_streams + 1});
    m_block_indices.set_shape({num_blocks});
    m_block_indices_begins.set_shape({num_streams + 1});
    m_sampled_tokens_indices.set_shape({num_streams});

    float* inputs_embeds = m_inputs_embeds.data<float>();
    int64_t* position_ids = m_position_ids.data<int64_t>();
    int32_t* past_lens = m_past_lens.data<int32_t>();
    int32_t* subsequence_begins = m_subsequence_begins.data<int32_t>();
    int32_t* block_indices = m_block_indices.data<int32_t>();
    int32_t* block_indices_begins = m_block_indices_begins.data<int32_t>();
    int64_t* sampled_tokens_indices = m_sampled_tokens_indices.data<int64_t>();
    subsequence_begins[0] = 0;
    block_indices_begins[0] = 0;
    for (size_t index = 0; index < num_streams; ++index) {
        const Job& job = *scheduled_jobs[index];
        const Stream& stream = *job.stream;
        const size_t stream_tokens = job.inputs_embeds.get_shape()[0];
        const size_t token_offset = static_cast<size_t>(subsequence_begins[index]);
        const size_t block_offset = static_cast<size_t>(block_indices_begins[index]);
        std::copy_n(job.inputs_embeds.data<const float>(),
                    stream_tokens * hidden_size,
                    inputs_embeds + token_offset * hidden_size);
        std::iota(position_ids + token_offset, position_ids + token_offset + stream_tokens, stream.m_next_position);
        std::copy(stream.m_blocks.begin(), stream.m_blocks.end(), block_indices + block_offset);
        past_lens[index] = static_cast<int32_t>(stream.m_cached_tokens);
        subsequence_begins[index + 1] = static_cast<int32_t>(token_offset + stream_tokens);
        block_indices_begins[index + 1] = static_cast<int32_t>(block_offset + stream.m_blocks.size());
        sampled_tokens_indices[index] = static_cast<int64_t>(token_offset + stream_tokens - 1);
    }
    m_max_context_len.data<int32_t>()[0] = static_cast<int32_t>(max_context_len);

    m_request.set_tensor("inputs_embeds", m_inputs_embeds);
    m_request.set_tensor("position_ids", m_position_ids);
    m_request.set_tensor("past_lens", m_past_lens);
    m_request.set_tensor("subsequence_begins", m_subsequence_begins);
    m_request.set_tensor("block_indices", m_block_indices);
    m_request.set_tensor("block_indices_begins", m_block_indices_begins);
    m_request.set_tensor("max_context_len", m_max_context_len);
    if (m_has_sampled_tokens_indices) {
        m_request.set_tensor(SAMPLED_TOKENS_INDICES_NAME, m_sampled_tokens_indices);
    }

    const TimePoint infer_start_time = std::chrono::steady_clock::now();
    m_request.infer();
    const TimePoint infer_end_time = std::chrono::steady_clock::now();
    const MicroSeconds inference_duration{PerfMetrics::get_microsec(infer_end_time - infer_start_time)};

    const ov::Tensor logits = m_request.get_tensor("logits");
    OPENVINO_ASSERT(logits.get_element_type() == ov::element::f32, "Voxtral text decoder logits must be f32.");
    const size_t num_rows = logits.get_shape().front();
    const size_t vocab_size = logits.get_shape().back();
    OPENVINO_ASSERT(logits.get_size() == num_rows * vocab_size, "Unexpected Voxtral text decoder logits shape.");
    const float* logits_data = logits.data<const float>();
    for (size_t index = 0; index < num_streams; ++index) {
        Job& job = *scheduled_jobs[index];
        Stream& stream = *job.stream;
        // Without the Gather before the LM head the model returns logits of every position.
        const size_t row = m_has_sampled_tokens_indices ? index : static_cast<size_t>(sampled_tokens_indices[index]);
        OPENVINO_ASSERT(row < num_rows, "Voxtral text decoder returned too few logits.");
        const TimePoint sampling_start_time = std::chrono::steady_clock::now();
        const float* row_data = logits_data + row * vocab_size;
        job.result.token_id = static_cast<int64_t>(std::max_element(row_data, row_data + vocab_size) - row_data);
        job.result.token_time = std::chrono::steady_clock::now();
        job.result.sampling_duration =
            MicroSeconds{PerfMetrics::get_microsec(job.result.token_time - sampling_start_time)};
        job.result.inference_duration = inference_duration;

        const size_t stream_tokens = job.inputs_embeds.get_shape()[0];
        stream.m_cached_tokens += stream_tokens;
        stream.m_next_position += static_cast<int64_t>(stream_tokens);
        stream.m_window_size.store(std::min(static_cast<size_t>(stream.m_next_position), m_sliding_window),
                                   std::memory_order_relaxed);
    }
}

void PagedTextDecoder::prepare_blocks(Stream& stream, const size_t new_tokens) {
    const size_t block_size = m_cache_manager->get_block_size();
    release_blocks(stream, count_blocks_outside_window(stream.m_cached_tokens, m_sliding_window, block_size));

    const size_t required_blocks = div_up(stream.m_cached_tokens + new_tokens, block_size);
    if (required_blocks <= stream.m_blocks.size()) {
        return;
    }
    const size_t missing_blocks = required_blocks - stream.m_blocks.size();
    const std::lock_guard<std::mutex> lock{m_blocks_mutex};
    if (m_free_blocks.size() < missing_blocks) {
        const size_t required_total = m_num_allocated_blocks + missing_blocks - m_free_blocks.size();
        // A stream decoding one position per step holds its window and a partially filled block at most, so the
        // cache isn't grown beyond that for the streams that exist.
        const size_t max_window_blocks = m_num_streams * div_up(m_sliding_window + block_size, block_size);
        size_t new_total =
            std::max(required_total, std::min(m_num_allocated_blocks * CACHE_GROWTH_FACTOR, max_window_blocks));
        if (m_max_num_blocks > 0) {
            new_total = std::min(new_total, m_max_num_blocks);
        }
        OPENVINO_ASSERT(new_total >= required_total,
                        "Voxtral KV cache of ",
                        m_max_num_blocks,
                        " blocks is exhausted. Increase SchedulerConfig::cache_size or num_kv_blocks, or destroy "
                        "streaming sessions that aren't used.");
        m_cache_manager->allocate_cache_if_needed(new_total);
        for (size_t block = m_num_allocated_blocks; block < new_total; ++block) {
            m_free_blocks.push_back(static_cast<int32_t>(block));
        }
        m_num_allocated_blocks = new_total;
    }
    stream.m_blocks.insert(stream.m_blocks.end(), m_free_blocks.end() - missing_blocks, m_free_blocks.end());
    m_free_blocks.resize(m_free_blocks.size() - missing_blocks);
    stream.m_num_blocks.store(stream.m_blocks.size(), std::memory_order_relaxed);
}

void PagedTextDecoder::release_blocks(Stream& stream, const size_t num_blocks) {
    if (num_blocks == 0) {
        return;
    }
    const auto released_end = stream.m_blocks.begin() + num_blocks;
    {
        const std::lock_guard<std::mutex> lock{m_blocks_mutex};
        m_free_blocks.insert(m_free_blocks.end(), stream.m_blocks.begin(), released_end);
    }
    stream.m_blocks.erase(stream.m_blocks.begin(), released_end);
    stream.m_cached_tokens -= std::min(stream.m_cached_tokens, num_blocks * m_cache_manager->get_block_size());
    stream.m_num_blocks.store(stream.m_blocks.size(), std::memory_order_relaxed);
}

}  // namespace ov::genai::voxtral_realtime
