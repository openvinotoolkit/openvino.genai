// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/scheduler_config.hpp"
#include "openvino/openvino.hpp"

namespace ov::genai {

class KVCacheManager;

namespace voxtral_realtime {

/**
 * @brief Number of whole KV-cache blocks at the beginning of a stream that the next decoding step doesn't attend to.
 * A query attends to the last sliding_window tokens including itself, so the first new token still needs the last
 * sliding_window - 1 cached tokens.
 */
size_t count_blocks_outside_window(size_t cached_tokens, size_t sliding_window, size_t block_size);

/**
 * @brief Limit attention of every PagedAttention operation of the model to the last sliding_window tokens.
 * SDPAToPagedAttention drops the sliding-window mask of the original model, so the limit is restored here.
 * @return Number of updated operations.
 */
size_t set_paged_attention_sliding_window(const std::shared_ptr<ov::Model>& model, size_t sliding_window);

struct TextDecoderStepResult {
    int64_t token_id = -1;
    MicroSeconds inference_duration{0.0f};
    MicroSeconds sampling_duration{0.0f};
    TimePoint token_time;
};

/**
 * @brief Voxtral Realtime text decoder on PagedAttention backend, shared by all streams of a pipeline.
 *
 * Voxtral decodes one text position per audio frame, and every position needs the audio embedding of its frame. A
 * stream therefore can't be submitted as a single ContinuousBatchingPipeline request the way Qwen3-ASR chunks are:
 * each Stream::decode() call submits the next positions of its stream instead. Concurrent decode() calls take turns
 * running a step, as Qwen3ASRContinuousBatchingAdapter::infer() does. A step decodes the positions of all active
 * streams in one batch: streams leave the batch between their positions to encode audio, so a step waits for the
 * positions of every stream that announced them with Stream::activate().
 *
 * Every stream owns the KV-cache blocks of its sliding window. Blocks that left the window are returned to the pool
 * shared by the streams.
 */
class PagedTextDecoder : public std::enable_shared_from_this<PagedTextDecoder> {
public:
    class Stream {
    public:
        struct Deactivator {
            void operator()(Stream* stream) const;
        };
        using ActiveScope = std::unique_ptr<Stream, Deactivator>;

        ~Stream();
        Stream(const Stream&) = delete;
        Stream& operator=(const Stream&) = delete;

        /**
         * @brief Announce that the stream keeps calling decode() until the returned scope is destroyed. Decoding
         * steps of other streams wait for the positions of this stream, so the scope must cover only the work
         * between its decode() calls, not the time the stream waits for audio.
         */
        ActiveScope activate();

        /**
         * @brief Decode the next positions of the stream.
         * @param inputs_embeds [positions, hidden_size] sums of text and audio embeddings.
         * @return Token with the highest logit at the last position.
         */
        TextDecoderStepResult decode(const ov::Tensor& inputs_embeds);

        /// Size in bytes of KV-cache blocks owned by the stream. Can be called from any thread.
        size_t get_kv_cache_size() const;
        /// Number of tokens the last decoded position attended to, including itself. Can be called from any thread.
        size_t get_window_size() const;

    private:
        friend class PagedTextDecoder;
        explicit Stream(std::shared_ptr<PagedTextDecoder> decoder);

        std::shared_ptr<PagedTextDecoder> m_decoder;
        // Physical block indices in the order of cached tokens.
        std::vector<int32_t> m_blocks;
        size_t m_cached_tokens = 0;
        // Positions keep growing after blocks are released: RoPE attention depends on position differences only.
        int64_t m_next_position = 0;
        // Guarded by PagedTextDecoder::m_step_mutex.
        bool m_active = false;
        std::atomic<size_t> m_num_blocks{0};
        std::atomic<size_t> m_window_size{0};
    };

    /**
     * @param model Stateful text decoder with inputs_embeds input. It's converted to PagedAttention in place.
     * @param scheduler_config num_kv_blocks or cache_size limit the KV cache shared by the streams. The cache grows
     * on demand if both are zero. Other fields aren't used.
     */
    PagedTextDecoder(const std::shared_ptr<ov::Model>& model,
                     const std::string& device,
                     const ov::AnyMap& properties,
                     const SchedulerConfig& scheduler_config,
                     size_t num_layers,
                     size_t sliding_window,
                     size_t num_delay_tokens);
    ~PagedTextDecoder();

    std::unique_ptr<Stream> create_stream();

private:
    struct Job {
        Stream* stream = nullptr;
        ov::Tensor inputs_embeds;
        TextDecoderStepResult result;
        std::exception_ptr error;
        bool done = false;
    };

    TextDecoderStepResult decode(Stream& stream, const ov::Tensor& inputs_embeds);
    void deactivate(Stream& stream);
    void step(const std::vector<Job*>& jobs);
    /// Release blocks that left the window and reserve blocks for new_tokens.
    void prepare_blocks(Stream& stream, size_t new_tokens);
    void release_blocks(Stream& stream, size_t num_blocks);

    const size_t m_sliding_window;
    ov::InferRequest m_request;
    std::unique_ptr<KVCacheManager> m_cache_manager;
    bool m_has_sampled_tokens_indices = false;

    // Decoding steps must not run concurrently. m_stepping marks a step in progress; m_step_cv is notified after every
    // step so that each waiting decode() call re-checks its own job.
    std::mutex m_step_mutex;
    std::condition_variable m_step_cv;
    bool m_stepping = false;
    std::vector<Job*> m_pending_jobs;
    // Streams that have submitted or are about to submit a job. Each of them has at most one pending job.
    size_t m_num_active_streams = 0;

    // Streams return their blocks from the threads that destroy them.
    std::mutex m_blocks_mutex;
    std::vector<int32_t> m_free_blocks;
    size_t m_num_allocated_blocks = 0;
    size_t m_num_streams = 0;
    // 0 if the cache grows on demand.
    size_t m_max_num_blocks = 0;

    ov::Tensor m_inputs_embeds;
    ov::Tensor m_position_ids;
    ov::Tensor m_past_lens;
    ov::Tensor m_subsequence_begins;
    ov::Tensor m_block_indices;
    ov::Tensor m_block_indices_begins;
    ov::Tensor m_max_context_len;
    ov::Tensor m_sampled_tokens_indices;
};

}  // namespace voxtral_realtime
}  // namespace ov::genai
