// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "visual_language/inputs_embedder.hpp"

#include "continuous_batching/cache_manager.hpp"
#include "sampling/sampler.hpp"
#include "continuous_batching/model_runner.hpp"
#include "continuous_batching/scheduler.hpp"
#include "continuous_batching/threaded_streamer.hpp"

namespace ov::genai {

enum class ModelInputType {
    TOKENS,
    EMBEDDINGS
};


/**
 * Base interface for all continuous batching based pipelines
 */
class ContinuousBatchingPipeline::IContinuousBatchingPipeline {
protected:
    Tokenizer m_tokenizer;

    // TODO (mzegla): GenerationConfig is request specific object
    // and pipeline only uses default rng_seed and some special tokens.
    GenerationConfig m_generation_config;

    PipelineMetrics m_pipeline_metrics;

    std::string m_device;

    struct PerfTime {
        float m_paged_attention_time_ms = 0.0f;
        float m_matmul_time_ms = 0.0f;
        float m_infer_total_ms = 0.0f;

        ~PerfTime() {
            // std::cout << "Inference requests aggregated statistic: " << std::endl;
            // std::cout << "Paged attention % of inference execution: " << (m_paged_attention_time_ms / m_infer_total_ms) * 100 << std::endl;
            // std::cout << "MatMul % of inference execution: " << (m_matmul_time_ms / m_infer_total_ms) * 100 << std::endl;
            // std::cout << "Total inference execution secs: " << m_infer_total_ms / 1000. << std::endl;
            // std::cout << std::endl;
        }
    } m_perf;

    bool m_is_chat_conversation = false;
    ChatHistory m_history;
    std::vector<ov::genai::EncodedImage> m_history_images;
    std::vector<size_t> m_history_image_ids;
    std::vector<ov::genai::EncodedVideo> m_history_videos;
    std::vector<size_t> m_history_video_ids;
    size_t m_image_id = 0;
    size_t m_video_id = 0;

    float m_load_time_ms = 0.0f;
    // to access m_load_time_ms
    friend class ContinuousBatchingPipeline;

    ModelInputType m_model_input_type = ModelInputType::TOKENS;
    std::shared_ptr<InputsEmbedder> m_inputs_embedder;
    std::mutex m_embeddings_mutex;

    void stream_tokens(const std::shared_ptr<ThreadedStreamerWrapper>& streamer_ptr, const GenerationHandle& handle);
public:
    GenerationConfig get_config() const;
    void set_config(const GenerationConfig& config);
    PipelineMetrics get_metrics() const;
    Tokenizer get_tokenizer();

    /**
     * Adds requests to awaiting queue using encoded inputs
     */
    virtual GenerationHandle add_request(uint64_t request_id,
                                         const ov::Tensor& input_ids,
                                         const GenerationConfig& sampling_params,
                                         std::optional<ov::Tensor> token_type_ids = std::nullopt) = 0;

    /**
     * Adds request to running queue based on string input
     * This step also performs tokenization's encode
     */
    virtual GenerationHandle add_request(uint64_t request_id,
                                         const std::string& prompt,
                                         const GenerationConfig& sampling_params) = 0;

    /**
     * Adds request to running queue based on string input and vector of images
     * This step also performs tokenization's encode
     */
    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 const std::vector<ov::Tensor>& images,
                                 GenerationConfig sampling_params);

    /**
     * Adds request to running queue based on string input and vector of images and videos
     * This step also performs tokenization's encode
     */
    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 const std::vector<ov::Tensor>& images,
                                 const std::vector<ov::Tensor>& videos,
                                 GenerationConfig sampling_params);

    /**
     * Checks whether server (pipeline) has non-finished requests and step() should be called within a loop
     */
    virtual bool has_non_finished_requests() = 0;

    /**
     * Performs a single inference step of all running (and pulls awaiting) requests
     */
    virtual void step() = 0;

    /**
     * Performs monolitic generation based on encoded prompts
     */
    virtual std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer,
             std::optional<std::vector<ov::Tensor>> token_type_ids = std::nullopt) = 0;

    /**
     * Performs monolitic generation based on text prompts
     */
    std::vector<GenerationResult>
    generate(const std::vector<std::string>& prompts,
             std::vector<GenerationConfig> sampling_params,
             const StreamerVariant& streamer);

    virtual std::vector<VLMDecodedResults>
    generate(
             const std::vector<std::string>& prompts,
             const std::vector<std::vector<ov::Tensor>>& rgbs,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer);

    virtual std::vector<VLMDecodedResults> generate(const std::vector<std::string>& prompts,
                                                    const std::vector<std::vector<ov::Tensor>>& images,
                                                    const std::vector<std::vector<ov::Tensor>>& video,
                                                    const std::vector<GenerationConfig>& sampling_params,
                                                    const StreamerVariant& streamer);

    
    /**
     * Performs monolitic generation based on ChatHistory objects
     */
    std::vector<GenerationResult>
    generate(const std::vector<ChatHistory>& histories,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer);

    /**
     * Starts chat with a given system prompt
     * 
     * In chat scenario prompts passed to `generate` method are accumulated inside the pipeline until `finish_chat` is called
     */
    void start_chat(const std::string& system_message);

    /**
     * Ends chat
     */
    void finish_chat();

    ~IContinuousBatchingPipeline();
};
}
