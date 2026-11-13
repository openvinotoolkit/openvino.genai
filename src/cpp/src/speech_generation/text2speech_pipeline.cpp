// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

#include <atomic>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iostream>

#include "json_utils.hpp"
#include "kokoro_tts_model.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_continuous_batching_pipeline_impl.hpp"
#include "speech_generation/qwen3_tts/qwen3_tts_model.hpp"
#include "speecht5_tts_model.hpp"
#include "utils.hpp"

namespace {

const std::string get_class_name(const std::filesystem::path& root_dir) {
    const std::filesystem::path model_index_path = root_dir / "config.json";
    std::ifstream file(model_index_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using ov::genai::utils::read_json_param;

    std::string arch = "none";
    if (data.contains("architectures") && data["architectures"].is_array() &&
        data["architectures"].get<std::vector<std::string>>().size() > 0) {
        arch = data["architectures"].get<std::vector<std::string>>()[0];
    }

    return arch;
}

enum class SpeechBackend {
    SpeechT5,
    Kokoro,
    Qwen3TTS,
};

SpeechBackend resolve_backend(const std::filesystem::path& root_dir,
                             const std::string& class_name) {
    if (class_name == "SpeechT5ForTextToSpeech") {
        return SpeechBackend::SpeechT5;
    }

    const bool architecture_mentions_qwen3_tts = class_name.find("Qwen3TTS") != std::string::npos;

    if (architecture_mentions_qwen3_tts) {
        return SpeechBackend::Qwen3TTS;
    }

    const bool has_openvino_model = std::filesystem::exists(root_dir / "openvino_model.xml");
    const bool architecture_mentions_kokoro = class_name.find("Kokoro") != std::string::npos;

    bool has_kokoro_vocab = false;
    const std::filesystem::path config_json_path = root_dir / "config.json";
    if (std::filesystem::exists(config_json_path)) {
        std::ifstream config_file(config_json_path);
        if (config_file.is_open()) {
            const nlohmann::json config = nlohmann::json::parse(config_file, nullptr, false);
            has_kokoro_vocab = !config.is_discarded() && config.contains("vocab") && config["vocab"].is_object();
        }
    }

    if (has_openvino_model && (architecture_mentions_kokoro || has_kokoro_vocab)) {
        return SpeechBackend::Kokoro;
    }

    OPENVINO_THROW("Unsupported text to speech generation pipeline '", class_name,
                   "'. Unable to auto-detect a supported backend from model metadata/files. "
                   "Expected SpeechT5 architecture metadata or Kokoro-specific model layout");
}

class Qwen3TTSSyncContinuousBatchingAdapter : public ov::genai::Text2SpeechPipelineImpl {
public:
    Qwen3TTSSyncContinuousBatchingAdapter(const std::filesystem::path& models_path,
                                          const std::string& device,
                                          const ov::AnyMap& properties,
                                          const ov::genai::SchedulerConfig& scheduler_config)
                : m_cb_impl(std::make_shared<ov::genai::Qwen3TTSContinuousBatchingPipelineImpl>(
              models_path,
              scheduler_config,
              device,
              properties)) {}

    ov::genai::Text2SpeechDecodedResults generate(const std::vector<std::string>& texts,
                                                  const ov::Tensor& speaker_embedding,
                                                  const ov::genai::SpeechGenerationConfig& generation_config) override {
        ov::genai::Text2SpeechDecodedResults result;
        result.output_sample_rate = m_cb_impl->get_output_sample_rate();

        const auto generation_start = std::chrono::steady_clock::now();
        std::vector<ov::genai::SpeechGenerationHandle> handles;
        handles.reserve(texts.size());
        ov::Tensor shared_speaker_embedding = speaker_embedding;
        ov::genai::SpeechGenerationConfig shared_generation_config = generation_config;
        if (!texts.empty()) {
            m_cb_impl->prepare_shared_voice_inputs(shared_speaker_embedding, shared_generation_config);
        }

        for (const auto& text : texts) {
            const uint64_t request_id = m_next_request_id.fetch_add(1, std::memory_order_relaxed);
            handles.push_back(
                m_cb_impl->add_request(request_id, text, shared_speaker_embedding, shared_generation_config));
        }

        while (m_cb_impl->has_non_finished_requests()) {
            m_cb_impl->step();
        }

        for (auto& handle : handles) {
            OPENVINO_ASSERT(
                handle->get_status() != ov::genai::GenerationStatus::IGNORED,
                "Qwen3-TTS request ran out of KV-cache memory. "
                "Increase SchedulerConfig.cache_size or reduce the batch size.");
            auto output = handle->read_all();
            result.perf_metrics.num_generated_samples += output.waveform.get_size();
            result.speeches.push_back(std::move(output.waveform));
            if (output.speaker_embedding) {
                result.speaker_embedding = output.speaker_embedding;
            }
            if (output.ref_codec_ids) {
                result.ref_codec_ids = output.ref_codec_ids;
            }
        }

        const auto generation_end = std::chrono::steady_clock::now();
        result.perf_metrics.raw_metrics.generate_durations.emplace_back(
            ov::genai::PerfMetrics::get_microsec(generation_end - generation_start));
        result.perf_metrics.evaluate_statistics();
        m_perf_metrics = result.perf_metrics;
        return result;
    }

    ov::Shape get_speaker_embedding_shape() const override {
        return m_cb_impl->get_speaker_embedding_shape();
    }

private:
    std::shared_ptr<ov::genai::Qwen3TTSContinuousBatchingPipelineImpl> m_cb_impl;
    std::atomic<uint64_t> m_next_request_id{1};
};

}  // namespace

namespace ov {
namespace genai {

Text2SpeechPipeline::Text2SpeechPipeline(const std::filesystem::path& root_dir,
                                         const std::string& device,
                                         const ov::AnyMap& properties)
    : m_speech_gen_config(utils::from_config_json_if_exists<SpeechGenerationConfig>(root_dir)) {
    const std::string class_name = get_class_name(root_dir);
    const bool is_npu_requested = ov::genai::utils::is_npu_requested(device, properties);
    const auto [pipeline_properties, attention_backend] = utils::extract_attention_backend(properties, is_npu_requested);

    const auto backend = resolve_backend(root_dir, class_name);
    if (backend == SpeechBackend::SpeechT5) {
        auto tokenizer = ov::genai::Tokenizer(root_dir);
        m_impl = std::make_shared<SpeechT5TTSImpl>(root_dir, device, pipeline_properties, tokenizer);
    } else if (backend == SpeechBackend::Qwen3TTS) {
        auto tokenizer = ov::genai::Tokenizer(root_dir);
        const bool explicitly_requires_pa = utils::explicitly_requires_paged_attention(properties, is_npu_requested);
        OPENVINO_ASSERT(!is_npu_requested || !explicitly_requires_pa,
                         "Qwen3-TTS PagedAttention is not supported on device '", device,
                         "'. Use ATTENTION_BACKEND='SDPA' without PagedAttention-specific properties.");
        if (is_npu_requested || attention_backend == ov::genai::SDPA_BACKEND) {
            m_impl = std::make_shared<Qwen3TTSImpl>(root_dir, device, pipeline_properties, tokenizer);
            std::cout << "[Qwen3-TTS] Selected SDPA backend ("
                      << (is_npu_requested ? "NPU requested" : "SDPA requested") << ")." << std::endl;
        } else if (explicitly_requires_pa) {
            auto [device_properties, scheduler_config] = utils::extract_scheduler_config(
                pipeline_properties,
                utils::get_latency_oriented_scheduler_config());
            m_impl = std::make_shared<Qwen3TTSSyncContinuousBatchingAdapter>(
                root_dir,
                device,
                device_properties,
                scheduler_config);
            std::cout << "[Qwen3-TTS] Selected PA backend (explicit PA requirement)." << std::endl;
        } else {
            try {
                m_impl = std::make_shared<Qwen3TTSSyncContinuousBatchingAdapter>(
                    root_dir,
                    device,
                    pipeline_properties,
                    utils::get_latency_oriented_scheduler_config());
                std::cout << "[Qwen3-TTS] Selected PA backend (default)." << std::endl;
            } catch (const ov::Exception& exception) {
                utils::log_paged_attention_fallback(exception);
                m_impl = std::make_shared<Qwen3TTSImpl>(root_dir, device, pipeline_properties, tokenizer);
                std::cout << "[Qwen3-TTS] Selected SDPA backend (PA initialization failed)." << std::endl;
            }
        }
    } else {
        m_impl = std::make_shared<KokoroTTSImpl>(root_dir, device, pipeline_properties);
    }
}

Text2SpeechDecodedResults Text2SpeechPipeline::generate(const std::vector<std::string>& texts,
                                                        const ov::Tensor& speaker_embedding,
                                                        const ov::AnyMap& properties) {
    SpeechGenerationConfig request_config = m_speech_gen_config;
    request_config.update_generation_config(properties);
    request_config.validate();
    return m_impl->generate(texts, speaker_embedding, request_config);
}

Text2SpeechDecodedResults Text2SpeechPipeline::generate(
    const std::vector<std::string>& texts,
    const ov::Tensor& speaker_embedding,
    const std::optional<SpeechGenerationConfig>& generation_config) {
    const SpeechGenerationConfig request_config = generation_config.has_value() ? *generation_config : m_speech_gen_config;
    request_config.validate();
    return m_impl->generate(texts, speaker_embedding, request_config);
}

SpeechGenerationConfig Text2SpeechPipeline::get_generation_config() const {
    return m_speech_gen_config;
}

void Text2SpeechPipeline::set_generation_config(const SpeechGenerationConfig& new_config) {
    m_speech_gen_config = new_config;
}

ov::Shape Text2SpeechPipeline::get_speaker_embedding_shape() const {
    return m_impl->get_speaker_embedding_shape();
}

}  // namespace genai
}  // namespace ov
