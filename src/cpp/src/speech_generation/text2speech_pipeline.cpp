// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <chrono>

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
                                          const ov::genai::Tokenizer& tokenizer,
                                          const ov::genai::SchedulerConfig& scheduler_config)
        : m_models_path(models_path),
          m_device(device),
          m_properties(properties),
          m_tokenizer(tokenizer),
          m_cb_impl(std::make_shared<ov::genai::Qwen3TTSContinuousBatchingPipelineImpl>(
              models_path,
              scheduler_config,
              device,
              properties)) {}

    ov::genai::Text2SpeechDecodedResults generate(const std::vector<std::string>& texts,
                                                  const ov::Tensor& speaker_embedding,
                                                  const ov::genai::SpeechGenerationConfig& generation_config) override {
        if (requires_sdpa(generation_config)) {
            return get_sdpa_impl().generate(texts, speaker_embedding, generation_config);
        }

        ov::genai::Text2SpeechDecodedResults result;
        result.output_sample_rate = m_cb_impl->get_output_sample_rate();

        const auto generation_start = std::chrono::steady_clock::now();
        std::vector<ov::genai::SpeechGenerationHandle> handles;
        handles.reserve(texts.size());

        uint64_t request_id = 1;
        for (const auto& text : texts) {
            handles.push_back(m_cb_impl->add_request(request_id++, text, speaker_embedding, generation_config));
        }

        while (m_cb_impl->has_non_finished_requests()) {
            m_cb_impl->step();
        }

        for (auto& handle : handles) {
            auto output = handle->read_all();
            result.perf_metrics.num_generated_samples += output.waveform.get_size();
            result.speeches.push_back(std::move(output.waveform));
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
    bool requires_sdpa(const ov::genai::SpeechGenerationConfig& generation_config) const {
        return !generation_config.ref_text.empty() || static_cast<bool>(generation_config.ref_audio) ||
               static_cast<bool>(generation_config.ref_codec_ids);
    }

    ov::genai::Qwen3TTSImpl& get_sdpa_impl() {
        if (!m_sdpa_impl) {
            m_sdpa_impl = std::make_shared<ov::genai::Qwen3TTSImpl>(m_models_path, m_device, m_properties, m_tokenizer);
        }
        return *m_sdpa_impl;
    }

    std::filesystem::path m_models_path;
    std::string m_device;
    ov::AnyMap m_properties;
    ov::genai::Tokenizer m_tokenizer;
    std::shared_ptr<ov::genai::Qwen3TTSContinuousBatchingPipelineImpl> m_cb_impl;
    std::shared_ptr<ov::genai::Qwen3TTSImpl> m_sdpa_impl;
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
        if (is_npu_requested || attention_backend == ov::genai::SDPA_BACKEND) {
            m_impl = std::make_shared<Qwen3TTSImpl>(root_dir, device, pipeline_properties, tokenizer);
        } else if (explicitly_requires_pa) {
            auto [device_properties, scheduler_config] = utils::extract_scheduler_config(
                pipeline_properties,
                utils::get_latency_oriented_scheduler_config());
            m_impl = std::make_shared<Qwen3TTSSyncContinuousBatchingAdapter>(
                root_dir,
                device,
                device_properties,
                tokenizer,
                scheduler_config);
        } else {
            try {
                m_impl = std::make_shared<Qwen3TTSSyncContinuousBatchingAdapter>(
                    root_dir,
                    device,
                    pipeline_properties,
                    tokenizer,
                    utils::get_latency_oriented_scheduler_config());
            } catch (const ov::Exception& exception) {
                utils::log_paged_attention_fallback(exception);
                m_impl = std::make_shared<Qwen3TTSImpl>(root_dir, device, pipeline_properties, tokenizer);
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
