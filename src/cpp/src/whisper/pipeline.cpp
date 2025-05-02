// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/whisper_pipeline.hpp"

#include <algorithm>
#include <filesystem>
#include <openvino/openvino.hpp>
#include <variant>

#include "openvino/genai/text_streamer.hpp"
#include "utils.hpp"
#include "whisper/context_tokens.hpp"
#include "whisper/models/decoder.hpp"
#include "whisper/whisper.hpp"
#include "whisper/config.hpp"
#include "whisper/feature_extractor.hpp"
#include "whisper/models.hpp"
#include "whisper/pipeline_base.hpp"
#include "whisper/pipeline_static.hpp"

namespace {
ov::genai::OptionalWhisperGenerationConfig get_config_from_map(const ov::AnyMap& config_map) {
    if (config_map.count("generation_config")) {
        return config_map.at("generation_config").as<ov::genai::WhisperGenerationConfig>();
    } else {
        return std::nullopt;
    }
}

ov::InferRequest init_model(ov::CompiledModel& compiled) {
    ov::InferRequest request = compiled.create_infer_request();

    try {
        ov::RemoteContext context = compiled.get_context();
        ov::Shape output_shape = request.get_output_tensor().get_shape();
        ov::RemoteTensor remote = context.create_tensor(ov::element::f32, output_shape);
        request.set_tensor("last_hidden_state", remote);
        return request;
    } catch (const ov::Exception&) {
        return request;
    }
}

}  // namespace

namespace ov {
namespace genai {

class WhisperPipeline::WhisperPipelineStatefulImpl : public WhisperPipeline::WhisperPipelineImplBase {
public:
    WhisperPipelineStatefulImpl(const std::filesystem::path& models_path,
                                const std::string& device,
                                const ov::AnyMap& properties)
        : WhisperPipelineImplBase{models_path},
          m_sampler(m_tokenizer) {
        ov::Core core = utils::singleton_core();

        ov::CompiledModel compiled_model =
            core.compile_model(models_path / "openvino_encoder_model.xml", device, properties);
        ov::genai::utils::print_compiled_model_properties(compiled_model, "whisper encoder model");
        m_encoder = init_model(compiled_model);

        m_decoder = WhisperDecoder::from_path(models_path, device, properties);

        // If eos_token_id was not provided, take value
        if (m_generation_config.eos_token_id == -1) {
            m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
        }

        m_sampler.set_seed(m_generation_config.rng_seed);
    }

    WhisperDecodedResults generate(const RawSpeechInput& raw_speech_input,
                                   OptionalWhisperGenerationConfig generation_config,
                                   const std::shared_ptr<StreamerBase> streamer) override {
        auto start_time = std::chrono::steady_clock::now();
        WhisperGenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;

        // If stop_token_ids were not provided, take value from default m_generation_config
        if (config.stop_token_ids.empty())
            config.stop_token_ids = m_generation_config.stop_token_ids;
        // If eos_token_id was not provided, take value from default m_generation_config
        if (config.eos_token_id == -1)
            config.set_eos_token_id(m_generation_config.eos_token_id);
        config.validate();

        auto [context_tokens, tokenization_duration_microseconds] = prepare_context_tokens(config, m_tokenizer);

        auto generate_result = ov::genai::whisper_generate(config,
                                                           m_model_config,
                                                           context_tokens,
                                                           raw_speech_input,
                                                           m_encoder,
                                                           m_decoder,
                                                           m_feature_extractor,
                                                           streamer,
                                                           m_sampler);
        auto decode_start_time = std::chrono::steady_clock::now();
        WhisperDecodedResults result{std::vector{m_tokenizer.decode(generate_result.output_tokens)}, std::vector{1.f}};
        generate_result.perf_metrics.raw_metrics.detokenization_durations.emplace_back(
            PerfMetrics::get_microsec(std::chrono::steady_clock::now() - decode_start_time));

        result.perf_metrics.raw_metrics.tokenization_durations.emplace_back(tokenization_duration_microseconds);

        result.perf_metrics = generate_result.perf_metrics;
        auto& segments = generate_result.segments;

        if (segments.has_value()) {
            std::vector<WhisperDecodedResultChunk> chunks;
            chunks.reserve((*segments).size());

            for (auto& segment : *segments) {
                decode_start_time = std::chrono::steady_clock::now();
                chunks.push_back(
                    WhisperDecodedResultChunk{segment.m_start, segment.m_end, m_tokenizer.decode(segment.m_tokens)});
                result.perf_metrics.raw_metrics.detokenization_durations.emplace_back(
                    PerfMetrics::get_microsec(std::chrono::steady_clock::now() - decode_start_time));
            }

            result.chunks = chunks;
        }

        auto& metrics = result.perf_metrics;
        metrics.load_time = this->m_load_time_ms;
        auto stop_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
        result.perf_metrics.raw_metrics.tokenization_durations.emplace_back(MicroSeconds(0.0f));
        metrics.evaluate_statistics(start_time);

        return result;
    }

private:
    ov::InferRequest m_encoder;
    std::shared_ptr<ov::genai::WhisperDecoder> m_decoder;
    Sampler m_sampler;
};

OPENVINO_SUPPRESS_DEPRECATED_START
std::pair<std::string, Any> streamer(std::shared_ptr<ChunkStreamerBase> func) {
    return {utils::STREAMER_ARG_NAME, Any::make<std::shared_ptr<ChunkStreamerBase>>(func)};
}
OPENVINO_SUPPRESS_DEPRECATED_END

std::pair<std::string, Any> generation_config(const WhisperGenerationConfig& config) {
    return {utils::CONFIG_ARG_NAME, Any::make<WhisperGenerationConfig>(config)};
}

}  // namespace genai
}  // namespace ov

ov::genai::WhisperPipeline::WhisperPipeline(const std::filesystem::path& models_path,
                                            const std::string& device,
                                            const ov::AnyMap& properties) {
    auto start_time = std::chrono::steady_clock::now();
    if (device == "NPU") {
        m_impl = std::make_unique<StaticWhisperPipeline>(models_path, properties);
    } else {
        m_impl = std::make_unique<WhisperPipelineStatefulImpl>(models_path, device, properties);
    }
    auto stop_time = std::chrono::steady_clock::now();
    m_impl->m_load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
}

ov::genai::WhisperDecodedResults ov::genai::WhisperPipeline::generate(const RawSpeechInput& raw_speech_input,
                                                                      OptionalWhisperGenerationConfig generation_config,
                                                                      StreamerVariant streamer) {
    auto base_streamer = utils::create_streamer(streamer, m_impl->m_tokenizer);

    return m_impl->generate(raw_speech_input, generation_config, base_streamer);
}

ov::genai::WhisperDecodedResults ov::genai::WhisperPipeline::generate(const RawSpeechInput& raw_speech_input,
                                                                      const ov::AnyMap& config_map) {
    auto config_arg = get_config_from_map(config_map);
    WhisperGenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
    config.update_generation_config(config_map);
    StreamerVariant streamer_variant = utils::get_streamer_from_map(config_map);
    const std::shared_ptr<StreamerBase> base_streamer = utils::create_streamer(streamer_variant, m_impl->m_tokenizer);

    return m_impl->generate(raw_speech_input, config, base_streamer);
}

ov::genai::WhisperGenerationConfig ov::genai::WhisperPipeline::get_generation_config() const {
    return m_impl->m_generation_config;
}

ov::genai::Tokenizer ov::genai::WhisperPipeline::get_tokenizer() {
    return m_impl->m_tokenizer;
}

void ov::genai::WhisperPipeline::set_generation_config(const WhisperGenerationConfig& config) {
    int64_t default_eos_token_id = m_impl->m_generation_config.eos_token_id;
    auto default_stop_token_ids = m_impl->m_generation_config.stop_token_ids;
    m_impl->m_generation_config = config;

    // If stop_token_ids were not provided, take value from default config
    if (config.stop_token_ids.empty())
        m_impl->m_generation_config.stop_token_ids = default_stop_token_ids;
    // if eos_token_id was not provided in config forward from default config
    if (config.eos_token_id == -1)
        m_impl->m_generation_config.set_eos_token_id(default_eos_token_id);

    m_impl->m_generation_config.validate();
}

ov::genai::WhisperPipeline::~WhisperPipeline() = default;

ov::genai::ChunkStreamerBase::~ChunkStreamerBase() = default;
