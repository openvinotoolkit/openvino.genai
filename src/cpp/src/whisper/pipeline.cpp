// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <filesystem>
#include <openvino/openvino.hpp>
#include <utility>
#include <variant>

#include "openvino/genai/text_streamer.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "utils.hpp"
#include "whisper/config.hpp"
#include "whisper/context_tokens.hpp"
#include "whisper/feature_extractor.hpp"
#include "whisper/models.hpp"
#include "whisper/models/decoder.hpp"
#include "whisper/pipeline_base.hpp"
#include "whisper/pipeline_static.hpp"
#include "whisper/whisper.hpp"
#include "whisper/word_level_timestamps.hpp"
#include "whisper_utils.hpp"

namespace {
ov::genai::OptionalWhisperGenerationConfig get_config_from_map(const ov::AnyMap& config_map) {
    if (config_map.count("generation_config")) {
        return config_map.at("generation_config").as<ov::genai::WhisperGenerationConfig>();
    } else {
        return std::nullopt;
    }
}

void erase_whisper_generation_config_keys(ov::AnyMap& config_map) {
    config_map.erase("word_timestamps");
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

void reshape_to_static_encoder(std::shared_ptr<ov::Model> model, const size_t batch_size, const size_t feature_size) {
    std::map<std::string, ov::PartialShape> new_shapes;
    for (auto input : model->inputs()) {
        const auto& input_name = input.get_any_name();
        ov::PartialShape new_shape;
        if (input_name.find("input_features") != std::string::npos) {
            const auto& partial_shape = input.get_partial_shape();
            OPENVINO_ASSERT(partial_shape.size() >= 3);
            new_shape = partial_shape;
            new_shape[0] = batch_size;  // batch_dim
            new_shape[1] = feature_size;
            new_shapes.emplace(input_name, new_shape);
        }
    }
    model->reshape(new_shapes);
}

}  // namespace

namespace ov {
namespace genai {

namespace {

class WhisperPipelineStatefulImpl : public WhisperPipelineImplBase {
public:
    WhisperPipelineStatefulImpl(const std::filesystem::path& models_path,
                                const std::string& device,
                                const ov::AnyMap& properties)
        : WhisperPipelineImplBase{models_path},
          m_device(device),
          m_sampler(m_tokenizer) {
        ov::AnyMap properties_copy = properties;
        m_generation_config.update_generation_config(properties_copy);
        erase_whisper_generation_config_keys(properties_copy);

        ov::Core core = utils::singleton_core();
        ov::CompiledModel compiled_model;
        if (device == "NPU") {
            auto encoder_model =
                core.read_model(models_path / "openvino_encoder_model.xml", {}, std::as_const(properties_copy));
            // NB: only batch_size == 1 is supported now for NPU
            reshape_to_static_encoder(encoder_model, 1, m_feature_extractor.feature_size);
            compiled_model = core.compile_model(encoder_model, "NPU", properties_copy);

            // WA: Push eos_token to NPUW to stop infer() if KVcache is full
            auto eos_token = m_generation_config.eos_token_id == -1 ? m_tokenizer.get_eos_token_id()
                                                                    : m_generation_config.eos_token_id;
            properties_copy.insert({"WHISPER_EOS_TOKEN", eos_token});

            if (m_generation_config.word_timestamps) {
                properties_copy.insert({"WHISPER_DECOMPOSE_SDPA", m_generation_config.word_timestamps});
            }
        } else {
            compiled_model = core.compile_model(models_path / "openvino_encoder_model.xml", device, properties_copy);
        }

        ov::genai::utils::print_compiled_model_properties(compiled_model, "whisper encoder model");
        m_encoder = init_model(compiled_model);
        if (device == "CPU") {
            m_inference_precision = compiled_model.get_property(ov::hint::inference_precision);
        }

        const bool decompose_cross_attention_spda_ops = m_generation_config.word_timestamps;
        m_decoder =
            WhisperDecoder::from_path(models_path,
                                      device,
                                      properties_copy,
                                      m_encoder.get_compiled_model().output("last_hidden_state").get_partial_shape(),
                                      decompose_cross_attention_spda_ops,
                                      m_inference_precision);

        // If eos_token_id was not provided, take value
        if (m_generation_config.eos_token_id == -1) {
            m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
        }

        m_sampler.set_seed(m_generation_config.rng_seed);
    }

    std::vector<WhisperDecodedResults> generate(const AudiosView& audios,
                                                OptionalWhisperGenerationConfig generation_config,
                                                const std::shared_ptr<StreamerBase> streamer) override {
        OPENVINO_ASSERT(!audios.empty(), "Whisper pipeline expects at least one audio input, got an empty batch.");
        auto start_time = std::chrono::steady_clock::now();
        WhisperGenerationConfig config = utils::prepare_per_generate_config(m_generation_config, generation_config);
        if (audios.size() > 1) {
            validate_batch_request(config, streamer, audios.size());
        }

        auto [context_tokens, tokenization_duration_microseconds] = prepare_context_tokens(config, m_tokenizer);

        auto generate_results = ov::genai::whisper_generate(config,
                                                            m_model_config,
                                                            context_tokens,
                                                            audios,
                                                            m_encoder,
                                                            m_decoder,
                                                            m_feature_extractor,
                                                            streamer,
                                                            m_sampler,
                                                            m_tokenizer);

        WhisperPerfMetrics metrics = std::move(generate_results.front().perf_metrics);
        std::vector<WhisperDecodedResults> results;
        results.reserve(generate_results.size());
        for (auto& generate_result : generate_results) {
            auto decode_start_time = std::chrono::steady_clock::now();
            WhisperDecodedResults result{std::vector{m_tokenizer.decode(generate_result.output_tokens)},
                                         std::vector{1.f}};
            metrics.raw_metrics.detokenization_durations.emplace_back(
                PerfMetrics::get_microsec(std::chrono::steady_clock::now() - decode_start_time));
            result.language = std::move(generate_result.language);
            result.words = std::move(generate_result.words);

            if (generate_result.segments.has_value()) {
                std::vector<WhisperDecodedResultChunk> chunks;
                chunks.reserve(generate_result.segments->size());
                for (const auto& segment : *generate_result.segments) {
                    decode_start_time = std::chrono::steady_clock::now();
                    chunks.push_back(WhisperDecodedResultChunk{segment.m_start,
                                                               segment.m_end,
                                                               m_tokenizer.decode(segment.m_tokens)});
                    metrics.raw_metrics.detokenization_durations.emplace_back(
                        PerfMetrics::get_microsec(std::chrono::steady_clock::now() - decode_start_time));
                }
                result.chunks = std::move(chunks);
            }
            results.push_back(std::move(result));
        }

        metrics.load_time = this->m_load_time_ms;
        metrics.raw_metrics.tokenization_durations.emplace_back(tokenization_duration_microseconds);
        metrics.raw_metrics.generate_durations.emplace_back(
            PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start_time));
        metrics.evaluate_statistics(start_time);
        for (auto& result : results) {
            result.perf_metrics = metrics;
        }
        return results;
    }

private:
    void validate_batch_request(const WhisperGenerationConfig& config,
                                const std::shared_ptr<StreamerBase>& streamer,
                                const size_t batch_size) const {
        OPENVINO_ASSERT(m_device == "CPU",
                        "Batched Whisper generation is supported on CPU only. Got device '",
                        m_device,
                        "' for ",
                        batch_size,
                        " audio inputs.");
        OPENVINO_ASSERT(WhisperDecoder::is_batch_validated_precision(m_inference_precision),
                        "Batched Whisper generation is validated only with f32 inference precision, but the "
                        "effective inference precision is ",
                        m_inference_precision,
                        ". Create the pipeline with ov::hint::inference_precision(ov::element::f32) "
                        "(INFERENCE_PRECISION_HINT='f32') or generate one audio at a time.");
        OPENVINO_ASSERT(m_decoder->supports_dynamic_batch(),
                        "Batched Whisper generation is not supported for this decoder model: it requires f32 "
                        "cross-attention states consumed only by scaled dot-product attention.");
        OPENVINO_ASSERT(!streamer,
                        "Streaming is only supported with a single audio input. Got ",
                        batch_size,
                        " audio inputs.");
        OPENVINO_ASSERT(!config.word_timestamps,
                        "'word_timestamps' is not supported for batched generation. Got ",
                        batch_size,
                        " audio inputs.");
        OPENVINO_ASSERT(config.num_beams == 1,
                        "Batched Whisper generation supports greedy decoding only. Got num_beams = ",
                        config.num_beams,
                        ".");
        OPENVINO_ASSERT(!config.do_sample,
                        "Batched Whisper generation supports greedy decoding only. Multinomial sampling "
                        "('do_sample') is not supported for ",
                        batch_size,
                        " audio inputs.");
    }

    std::string m_device;
    ov::element::Type m_inference_precision = ov::element::dynamic;
    ov::InferRequest m_encoder;
    std::shared_ptr<ov::genai::WhisperDecoder> m_decoder;
    Sampler m_sampler;
};

}  // namespace

std::unique_ptr<WhisperPipelineImplBase> create_whisper_pipeline_impl(const std::filesystem::path& models_path,
                                                                      const std::string& device,
                                                                      const ov::AnyMap& properties) {
    if (device == "NPU") {
        auto properties_copy = properties;
        const bool use_static_pipeline = utils::pop_or_default(properties_copy, "STATIC_PIPELINE", false);
        if (!use_static_pipeline) {
            return std::make_unique<WhisperPipelineStatefulImpl>(models_path, device, properties_copy);
        }
        return std::make_unique<StaticWhisperPipeline>(models_path, properties_copy);
    }
    return std::make_unique<WhisperPipelineStatefulImpl>(models_path, device, properties);
}

std::pair<std::string, Any> generation_config(const WhisperGenerationConfig& config) {
    return {utils::CONFIG_ARG_NAME, Any::make<WhisperGenerationConfig>(config)};
}

}  // namespace genai
}  // namespace ov

ov::genai::WhisperPipeline::WhisperPipeline(const std::filesystem::path& models_path,
                                            const std::string& device,
                                            const ov::AnyMap& properties) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl = create_whisper_pipeline_impl(models_path, device, properties);
    auto stop_time = std::chrono::steady_clock::now();
    m_impl->m_load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
}

ov::genai::WhisperDecodedResults ov::genai::WhisperPipeline::generate(const RawSpeechInput& raw_speech_input,
                                                                      OptionalWhisperGenerationConfig generation_config,
                                                                      StreamerVariant streamer) {
    auto base_streamer = utils::create_streamer(streamer, m_impl->m_tokenizer);

    auto results = m_impl->generate(AudiosView{std::cref(raw_speech_input)}, generation_config, base_streamer);
    return std::move(results.front());
}

ov::genai::WhisperDecodedResults ov::genai::WhisperPipeline::generate(const RawSpeechInput& raw_speech_input,
                                                                      const ov::AnyMap& config_map) {
    auto config_arg = get_config_from_map(config_map);
    WhisperGenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
    config.update_generation_config(config_map);
    StreamerVariant streamer_variant = utils::get_streamer_from_map(config_map);
    const std::shared_ptr<StreamerBase> base_streamer = utils::create_streamer(streamer_variant, m_impl->m_tokenizer);

    auto results = m_impl->generate(AudiosView{std::cref(raw_speech_input)}, config, base_streamer);
    return std::move(results.front());
}

std::vector<ov::genai::WhisperDecodedResults> ov::genai::WhisperPipeline::generate(
    const std::vector<RawSpeechInput>& raw_speech_inputs,
    OptionalWhisperGenerationConfig generation_config,
    StreamerVariant streamer) {
    auto base_streamer = utils::create_streamer(streamer, m_impl->m_tokenizer);

    return m_impl->generate(AudiosView(raw_speech_inputs.begin(), raw_speech_inputs.end()),
                            generation_config,
                            base_streamer);
}

std::vector<ov::genai::WhisperDecodedResults> ov::genai::WhisperPipeline::generate(
    const std::vector<RawSpeechInput>& raw_speech_inputs,
    const ov::AnyMap& config_map) {
    auto config_arg = get_config_from_map(config_map);
    WhisperGenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
    config.update_generation_config(config_map);
    StreamerVariant streamer_variant = utils::get_streamer_from_map(config_map);
    const std::shared_ptr<StreamerBase> base_streamer = utils::create_streamer(streamer_variant, m_impl->m_tokenizer);

    return m_impl->generate(AudiosView(raw_speech_inputs.begin(), raw_speech_inputs.end()), config, base_streamer);
}

ov::genai::WhisperGenerationConfig ov::genai::WhisperPipeline::get_generation_config() const {
    return m_impl->m_generation_config;
}

ov::genai::Tokenizer ov::genai::WhisperPipeline::get_tokenizer() {
    return m_impl->m_tokenizer;
}

void ov::genai::WhisperPipeline::set_generation_config(const WhisperGenerationConfig& config) {
    m_impl->set_generation_config(config);
}

ov::genai::WhisperPipeline::~WhisperPipeline() = default;
