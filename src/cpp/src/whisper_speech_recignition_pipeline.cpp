// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <openvino/openvino.hpp>
#include <variant>

#include "llm_pipeline_base.hpp"
#include "llm_pipeline_static.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_speech_recognition_pipeline.hpp"
#include "text_callback_streamer.hpp"
#include "utils.hpp"
#include "whisper/whisper_models.hpp"

namespace ov {
namespace genai {

std::vector<int64_t> whisper_generate(ov::genai::WhisperGenerationConfig& config,
                                      std::vector<float> pcmf32,
                                      ov::genai::WhisperInitializedModels& models);

class WhisperSpeechRecognitionPipeline::Impl {
    // todo: move to utils
    ov::genai::WhisperGenerationConfig from_config_json_if_exists(const std::filesystem::path& model_path) {
        auto config_file_path = model_path / "generation_config.json";
        if (std::filesystem::exists(config_file_path)) {
            return ov::genai::WhisperGenerationConfig((config_file_path).string());
        } else {
            return ov::genai::WhisperGenerationConfig{};
        }
    }

    std::string m_model_path;
    ov::Core m_core;

public:
    ov::genai::WhisperGenerationConfig m_generation_config;
    ov::genai::WhisperInitializedModels m_models;
    Tokenizer m_tokenizer;
    float m_load_time_ms = 0;

    std::optional<int32_t> m_selected_beam = std::nullopt;

    // Impl(const ov::InferRequest& request,
    //      const ov::genai::Tokenizer& tokenizer,
    //      OptionalWhisperGenerationConfig generation_config = std::nullopt)
    //     : m_model_runner(request) {
    //     WhisperGenerationConfig default_config;
    //     m_generation_config = (generation_config.has_value()) ? *generation_config : default_config;
    // }

    Impl(const std::filesystem::path& model_path,
         const ov::genai::Tokenizer& tokenizer,
         const std::string& device,
         const ov::AnyMap& plugin_config)
        : m_generation_config{from_config_json_if_exists(model_path)},
          m_tokenizer{tokenizer} {
        ov::Core core;
        core.set_property(device, plugin_config);

        m_core = core;
        m_model_path = model_path;

        std::string encoder_model_path = model_path / "openvino_encoder_model.xml";
        std::string decoder_model_path = model_path / "openvino_decoder_model.xml";
        std::string decoder_with_past_model_path = model_path / "openvino_decoder_with_past_model.xml";

        m_models.encoder = core.compile_model(encoder_model_path, device).create_infer_request();
        m_models.decoder_compiled = core.compile_model(decoder_model_path, device);
        m_models.decoder = m_models.decoder_compiled.create_infer_request();
        m_models.decoder_with_past_compiled = core.compile_model(decoder_with_past_model_path, device);
        m_models.decoder_with_past = m_models.decoder_with_past_compiled.create_infer_request();

        // If eos_token_id was not provided, take value
        if (m_generation_config.eos_token_id == -1) {
            m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
        }

        m_model_path = model_path;
    }

    Impl(const std::filesystem::path& model_path, const std::string& device, const ov::AnyMap& plugin_config)
        : Impl{model_path, Tokenizer(model_path.string()), device, plugin_config} {}

    DecodedResults generate(PCMf32AudioDataInput& inputs,
                            OptionalWhisperGenerationConfig generation_config,
                            StreamerVariant streamer) {
        auto start_time = std::chrono::steady_clock::now();
        WhisperGenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;

        auto tokens = ov::genai::whisper_generate(config, inputs, m_models);

        // if (auto input_vector = std::get_if<std::vector<std::string>>(&inputs)) {
        //     encoded_input = m_tokenizer.encode(*input_vector);
        // } else if (auto input_prompt = std::get_if<std::string>(&inputs)) {
        //     std::string& prompt = *input_prompt;
        //     encoded_input = m_tokenizer.encode(prompt);
        // }
        // auto encode_stop_time = std::chrono::steady_clock::now();
        // auto encoded_results = generate(encoded_input, config, streamer);

        // auto decode_start_time = std::chrono::steady_clock::now();
        // DecodedResults decoded_results = {m_tokenizer.decode(encoded_results.tokens), encoded_results.scores};
        // auto decode_stop_time = std::chrono::steady_clock::now();

        // // generate_durations
        // decoded_results.perf_metrics = encoded_results.perf_metrics;

        // auto& raw_counters = decoded_results.perf_metrics.raw_metrics;
        // auto stop_time = std::chrono::steady_clock::now();
        // raw_counters.generate_durations = std::vector<MicroSeconds>();
        // raw_counters.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
        // raw_counters.tokenization_durations.emplace_back(PerfMetrics::get_microsec(encode_stop_time -
        // start_time)); raw_counters.detokenization_durations.emplace_back(
        //     PerfMetrics::get_microsec(decode_stop_time - decode_start_time));

        // decoded_results.perf_metrics.evaluate_statistics(start_time);

        DecodedResults decoded_results{std::vector{m_tokenizer.decode(tokens)}, std::vector{1.f}};
        return decoded_results;
    }

    // todo: move to utils
    ov::genai::OptionalWhisperGenerationConfig get_config_from_map(const ov::AnyMap& config_map) {
        if (config_map.count("generation_config")) {
            return config_map.at("generation_config").as<ov::genai::WhisperGenerationConfig>();
        } else {
            return std::nullopt;
        }
    }

    // EncodedResults generate(const EncodedInputs& inputs,
    //                         OptionalWhisperGenerationConfig generation_config,
    //                         StreamerVariant streamer) override {
    //     auto start_time = std::chrono::steady_clock::now();
    //     ov::Tensor input_ids;
    //     ov::Tensor attention_mask;
    //     if (auto data = std::get_if<ov::Tensor>(&inputs)) {
    //         input_ids = *data;
    //         attention_mask = ov::genai::utils::init_attention_mask(input_ids);
    //     } else if (auto data = std::get_if<TokenizedInputs>(&inputs)) {
    //         input_ids = data->input_ids;
    //         attention_mask = data->attention_mask;
    //     }

    //     WhisperGenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;

    //     // If eos_token_id was not provided, take value from default m_generation_config
    //     if (config.eos_token_id == -1)
    //         config.eos_token_id = m_generation_config.eos_token_id;
    //     config.validate();

    //     std::shared_ptr<StreamerBase> streamer_ptr;
    //     if (auto streamer_obj = std::get_if<std::monostate>(&streamer)) {
    //         streamer_ptr = nullptr;
    //     } else if (auto streamer_obj = std::get_if<std::shared_ptr<StreamerBase>>(&streamer)) {
    //         streamer_ptr = *streamer_obj;
    //     } else if (auto callback = std::get_if<std::function<bool(std::string)>>(&streamer)) {
    //         streamer_ptr = std::make_shared<TextCallbackStreamer>(m_tokenizer, *callback);
    //     }

    //     auto batch_size = input_ids.get_shape().at(0);
    //     if ((batch_size != 1 || !(config.is_greedy_decoding() || config.is_multinomial())) && streamer_ptr) {
    //         OPENVINO_THROW("Currently streaming is possible only with batch size=1 and "
    //                        "only for greedy or multinomial decoding");
    //     }

    //     auto num_inputs = m_model_runner.get_compiled_model().inputs().size();
    //     OPENVINO_ASSERT(num_inputs == 4 || num_inputs == 3,
    //                     "Model should have 3 or 4 inputs: "
    //                     "either (input_ids, attention_mask, beam_idx) or "
    //                     "(input_ids, attention_mask, position_ids, beam_idx) "
    //                     "but you have '" +
    //                         std::to_string(num_inputs) + "' inputs");

    //     size_t kv_cache_len = 0;
    //     ov::Tensor concatenated_attention_mask = attention_mask;

    //     bool position_ids_available = (num_inputs == 4);
    //     std::optional<ov::Tensor> position_ids = std::nullopt;
    //     if (position_ids_available) {
    //         position_ids = ov::Tensor{ov::element::i64, input_ids.get_shape()};
    //         utils::initialize_position_ids(*position_ids, attention_mask, kv_cache_len);
    //     }

    //     ov::genai::EncodedResults result;
    //     if (config.is_greedy_decoding()) {
    //         result = ov::genai::greedy_decoding(m_model_runner,
    //                                             input_ids,
    //                                             concatenated_attention_mask,
    //                                             config,
    //                                             streamer_ptr,
    //                                             position_ids);
    //         m_selected_beam = 0;
    //     } else if (config.is_beam_search()) {
    //         std::tie(result, m_selected_beam) = beam_search(m_model_runner,
    //                                                         input_ids,
    //                                                         concatenated_attention_mask,
    //                                                         config,
    //                                                         position_ids,
    //                                                         m_selected_beam);
    //     } else if (config.is_multinomial()) {
    //         result = multinominal_decoding(m_model_runner,
    //                                        input_ids,
    //                                        concatenated_attention_mask,
    //                                        config,
    //                                        streamer_ptr,
    //                                        position_ids);
    //         m_selected_beam = 0;
    //     } else {
    //         OPENVINO_THROW("No decoding algorithm found for provided configuration parameters.");
    //     }

    //     m_model_runner.reset_state();
    //     m_selected_beam = std::nullopt;
    //     auto stop_time = std::chrono::steady_clock::now();

    //     // If is called without tokenization then that stat will not be reported.
    //     auto& metrics = result.perf_metrics;
    //     metrics.num_input_tokens = batch_size * input_ids.get_shape().at(1);
    //     metrics.load_time = this->m_load_time_ms;
    //     metrics.raw_metrics.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
    //     metrics.evaluate_statistics(start_time);
    //     return result;
    // }
};

DecodedResults WhisperSpeechRecognitionPipeline::generate(PCMf32AudioDataInput inputs,
                                                          OptionalWhisperGenerationConfig generation_config,
                                                          StreamerVariant streamer) {
    return m_impl->generate(inputs, generation_config, streamer);
}

DecodedResults WhisperSpeechRecognitionPipeline::generate(PCMf32AudioDataInput inputs, const ov::AnyMap& config_map) {
    auto config_arg = m_impl->get_config_from_map(config_map);
    WhisperGenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
    config.update_generation_config(config_map);

    return m_impl->generate(inputs, config, utils::get_streamer_from_map(config_map));
}

// EncodedResults WhisperSpeechRecognitionPipeline::generate(const EncodedInputs& inputs,
//                                      OptionalWhisperGenerationConfig generation_config,
//                                      StreamerVariant streamer) {
//     return m_impl->generate(inputs, generation_config, streamer);
// }

// EncodedResults WhisperSpeechRecognitionPipeline::generate(const EncodedInputs& inputs, const ov::AnyMap& config_map)
// {
//     auto config_arg = m_impl->get_config_from_map(config_map);
//     GenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
//     config.update_generation_config(config_map);

//     return m_impl->generate(inputs, config, utils::get_streamer_from_map(config_map));
// }

// std::pair<std::string, Any> streamer(StreamerVariant func) {
//     if (auto streamer_obj = std::get_if<std::shared_ptr<StreamerBase>>(&func)) {
//         return {utils::STREAMER_ARG_NAME, Any::make<std::shared_ptr<StreamerBase>>(*streamer_obj)};
//     } else {
//         auto callback = std::get<std::function<bool(std::string)>>(func);
//         return {utils::STREAMER_ARG_NAME, Any::make<std::function<bool(std::string)>>(callback)};
//     }
// }

// std::pair<std::string, Any> generation_config(const WhisperGenerationConfig& config) {
//     return {utils::CONFIG_ARG_NAME, Any::make<WhisperGenerationConfig>(config)};
// }

}  // namespace genai
}  // namespace ov

ov::genai::WhisperSpeechRecognitionPipeline::WhisperSpeechRecognitionPipeline(
    const ov::InferRequest& request,
    const ov::genai::Tokenizer& tokenizer,
    OptionalWhisperGenerationConfig generation_config) {
    auto start_time = std::chrono::steady_clock::now();
    // m_impl = std::make_unique<WhisperSpeechRecognitionPipeline::Impl>(request, tokenizer, generation_config);
    auto stop_time = std::chrono::steady_clock::now();
    m_impl->m_load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
}

ov::genai::WhisperSpeechRecognitionPipeline::WhisperSpeechRecognitionPipeline(const std::string& model_path,
                                                                              const ov::genai::Tokenizer& tokenizer,
                                                                              const std::string& device,
                                                                              const ov::AnyMap& plugin_config) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl = std::make_unique<WhisperSpeechRecognitionPipeline::Impl>(model_path, tokenizer, device, plugin_config);
    auto stop_time = std::chrono::steady_clock::now();
    m_impl->m_load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
}

ov::genai::WhisperSpeechRecognitionPipeline::WhisperSpeechRecognitionPipeline(const std::string& path,
                                                                              const std::string& device,
                                                                              const ov::AnyMap& config) {
    auto start_time = std::chrono::steady_clock::now();
    m_impl = std::make_unique<WhisperSpeechRecognitionPipeline::Impl>(path, device, config);
    auto stop_time = std::chrono::steady_clock::now();
    m_impl->m_load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
}

ov::genai::WhisperGenerationConfig ov::genai::WhisperSpeechRecognitionPipeline::get_generation_config() const {
    return m_impl->m_generation_config;
}

ov::genai::Tokenizer ov::genai::WhisperSpeechRecognitionPipeline::get_tokenizer() {
    return m_impl->m_tokenizer;
}

void ov::genai::WhisperSpeechRecognitionPipeline::set_generation_config(const WhisperGenerationConfig& config) {
    int64_t default_eos_token_id = m_impl->m_generation_config.eos_token_id;
    m_impl->m_generation_config = config;
    // if eos_token_id was not provided in config forward from default config
    if (config.eos_token_id == -1)
        m_impl->m_generation_config.eos_token_id = default_eos_token_id;

    m_impl->m_generation_config.validate();
}

ov::genai::WhisperSpeechRecognitionPipeline::~WhisperSpeechRecognitionPipeline() = default;
