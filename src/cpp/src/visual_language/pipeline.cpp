// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <random>

#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/text_streamer.hpp"

#include "visual_language/vlm_config.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "visual_language/embedding_model.hpp"
#include "visual_language/pipeline_base.hpp"
#include "visual_language/continuous_batching_adapter.hpp"

#include "sampler.hpp"
#include "utils.hpp"
#include "lm_encoding.hpp"

using namespace ov::genai;

class VLMPipeline::VLMPipelineImpl : public VLMPipelineBase{
    // A config to follow for text generation.
    GenerationConfig m_generation_config;
    // A tokenizer encoding a prompt.
    Tokenizer m_tokenizer;
    // A model to compute token embeddings.
    // Input shape: [N, conversation length].
    // Output shape: [1, conversation length, hidden_size].
    EmbeddingsModel m_embedding;
    // A language model used to generate a response.
    // Input shapes: inputs_embeds[N, conversation length, hidden_size],
    // position_ids[N, conversation length], beam_idx[N].
    // Output shape: logits[N, conversation length, vocab_size].
    ov::InferRequest m_language;
    // True if chat mode is activated to save conversation
    // history between generate() calls.
    bool m_is_chat_conversation = false;
    // InputsEmbedder
    std::shared_ptr<InputsEmbedder> m_inputs_embedder;
    // Component for applying sampling to lm outputs
    Sampler m_sampler;
    size_t m_max_kv_cache_size = std::numeric_limits<size_t>::max();
    bool m_is_npu = false;
public:
    VLMPipelineImpl(
        const std::filesystem::path& models_dir,
        const std::string& device,
        const ov::AnyMap& properties
    ) :
        m_generation_config{
            utils::from_config_json_if_exists<GenerationConfig>(
                models_dir, "generation_config.json"
            )
        } {
        m_is_npu = device.find("NPU") != std::string::npos;

        auto properties_copy = properties;
        auto language_model_path = models_dir / "openvino_language_model.xml";
        auto language_model =  utils::singleton_core().read_model(language_model_path, {}, properties_copy);
        auto kv_pos = ov::genai::utils::get_kv_axes_pos(language_model);

        // In case user provided properties per-device
        // {
        //     ov::device::properties("NPU", ...),
        //     ov::device::properties("CPU", ...)
        // }
        auto device_propertes = utils::pop_or_default<ov::AnyMap>(
            properties_copy, ov::device::properties.name(), { }
        );
        // Otherwise, the same properties are used for all models and devices
        auto lm_properties = device_propertes.empty()
            ? properties_copy
            : utils::pop_or_default<ov::AnyMap>(device_propertes, device, {});

        ov::CompiledModel compiled_language_model;
        auto embedder_device = device;
        if (m_is_npu) {
            embedder_device = "CPU";
            utils::KVDesc kv_desc;
            std::tie(compiled_language_model, kv_desc) = utils::compile_decoder_for_npu(
                language_model, lm_properties, kv_pos, language_model_path
            );
            m_max_kv_cache_size = kv_desc.max_prompt_len + kv_desc.min_response_len;
        } else {
            compiled_language_model = utils::singleton_core().compile_model(language_model, device, lm_properties);
        }
        ov::genai::utils::print_compiled_model_properties(compiled_language_model, "VLM language model");

        m_language = compiled_language_model.create_infer_request();

        utils::KVCacheState& kv_cache_state = m_inputs_embedder->get_kv_cache_state();
        kv_cache_state.seq_length_axis = kv_pos.seq_len;
        m_language.get_tensor("attention_mask").set_shape({1, 0});

        auto embedder_properties = device_propertes.empty()
            ? properties_copy
            : utils::pop_or_default<ov::AnyMap>(device_propertes, embedder_device, {});
        m_inputs_embedder = std::make_shared<InputsEmbedder>(models_dir, embedder_device, embedder_properties);
        m_tokenizer = m_inputs_embedder->get_tokenizer();
        m_embedding = m_inputs_embedder->get_embedding_model();

        // If eos_token_id was not provided, take value
        if (m_generation_config.eos_token_id == -1) {
            m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
        }

        m_sampler.set_tokenizer(m_tokenizer);
        m_sampler.set_seed(m_generation_config.rng_seed);
    }


    VLMPipelineImpl(
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap& properties,
        const GenerationConfig& generation_config
    ) :
        m_generation_config{generation_config} {
        m_is_npu = device.find("NPU") != std::string::npos;
        OPENVINO_ASSERT(m_is_npu,
            "VLMPipeline initialization from string isn't supported for NPU device");

        m_inputs_embedder = std::make_shared<InputsEmbedder>(models_map, tokenizer, config_dir_path, device, properties);

        m_tokenizer = m_inputs_embedder->get_tokenizer();
        m_embedding = m_inputs_embedder->get_embedding_model();

        auto m_language_pair = utils::get_model_weights_pair(models_map, "language");
        m_language = utils::singleton_core().compile_model(
            m_language_pair.first, m_language_pair.second, device, properties
        ).create_infer_request();

        m_language.get_tensor("attention_mask").set_shape({1, 0});

        // If eos_token_id was not provided, take value
        if (m_generation_config.eos_token_id == -1) {
            m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
        }

        m_sampler.set_tokenizer(m_tokenizer);
        m_sampler.set_seed(m_generation_config.rng_seed);
    }

    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& rgbs,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        auto generate_start_time = std::chrono::steady_clock::now();
        VLMPerfMetrics perf_metrics;
        auto& raw_counters = perf_metrics.raw_metrics;
        auto& raw_vlm_counters = perf_metrics.vlm_raw_metrics;

        if (!m_is_chat_conversation) {
            m_language.reset_state();
            m_language.get_tensor("attention_mask").set_shape({1, 0});
        }

        // If stop_token_ids were not provided, take value from default m_generation_config
        if (generation_config.stop_token_ids.empty())
            generation_config.stop_token_ids = m_generation_config.stop_token_ids;

        // If eos_token_id was not provided, take value from default m_generation_config
        if (generation_config.eos_token_id == -1)
            generation_config.set_eos_token_id(m_generation_config.eos_token_id);
        generation_config.validate();

        if (m_is_npu) {
            OPENVINO_ASSERT(rgbs.size() == 1u, "Currently only batch size equal to 1 is supported for NPU device!");
            OPENVINO_ASSERT(generation_config.is_greedy_decoding() || generation_config.is_multinomial(),
                "Currently only greedy and multinomial decoding are supported for NPU device!");
            OPENVINO_ASSERT(generation_config.num_return_sequences == 1u,
                "Currently only \"num_return_sequences\" equal to 1 is supported for NPU device!");
        }

        m_inputs_embedder->set_apply_chat_template_status(generation_config.apply_chat_template);

        auto start_get_inputs_embeds = std::chrono::steady_clock::now();
        ov::Tensor inputs_embeds = m_inputs_embedder->get_inputs_embeds(prompt, rgbs, perf_metrics);
        auto end_get_inputs_embeds = std::chrono::steady_clock::now();

        utils::KVCacheState& kv_cache_state = m_inputs_embedder->get_kv_cache_state();
        if (m_is_chat_conversation)
            utils::trim_kv_cache(m_language, kv_cache_state, std::nullopt);

        std::vector<SequenceGroup::Ptr> requests;
        size_t request_id = 0;
        size_t block_size = 1; // not used

        size_t history_size = m_language.get_tensor("attention_mask").get_shape().at(1) - kv_cache_state.num_tokens_to_trim;
        size_t inputs_embeds_size = inputs_embeds.get_shape().at(1);

        std::vector<int64_t> tokenized_history = kv_cache_state.get_state();
        ov::Tensor prompt_ids(ov::element::i64, { history_size + inputs_embeds_size });
        OPENVINO_ASSERT(prompt_ids.get_size() >= tokenized_history.size(), "Prompt ids size is less than tokenized history size");
        std::fill_n(prompt_ids.data<int64_t>(), prompt_ids.get_size(), m_tokenizer.get_pad_token_id());
        std::copy(tokenized_history.begin(), tokenized_history.end(), prompt_ids.data<int64_t>());

        SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(request_id, prompt_ids, generation_config, block_size);
        requests.push_back(sequence_group);

        std::shared_ptr<StreamerBase> streamer_ptr = utils::create_streamer(streamer, m_tokenizer);

        OPENVINO_ASSERT(streamer_ptr == nullptr || generation_config.num_return_sequences == 1 &&
            (generation_config.is_greedy_decoding() || generation_config.is_multinomial()),
            "Currently streaming is possible only with batch size=1 and only for greedy or multinomial decoding");

        ov::Tensor new_atten_mask = ov::Tensor{ov::element::i64, { 1, history_size + inputs_embeds_size }};
        std::fill_n(new_atten_mask.data<int64_t>(), new_atten_mask.get_size(), 1);

        ov::Tensor position_ids;
        std::optional<int64_t> rope_delta;
        std::tie(position_ids, rope_delta) = m_inputs_embedder->get_position_ids(inputs_embeds_size, history_size);

        if (m_sampler.get_seed() != generation_config.rng_seed) {
            m_sampler.set_seed(generation_config.rng_seed);
        }

        ov::genai::utils::GenerationFinishInfo finish_info = ov::genai::get_lm_encoded_results(m_language, inputs_embeds, new_atten_mask, streamer_ptr, m_sampler, requests,
                                                                                               position_ids, kv_cache_state, m_embedding, rope_delta, m_max_kv_cache_size);
        EncodedResults& encoded_result = finish_info.results;

        auto decode_start_time = std::chrono::steady_clock::now();
        VLMDecodedResults decoded;
        for (size_t idx = 0; idx < encoded_result.tokens.size(); ++idx) {
            decoded.texts.push_back(m_tokenizer.decode(encoded_result.tokens.at(idx)));
            decoded.scores.push_back(encoded_result.scores.at(idx));
        }
        auto decode_end_time = std::chrono::steady_clock::now();

        std::string decoded_results = decoded.texts.at(0);
        if (m_is_chat_conversation)
            m_inputs_embedder->update_chat_history(decoded_results, finish_info.streaming_finish_status);
        else
            kv_cache_state.reset_state();

        auto generate_end_time = std::chrono::steady_clock::now();
        decoded.perf_metrics = encoded_result.perf_metrics;

        // Common perf metrics
        auto& res_raw_counters = decoded.perf_metrics.raw_metrics;
        decoded.perf_metrics.num_input_tokens = prompt_ids.get_size();
        decoded.perf_metrics.load_time = this->get_load_time();
        res_raw_counters.generate_durations.emplace_back(PerfMetrics::get_microsec(generate_end_time - generate_start_time));
        res_raw_counters.detokenization_durations.emplace_back(PerfMetrics::get_microsec(decode_end_time - decode_start_time));
        res_raw_counters.tokenization_durations.insert(res_raw_counters.tokenization_durations.end(), raw_counters.tokenization_durations.begin(), raw_counters.tokenization_durations.end());

        // VLM specific perf metrics
        decoded.perf_metrics.vlm_raw_metrics.prepare_embeddings_durations.emplace_back(PerfMetrics::get_microsec(end_get_inputs_embeds - start_get_inputs_embeds));

        // Evaluate statistics
        decoded.perf_metrics.m_evaluated = false;
        decoded.perf_metrics.evaluate_statistics(generate_start_time);

        return decoded;
    }

    void start_chat(const std::string& system_message) override {
        OPENVINO_ASSERT(!m_is_npu, "start_chat() isn't supported in VLMPipeline for NPU device");
        m_is_chat_conversation = true;
        bool have_state = 0 != m_language.get_tensor("attention_mask").get_size();
        if (have_state) {
            // Resetting state may be slow.
            m_language.reset_state();
            // Since if is already introduced, move all resetting here.
            m_language.get_tensor("attention_mask").set_shape({0, 0});
        }
        m_inputs_embedder->start_chat(system_message);
    }

    void finish_chat() override {
        OPENVINO_ASSERT(!m_is_npu, "finish_chat() isn't supported in VLMPipeline for NPU device");
        m_is_chat_conversation = false;
        // Resetting state may be slow.
        m_language.reset_state();
        m_language.get_tensor("attention_mask").set_shape({0, 0});
        // clear all chat history
        m_inputs_embedder->finish_chat();
    }

    Tokenizer get_tokenizer() const override {
        return m_tokenizer;
    }

    void set_chat_template(const std::string& new_template) override {
        OPENVINO_ASSERT(!m_is_chat_conversation, "Chat template cannot be changed once start_chat() is called. Please, finish current chat via finish_chat()");
        m_tokenizer.set_chat_template(new_template);
    }

    GenerationConfig get_generation_config() const override {
        return m_generation_config;
    }

    void set_generation_config(const GenerationConfig& new_config) override {
        int64_t default_eos_token_id = m_generation_config.eos_token_id;
        auto default_stop_token_ids = m_generation_config.stop_token_ids;
        m_generation_config = new_config;

        // If stop_token_ids were not provided, take value from default config
        if (m_generation_config.stop_token_ids.empty())
            m_generation_config.stop_token_ids = default_stop_token_ids;
        // if eos_token_id was not provided in config forward from default config
        if (m_generation_config.eos_token_id == -1)
            m_generation_config.set_eos_token_id(default_eos_token_id);

        m_generation_config.validate();
    }
};

VLMPipeline::VLMPipeline(
    const std::filesystem::path& models_dir,
    const std::string& device,
    const ov::AnyMap& properties
) {
    auto start_time = std::chrono::steady_clock::now();

    if (properties.find(scheduler_config.name()) != properties.end() ||
        properties.find(utils::DRAFT_MODEL_ARG_NAME) != properties.end() ||
        properties.find(prompt_lookup.name()) != properties.end()) {
        auto [plugin_config, scheduler_config] = utils::extract_scheduler_config(properties);
        m_pimpl = std::make_unique<VLMContinuousBatchingAdapter>(models_dir, scheduler_config, device, plugin_config);
    }
    else {
        m_pimpl = std::make_unique<VLMPipelineImpl>(models_dir, device, properties);
    }
    auto stop_time = std::chrono::steady_clock::now();
    m_pimpl->set_load_time(std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count());
}

VLMPipeline::VLMPipeline(
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap& properties,
    const GenerationConfig& generation_config
) {
    auto start_time = std::chrono::steady_clock::now();
    if (properties.find(scheduler_config.name()) != properties.end() ||
        properties.find(utils::DRAFT_MODEL_ARG_NAME) != properties.end() ||
        properties.find(prompt_lookup.name()) != properties.end()) {
        auto [plugin_config, scheduler_config] = utils::extract_scheduler_config(properties);
        m_pimpl = std::make_unique<VLMContinuousBatchingAdapter>(models_map, tokenizer, config_dir_path, scheduler_config, device, plugin_config, generation_config);
    }
    else {
        m_pimpl = std::make_unique<VLMPipelineImpl>(models_map, tokenizer, config_dir_path, device, properties, generation_config);
    }
    auto stop_time = std::chrono::steady_clock::now();
    m_pimpl->set_load_time(std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count());
}

VLMPipeline::~VLMPipeline() = default;

VLMDecodedResults VLMPipeline::generate(
    const std::string& prompt,
    const std::vector<ov::Tensor>& rgbs,
    const GenerationConfig& generation_config,
    const StreamerVariant& streamer
) {
    return m_pimpl->generate(prompt, rgbs, generation_config, streamer);
}

VLMDecodedResults VLMPipeline::generate(
    const std::string& prompt,
    const ov::Tensor& rgb,
    const GenerationConfig& generation_config,
    const StreamerVariant& streamer
) {
    return m_pimpl->generate(prompt, {rgb}, generation_config, streamer);
}

VLMDecodedResults VLMPipeline::generate(
    const std::string& prompt,
    const ov::AnyMap& config_map
) {
    return m_pimpl->generate(prompt, config_map);
}

void VLMPipeline::start_chat(const std::string& system_message) {
    m_pimpl->start_chat(system_message);
}

void VLMPipeline::finish_chat() {
    m_pimpl->finish_chat();
}

void VLMPipeline::set_chat_template(const std::string& new_template) {
    m_pimpl->set_chat_template(new_template);
}

Tokenizer VLMPipeline::get_tokenizer() const {
    return m_pimpl->get_tokenizer();
}

GenerationConfig VLMPipeline::get_generation_config() const {
    return m_pimpl->get_generation_config();
}

void VLMPipeline::set_generation_config(const GenerationConfig& new_config) {
    m_pimpl->set_generation_config(new_config);
}
