// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/visual_language/pipeline.hpp"

#include <optional>
#include <random>

#include "lm_encoding.hpp"
#include "logger.hpp"
#include "lora/helper.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "openvino/runtime/auto/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "sampling/sampler.hpp"
#include "utils.hpp"
#include "visual_language/continuous_batching_adapter.hpp"
#include "visual_language/embedding_model.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "visual_language/pipeline_base.hpp"
#include "visual_language/vision_registry.hpp"
#include "visual_language/vlm_chat_context.hpp"
#include "visual_language/vlm_config.hpp"

using namespace ov::genai;

namespace {
void update_npu_properties(const std::filesystem::path& models_dir, ov::AnyMap& properties) {
    auto vlm_config = utils::from_config_json_if_exists<VLMConfig>(models_dir, "config.json");
    switch (vlm_config.model_type) {
        case VLMModelType::GEMMA3:
            properties.insert({"NPUW_LLM_PREFILL_HINT", "STATIC"});
            break;
        default:
            break;
    }
}

void npu_auto_default_properties(ov::AnyMap& device_properties) {
    auto auto_properties = utils::pop_or_default<ov::AnyMap>(device_properties, "AUTO", {});
    auto_properties.insert(ov::device::priorities("CPU"));
    auto_properties.insert(ov::intel_auto::enable_startup_fallback(false));

    device_properties["AUTO"] = auto_properties;
}

void apply_linear_attention_backend_constraints(
    const std::shared_ptr<ov::Model>& language_model,
    const ov::AnyMap& user_properties,
    std::string& attention_backend
) {
    if (attention_backend != PA_BACKEND || !utils::has_linear_attention_states(language_model)) {
        return;
    }

    if (utils::explicitly_requires_paged_attention(user_properties)
        || user_properties.find("ATTENTION_BACKEND") != user_properties.end()) {
        GENAI_WARN("PA backend does not support models with linear attention states. The model may work incorrectly.");
    } else {
        attention_backend = SDPA_BACKEND;
    }
}

}

class VLMPipeline::VLMPipelineImpl : public VLMPipelineBase{
    // A config to follow for text generation.
    GenerationConfig m_generation_config;
    // A tokenizer encoding a prompt.
    Tokenizer m_tokenizer;
    // A model to compute token embeddings.
    // Input shape: [N, conversation length].
    // Output shape: [1, conversation length, hidden_size].
    EmbeddingsModel::Ptr m_embedding;
    // A language model used to generate a response.
    // Input shapes: inputs_embeds[N, conversation length, hidden_size],
    // position_ids[N, conversation length], beam_idx[N].
    // Output shape: logits[N, conversation length, vocab_size] for NPU,
    // or logits[N, 1, vocab_size] for non-NPU if slice_before_matmul transformation
    // is successfully applied (pattern may not match all model architectures).
    ov::InferRequest m_language;
    // LoRA adapter controller
    std::optional<AdapterController> m_adapter_controller;
    // True if chat mode is activated to save conversation
    // history between generate() calls.
    bool m_is_chat_conversation = false;
    // InputsEmbedder
    std::shared_ptr<InputsEmbedder> m_inputs_embedder;
    // Component for applying sampling to lm outputs
    Sampler m_sampler;
    size_t m_max_prompt_len = std::numeric_limits<size_t>::max();
    size_t m_max_kv_cache_size = std::numeric_limits<size_t>::max();
    bool m_is_npu = false;
    size_t m_image_id = 0;
    size_t m_video_id = 0;
    ChatHistory m_history;

    // if True, full history will be used as prompt on each chat generation
    bool m_use_full_chat_history = false;
    // It stores encoded images, videos and vision count in case when m_use_full_chat_history is true
    std::vector<ov::genai::EncodedImage> m_encoded_images;
    std::vector<ov::genai::EncodedVideo> m_encoded_videos;
    std::vector<std::pair<std::size_t, std::size_t>> m_history_vision_count;  // pair<video count, image count>

    std::string m_system_message;
    std::shared_ptr<VisionRegistry> m_vision_registry;
private:
    void finalize_initialization(
        const std::shared_ptr<ov::Model>& language_model,
        const utils::KVAxesPosition& kv_pos
    ) {
        m_tokenizer = m_inputs_embedder->get_tokenizer();
        m_embedding = m_inputs_embedder->get_embedding_model();

        utils::CacheState& cache_state = m_inputs_embedder->get_cache_state();
        cache_state.set_cache_types(utils::get_cache_types(*language_model));
        cache_state.seq_length_axis = kv_pos.seq_len;

        if (m_generation_config.eos_token_id == -1) {
            m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
        }

        m_sampler.set_tokenizer(m_tokenizer);
        m_sampler.set_seed(m_generation_config.rng_seed);

        m_vision_registry = std::make_shared<VisionRegistry>();

        // NPU does not support history, so use full chat history on each chat iteration.
        // Linear attention forces full KV cache reset, need to provide all image/video embeddings.
        m_use_full_chat_history = m_is_npu || cache_state.has_linear();
    }

    void initialize_from_model_and_dir(
        const std::shared_ptr<ov::Model>& language_model,
        const std::filesystem::path& models_dir,
        const std::string& device,
        const ov::AnyMap& properties
    ) {
        m_is_npu = device.find("NPU") != std::string::npos;

        auto filtered_properties = extract_adapters_from_properties(properties, &m_generation_config.adapters);
        auto& properties_copy = filtered_properties.fork();
        auto kv_pos = ov::genai::utils::get_kv_axes_pos(language_model);

        // In case user provided properties per-device
        // {
        //     ov::device::properties("NPU", ...),
        //     ov::device::properties("CPU", ...)
        // }
        auto device_properties = utils::pop_or_default<ov::AnyMap>(
            properties_copy, ov::device::properties.name(), { }
        );
        // Otherwise, the same properties are used for all models and devices
        auto lm_properties = device_properties.empty()
            ? properties_copy
            : utils::pop_or_default<ov::AnyMap>(device_properties, device, {});

        if (m_generation_config.adapters) {
            m_generation_config.adapters->set_tensor_name_prefix(
                m_generation_config.adapters->get_tensor_name_prefix().value_or("base_model.model.")
            );
            m_adapter_controller = AdapterController(language_model, *m_generation_config.adapters, device);
        }

        ov::CompiledModel compiled_language_model;
        auto embedder_device = device;
        if (m_is_npu) {
            embedder_device = "AUTO";
            utils::KVDesc kv_desc;
            update_npu_properties(models_dir, lm_properties);
            std::tie(compiled_language_model, kv_desc) = utils::compile_decoder_for_npu(language_model, lm_properties, kv_pos);
            m_max_prompt_len = kv_desc.max_prompt_len;
            m_max_kv_cache_size = kv_desc.max_prompt_len + kv_desc.min_response_len;
            npu_auto_default_properties(device_properties);
        } else {
            // Slice-before-matmul rewrites LM logits to be produced only for the last token.
            // After this transformation, the non-NPU path returns logits with seq_len == 1,
            // i.e. [N, 1, vocab_size], not [N, conversation length, vocab_size].
            utils::apply_slice_before_matmul_transformation(language_model);
            compiled_language_model = utils::singleton_core().compile_model(language_model, device, lm_properties);
        }
        ov::genai::utils::print_compiled_model_properties(compiled_language_model, "VLM language model");

        m_language = compiled_language_model.create_infer_request();
        m_language.get_tensor("attention_mask").set_shape({1, 0});

        auto embedder_properties = device_properties.empty()
            ? properties_copy
            : utils::pop_or_default<ov::AnyMap>(device_properties, embedder_device, {});

        m_inputs_embedder = std::make_shared<InputsEmbedder>(models_dir, embedder_device, embedder_properties);

        finalize_initialization(language_model, kv_pos);
    }

    void initialize_from_model_and_map(
        const std::shared_ptr<ov::Model>& language_model,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap& properties
    ) {
        m_is_npu = device.find("NPU") != std::string::npos;
        OPENVINO_ASSERT(!m_is_npu,
            "VLMPipeline initialization from string isn't supported for NPU device");

        auto filtered_properties = extract_adapters_from_properties(properties, &m_generation_config.adapters);
        auto& properties_copy = filtered_properties.fork();

        m_inputs_embedder = std::make_shared<InputsEmbedder>(models_map, tokenizer, config_dir_path, device, properties_copy);

        m_tokenizer = m_inputs_embedder->get_tokenizer();
        m_embedding = m_inputs_embedder->get_embedding_model();

        const auto kv_pos = ov::genai::utils::get_kv_axes_pos(language_model);

        if (m_generation_config.adapters) {
            m_generation_config.adapters->set_tensor_name_prefix(
                m_generation_config.adapters->get_tensor_name_prefix().value_or("base_model.model.")
            );
            m_adapter_controller = AdapterController(language_model, *m_generation_config.adapters, device);
        }

        // Slice-before-matmul rewrites LM logits to be produced only for the last token.
        // After this transformation, default path returns logits with seq_len == 1,
        // i.e. [N, 1, vocab_size], not [N, conversation length, vocab_size].
        utils::apply_slice_before_matmul_transformation(language_model);
        m_language = utils::singleton_core().compile_model(language_model, device, properties_copy
        ).create_infer_request();
        m_language.get_tensor("attention_mask").set_shape({1, 0});
        finalize_initialization(language_model, kv_pos);
    }
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
        auto language_model_path = models_dir / "openvino_language_model.xml";
        auto properties_copy = properties;

        utils::extract_extensions_to_core(properties_copy);
        auto language_model = utils::singleton_core().read_model(language_model_path, {}, properties_copy);
        initialize_from_model_and_dir(language_model, models_dir, device, properties_copy);
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
        auto properties_copy = properties;
        utils::extract_extensions_to_core(properties_copy);
        const auto& language_pair = utils::get_model_weights_pair(models_map, "language");
        auto language_model = utils::singleton_core().read_model(language_pair.first, language_pair.second);
        initialize_from_model_and_map(language_model, models_map, tokenizer, config_dir_path, device, properties_copy);
    }

    VLMPipelineImpl(
        const std::shared_ptr<ov::Model>& language_model,
        const std::filesystem::path& models_dir,
        const std::string& device,
        const ov::AnyMap& properties
    ) :
        m_generation_config{
            utils::from_config_json_if_exists<GenerationConfig>(
                models_dir, "generation_config.json"
            )
        } {
        initialize_from_model_and_dir(language_model, models_dir, device, properties);
    }

    VLMPipelineImpl(
        const std::shared_ptr<ov::Model>& language_model,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap& properties,
        const GenerationConfig& generation_config
    ) :
        m_generation_config{generation_config} {
        initialize_from_model_and_map(language_model, models_map, tokenizer, config_dir_path, device, properties);
    }

    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        return generate(prompt, images, {}, std::move(generation_config), streamer);
    }

    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        return generate(prompt, images, videos, {}, std::move(generation_config), streamer);
    }

    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const std::vector<VideoMetadata>& videos_metadata,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        auto generate_start_time = std::chrono::steady_clock::now();
        VLMPerfMetrics perf_metrics;
        auto& raw_counters = perf_metrics.raw_metrics;

        if (!m_is_chat_conversation) {
            reset_language_state();
            m_language.get_tensor("attention_mask").set_shape({1, 0});
        }

        setup_generation_config(generation_config);

        bool intermediate_remote_tensor = true;
        if (m_is_npu) {
            validate_inputs_for_npu(images, videos, generation_config);
            intermediate_remote_tensor = false;
        }

        m_inputs_embedder->set_vision_token_pruning_config(generation_config.pruning_ratio,
                                                           generation_config.relevance_weight);

        auto encoded_images = m_inputs_embedder->encode_images(images);
        auto encoded_videos = m_inputs_embedder->encode_videos(videos, videos_metadata);
        auto [unified_prompt, image_sequence, video_sequence] = m_inputs_embedder->normalize_prompt(prompt, m_image_id, m_video_id, encoded_images, encoded_videos);

        if (m_is_chat_conversation) {
            m_history.push_back({{"role", "user"}, {"content", unified_prompt}});

            unified_prompt = m_tokenizer.apply_chat_template(m_history, true);

            if (m_use_full_chat_history) {
                m_history_vision_count.emplace_back(std::make_pair(video_sequence.size(), image_sequence.size()));

                m_encoded_images.reserve(m_encoded_images.size() + encoded_images.size());
                m_encoded_images.insert(m_encoded_images.end(), encoded_images.begin(), encoded_images.end());
                image_sequence.resize(m_encoded_images.size());
                std::iota(image_sequence.begin(), image_sequence.end(), 0);
                encoded_images = m_encoded_images;

                m_encoded_videos.reserve(m_encoded_videos.size() + encoded_videos.size());
                m_encoded_videos.insert(m_encoded_videos.end(), encoded_videos.begin(), encoded_videos.end());
                video_sequence.resize(m_encoded_videos.size());
                std::iota(video_sequence.begin(), video_sequence.end(), 0);
                encoded_videos = m_encoded_videos;

                m_inputs_embedder->start_chat(m_system_message);
            } else {
                for (size_t idx = 0; idx < image_sequence.size(); idx++) {
                   image_sequence[idx] -= m_image_id;
                }
                for (size_t idx = 0; idx < video_sequence.size(); idx++) {
                    video_sequence[idx] -= m_video_id;
                }
            }
        } else {
            m_inputs_embedder->set_apply_chat_template_status(generation_config.apply_chat_template);
        }

        auto finish_info = prepare_inputs_and_generate(
            unified_prompt,
            encoded_images,
            encoded_videos,
            image_sequence,
            video_sequence,
            m_history_vision_count,
            generation_config,
            perf_metrics,
            streamer,
            intermediate_remote_tensor
        );

        EncodedResults& encoded_result = finish_info.results;

        auto decode_start_time = std::chrono::steady_clock::now();
        VLMDecodedResults decoded;
        for (size_t idx = 0; idx < encoded_result.tokens.size(); ++idx) {
            decoded.texts.push_back(m_tokenizer.decode(encoded_result.tokens.at(idx)));
            decoded.scores.push_back(encoded_result.scores.at(idx));
        }
        decoded.finish_reasons = encoded_result.finish_reasons;
        auto decode_end_time = std::chrono::steady_clock::now();

        std::string decoded_results = decoded.texts.at(0);
        if (m_is_chat_conversation) {
            // Update m_history with pruned content if pruning is enabled
            if (generation_config.pruning_ratio > 0) {
                auto history_state = ChatHistoryInternalState::get_or_create(m_history, m_vision_registry);
                size_t last_user_idx = history_state->get_last_user_message_index();

                // Replace the original prompt with the pruned prompt after CDPruner
                auto user_message = m_history[last_user_idx];
                std::string original_prompt = user_message["content"].get_string();
                user_message["content"] = m_inputs_embedder->get_last_pruned_prompt(original_prompt);
                m_history[last_user_idx] = user_message;
            }

            m_inputs_embedder->update_chat_history(decoded_results, finish_info.streaming_finish_status);

            if (finish_info.streaming_finish_status != ov::genai::GenerationStatus::CANCEL) {
                // using here images.size() instead of encoded_images.size() since
                // encoded_images could be overriden when m_use_full_chat_history is true
                m_image_id += images.size();
                m_video_id += videos.size();
                // Tail of chat template is missing in KV cache.
                // Find the tail to concatenate it with the next input prompt.
                m_history.push_back({{"role", "assistant"}, {"content", decoded_results}});
            } else {
                m_history.pop_back();
                if (m_use_full_chat_history) {
                    OPENVINO_ASSERT(images.size() <= m_encoded_images.size(), "Number of images to remove is more than stored images!");
                    m_encoded_images.resize(m_encoded_images.size() - images.size());

                    OPENVINO_ASSERT(videos.size() <= m_encoded_videos.size(), "Number of videos to remove is more than stored videos!");
                    m_encoded_videos.resize(m_encoded_videos.size() - videos.size());

                    m_history_vision_count.pop_back();
                }
            }
        } else {
            utils::CacheState& cache_state = m_inputs_embedder->get_cache_state();
            cache_state.reset_state();
        }

        if (!(m_is_chat_conversation && m_use_full_chat_history)) {
            m_encoded_images.clear();
            m_encoded_videos.clear();
            m_history_vision_count.clear();
        }

        auto generate_end_time = std::chrono::steady_clock::now();
        decoded.perf_metrics = encoded_result.perf_metrics;

        // Common perf metrics
        auto& res_raw_counters = decoded.perf_metrics.raw_metrics;
        decoded.perf_metrics.num_input_tokens = perf_metrics.num_input_tokens;
        decoded.perf_metrics.load_time = this->get_load_time();
        res_raw_counters.generate_durations.emplace_back(PerfMetrics::get_microsec(generate_end_time - generate_start_time));
        res_raw_counters.detokenization_durations.emplace_back(PerfMetrics::get_microsec(decode_end_time - decode_start_time));
        res_raw_counters.tokenization_durations.insert(res_raw_counters.tokenization_durations.end(), raw_counters.tokenization_durations.begin(), raw_counters.tokenization_durations.end());

        // VLM specific perf metrics
        decoded.perf_metrics.vlm_raw_metrics.prepare_embeddings_durations.insert(
            decoded.perf_metrics.vlm_raw_metrics.prepare_embeddings_durations.end(),
            perf_metrics.vlm_raw_metrics.prepare_embeddings_durations.begin(),
            perf_metrics.vlm_raw_metrics.prepare_embeddings_durations.end()
        );

        // Evaluate statistics
        decoded.perf_metrics.m_evaluated = false;
        decoded.perf_metrics.evaluate_statistics(generate_start_time);

        return decoded;
    }

    VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        return generate(history, images, {}, std::move(generation_config), streamer);
    }

    VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        return generate(history, images, videos, {}, std::move(generation_config), streamer);
    }

    VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const std::vector<VideoMetadata>& videos_metadata,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        auto generate_start_time = std::chrono::steady_clock::now();
        VLMPerfMetrics perf_metrics;
        auto& raw_counters = perf_metrics.raw_metrics;

        m_is_chat_conversation = true;

        setup_generation_config(generation_config);

        bool intermediate_remote_tensor = true;
        if (m_is_npu) {
            validate_inputs_for_npu(images, videos, generation_config);
            intermediate_remote_tensor = false;
        }

        m_inputs_embedder->set_vision_token_pruning_config(generation_config.pruning_ratio,
                                                           generation_config.relevance_weight);

        VLMChatContext chat_context(history, m_vision_registry, *m_inputs_embedder);

        auto processed_chat_data = chat_context.process(images, videos, videos_metadata);

        bool use_full_history = processed_chat_data.needs_kv_cache_reset || m_use_full_chat_history;

        if (use_full_history) {
            reset_language_state();
            m_language.get_tensor("attention_mask").set_shape({1, 0});
            m_inputs_embedder->start_chat("");
        }

        std::string templated_history = m_tokenizer.apply_chat_template(
            processed_chat_data.normalized_history,
            true
        );

        ov::genai::utils::GenerationFinishInfo generation_finish_info;

        const auto& images_embeds = use_full_history
            ? processed_chat_data.encoded_images
            : processed_chat_data.new_encoded_images;
        const auto& videos_embeds = use_full_history
            ? processed_chat_data.encoded_videos
            : processed_chat_data.new_encoded_videos;
        const auto& image_seq = use_full_history
            ? processed_chat_data.image_sequence
            : processed_chat_data.new_image_sequence;
        const auto& video_seq = use_full_history
            ? processed_chat_data.video_sequence
            : processed_chat_data.new_video_sequence;
        const auto& vision_counts = use_full_history
            ? processed_chat_data.vision_counts
            : std::vector<std::pair<std::size_t, std::size_t>>{ {video_seq.size(), image_seq.size()} };

        generation_finish_info = prepare_inputs_and_generate(
            templated_history,
            images_embeds,
            videos_embeds,
            image_seq,
            video_seq,
            vision_counts,
            generation_config,
            perf_metrics,
            streamer,
            intermediate_remote_tensor
        );

        EncodedResults& encoded_result = generation_finish_info.results;
        
        // Update pruned content after generation (CDPruner has run during prepare_inputs_and_generate)
        if (generation_config.pruning_ratio > 0) {
            chat_context.apply_pruning_to_last_message();
        }

        auto decode_start_time = std::chrono::steady_clock::now();
        VLMDecodedResults decoded;
        for (size_t idx = 0; idx < encoded_result.tokens.size(); ++idx) {
            decoded.texts.push_back(m_tokenizer.decode(encoded_result.tokens.at(idx)));
            decoded.scores.push_back(encoded_result.scores.at(idx));
        }
        decoded.finish_reasons = encoded_result.finish_reasons;
        auto decode_end_time = std::chrono::steady_clock::now();

        std::string decoded_text = decoded.texts.at(0);

        m_inputs_embedder->update_chat_history(decoded_text, generation_finish_info.streaming_finish_status);

        if (generation_finish_info.streaming_finish_status == ov::genai::GenerationStatus::CANCEL) {
            chat_context.rollback();
        }

        auto generate_end_time = std::chrono::steady_clock::now();
        decoded.perf_metrics = encoded_result.perf_metrics;

        // Common perf metrics
        auto& res_raw_counters = decoded.perf_metrics.raw_metrics;
        decoded.perf_metrics.num_input_tokens = perf_metrics.num_input_tokens;
        decoded.perf_metrics.load_time = this->get_load_time();
        res_raw_counters.generate_durations.emplace_back(PerfMetrics::get_microsec(generate_end_time - generate_start_time));
        res_raw_counters.detokenization_durations.emplace_back(PerfMetrics::get_microsec(decode_end_time - decode_start_time));
        res_raw_counters.tokenization_durations.insert(res_raw_counters.tokenization_durations.end(), raw_counters.tokenization_durations.begin(), raw_counters.tokenization_durations.end());

        // VLM specific perf metrics
        decoded.perf_metrics.vlm_raw_metrics.prepare_embeddings_durations.insert(
            decoded.perf_metrics.vlm_raw_metrics.prepare_embeddings_durations.end(),
            perf_metrics.vlm_raw_metrics.prepare_embeddings_durations.begin(),
            perf_metrics.vlm_raw_metrics.prepare_embeddings_durations.end()
        );

        // Evaluate statistics
        decoded.perf_metrics.m_evaluated = false;
        decoded.perf_metrics.evaluate_statistics(generate_start_time);

        return decoded;
    }

    void start_chat(const std::string& system_message) override {
        m_is_chat_conversation = true;
        m_system_message = system_message;
        m_inputs_embedder->start_chat(m_system_message);
        if (system_message.empty()) {
            return;
        }
        m_history.clear();
        m_history.push_back({{"role", "system"}, {"content", m_system_message}});
    }

    void finish_chat() override {
        m_is_chat_conversation = false;
        m_image_id = 0;
        m_video_id = 0;
        // Resetting state may be slow.
        reset_language_state();
        m_language.get_tensor("attention_mask").set_shape({0, 0});
        // clear all chat history
        m_inputs_embedder->finish_chat();
        m_history.clear();
        m_encoded_images.clear();
        m_encoded_videos.clear();
        m_history_vision_count.clear();
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

private:
    void reset_language_state() {
        if (m_adapter_controller) {
            // Preserve adapter-owned state variables
            for (auto& state : m_language.query_state()) {
                if (!m_adapter_controller->has_state_name(state.get_name())) {
                    state.reset();
                }
            }
        } else {
            m_language.reset_state();
        }
    }

    void setup_generation_config(GenerationConfig& generation_config) {
        // If stop_token_ids were not provided, take value from default m_generation_config
        if (generation_config.stop_token_ids.empty())
            generation_config.stop_token_ids = m_generation_config.stop_token_ids;

        // If eos_token_id was not provided, take value from default m_generation_config
        if (generation_config.eos_token_id == -1)
            generation_config.set_eos_token_id(m_generation_config.eos_token_id);
        generation_config.validate();
    }

    void validate_inputs_for_npu(
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const GenerationConfig& generation_config
    ) {
        OPENVINO_ASSERT(generation_config.is_greedy_decoding() || generation_config.is_multinomial(),
            "Currently only greedy and multinomial decoding are supported for NPU device!");
        OPENVINO_ASSERT(generation_config.num_return_sequences == 1u,
            "Currently only \"num_return_sequences\" equal to 1 is supported for NPU device!");
        if (m_is_chat_conversation)
            OPENVINO_ASSERT(videos.empty(), "Chat mode is currently not supported with video input for NPU device!");
    }

    ov::genai::utils::GenerationFinishInfo prepare_inputs_and_generate(
        const std::string& unified_prompt,
        const std::vector<ov::genai::EncodedImage>& encoded_images,
        const std::vector<ov::genai::EncodedVideo>& encoded_videos,
        const std::vector<size_t>& image_sequence,
        const std::vector<size_t>& video_sequence,
        const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count,
        GenerationConfig& generation_config,
        VLMPerfMetrics& perf_metrics,
        const StreamerVariant& streamer,
        const bool use_intermediate_remote_tensor
    ) {
        ov::Tensor inputs_embeds;
        std::optional<ov::Tensor> token_type_ids;
        bool recalculate_merged_embeddings = encoded_images.size() > 0 || encoded_videos.size() > 0;

        auto start_get_inputs_embeds = std::chrono::steady_clock::now();
        if (m_inputs_embedder->has_token_type_ids()) {
            std::tie(inputs_embeds, token_type_ids) =
                m_inputs_embedder->get_inputs_embeds_with_token_type_ids(
                    unified_prompt,
                    encoded_images,
                    encoded_videos,
                    perf_metrics,
                    recalculate_merged_embeddings,
                    image_sequence,
                    video_sequence,
                    history_vision_count
                );
        } else {
            inputs_embeds = m_inputs_embedder->get_inputs_embeds(
                unified_prompt,
                encoded_images, 
                encoded_videos,
                perf_metrics,
                recalculate_merged_embeddings,
                image_sequence,
                video_sequence,
                history_vision_count
            );
        }
        auto end_get_inputs_embeds = std::chrono::steady_clock::now();
        perf_metrics.vlm_raw_metrics.prepare_embeddings_durations.emplace_back(PerfMetrics::get_microsec(end_get_inputs_embeds - start_get_inputs_embeds));

        if (m_is_npu) {
            // Prefill model in NPU is reshaped to NPUW_LLM_MAX_PROMPT_LEN x NPUW_LLM_MAX_PROMPT_LEN
            OPENVINO_ASSERT(inputs_embeds.get_shape().at(1) <= m_max_prompt_len,
                "VLM pipeline on NPU may only process input embeddings up to ", m_max_prompt_len,
                " tokens. ", inputs_embeds.get_shape().at(1), " is passed.\nSet the \"MAX_PROMPT_LEN\""
                " config option to increase the limit.");
        }

        utils::CacheState& cache_state = m_inputs_embedder->get_cache_state();

        if (m_is_chat_conversation) {
            if (m_use_full_chat_history) {
                cache_state.reset_state();
                m_language.reset_state();
                m_language.get_tensor("attention_mask").set_shape({1, 0});
            } else {
                bool needs_full_reset = cache_state.needs_reset();
                utils::trim_kv_cache(m_language, cache_state, m_adapter_controller);
                if (needs_full_reset) {
                    m_language.get_tensor("attention_mask").set_shape({1, 0});
                }
            }
        }

        if (m_adapter_controller) {
            m_adapter_controller->apply(m_language, generation_config.adapters);
        }

        std::vector<SequenceGroup::Ptr> requests;
        size_t request_id = 0;
        size_t block_size = 1; // not used

        const size_t history_size = m_language.get_tensor("attention_mask").get_shape().at(1) - cache_state.num_tokens_to_trim;
        const size_t inputs_embeds_size = inputs_embeds.get_shape().at(1);

        std::vector<int64_t> tokenized_history = cache_state.get_state();
        ov::Tensor prompt_ids(ov::element::i64, { history_size + inputs_embeds_size });
        OPENVINO_ASSERT(prompt_ids.get_size() >= tokenized_history.size(), "Prompt ids size is less than tokenized history size");
        std::fill_n(prompt_ids.data<int64_t>(), prompt_ids.get_size(), m_tokenizer.get_pad_token_id());
        std::copy(tokenized_history.begin(), tokenized_history.end(), prompt_ids.data<int64_t>());

        // Update perf metrics with num_input_tokens
        perf_metrics.num_input_tokens = prompt_ids.get_size();

        SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(request_id, prompt_ids, generation_config, block_size);
        requests.push_back(std::move(sequence_group));

        std::shared_ptr<StreamerBase> streamer_ptr = utils::create_streamer(streamer, m_tokenizer);

        OPENVINO_ASSERT(streamer_ptr == nullptr || generation_config.num_return_sequences == 1 &&
            (generation_config.is_greedy_decoding() || generation_config.is_multinomial()),
            "Currently streaming is possible only with batch size=1 and only for greedy or multinomial decoding");

        ov::Tensor new_atten_mask = ov::Tensor{ov::element::i64, { 1, history_size + inputs_embeds_size }};
        std::fill_n(new_atten_mask.data<int64_t>(), new_atten_mask.get_size(), 1);

        ov::Tensor position_ids;
        std::optional<int64_t> rope_delta;
        std::tie(position_ids, rope_delta) = m_inputs_embedder->get_position_ids(inputs_embeds_size, history_size);

        const auto& lm_extra_inputs = m_inputs_embedder->get_lm_extra_inputs();

        auto per_layer_callback = m_inputs_embedder->get_per_layer_embeddings_callback();

        if (m_sampler.get_seed() != generation_config.rng_seed) {
            m_sampler.set_seed(generation_config.rng_seed);
        }

        return ov::genai::get_lm_encoded_results(m_language,
                                                 inputs_embeds,
                                                 new_atten_mask,
                                                 streamer_ptr,
                                                 m_sampler,
                                                 std::move(requests),
                                                 position_ids,
                                                 token_type_ids,
                                                 cache_state,
                                                 m_embedding,
                                                 rope_delta,
                                                 m_max_kv_cache_size,
                                                 use_intermediate_remote_tensor,
                                                 lm_extra_inputs,
                                                 std::move(per_layer_callback));
    }
};

bool requires_sdpa(const std::filesystem::path& models_dir) {
    auto vlm_config = utils::from_config_json_if_exists<VLMConfig>(models_dir, "config.json");
    // TODO: remove it when GEMMA3 ticket-171180 is fixed
    return vlm_config.model_type == VLMModelType::GEMMA3
        // ticket: 183493
        || vlm_config.model_type == VLMModelType::GEMMA4
        // TODO: remove Qwen3.5 limitation once ticket-183791 is fixed
        || vlm_config.model_type == VLMModelType::QWEN3_5
        || vlm_config.model_type == VLMModelType::QWEN3_5_MOE;
}

VLMPipeline::VLMPipeline(
    const std::filesystem::path& models_dir,
    const std::string& device,
    const ov::AnyMap& user_properties
) {
    auto start_time = std::chrono::steady_clock::now();

    auto [properties, attention_backend] = utils::extract_attention_backend(user_properties);
    utils::clear_false_prompt_lookup_from_config(properties);
    if (device == "NPU") {
        auto it = properties.find("scheduler_config");
        OPENVINO_ASSERT(it == properties.end(), "scheduler_config should be removed for VLMPipeline initialization");
        m_pimpl = std::make_unique<VLMPipelineImpl>(models_dir, device, properties);
    } else {
        utils::extract_extensions_to_core(properties);
        auto language_model_path = models_dir / "openvino_language_model.xml";
        auto language_model = utils::singleton_core().read_model(language_model_path, {}, properties);
        apply_linear_attention_backend_constraints(language_model, user_properties, attention_backend);

        // If CB is invoked explicitly, create CB adapter as is and re-throw in case if internal issues
        if (utils::explicitly_requires_paged_attention(user_properties)) {
            auto [plugin_properties, scheduler_config] = utils::extract_scheduler_config(properties, utils::get_latency_oriented_scheduler_config());
            m_pimpl = std::make_unique<VLMContinuousBatchingAdapter>(language_model, models_dir, scheduler_config, device, plugin_properties);
        } else if (attention_backend == PA_BACKEND && !requires_sdpa(models_dir)) {
            // try to call CB adapter one more time, but with safe guard to silent exception
            try {
                auto [plugin_properties, scheduler_config] = utils::extract_scheduler_config(properties, utils::get_latency_oriented_scheduler_config());
                // we need use CB only for x86 and arm64, as for other architectures like risc-v we can create Paged Attention based model
                // but cannot perform its inference later
    #if defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_ARM64)
                m_pimpl = std::make_unique<VLMContinuousBatchingAdapter>(language_model, models_dir, scheduler_config, device, plugin_properties);
#endif
            } catch (ov::Exception&) {
                // ignore exceptions from PA
            }
        }

        if (m_pimpl == nullptr) {
            m_pimpl = std::make_unique<VLMPipelineImpl>(language_model, models_dir, device, properties);
        }
    }

    auto stop_time = std::chrono::steady_clock::now();
    m_pimpl->set_load_time(std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count());
}

VLMPipeline::VLMPipeline(
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap& user_properties,
    const GenerationConfig& generation_config
) {
    auto start_time = std::chrono::steady_clock::now();

    auto [properties, attention_backend] = utils::extract_attention_backend(user_properties);
    utils::clear_false_prompt_lookup_from_config(properties);
    if (device == "NPU") {
        auto it = properties.find("scheduler_config");
        OPENVINO_ASSERT(it == properties.end(), "scheduler_config should be removed for VLMPipeline initialization");
        m_pimpl = std::make_unique<VLMPipelineImpl>(models_map, tokenizer, config_dir_path, device, properties, generation_config);
    } else {
        utils::extract_extensions_to_core(properties);
        const auto& [model_str, weights] = utils::get_model_weights_pair(models_map, "language");
        auto language_model = utils::singleton_core().read_model(model_str, weights);
        apply_linear_attention_backend_constraints(language_model, user_properties, attention_backend);

        // If CB is invoked explicitly, create CB adapter as is and re-throw in case if internal issues
        if (utils::explicitly_requires_paged_attention(user_properties)) {
            auto [plugin_properties, scheduler_config] = utils::extract_scheduler_config(properties, utils::get_latency_oriented_scheduler_config());
            m_pimpl = std::make_unique<VLMContinuousBatchingAdapter>(language_model, models_map, tokenizer, config_dir_path, scheduler_config, device, plugin_properties, generation_config);
        } else if (attention_backend == PA_BACKEND && !requires_sdpa(config_dir_path)) {
            // try to call CB adapter one more time, but with safe guard to silent exception
            try {
                auto [plugin_properties, scheduler_config] = utils::extract_scheduler_config(properties, utils::get_latency_oriented_scheduler_config());
                // we need use CB only for x86 and arm64, as for other architectures like risc-v we can create Paged Attention based model
                // but cannot perform its inference later
    #if defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_ARM64)
                m_pimpl = std::make_unique<VLMContinuousBatchingAdapter>(language_model, models_map, tokenizer, config_dir_path, scheduler_config, device, plugin_properties, generation_config);
    #endif
            } catch (ov::Exception&) {
                // ignore exceptions from PA
            }
        }

        if (m_pimpl == nullptr) {
            m_pimpl = std::make_unique<VLMPipelineImpl>(language_model, models_map, tokenizer, config_dir_path, device, properties, generation_config);
        }

    }

    auto stop_time = std::chrono::steady_clock::now();
    m_pimpl->set_load_time(std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count());
}

VLMPipeline::~VLMPipeline() = default;

VLMDecodedResults VLMPipeline::generate(
    const std::string& prompt,
    const std::vector<ov::Tensor>& images,
    const std::vector<ov::Tensor>& videos,
    const GenerationConfig& generation_config,
    const StreamerVariant& streamer
) {
    return m_pimpl->generate(prompt, images, videos, generation_config, streamer);
}

VLMDecodedResults VLMPipeline::generate(
    const std::string& prompt,
    const std::vector<ov::Tensor>& images,
    const GenerationConfig& generation_config,
    const StreamerVariant& streamer
) {
    return m_pimpl->generate(prompt, images, generation_config, streamer);
}

VLMDecodedResults VLMPipeline::generate(
    const std::string& prompt,
    const ov::Tensor& image,
    const GenerationConfig& generation_config,
    const StreamerVariant& streamer
) {
    return m_pimpl->generate(prompt, {image}, generation_config, streamer);
}

VLMDecodedResults VLMPipeline::generate(
    const std::string& prompt,
    const ov::AnyMap& config_map
) {
    return m_pimpl->generate(prompt, config_map);
}

VLMDecodedResults VLMPipeline::generate(
    const ChatHistory& history,
    const std::vector<ov::Tensor>& images,
    const std::vector<ov::Tensor>& videos,
    const GenerationConfig& generation_config,
    const StreamerVariant& streamer
) {
    return m_pimpl->generate(history, images, videos, generation_config, streamer);
}

VLMDecodedResults VLMPipeline::generate(
    const ChatHistory& history,
    const std::vector<ov::Tensor>& images,
    const GenerationConfig& generation_config,
    const StreamerVariant& streamer
) {
    return m_pimpl->generate(history, images, generation_config, streamer);
}

VLMDecodedResults VLMPipeline::generate(
    const ChatHistory& history,
    const ov::Tensor& image,
    const GenerationConfig& generation_config,
    const StreamerVariant& streamer
) {
    return m_pimpl->generate(history, {image}, generation_config, streamer);
}

VLMDecodedResults VLMPipeline::generate(
    const ChatHistory& history,
    const ov::AnyMap& config_map
) {
    return m_pimpl->generate(history, config_map);
}

void VLMPipeline::start_chat(const std::string& system_message) {
    m_pimpl->finish_chat();
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
